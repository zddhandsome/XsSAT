"""
XsSAT 多卡分布式训练脚本 (PyTorch DDP)

使用方式:
    # 单机 6 卡
    torchrun --nproc_per_node=6 train_ddp.py --config config/sr40_100.yaml

    # 单机 8 卡
    torchrun --nproc_per_node=8 train_ddp.py --config config/sr40_100.yaml

    # 指定 GPU
    CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=4 train_ddp.py --config config/sr40_100.yaml

    # 从检查点恢复 (兼容单卡 train.py 保存的 checkpoint)
    torchrun --nproc_per_node=4 train_ddp.py --config config/sr40_100.yaml --checkpoint checkpoints/sr40_100/checkpoint_latest.pt

    # 手动指定梯度累积步数 (默认自动按卡数缩减)
    torchrun --nproc_per_node=4 train_ddp.py --config config/sr40_100.yaml --accum_steps 4

与单卡 train.py 的兼容性:
    - checkpoint 格式完全兼容，可互相加载
    - 配置文件通用，无需修改 yaml
    - accumulation_steps 自动按卡数缩减以保持相同有效 batch size
"""

import os
import sys
import math
import yaml
import argparse
import random
from dataclasses import asdict
from pathlib import Path
from typing import Dict, Optional, Any

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.cuda.amp import autocast, GradScaler
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.models import XsSAT
from src.data import (
    SATDataset, PackedSATDataset, SATAugmentation,
    OnlineSATDataset, MixedSATDataset,
)
from src.data.dataset import SATCollator
from src.training.trainer import TrainerConfig
from src.training.logger import Logger
from src.losses import MultiTaskLoss


# ======================== 分布式工具函数 ========================

def is_main_process():
    return dist.get_rank() == 0


def print_rank0(*args, **kwargs):
    if is_main_process():
        print(*args, **kwargs)


def reduce_scalar(value, device, world_size):
    """AllReduce 一个标量值，返回全局平均。"""
    if isinstance(value, torch.Tensor):
        t = value.clone().detach().to(device)
    else:
        t = torch.tensor(value, dtype=torch.float64, device=device)
    dist.all_reduce(t, op=dist.ReduceOp.SUM)
    return (t / world_size).item()


def reduce_sum(value, device):
    """AllReduce 求和 (不除以 world_size)。"""
    t = torch.tensor(value, dtype=torch.float64, device=device)
    dist.all_reduce(t, op=dist.ReduceOp.SUM)
    return t.item()


# ======================== DDP DataLoader ========================

def create_ddp_dataloader(
    dataset,
    batch_size: int,
    num_workers: int,
    pin_memory: bool,
    is_train: bool = True,
    worker_init_fn=None,
    seed: int = 42,
) -> DataLoader:
    """创建支持 DDP 的 DataLoader (DistributedSampler)。"""
    max_clauses = getattr(dataset, 'max_clauses', 550)
    max_vars = getattr(dataset, 'max_vars', 100)
    collator = SATCollator(max_clauses, max_vars)

    sampler = DistributedSampler(
        dataset,
        num_replicas=dist.get_world_size(),
        rank=dist.get_rank(),
        shuffle=is_train,
        seed=seed,
    )

    return DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=collator,
        worker_init_fn=worker_init_fn,
        drop_last=is_train,
        persistent_workers=(num_workers > 0),
    )


# ======================== DDP Trainer ========================

class DDPTrainer:
    """
    XsSAT 分布式训练器

    与单卡 Trainer 的核心区别:
    - 模型包裹 DistributedDataParallel，梯度自动跨卡同步
    - Loss / metrics 通过 AllReduce 聚合后记录，保证日志准确
    - 只在 rank 0 写日志、TensorBoard、保存 checkpoint
    - Checkpoint 保存 module.state_dict()，与单卡格式兼容
    """

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader],
        config: TrainerConfig,
        local_rank: int,
    ):
        self.config = config
        self.local_rank = local_rank
        self.rank = dist.get_rank()
        self.world_size = dist.get_world_size()
        self.is_main = (self.rank == 0)

        # 设备
        self.device = torch.device(f'cuda:{local_rank}')

        # 模型 → DDP
        self.model = model.to(self.device)
        self.model = DDP(
            self.model,
            device_ids=[local_rank],
            output_device=local_rank,
            # 通过图连接的零损失回退保证每次迭代都能追踪到所有可训练参数，
            # 因此不再需要 DDP 额外遍历 autograd 图去检测 unused params。
            find_unused_parameters=False,
        )

        self.train_loader = train_loader
        self.val_loader = val_loader

        # 损失函数
        self.criterion = MultiTaskLoss(**self.config.loss_config)

        # 优化器 & 调度器
        self.optimizer = self._create_optimizer()
        self.total_steps = (
            len(train_loader) * self.config.num_epochs
            // self.config.gradient_accumulation_steps
        )
        self.scheduler = self._create_scheduler()

        # 混合精度
        if self.config.use_amp:
            self.scaler = GradScaler()
            self.amp_dtype = (
                torch.float16 if self.config.amp_dtype == 'float16' else torch.bfloat16
            )
        else:
            self.scaler = None
            self.amp_dtype = torch.float32

        # 日志 (仅 rank 0)
        if self.is_main:
            Path(self.config.save_dir).mkdir(parents=True, exist_ok=True)
            Path(self.config.log_dir).mkdir(parents=True, exist_ok=True)
            self.logger = Logger(
                save_dir=self.config.log_dir,
                config=asdict(self.config),
            )
        else:
            self.logger = None

        # 训练状态
        self.current_epoch = 0
        self.global_step = 0
        self.best_metric = float('-inf')
        self.patience_counter = 0
        self.training_history = []

    # ---------- 优化器 / 调度器 ----------

    def _create_optimizer(self) -> optim.Optimizer:
        decay_params = []
        no_decay_params = []
        for name, param in self.model.named_parameters():
            if not param.requires_grad:
                continue
            if 'bias' in name or 'norm' in name or 'LayerNorm' in name:
                no_decay_params.append(param)
            else:
                decay_params.append(param)

        param_groups = [
            {'params': decay_params, 'weight_decay': self.config.weight_decay},
            {'params': no_decay_params, 'weight_decay': 0.0},
        ]

        if self.config.optimizer == 'adamw':
            return optim.AdamW(
                param_groups,
                lr=self.config.learning_rate,
                betas=(self.config.adam_beta1, self.config.adam_beta2),
                eps=self.config.adam_epsilon,
            )
        elif self.config.optimizer == 'adam':
            return optim.Adam(
                param_groups,
                lr=self.config.learning_rate,
                betas=(self.config.adam_beta1, self.config.adam_beta2),
                eps=self.config.adam_epsilon,
            )
        else:
            raise ValueError(f"Unknown optimizer: {self.config.optimizer}")

    def _create_scheduler(self) -> optim.lr_scheduler._LRScheduler:
        warmup_steps = self.config.warmup_steps
        total_steps = self.total_steps

        if self.config.lr_scheduler == 'cosine':
            def lr_lambda(step):
                if step < warmup_steps:
                    return step / max(warmup_steps, 1)
                progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
                return self.config.lr_min_ratio + 0.5 * (1 - self.config.lr_min_ratio) * (
                    1 + math.cos(progress * math.pi)
                )
            return optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda)

        elif self.config.lr_scheduler == 'constant':
            def lr_lambda(step):
                if step < warmup_steps:
                    return step / max(warmup_steps, 1)
                return 1.0
            return optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda)

        else:
            raise ValueError(f"Unknown scheduler: {self.config.lr_scheduler}")

    # ---------- 工具方法 ----------

    def _move_to_device(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        return {
            k: v.to(self.device, non_blocking=True) if torch.is_tensor(v) else v
            for k, v in batch.items()
        }

    # ---------- 训练主循环 ----------

    def train(self) -> list:
        if self.is_main:
            self.logger.write("Starting DDP training...\n")
            self.logger.write(f"  World size: {self.world_size} GPUs\n")
            self.logger.write(f"  Per-GPU batch size: {self.config.batch_size}\n")
            self.logger.write(f"  Accumulation steps: {self.config.gradient_accumulation_steps}\n")
            eff = (self.config.batch_size * self.world_size
                   * self.config.gradient_accumulation_steps)
            self.logger.write(f"  Effective batch size: {eff}\n")

        val_metrics = {}
        for epoch in range(self.current_epoch, self.config.num_epochs):
            self.current_epoch = epoch

            # 每个 epoch 设置 sampler 以保证各卡数据不同
            if hasattr(self.train_loader.sampler, 'set_epoch'):
                self.train_loader.sampler.set_epoch(epoch)

            train_metrics = self._train_epoch()

            val_metrics = {}
            if self.val_loader and (epoch + 1) % self.config.eval_every_n_epochs == 0:
                if hasattr(self.val_loader.sampler, 'set_epoch'):
                    self.val_loader.sampler.set_epoch(epoch)
                val_metrics = self._validate()

            if self.is_main:
                self._log_epoch(epoch, train_metrics, val_metrics)
                self._maybe_save_best_checkpoint(epoch, val_metrics)
                if (epoch + 1) % self.config.save_every_n_epochs == 0:
                    self._save_checkpoint(epoch, val_metrics)

            if self._check_early_stopping(val_metrics):
                if self.is_main:
                    self.logger.write(f"Early stopping at epoch {epoch}\n")
                break

            dist.barrier()

        if self.is_main:
            self._save_checkpoint(self.current_epoch, val_metrics, is_final=True)
            self.logger.write("Training completed!\n")
            self.logger.close()

        return self.training_history

    def _train_epoch(self) -> Dict[str, float]:
        self.model.train()

        epoch_loss = 0.0
        epoch_metrics = {
            'sat_loss': 0.0,
            'core_loss': 0.0,
            'negation_loss': 0.0,
            'sat_logit_mean': 0.0,
            'clause_vote_mean': 0.0,
            'delta_logit_mean': 0.0,
            'lambda_delta': 0.0,
        }
        num_batches = 0

        loader = self.train_loader
        if self.is_main:
            loader = tqdm(loader, desc=f"Epoch {self.current_epoch}")

        for batch_idx, batch in enumerate(loader):
            batch = self._move_to_device(batch)

            with autocast(enabled=self.config.use_amp, dtype=self.amp_dtype):
                outputs = self.model(
                    batch['vsm'],
                    batch.get('clause_mask'),
                    batch.get('var_mask'),
                )
                loss, loss_dict = self.criterion(outputs, batch)
                loss = loss / self.config.gradient_accumulation_steps
                output_stats = self._extract_output_stats(outputs)

            if self.scaler:
                self.scaler.scale(loss).backward()
            else:
                loss.backward()

            if (batch_idx + 1) % self.config.gradient_accumulation_steps == 0:
                if self.scaler:
                    self.scaler.unscale_(self.optimizer)
                    nn.utils.clip_grad_norm_(
                        self.model.parameters(), self.config.max_grad_norm
                    )
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    nn.utils.clip_grad_norm_(
                        self.model.parameters(), self.config.max_grad_norm
                    )
                    self.optimizer.step()

                self.scheduler.step()
                self.optimizer.zero_grad(set_to_none=True)
                self.global_step += 1

                # 周期性记录 step 级日志
                # 注: step 级只记录 rank 0 的本地 loss (避免 AllReduce 死锁)
                #     epoch 级会做精确的全局聚合
                if self.is_main and self.global_step % self.config.log_every_n_steps == 0:
                    local = {}
                    for k, v in loss_dict.items():
                        local[k] = v.item() if torch.is_tensor(v) else v
                    self._log_step(local)

            epoch_loss += loss.item() * self.config.gradient_accumulation_steps
            for k, v in loss_dict.items():
                if k in epoch_metrics:
                    epoch_metrics[k] += v.item() if torch.is_tensor(v) else v
            for k, v in output_stats.items():
                if k in epoch_metrics:
                    epoch_metrics[k] += v
            num_batches += 1

            if self.is_main and hasattr(loader, 'set_postfix'):
                loader.set_postfix({
                    'loss': f"{epoch_loss / num_batches:.4f}",
                    'lr': f"{self.scheduler.get_last_lr()[0]:.2e}",
                })

        # Epoch 结束: 批量 AllReduce 所有 metrics (1次通信替代多次)
        epoch_metrics = {k: v / max(num_batches, 1) for k, v in epoch_metrics.items()}
        epoch_metrics['total_loss'] = epoch_loss / max(num_batches, 1)

        keys = sorted(epoch_metrics.keys())
        vals = torch.tensor([epoch_metrics[k] for k in keys],
                            dtype=torch.float64, device=self.device)
        dist.all_reduce(vals, op=dist.ReduceOp.SUM)
        vals /= self.world_size
        reduced_metrics = {k: vals[i].item() for i, k in enumerate(keys)}

        return reduced_metrics

    # ---------- 验证 ----------

    def _validate(self) -> Dict[str, float]:
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0
        sat_correct = 0
        sat_total = 0
        unsat_correct = 0
        unsat_total = 0
        num_batches = 0
        sat_logit_mean = 0.0
        clause_vote_mean = 0.0
        delta_logit_mean = 0.0
        lambda_delta = 0.0

        with torch.no_grad():
            for batch in self.val_loader:
                batch = self._move_to_device(batch)

                with autocast(enabled=self.config.use_amp, dtype=self.amp_dtype):
                    outputs = self.model(
                        batch['vsm'],
                        batch.get('clause_mask'),
                        batch.get('var_mask'),
                    )
                    loss, _ = self.criterion(outputs, batch)
                    output_stats = self._extract_output_stats(outputs)

                total_loss += loss.item()
                preds = (torch.sigmoid(outputs['sat_pred'].squeeze(-1)) > 0.5).long()
                labels = batch['sat_label'].squeeze(-1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)
                sat_mask = labels == 1
                unsat_mask = labels == 0
                if sat_mask.any():
                    sat_correct += (preds[sat_mask] == labels[sat_mask]).sum().item()
                    sat_total += sat_mask.sum().item()
                if unsat_mask.any():
                    unsat_correct += (preds[unsat_mask] == labels[unsat_mask]).sum().item()
                    unsat_total += unsat_mask.sum().item()
                sat_logit_mean += output_stats['sat_logit_mean']
                clause_vote_mean += output_stats['clause_vote_mean']
                delta_logit_mean += output_stats['delta_logit_mean']
                lambda_delta += output_stats['lambda_delta']
                num_batches += 1

        # AllReduce 验证指标: 用 SUM 聚合计数后再算全局指标
        global_loss = reduce_sum(total_loss, self.device)
        global_correct = reduce_sum(correct, self.device)
        global_total = reduce_sum(total, self.device)
        global_sat_correct = reduce_sum(sat_correct, self.device)
        global_sat_total = reduce_sum(sat_total, self.device)
        global_unsat_correct = reduce_sum(unsat_correct, self.device)
        global_unsat_total = reduce_sum(unsat_total, self.device)
        global_num_batches = reduce_sum(num_batches, self.device)
        global_sat_logit_mean = reduce_sum(sat_logit_mean, self.device)
        global_clause_vote_mean = reduce_sum(clause_vote_mean, self.device)
        global_delta_logit_mean = reduce_sum(delta_logit_mean, self.device)
        global_lambda_delta = reduce_sum(lambda_delta, self.device)

        val_accuracy = global_correct / max(global_total, 1)
        val_sat_accuracy = global_sat_correct / max(global_sat_total, 1)
        val_unsat_accuracy = global_unsat_correct / max(global_unsat_total, 1)
        val_balanced_accuracy = 0.5 * (val_sat_accuracy + val_unsat_accuracy)

        return {
            'val_loss': global_loss / max(global_num_batches, 1),
            'val_accuracy': val_accuracy,
            'val_sat_accuracy': val_sat_accuracy,
            'val_unsat_accuracy': val_unsat_accuracy,
            'val_balanced_accuracy': val_balanced_accuracy,
            'val_sat_logit_mean': global_sat_logit_mean / max(global_num_batches, 1),
            'val_clause_vote_mean': global_clause_vote_mean / max(global_num_batches, 1),
            'val_delta_logit_mean': global_delta_logit_mean / max(global_num_batches, 1),
            'val_lambda_delta': global_lambda_delta / max(global_num_batches, 1),
        }

    @staticmethod
    def _extract_output_stats(outputs: Dict[str, torch.Tensor]) -> Dict[str, float]:
        sat_pred = outputs['sat_pred'].detach()
        stats = {
            'sat_logit_mean': sat_pred.mean().item(),
            'clause_vote_mean': 0.0,
            'delta_logit_mean': 0.0,
            'lambda_delta': 0.0,
        }
        if 'clause_vote' in outputs:
            stats['clause_vote_mean'] = outputs['clause_vote'].detach().mean().item()
        if 'delta_logit' in outputs:
            stats['delta_logit_mean'] = outputs['delta_logit'].detach().mean().item()
        if 'lambda_delta' in outputs:
            stats['lambda_delta'] = outputs['lambda_delta'].detach().mean().item()
        return stats

    # ---------- 日志 ----------

    def _log_step(self, loss_dict: Dict[str, float]):
        if self.logger and self.logger.writer:
            for k, v in loss_dict.items():
                self.logger.scalar_summary(f'train/{k}', v, self.global_step)
            self.logger.scalar_summary(
                'train/lr', self.scheduler.get_last_lr()[0], self.global_step
            )

    def _log_epoch(self, epoch: int, train_metrics: Dict, val_metrics: Dict):
        train_log_keys = ['core_loss', 'sat_loss', 'total_loss']
        val_log_keys = ['val_loss', 'val_accuracy', 'val_sat_accuracy', 'val_unsat_accuracy']

        self.logger.write(f'epoch: {epoch} | Train | ')
        for k in train_log_keys:
            if k not in train_metrics:
                continue
            v = train_metrics[k]
            self.logger.scalar_summary(f'train_{k}', v, epoch)
            self.logger.write(f'{k} {v:8f} | ')
        self.logger.write('\n')

        if val_metrics:
            self.logger.write(f'epoch: {epoch} | Val | ')
            for k in val_log_keys:
                if k not in val_metrics:
                    continue
                v = val_metrics[k]
                self.logger.scalar_summary(f'val_{k}', v, epoch)
                self.logger.write(f'{k} {v:8f} | ')
            self.logger.write('\n')

        self.training_history.append({
            'epoch': epoch,
            'train': train_metrics,
            'val': val_metrics,
        })

    # ---------- Checkpoint ----------

    def _build_checkpoint(self, epoch: int, metrics: Dict) -> Dict[str, Any]:
        # 保存内部模型 (去掉 DDP 的 'module.' 前缀)，与单卡 checkpoint 格式兼容
        model_state = self.model.module.state_dict()

        checkpoint = {
            'epoch': epoch,
            'global_step': self.global_step,
            'model_state_dict': model_state,
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'config': asdict(self.config),
            'metrics': metrics,
            'best_metric': self.best_metric,
            'training_history': self.training_history,
        }
        if self.scaler:
            checkpoint['scaler_state_dict'] = self.scaler.state_dict()

        return checkpoint

    def _maybe_save_best_checkpoint(self, epoch: int, metrics: Dict):
        if not metrics:
            return

        current_metric = metrics.get(self.config.best_metric, float('-inf'))
        if current_metric <= self.best_metric:
            return

        self.best_metric = current_metric
        checkpoint = self._build_checkpoint(epoch, metrics)
        checkpoint['best_metric'] = self.best_metric
        best_path = os.path.join(self.config.save_dir, 'checkpoint_best.pt')
        torch.save(checkpoint, best_path)
        self.logger.write(
            f"New best model saved! {self.config.best_metric}: {current_metric:.4f}\n"
        )

    def _save_checkpoint(self, epoch: int, metrics: Dict, is_final: bool = False):
        checkpoint = self._build_checkpoint(epoch, metrics)

        latest_path = os.path.join(self.config.save_dir, 'checkpoint_latest.pt')
        torch.save(checkpoint, latest_path)

        if not self.config.save_best_only:
            epoch_path = os.path.join(self.config.save_dir, f'checkpoint_epoch_{epoch}.pt')
            torch.save(checkpoint, epoch_path)

        if is_final:
            final_path = os.path.join(self.config.save_dir, 'checkpoint_final.pt')
            torch.save(checkpoint, final_path)

    def load_checkpoint(self, checkpoint_path: str, weights_only: bool = False):
        checkpoint = torch.load(checkpoint_path, map_location=self.device)

        # 1) 恢复模型权重
        load_result = self.model.module.load_state_dict(
            checkpoint['model_state_dict'],
            strict=False,
        )
        if load_result.missing_keys:
            print_rank0(f"Missing keys: {load_result.missing_keys}")
        if load_result.unexpected_keys:
            print_rank0(f"Unexpected keys: {load_result.unexpected_keys}")

        if weights_only:
            self.current_epoch = 0
            self.global_step = 0
            self.best_metric = float('-inf')
            self.training_history = []
            if self.is_main:
                self.logger.write(
                    f"Loaded model weights from checkpoint epoch {checkpoint['epoch']} "
                    "without optimizer/scheduler state\n"
                )
            return

        # 2) 恢复优化器状态 (保留 Adam 的一阶/二阶矩估计)
        try:
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        except ValueError as exc:
            print_rank0(f"Warning: optimizer state not restored: {exc}")

        # 3) 恢复调度器状态 (保留 cosine 进度)
        try:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        except ValueError as exc:
            print_rank0(f"Warning: scheduler state not restored: {exc}")

        # 4) 如果当前配置的 LR 与 checkpoint 不同，更新 base_lrs
        new_base_lrs = [self.config.learning_rate] * len(self.scheduler.base_lrs)
        old_base_lrs = self.scheduler.base_lrs

        if old_base_lrs != new_base_lrs:
            self.scheduler.base_lrs = new_base_lrs
            for param_group, lr in zip(self.optimizer.param_groups, self.scheduler.get_lr()):
                param_group['lr'] = lr
            if self.is_main:
                self.logger.write(
                    f"  LR base updated: {old_base_lrs[0]:.2e} -> {new_base_lrs[0]:.2e}\n"
                )

        # 5) 恢复混合精度缩放器
        if self.scaler and 'scaler_state_dict' in checkpoint:
            self.scaler.load_state_dict(checkpoint['scaler_state_dict'])

        # 6) 恢复训练状态
        self.current_epoch = checkpoint['epoch'] + 1
        self.best_metric = checkpoint.get('best_metric', float('-inf'))
        self.training_history = checkpoint.get('training_history', [])

        # 7) 修正 global_step: 基于当前 steps_per_epoch 重新计算
        #    防止因 effective batch size 变化导致 cosine 调度器进度错位
        old_global_step = checkpoint['global_step']
        current_steps_per_epoch = (
            len(self.train_loader) // self.config.gradient_accumulation_steps
        )
        correct_global_step = checkpoint['epoch'] * current_steps_per_epoch
        self.global_step = correct_global_step

        if old_global_step != correct_global_step and self.is_main:
            self.logger.write(
                f"  global_step recalculated: {old_global_step} -> {correct_global_step}\n"
                f"  (steps_per_epoch={current_steps_per_epoch}, epoch={checkpoint['epoch']})\n"
                f"  Cosine schedule progress corrected to match actual epoch progress.\n"
            )

        # 同步调度器内部计数器到修正后的 global_step
        self.scheduler.last_epoch = correct_global_step
        self.scheduler._step_count = correct_global_step + 1
        # 更新优化器 LR 以匹配修正后的调度位置
        for param_group, lr in zip(self.optimizer.param_groups, self.scheduler.get_lr()):
            param_group['lr'] = lr

        # 8) 检查配置变化并警告
        old_config = checkpoint.get('config', {})
        if old_config and self.is_main:
            new_accum = self.config.gradient_accumulation_steps
            old_accum = old_config.get('gradient_accumulation_steps', new_accum)
            new_lr = self.config.learning_rate
            old_lr = old_config.get('learning_rate', new_lr)
            if old_accum != new_accum:
                self.logger.write(
                    f"  WARNING: accumulation_steps changed: {old_accum} -> {new_accum}\n"
                    f"  Please verify effective batch size is consistent!\n"
                )
            if abs(old_lr - new_lr) / max(old_lr, 1e-10) > 0.01:
                self.logger.write(
                    f"  WARNING: learning_rate changed: {old_lr:.2e} -> {new_lr:.2e}\n"
                )

        if self.is_main:
            self.logger.write(f"Loaded checkpoint from epoch {checkpoint['epoch']}\n")
            self.logger.write(
                f"  Resumed LR: {self.optimizer.param_groups[0]['lr']:.6e}\n"
            )

    # ---------- Early Stopping ----------

    def _check_early_stopping(self, val_metrics: Dict) -> bool:
        if not val_metrics:
            return False

        current_metric = val_metrics.get(self.config.best_metric, float('-inf'))

        if current_metric > self.best_metric:
            self.patience_counter = 0
        else:
            self.patience_counter += 1

        # 所有进程同步 early stopping 决策 (由 rank 0 广播)
        should_stop = torch.tensor(
            int(self.patience_counter >= self.config.early_stopping_patience),
            device=self.device,
        )
        dist.broadcast(should_stop, src=0)

        return should_stop.item() == 1


# ======================== 主函数 ========================

def load_config(config_path: str) -> dict:
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def main():
    parser = argparse.ArgumentParser(description='Train XsSAT (DDP)')
    parser.add_argument('--config', type=str, default='config/ablation/A0_full.yaml',
                        help='配置文件路径')
    parser.add_argument('--train_path', type=str, default='data/data_neuro_like_10_40/train.pt',
                        help='训练数据路径 (.pt 文件)')
    parser.add_argument('--val_path', type=str, default='data/data_neuro_like_10_40/val.pt',
                        help='验证数据路径 (.pt 文件)')
    parser.add_argument('--checkpoint', type=str, default=None,
                        help='从检查点恢复训练 (兼容单卡/多卡 checkpoint)')
    parser.add_argument('--weights_only', action='store_true',
                        help='仅加载模型权重，用于微调新任务，不恢复优化器/epoch状态')
    parser.add_argument('--epochs', type=int, default=None)
    parser.add_argument('--batch_size', type=int, default=None)
    parser.add_argument('--lr', type=float, default=None)
    parser.add_argument('--accum_steps', type=int, default=None,
                        help='手动指定梯度累积步数 (默认: 原始值 / 卡数)')
    parser.add_argument('--seed', type=int, default=42,
                        help='随机种子 (默认: 42)')
    args = parser.parse_args()

    # ====== 初始化分布式环境 ======
    dist.init_process_group(backend='nccl')
    local_rank = int(os.environ['LOCAL_RANK'])
    torch.cuda.set_device(local_rank)
    world_size = dist.get_world_size()
    rank = dist.get_rank()

    # 种子: 基础种子相同保证模型初始化一致，数据加载种子按 rank 区分
    seed = args.seed
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    # ====== 加载配置 ======
    config = load_config(args.config)

    if args.epochs:
        config['training']['num_epochs'] = args.epochs
    if args.batch_size:
        config['training']['batch_size'] = args.batch_size
    if args.lr:
        config['training']['learning_rate'] = args.lr

    # 自动调整梯度累积步数: 原始值 / 卡数 (保持有效 batch 不变)
    train_config = config['training']
    original_accum = train_config.get('accumulation_steps', 1)
    if args.accum_steps is not None:
        adjusted_accum = args.accum_steps
    else:
        adjusted_accum = max(1, original_accum // world_size)
    train_config['accumulation_steps'] = adjusted_accum

    eff_batch = train_config['batch_size'] * world_size * adjusted_accum

    if rank == 0:
        print("=" * 60)
        print("XsSAT DDP Training")
        print("=" * 60)
        print(f"  World size:           {world_size} GPUs")
        for i in range(world_size):
            print(f"    GPU {i}: {torch.cuda.get_device_name(i)}")
        print(f"  Per-GPU batch size:   {train_config['batch_size']}")
        print(f"  Accumulation steps:   {adjusted_accum}  (原始: {original_accum})")
        print(f"  Effective batch size: {eff_batch}")
        print("=" * 60)

    # ====== 创建模型 ======
    print_rank0("\n[1/4] Creating model...")
    model_config = config['model']
    ablation_cfg = config.get('ablation', {})
    model = XsSAT(
        max_clauses=model_config['max_clauses'],
        max_vars=model_config['max_vars'],
        embed_dim=model_config['embed_dim'],
        num_layers=model_config['num_layers'],
        num_heads=model_config['num_heads'],
        ffn_ratio=model_config.get('ffn_ratio', 4),
        dropout=model_config['dropout'],
        tau_init=model_config.get('tau_init', 1.0),
        neg_samples=model_config.get('neg_samples', 256),
        use_gradient_checkpoint=model_config.get('gradient_checkpoint', False),
        use_multichannel_vsm=ablation_cfg.get('use_multichannel_vsm', True),
        use_negation=ablation_cfg.get('use_negation', True),
        attention_type=ablation_cfg.get('attention_type', 'axial'),
        readout_type=ablation_cfg.get('readout_type', 'semantic'),
        use_polarity_offset=ablation_cfg.get('use_polarity_offset', True),
        use_periodic_global_token=ablation_cfg.get('use_periodic_global_token', False),
        global_token_every_n_layers=ablation_cfg.get('global_token_every_n_layers', 2),
        global_token_writeback_scale=ablation_cfg.get('global_token_writeback_scale', 0.1),
        use_clause_literal_fusion=ablation_cfg.get('use_clause_literal_fusion', False),
        use_multiscale_clause_context=ablation_cfg.get('use_multiscale_clause_context', False),
        clause_hierarchy_levels=ablation_cfg.get('clause_hierarchy_levels', 2),
        clause_hierarchy_window=ablation_cfg.get('clause_hierarchy_window', 4),
        clause_context_prototypes=ablation_cfg.get('clause_context_prototypes', 4),
        detach_core_backbone=ablation_cfg.get('detach_core_backbone', False),
        use_polarity_pair_mixer=ablation_cfg.get('use_polarity_pair_mixer', False),
        pair_mixer_every_n_layers=ablation_cfg.get('pair_mixer_every_n_layers', 2),
        pair_mixer_writeback_scale=ablation_cfg.get('pair_mixer_writeback_scale', 0.1),
        hard_clause_topk_ratio=ablation_cfg.get('hard_clause_topk_ratio', 0.1),
        hard_clause_min_topk=ablation_cfg.get('hard_clause_min_topk', 8),
        use_structural_features=ablation_cfg.get('use_structural_features', False),
        clause_short_threshold=ablation_cfg.get('clause_short_threshold', 4),
        use_recurrent_axial=ablation_cfg.get('use_recurrent_axial', False),
        recurrent_steps=ablation_cfg.get('recurrent_steps', 4),
        recurrent_base_layers=ablation_cfg.get('recurrent_base_layers', 2),
    )

    if rank == 0:
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"Total parameters: {total_params:,}")
        print(f"Trainable parameters: {trainable_params:,}")

    # NOTE: torch.compile disabled — current PyTorch/Triton version has a
    # codegen bug (int1 vs int32 type mismatch) on boolean reductions in
    # CellEmbedding. Re-enable after upgrading PyTorch >= 2.3.
    # if hasattr(torch, 'compile'):
    #     model = torch.compile(model, mode='reduce-overhead')

    # ====== 加载数据 ======
    print_rank0("\n[2/4] Loading datasets...")

    online_cfg = config.get('data', {}).get('online_generation', {})
    use_online = online_cfg.get('enabled', False)

    # 数据增强
    augmentation = config.get('data', {}).get('augmentation', {})
    if augmentation.get('enabled', True):
        train_transform = SATAugmentation(
            variable_permutation=augmentation.get('variable_permutation', True),
            clause_permutation=augmentation.get('clause_permutation', True),
            polarity_flip_prob=augmentation.get('polarity_flip_prob',
                                                 augmentation.get('polarity_flip', 0.1)),
        )
    else:
        train_transform = None

    if use_online:
        online_dataset = OnlineSATDataset(
            epoch_size=online_cfg.get('epoch_size', 10000),
            max_clauses=model_config['max_clauses'],
            max_vars=model_config['max_vars'],
            min_vars=online_cfg.get('min_vars', 3),
            max_vars_gen=online_cfg.get('max_vars_gen', model_config['max_vars']),
            k=online_cfg.get('k', 3),
            cv_ratio_range=tuple(online_cfg.get('cv_ratio_range', [3.5, 5.0])),
            init_size=online_cfg.get('init_size', 4),
            depth=online_cfg.get('depth', 2),
            bloom_choice=online_cfg.get('bloom_choice', 2),
            pad_k=online_cfg.get('pad_k', 3),
            sat_ratio=online_cfg.get('sat_ratio', 0.5),
            transform=train_transform,
        )

        mix_real = online_cfg.get('mix_real', False)
        if mix_real and args.train_path:
            real_train_dataset = PackedSATDataset(args.train_path, transform=train_transform)
            real_ratio = online_cfg.get('real_ratio', 0.5)
            train_dataset = MixedSATDataset(
                real_dataset=real_train_dataset,
                online_dataset=online_dataset,
                real_ratio=real_ratio,
            )
            print_rank0(f"Mixed mode: real_ratio={real_ratio}")
        else:
            train_dataset = online_dataset
            print_rank0(f"Online-only mode: epoch_size={len(train_dataset)}")

        # 验证集
        if args.val_path and os.path.exists(args.val_path):
            val_dataset = PackedSATDataset(args.val_path)
        else:
            val_dataset = OnlineSATDataset(
                epoch_size=online_cfg.get('epoch_size', 10000) // 5,
                max_clauses=model_config['max_clauses'],
                max_vars=model_config['max_vars'],
                min_vars=online_cfg.get('min_vars', 3),
                max_vars_gen=online_cfg.get('max_vars_gen', model_config['max_vars']),
                k=online_cfg.get('k', 3),
                cv_ratio_range=tuple(online_cfg.get('cv_ratio_range', [3.5, 5.0])),
                init_size=online_cfg.get('init_size', 4),
                depth=online_cfg.get('depth', 2),
                bloom_choice=online_cfg.get('bloom_choice', 2),
                pad_k=online_cfg.get('pad_k', 3),
                sat_ratio=online_cfg.get('sat_ratio', 0.5),
            )
            print_rank0(f"Online validation: {len(val_dataset)} instances")

    elif args.train_path and args.train_path.endswith('.pt'):
        train_dataset = PackedSATDataset(args.train_path, transform=train_transform)
        val_dataset = PackedSATDataset(args.val_path)
    else:
        train_dataset = SATDataset(
            data_path=args.train_path,
            max_clauses=model_config['max_clauses'],
            max_vars=model_config['max_vars'],
        )
        val_dataset = SATDataset(
            data_path=args.val_path,
            max_clauses=model_config['max_clauses'],
            max_vars=model_config['max_vars'],
        )

    print_rank0(f"Train samples: {len(train_dataset)}  ({len(train_dataset) // world_size} per GPU)")
    print_rank0(f"Val samples:   {len(val_dataset)}")

    # ====== 创建 DDP DataLoader ======
    print_rank0("\n[3/4] Creating DDP dataloaders...")

    num_workers = config['data'].get('num_workers', 4)
    pin_memory = config['data'].get('pin_memory', True)

    def ddp_worker_init_fn(worker_id):
        """每个 worker 的独立随机种子 (结合 rank 避免跨卡重复)。"""
        worker_seed = seed + rank * 1000 + worker_id
        np.random.seed(worker_seed % (2 ** 32))
        random.seed(worker_seed)

    use_online_val = use_online and not (args.val_path and os.path.exists(str(args.val_path or '')))

    train_loader = create_ddp_dataloader(
        train_dataset,
        batch_size=train_config['batch_size'],
        num_workers=num_workers,
        pin_memory=pin_memory,
        is_train=True,
        worker_init_fn=ddp_worker_init_fn if use_online else None,
        seed=seed,
    )

    val_loader = create_ddp_dataloader(
        val_dataset,
        batch_size=train_config['batch_size'],
        num_workers=num_workers,
        pin_memory=pin_memory,
        is_train=False,
        worker_init_fn=ddp_worker_init_fn if use_online_val else None,
        seed=seed,
    )

    # ====== 损失配置 ======
    loss_config = {}
    loss_cfg = config.get('loss', {})
    loss_config['alpha_core'] = loss_cfg.get('alpha_core', 0.2)
    loss_config['alpha_neg'] = loss_cfg.get('alpha_neg', 0.1)
    if 'sat_loss_config' in loss_cfg:
        loss_config['sat_loss_config'] = loss_cfg['sat_loss_config']
    if 'clause_loss_config' in loss_cfg:
        loss_config['clause_loss_config'] = loss_cfg['clause_loss_config']

    # ====== 创建训练器 ======
    print_rank0("\n[4/4] Starting training...")

    warmup_steps = (
        train_config.get('warmup_epochs', 5) * len(train_loader)
        // max(adjusted_accum, 1)
    )

    trainer_config = TrainerConfig(
        num_epochs=train_config['num_epochs'],
        learning_rate=float(train_config['learning_rate']),
        weight_decay=float(train_config['weight_decay']),
        warmup_steps=warmup_steps,
        max_grad_norm=train_config.get('gradient_clip', 1.0),
        optimizer=train_config['optimizer']['type'],
        adam_beta1=train_config['optimizer'].get('betas', [0.9, 0.999])[0],
        adam_beta2=train_config['optimizer'].get('betas', [0.9, 0.999])[1],
        lr_scheduler=train_config['scheduler']['type'],
        batch_size=train_config['batch_size'],
        gradient_accumulation_steps=adjusted_accum,
        use_amp=config.get('device', {}).get('mixed_precision', True),
        save_dir=config['logging']['checkpoint_dir'],
        log_dir=config['logging']['log_dir'],
        save_every_n_epochs=config['logging'].get('save_every', 5),
        log_every_n_steps=config['logging'].get('log_every', 100),
        use_tensorboard=config['logging'].get('tensorboard', True),
        early_stopping_patience=train_config['early_stopping']['patience'],
        best_metric=train_config.get('best_metric', 'val_balanced_accuracy'),
        loss_config=loss_config,
        device=f'cuda:{local_rank}',
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    os.makedirs(trainer_config.save_dir, exist_ok=True)
    os.makedirs(trainer_config.log_dir, exist_ok=True)

    trainer = DDPTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        config=trainer_config,
        local_rank=local_rank,
    )

    if args.checkpoint:
        print_rank0(f"Loading checkpoint: {args.checkpoint}")
        trainer.load_checkpoint(args.checkpoint, weights_only=args.weights_only)

    print_rank0(f"\n{'=' * 60}")
    print_rank0(f"Training for {trainer_config.num_epochs} epochs")
    print_rank0(f"Batch size: {trainer_config.batch_size} × {world_size} GPUs × {adjusted_accum} accum = {eff_batch}")
    print_rank0(f"Learning rate: {trainer_config.learning_rate}")
    print_rank0(f"Batches per GPU per epoch: {len(train_loader)}")
    print_rank0(f"{'=' * 60}\n")

    history = trainer.train()

    if rank == 0:
        print("\nTraining completed!")
        print(f"Best validation {trainer_config.best_metric}: {trainer.best_metric:.4f}")
        print(f"Checkpoints saved to: {trainer_config.save_dir}")

    dist.destroy_process_group()


if __name__ == '__main__':
    main()
