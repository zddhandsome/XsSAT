"""
XsSAT Trainer Module

- 多通道VSM输入
- 三项损失: L_sat + α_core·L_core + α_neg·L_negation
- 无VSIDS, 无对比损失
"""

import os
import time
import json
from dataclasses import dataclass, field, asdict
from typing import Dict, Optional, List, Callable, Any, Tuple
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler
from tqdm import tqdm

from ..losses import MultiTaskLoss
from .logger import Logger


OBSOLETE_METRIC_KEYS = {
    'diff_norm_mean',
    'register_summary_norm_mean',
    'val_diff_norm_mean',
    'val_register_summary_norm_mean',
}


def _sanitize_metric_dict(metrics: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """Drop obsolete metric keys from checkpoint/log payloads."""
    if not isinstance(metrics, dict):
        return {}
    return {
        key: value for key, value in metrics.items()
        if key not in OBSOLETE_METRIC_KEYS
    }


def _sanitize_training_history(history: Optional[List[Dict[str, Any]]]) -> List[Dict[str, Any]]:
    """Normalize historical entries so resumed runs do not keep stale metrics."""
    if not isinstance(history, list):
        return []

    sanitized_history = []
    for entry in history:
        if not isinstance(entry, dict):
            continue

        sanitized_entry = dict(entry)
        if 'train' in sanitized_entry:
            sanitized_entry['train'] = _sanitize_metric_dict(sanitized_entry.get('train'))
        if 'val' in sanitized_entry:
            sanitized_entry['val'] = _sanitize_metric_dict(sanitized_entry.get('val'))
        sanitized_history.append(sanitized_entry)

    return sanitized_history


def _summarize_key_list(label: str, keys: List[str], max_items: int = 8):
    """Keep compatibility logs short when loading older checkpoints."""
    if not keys:
        return

    preview = ', '.join(keys[:max_items])
    suffix = '' if len(keys) <= max_items else ', ...'
    print(f"{label}: {len(keys)} ({preview}{suffix})")


@dataclass
class TrainerConfig:
    """训练配置"""
    # 基础训练参数
    num_epochs: int = 100
    learning_rate: float = 1e-4
    weight_decay: float = 0.01
    warmup_steps: int = 1000
    max_grad_norm: float = 1.0

    # 优化器
    optimizer: str = 'adamw'
    adam_beta1: float = 0.9
    adam_beta2: float = 0.999
    adam_epsilon: float = 1e-8

    # 学习率调度
    lr_scheduler: str = 'cosine'
    lr_min_ratio: float = 0.01

    # 批量
    batch_size: int = 32
    gradient_accumulation_steps: int = 1

    # 混合精度
    use_amp: bool = True
    amp_dtype: str = 'float16'

    # 检查点
    save_dir: str = './checkpoints'
    save_every_n_epochs: int = 5
    save_best_only: bool = True
    best_metric: str = 'val_accuracy'

    # 日志
    log_dir: str = './logs'
    log_every_n_steps: int = 100
    use_tensorboard: bool = True

    # 验证
    eval_every_n_epochs: int = 1
    early_stopping_patience: int = 10

    # 损失权重
    loss_config: Dict = field(default_factory=lambda: {
        'alpha_core': 0.2,
        'alpha_neg': 0.1,
    })

    # 设备
    device: str = 'cuda'
    num_workers: int = 4
    pin_memory: bool = True


class Trainer:
    """
    XsSAT 训练器
    """

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        config: Optional[TrainerConfig] = None,
    ):
        self.config = config or TrainerConfig()
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader

        # 设备
        self.device = torch.device(self.config.device if torch.cuda.is_available() else 'cpu')
        self.model = self.model.to(self.device)

        # 损失函数
        self.criterion = MultiTaskLoss(**self.config.loss_config)

        # 优化器
        self.optimizer = self._create_optimizer()

        # 学习率调度器
        self.total_steps = len(train_loader) * self.config.num_epochs // self.config.gradient_accumulation_steps
        self.scheduler = self._create_scheduler()

        # 混合精度
        if self.config.use_amp:
            self.scaler = GradScaler()
            self.amp_dtype = torch.float16 if self.config.amp_dtype == 'float16' else torch.bfloat16
        else:
            self.scaler = None
            self.amp_dtype = torch.float32

        # 目录
        Path(self.config.save_dir).mkdir(parents=True, exist_ok=True)
        Path(self.config.log_dir).mkdir(parents=True, exist_ok=True)

        # 日志
        self.logger = Logger(
            save_dir=self.config.log_dir,
            config=asdict(self.config)
        )

        # 训练状态
        self.current_epoch = 0
        self.global_step = 0
        self.best_metric = float('-inf')
        self.patience_counter = 0
        self.training_history = []

    def _create_optimizer(self) -> optim.Optimizer:
        """创建优化器"""
        # 分组: 对bias和LayerNorm不使用weight decay
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
                eps=self.config.adam_epsilon
            )
        elif self.config.optimizer == 'adam':
            return optim.Adam(
                param_groups,
                lr=self.config.learning_rate,
                betas=(self.config.adam_beta1, self.config.adam_beta2),
                eps=self.config.adam_epsilon
            )
        else:
            raise ValueError(f"Unknown optimizer: {self.config.optimizer}")

    def _create_scheduler(self) -> optim.lr_scheduler._LRScheduler:
        """创建学习率调度器 (cosine with warmup)"""
        warmup_steps = self.config.warmup_steps
        total_steps = self.total_steps

        if self.config.lr_scheduler == 'cosine':
            def lr_lambda(step):
                if step < warmup_steps:
                    return step / max(warmup_steps, 1)
                progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
                return self.config.lr_min_ratio + 0.5 * (1 - self.config.lr_min_ratio) * (
                    1 + torch.cos(torch.tensor(progress * 3.14159)).item()
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

    def train(self) -> Dict[str, List[float]]:
        """执行完整训练"""
        self.logger.write("Starting training...\n")
        val_metrics = {}

        if self.current_epoch >= self.config.num_epochs:
            self.logger.write(
                "Current epoch is already beyond configured num_epochs; "
                "skipping training loop.\n"
            )
            self._save_checkpoint(self.current_epoch, val_metrics, is_final=True)
            return self.training_history

        for epoch in range(self.current_epoch, self.config.num_epochs):
            self.current_epoch = epoch

            train_metrics = self._train_epoch()

            val_metrics = {}
            if self.val_loader and (epoch + 1) % self.config.eval_every_n_epochs == 0:
                val_metrics = self._validate()

            self._log_epoch(epoch, train_metrics, val_metrics)

            if (epoch + 1) % self.config.save_every_n_epochs == 0:
                self._save_checkpoint(epoch, val_metrics)

            if self._check_early_stopping(val_metrics):
                self.logger.write(f"Early stopping at epoch {epoch}\n")
                break

        self.logger.write("Training completed!\n")
        self.logger.close()

        self._save_checkpoint(self.current_epoch, val_metrics, is_final=True)

        return self.training_history

    def _train_epoch(self) -> Dict[str, float]:
        """训练一个epoch"""
        self.model.train()

        epoch_loss = 0.0
        epoch_metrics = {
            'sat_loss': 0.0,
            'core_loss': 0.0,
            'negation_loss': 0.0,
        }
        num_batches = 0

        pbar = tqdm(self.train_loader, desc=f"Epoch {self.current_epoch}")

        for batch_idx, batch in enumerate(pbar):
            batch = self._move_to_device(batch)

            with autocast(enabled=self.config.use_amp, dtype=self.amp_dtype):
                outputs = self.model(
                    batch['vsm'],
                    batch.get('clause_mask'),
                    batch.get('var_mask'),
                )

                loss, loss_dict = self.criterion(outputs, batch)
                loss = loss / self.config.gradient_accumulation_steps

            if self.scaler:
                self.scaler.scale(loss).backward()
            else:
                loss.backward()

            if (batch_idx + 1) % self.config.gradient_accumulation_steps == 0:
                if self.scaler:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.max_grad_norm)
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.max_grad_norm)
                    self.optimizer.step()

                self.optimizer.zero_grad()
                self.scheduler.step()
                self.global_step += 1

                if self.global_step % self.config.log_every_n_steps == 0:
                    self._log_step(loss_dict)

            epoch_loss += loss.item() * self.config.gradient_accumulation_steps
            for k, v in loss_dict.items():
                if k in epoch_metrics:
                    epoch_metrics[k] += v.item() if torch.is_tensor(v) else v
            num_batches += 1

            pbar.set_postfix({
                'loss': f"{epoch_loss / num_batches:.4f}",
                'lr': f"{self.scheduler.get_last_lr()[0]:.2e}"
            })

        epoch_metrics = {k: v / max(num_batches, 1) for k, v in epoch_metrics.items()}
        epoch_metrics['total_loss'] = epoch_loss / max(num_batches, 1)

        return epoch_metrics

    def _validate(self) -> Dict[str, float]:
        """验证"""
        self.model.eval()
        return self._simple_validate()

    def _simple_validate(self) -> Dict[str, float]:
        """简单验证"""
        total_loss = 0.0
        correct = 0
        total = 0
        sat_correct = 0
        sat_total = 0
        unsat_correct = 0
        unsat_total = 0

        with torch.no_grad():
            for batch in self.val_loader:
                batch = self._move_to_device(batch)

                with autocast(enabled=self.config.use_amp, dtype=self.amp_dtype):
                    outputs = self.model(batch['vsm'], batch.get('clause_mask'), batch.get('var_mask'))
                    loss, _ = self.criterion(outputs, batch)

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

        val_accuracy = correct / max(total, 1)
        val_sat_accuracy = sat_correct / max(sat_total, 1)
        val_unsat_accuracy = unsat_correct / max(unsat_total, 1)

        return {
            'val_loss': total_loss / max(len(self.val_loader), 1),
            'val_accuracy': val_accuracy,
            'val_sat_accuracy': val_sat_accuracy,
            'val_unsat_accuracy': val_unsat_accuracy,
            'val_balanced_accuracy': 0.5 * (val_sat_accuracy + val_unsat_accuracy),
        }

    def _move_to_device(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """移动批次数据到设备"""
        return {
            k: v.to(self.device) if torch.is_tensor(v) else v
            for k, v in batch.items()
        }

    def _log_step(self, loss_dict: Dict[str, Any]):
        """记录训练步骤"""
        if self.logger.writer:
            for k, v in loss_dict.items():
                val = v.item() if torch.is_tensor(v) else v
                self.logger.scalar_summary(f'train/{k}', val, self.global_step)
            self.logger.scalar_summary('train/lr', self.scheduler.get_last_lr()[0], self.global_step)

    def _log_epoch(self, epoch: int, train_metrics: Dict, val_metrics: Dict):
        """记录epoch"""
        train_metrics = _sanitize_metric_dict(train_metrics)
        val_metrics = _sanitize_metric_dict(val_metrics)
        train_log_keys = ['core_loss', 'sat_loss', 'total_loss']
        val_log_keys = ['val_loss', 'val_accuracy', 'val_sat_accuracy', 'val_unsat_accuracy']

        self.logger.write('epoch: {} | Train | '.format(epoch))
        for k in train_log_keys:
            if k not in train_metrics:
                continue
            v = train_metrics[k]
            self.logger.scalar_summary('train_{}'.format(k), v, epoch)
            self.logger.write('{} {:8f} | '.format(k, v))
        self.logger.write('\n')

        if val_metrics:
            self.logger.write('epoch: {} | Val | '.format(epoch))
            for k in val_log_keys:
                if k not in val_metrics:
                    continue
                v = val_metrics[k]
                self.logger.scalar_summary('val_{}'.format(k), v, epoch)
                self.logger.write('{} {:8f} | '.format(k, v))
            self.logger.write('\n')

        self.training_history.append({
            'epoch': epoch,
            'train': train_metrics,
            'val': val_metrics
        })

    def _save_checkpoint(self, epoch: int, metrics: Dict, is_final: bool = False):
        """保存检查点"""
        sanitized_metrics = _sanitize_metric_dict(metrics)
        sanitized_history = _sanitize_training_history(self.training_history)
        checkpoint = {
            'epoch': epoch,
            'global_step': self.global_step,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'config': asdict(self.config),
            'metrics': sanitized_metrics,
            'best_metric': self.best_metric,
            'training_history': sanitized_history,
        }

        if self.scaler:
            checkpoint['scaler_state_dict'] = self.scaler.state_dict()

        latest_path = os.path.join(self.config.save_dir, 'checkpoint_latest.pt')
        torch.save(checkpoint, latest_path)

        current_metric = sanitized_metrics.get(self.config.best_metric, 0)
        if current_metric > self.best_metric:
            self.best_metric = current_metric
            best_path = os.path.join(self.config.save_dir, 'checkpoint_best.pt')
            torch.save(checkpoint, best_path)
            self.logger.write(f"New best model saved! {self.config.best_metric}: {current_metric:.4f}\n")

        if not self.config.save_best_only:
            epoch_path = os.path.join(self.config.save_dir, f'checkpoint_epoch_{epoch}.pt')
            torch.save(checkpoint, epoch_path)

        if is_final:
            final_path = os.path.join(self.config.save_dir, 'checkpoint_final.pt')
            torch.save(checkpoint, final_path)

    def load_checkpoint(self, checkpoint_path: str, weights_only: bool = False):
        """加载检查点"""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)

        load_result = self.model.load_state_dict(
            checkpoint['model_state_dict'],
            strict=False,
        )
        _summarize_key_list("Missing keys", load_result.missing_keys)
        _summarize_key_list("Unexpected keys", load_result.unexpected_keys)

        if weights_only:
            self.current_epoch = 0
            self.global_step = 0
            self.best_metric = float('-inf')
            self.training_history = []
            self.logger.write(
                f"Loaded model weights from checkpoint epoch {checkpoint['epoch']} "
                "without optimizer/scheduler state\n"
            )
            return

        try:
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        except ValueError as exc:
            print(f"Warning: optimizer state not restored: {exc}")
        try:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        except ValueError as exc:
            print(f"Warning: scheduler state not restored: {exc}")

        if self.scaler and 'scaler_state_dict' in checkpoint:
            self.scaler.load_state_dict(checkpoint['scaler_state_dict'])

        self.current_epoch = checkpoint['epoch'] + 1
        self.global_step = checkpoint['global_step']
        self.best_metric = checkpoint.get('best_metric', float('-inf'))
        self.training_history = _sanitize_training_history(checkpoint.get('training_history', []))

        self.logger.write(f"Loaded checkpoint from epoch {checkpoint['epoch']}\n")

    def _check_early_stopping(self, val_metrics: Dict) -> bool:
        """检查是否早停"""
        if not val_metrics:
            return False

        current_metric = val_metrics.get(self.config.best_metric, 0)

        if current_metric > self.best_metric:
            self.patience_counter = 0
        else:
            self.patience_counter += 1

        return self.patience_counter >= self.config.early_stopping_patience
