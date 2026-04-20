"""
XsSAT 训练脚本

多通道VSM + 稀疏轴向注意力 + 语义对齐读出
"""

import os
import sys
import yaml
import argparse
import torch
import numpy as np
import random

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.models import XsSAT
from src.data import (
    SATDataset, PackedSATDataset, SATAugmentation,
    create_dataloader,
    OnlineSATDataset, MixedSATDataset, online_worker_init_fn
)
from src.training.trainer import Trainer, TrainerConfig


def load_config(config_path: str) -> dict:
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def main():
    parser = argparse.ArgumentParser(description='Train XsSAT')
    parser.add_argument('--config', type=str, default='config/ablation/A0_full.yaml',
                        help='配置文件路径')
    parser.add_argument('--train_path', type=str, default='data_test_3_10/train.pt',
                        help='训练数据路径 (.pt 文件)')
    parser.add_argument('--val_path', type=str, default='data_test_3_10/val.pt',
                        help='验证数据路径 (.pt 文件)')
    parser.add_argument('--checkpoint', type=str, default=None,
                        help='从检查点恢复训练')
    parser.add_argument('--weights_only', action='store_true',
                        help='仅加载模型权重，用于微调新任务，不恢复优化器/epoch状态')
    parser.add_argument('--seed', type=int, default=None,
                        help='随机种子 (覆盖 config 中的 seed)')
    parser.add_argument('--epochs', type=int, default=None)
    parser.add_argument('--batch_size', type=int, default=None)
    parser.add_argument('--lr', type=float, default=None)
    args = parser.parse_args()

    config = load_config(args.config)

    # 种子初始化
    seed = config.get('seed', 42)
    if args.seed is not None:
        seed = args.seed
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

    if args.epochs:
        config['training']['num_epochs'] = args.epochs
    if args.batch_size:
        config['training']['batch_size'] = args.batch_size
    if args.lr:
        config['training']['learning_rate'] = args.lr

    print("=" * 60)
    print("XsSAT Training")
    print("  Multi-Channel VSM + Axial Attention + Semantic Readout")
    print("=" * 60)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")
    if device == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    # ====== 创建模型 ======
    print("\n[1/4] Creating model...")
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

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")

    # ====== 加载数据 ======
    print("\n[2/4] Loading datasets...")

    online_cfg = config.get('data', {}).get('online_generation', {})
    use_online = online_cfg.get('enabled', False)
    train_config = config['training']

    # 数据增强
    augmentation = config.get('data', {}).get('augmentation', {})
    if augmentation.get('enabled', True):
        train_transform = SATAugmentation(
            variable_permutation=augmentation.get('variable_permutation', True),
            clause_permutation=augmentation.get('clause_permutation', True),
            polarity_flip_prob=augmentation.get('polarity_flip_prob',
                                                 augmentation.get('polarity_flip', 0.1))
        )
        print(f"Data augmentation enabled")
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
            print(f"Mixed mode: real_ratio={real_ratio}")
        else:
            train_dataset = online_dataset
            print(f"Online-only mode: epoch_size={len(train_dataset)}")

        # 验证集: 也用在线生成 (或从文件加载)
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
            print(f"Online validation: {len(val_dataset)} instances")

    elif args.train_path and args.train_path.endswith('.pt'):
        train_dataset = PackedSATDataset(args.train_path, transform=train_transform)
        val_dataset = PackedSATDataset(args.val_path)
    else:
        train_dataset = SATDataset(
            data_path=args.train_path,
            max_clauses=model_config['max_clauses'],
            max_vars=model_config['max_vars']
        )
        val_dataset = SATDataset(
            data_path=args.val_path,
            max_clauses=model_config['max_clauses'],
            max_vars=model_config['max_vars']
        )

    print(f"Train samples: {len(train_dataset)}")
    print(f"Val samples: {len(val_dataset)}")

    # ====== 创建 DataLoader ======
    print("\n[3/4] Creating dataloaders...")

    train_loader = create_dataloader(
        train_dataset,
        batch_size=train_config['batch_size'],
        shuffle=not use_online,
        num_workers=config['data'].get('num_workers', 4),
        pin_memory=config['data'].get('pin_memory', True),
        worker_init_fn=online_worker_init_fn if use_online else None,
    )

    val_loader = create_dataloader(
        val_dataset,
        batch_size=train_config['batch_size'],
        shuffle=False,
        num_workers=config['data'].get('num_workers', 4),
        pin_memory=config['data'].get('pin_memory', True),
        worker_init_fn=online_worker_init_fn if (use_online and not (args.val_path and os.path.exists(str(args.val_path or '')))) else None,
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
    warmup_steps = train_config.get('warmup_epochs', 5) * len(train_loader) // max(train_config.get('accumulation_steps', 1), 1)

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
        gradient_accumulation_steps=train_config.get('accumulation_steps', 1),
        use_amp=config.get('device', {}).get('mixed_precision', True),
        save_dir=config['logging']['checkpoint_dir'],
        best_metric=train_config.get('best_metric', 'val_accuracy'),
        log_dir=config['logging']['log_dir'],
        save_every_n_epochs=config['logging'].get('save_every', 5),
        log_every_n_steps=config['logging'].get('log_every', 100),
        use_tensorboard=config['logging'].get('tensorboard', True),
        early_stopping_patience=train_config['early_stopping']['patience'],
        loss_config=loss_config,
        device=device,
        num_workers=config['data'].get('num_workers', 4),
        pin_memory=config['data'].get('pin_memory', True)
    )

    os.makedirs(trainer_config.save_dir, exist_ok=True)
    os.makedirs(trainer_config.log_dir, exist_ok=True)

    print("\n[4/4] Starting training...")
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        config=trainer_config
    )

    if args.checkpoint:
        print(f"Loading checkpoint: {args.checkpoint}")
        trainer.load_checkpoint(args.checkpoint, weights_only=args.weights_only)

    print("\n" + "=" * 60)
    print(f"Training for {trainer_config.num_epochs} epochs")
    print(f"Batch size: {trainer_config.batch_size}")
    print(f"Learning rate: {trainer_config.learning_rate}")
    print(f"Model: {total_params:,} params, {model_config['num_layers']} axial layers")
    print("=" * 60 + "\n")

    history = trainer.train()

    print("\nTraining completed!")
    print(f"Best validation {trainer_config.best_metric}: {trainer.best_metric:.4f}")
    print(f"Checkpoints saved to: {trainer_config.save_dir}")


if __name__ == '__main__':
    main()
