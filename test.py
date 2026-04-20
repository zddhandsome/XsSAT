"""
XsSAT 测试脚本

支持三种模式:
1. 从 .pt 文件加载测试集评估
2. 在线生成测试实例评估 (无需预备数据)
3. 从 CNF 目录 + CSV 标签评估
"""

import os
import sys
import yaml
import argparse
import time
import torch
import numpy as np
from sklearn.metrics import roc_curve, auc
from torch.cuda.amp import autocast

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.models import XsSAT
from src.data import (
    CNFCSVTestDataset,
    PackedSATDataset,
    OnlineSATDataset,
    create_dataloader,
)


OBSOLETE_METRIC_KEYS = {
    'diff_norm_mean',
    'register_summary_norm_mean',
    'val_diff_norm_mean',
    'val_register_summary_norm_mean',
}


def load_config(config_path: str) -> dict:
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def _sanitize_metric_dict(metrics):
    if not isinstance(metrics, dict):
        return {}
    return {
        key: value for key, value in metrics.items()
        if key not in OBSOLETE_METRIC_KEYS
    }


def _sanitize_training_history(history):
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


def _sanitize_checkpoint_metadata(checkpoint: dict) -> dict:
    if not isinstance(checkpoint, dict):
        return checkpoint

    sanitized = dict(checkpoint)
    sanitized['metrics'] = _sanitize_metric_dict(checkpoint.get('metrics'))
    if 'training_history' in checkpoint:
        sanitized['training_history'] = _sanitize_training_history(
            checkpoint.get('training_history')
        )
    return sanitized


def _print_key_summary(label: str, keys, max_items: int = 8):
    if not keys:
        return

    preview = ', '.join(keys[:max_items])
    suffix = '' if len(keys) <= max_items else ', ...'
    print(f"  {label}: {len(keys)} ({preview}{suffix})")


def _print_cnf_dataset_summary(dataset: CNFCSVTestDataset):
    summary = getattr(dataset, 'summary', None)
    if not summary:
        return

    print("  CNF+CSV mode enabled")
    print(f"  Matched labels:       {summary['matched_labels']}")
    print(f"  Vars > max_vars:      {summary['vars_truncated']}")
    print(f"  Clauses > max_clauses:{summary['clauses_truncated']:>6d}")
    if summary['any_truncated']:
        print("  WARNING: some samples exceed model limits and will be truncated during encoding")


def _validate_dataset_shape(dataset, model_max_clauses: int, model_max_vars: int):
    dataset_max_clauses = getattr(dataset, 'max_clauses', None)
    dataset_max_vars = getattr(dataset, 'max_vars', None)

    mismatches = []
    if dataset_max_clauses is not None and dataset_max_clauses > model_max_clauses:
        mismatches.append(
            f"dataset max_clauses={dataset_max_clauses} exceeds model max_clauses={model_max_clauses}"
        )
    if dataset_max_vars is not None and dataset_max_vars > model_max_vars:
        mismatches.append(
            f"dataset max_vars={dataset_max_vars} exceeds model max_vars={model_max_vars}"
        )

    if mismatches:
        raise ValueError(
            "Incompatible test data for current config/checkpoint: "
            + "; ".join(mismatches)
            + ". Please use a matching dataset or override --test_path/--test_dir."
        )


def _resolve_amp_dtype(dtype_name: str) -> torch.dtype:
    if dtype_name == 'bfloat16':
        return torch.bfloat16
    return torch.float16


def _is_cuda_oom(exc: BaseException) -> bool:
    return 'out of memory' in str(exc).lower()


def _is_cublas_heuristic_failure(exc: BaseException) -> bool:
    text = str(exc)
    return (
        'CUBLAS_STATUS_NOT_SUPPORTED' in text
        or 'cublasLtMatmulAlgoGetHeuristic' in text
    )


def _merge_split_outputs(outputs):
    first = outputs[0]

    if isinstance(first, dict):
        return {
            key: _merge_split_outputs([output[key] for output in outputs])
            for key in first
        }

    if torch.is_tensor(first):
        if first.ndim == 0:
            return torch.stack(outputs).mean()
        return torch.cat(outputs, dim=0)

    return first


def _slice_optional_tensor(tensor, start: int, end: int):
    if tensor is None:
        return None
    return tensor[start:end]


def _forward_with_auto_split(
    model,
    vsm,
    clause_mask,
    var_mask,
    amp_enabled: bool,
    amp_dtype: torch.dtype,
):
    try:
        with autocast(enabled=amp_enabled, dtype=amp_dtype):
            return model(vsm, clause_mask, var_mask)
    except (torch.cuda.OutOfMemoryError, RuntimeError) as exc:
        if amp_enabled and _is_cublas_heuristic_failure(exc):
            print(
                "  CUDA AMP matmul heuristic failed; retrying this batch in fp32"
            )
            return model(vsm, clause_mask, var_mask)

        if not _is_cuda_oom(exc) or vsm.shape[0] <= 1:
            raise

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        split_point = max(vsm.shape[0] // 2, 1)
        print(
            f"  CUDA OOM on batch size {vsm.shape[0]}; retrying with "
            f"sub-batches {split_point} + {vsm.shape[0] - split_point}"
        )

        outputs = []
        for start, end in ((0, split_point), (split_point, vsm.shape[0])):
            if start == end:
                continue
            outputs.append(
                _forward_with_auto_split(
                    model,
                    vsm[start:end],
                    _slice_optional_tensor(clause_mask, start, end),
                    _slice_optional_tensor(var_mask, start, end),
                    amp_enabled=amp_enabled,
                    amp_dtype=amp_dtype,
                )
            )
        return _merge_split_outputs(outputs)


def evaluate(model, dataloader, device, threshold=0.5, amp_enabled=False, amp_dtype=torch.float16):
    """评估模型 SAT/UNSAT 分类 + UNSAT Core 识别"""
    model.eval()

    all_probs = []
    all_preds = []
    all_labels = []
    all_core_probs = []
    all_core_labels = []
    all_clause_masks = []

    total_time = 0
    num_samples = 0

    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            vsm = batch['vsm'].to(device)
            clause_mask = batch.get('clause_mask')
            var_mask = batch.get('var_mask')
            sat_label = batch['sat_label'].to(device)

            if clause_mask is not None:
                clause_mask = clause_mask.to(device)
            if var_mask is not None:
                var_mask = var_mask.to(device)

            start_time = time.time()
            outputs = _forward_with_auto_split(
                model,
                vsm,
                clause_mask,
                var_mask,
                amp_enabled=amp_enabled,
                amp_dtype=amp_dtype,
            )
            if device.type == 'cuda':
                torch.cuda.synchronize()
            end_time = time.time()

            total_time += (end_time - start_time) * 1000
            num_samples += vsm.shape[0]

            # SAT 预测
            sat_logits = outputs['sat_pred'].squeeze(-1)
            probs = torch.sigmoid(sat_logits)
            preds = (probs > threshold).long()

            labels = sat_label.squeeze(-1)

            all_probs.extend(probs.cpu().numpy().tolist())
            all_preds.extend(preds.cpu().numpy().tolist())
            all_labels.extend(labels.cpu().numpy().tolist())

            # UNSAT Core 预测 (收集用于后续评估)
            if 'core_labels' in batch:
                all_core_probs.append(outputs['clause_scores'].cpu())
                all_core_labels.append(batch['core_labels'])
                if clause_mask is not None:
                    all_clause_masks.append(clause_mask.cpu())

            batch_acc = (preds == labels).float().mean().item()
            if (batch_idx + 1) % 10 == 0 or batch_idx == 0:
                print(f'  Batch {batch_idx}: acc={batch_acc:.4f} ({(end_time - start_time)*1000:.1f}ms)')

    results = {
        'probs': np.array(all_probs),
        'preds': np.array(all_preds),
        'labels': np.array(all_labels),
        'total_time': total_time,
        'num_samples': num_samples,
    }

    # UNSAT Core 评估
    if all_core_probs:
        # 不同 batch 的 clause 维度可能不同（动态 padding），需统一到最大尺寸
        max_c = max(t.shape[1] for t in all_core_probs)
        def _pad_to_max(tensors, val=0):
            return [torch.nn.functional.pad(t, (0, max_c - t.shape[1]), value=val)
                    if t.shape[1] < max_c else t for t in tensors]
        results['core_probs'] = torch.cat(_pad_to_max(all_core_probs, 0), dim=0)
        results['core_labels'] = torch.cat(_pad_to_max(all_core_labels, 0), dim=0)
        if all_clause_masks:
            results['clause_masks'] = torch.cat(_pad_to_max(all_clause_masks, 0), dim=0)

    return results


def print_metrics(results, threshold=0.5):
    """打印评估指标"""
    probs = results['probs']
    preds = results['preds']
    labels = results['labels']
    total_time = results['total_time']
    num_samples = results['num_samples']

    tp = ((preds == 1) & (labels == 1)).sum()
    tn = ((preds == 0) & (labels == 0)).sum()
    fp = ((preds == 1) & (labels == 0)).sum()
    fn = ((preds == 0) & (labels == 1)).sum()

    accuracy = (tp + tn) / len(labels)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    # Per-class accuracy
    sat_mask = labels == 1
    unsat_mask = labels == 0
    sat_acc = (preds[sat_mask] == labels[sat_mask]).mean() if sat_mask.sum() > 0 else 0
    unsat_acc = (preds[unsat_mask] == labels[unsat_mask]).mean() if unsat_mask.sum() > 0 else 0
    balanced_acc = 0.5 * (sat_acc + unsat_acc)

    # ROC-AUC
    if len(np.unique(labels)) > 1:
        fpr, tpr, thresholds = roc_curve(labels, probs, pos_label=1)
        roc_auc = auc(fpr, tpr)
        opt_threshold = thresholds[np.argmax(tpr - fpr)]
    else:
        roc_auc = float('nan')
        opt_threshold = threshold

    print("\n" + "=" * 55)
    print(f"  SAT/UNSAT Classification (threshold={threshold})")
    print("=" * 55)
    print(f"  Balanced Acc:  {balanced_acc*100:6.2f}%")
    print(f"  SAT Accuracy:  {sat_acc*100:6.2f}%  ({sat_mask.sum()} samples)")
    print(f"  UNSAT Accuracy:{unsat_acc*100:6.2f}%  ({unsat_mask.sum()} samples)")
    print(f"  Accuracy:      {accuracy*100:6.2f}%")
    print(f"  Precision:     {precision*100:6.2f}%")
    print(f"  Recall:        {recall*100:6.2f}%")
    print(f"  F1 Score:      {f1*100:6.2f}%")
    print(f"  ROC-AUC:       {roc_auc:.4f}")
    print("-" * 55)
    print(f"  Confusion Matrix:")
    print(f"              Pred UNSAT  Pred SAT")
    print(f"  True UNSAT    {tn:5d}       {fp:5d}")
    print(f"  True SAT      {fn:5d}       {tp:5d}")
    print("-" * 55)
    print(f"  Total samples:        {num_samples}")
    print(f"  Total time:           {total_time:.1f}ms")
    print(f"  Avg time per sample:  {total_time/max(num_samples,1):.2f}ms")
    print(f"  Optimal threshold:    {opt_threshold:.4f}")

    # Optimal threshold results
    opt_preds = (probs > opt_threshold).astype(int)
    opt_acc = (opt_preds == labels).mean()
    print(f"  Accuracy @opt:        {opt_acc*100:.2f}%")
    print("=" * 55)

    # UNSAT Core evaluation
    if 'core_probs' in results:
        _print_core_metrics(results)

    return {
        'balanced_accuracy': balanced_acc,
        'sat_accuracy': sat_acc,
        'unsat_accuracy': unsat_acc,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'roc_auc': roc_auc,
        'optimal_threshold': opt_threshold,
    }


def _print_core_metrics(results):
    """评估 UNSAT Core 识别"""
    core_scores = results['core_probs']       # [N, C]
    core_labels = results['core_labels']       # [N, C]
    labels = results['labels']                 # [N]

    # 只在 UNSAT 样本上评估
    unsat_mask = (labels == 0)
    if unsat_mask.sum() == 0:
        return

    unsat_scores = core_scores[unsat_mask]
    unsat_core = core_labels[unsat_mask]

    clause_masks = results.get('clause_masks')
    if clause_masks is not None:
        unsat_cmask = clause_masks[unsat_mask]
    else:
        unsat_cmask = None

    # Core clause 应该有更低的 clause_score (soft-min语义)
    # 使用 -score 作为 core 预测概率
    core_pred_probs = torch.sigmoid(-unsat_scores)

    # 计算每个样本的 core F1
    f1_scores = []
    iou_scores = []
    for i in range(len(unsat_scores)):
        true_core = unsat_core[i] > 0.5
        if unsat_cmask is not None:
            valid = unsat_cmask[i]
            true_core = true_core & valid

        if true_core.sum() == 0:
            continue

        pred_core = core_pred_probs[i] > 0.5
        if unsat_cmask is not None:
            pred_core = pred_core & valid

        tp = (pred_core & true_core).float().sum()
        fp = (pred_core & ~true_core).float().sum()
        fn = (~pred_core & true_core).float().sum()

        p = tp / (tp + fp) if (tp + fp) > 0 else 0
        r = tp / (tp + fn) if (tp + fn) > 0 else 0
        f = 2 * p * r / (p + r) if (p + r) > 0 else 0
        f1_scores.append(f.item() if torch.is_tensor(f) else f)

        intersection = (pred_core & true_core).float().sum()
        union = (pred_core | true_core).float().sum()
        iou = intersection / union if union > 0 else 0
        iou_scores.append(iou.item() if torch.is_tensor(iou) else iou)

    if f1_scores:
        print(f"\n  UNSAT Core Identification ({len(f1_scores)} UNSAT samples)")
        print("-" * 55)
        print(f"  Avg Core F1:   {np.mean(f1_scores)*100:6.2f}%")
        print(f"  Avg Core IoU:  {np.mean(iou_scores)*100:6.2f}%")
        print("=" * 55)


def main():
    parser = argparse.ArgumentParser(description='Test XsSAT')
    parser.add_argument('--config', type=str, default='config/sr10_40_vsm_plus.yaml')
    parser.add_argument('--checkpoint', type=str, default='/root/autodl-tmp/GeoSATformer/checkpoints/sr10_40_vsm_plus/checkpoint_best.pt',
                        help='模型检查点路径')
    parser.add_argument('--test_path', type=str, default='data/data_neuro_like_90_100/test.pt',
                        help='测试数据路径 (.pt 文件)')
    parser.add_argument('--test_dir', type=str, default=None,
                        help='CNF 测试目录 (与 --label_csv 配合使用)')
    parser.add_argument('--label_csv', type=str, default=None,
                        help='CNF 测试标签 CSV 路径')
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--threshold', type=float, default=0.5)
    parser.add_argument('--num_test', type=int, default=2000,
                        help='在线生成的测试实例数量')
    parser.add_argument('--disable_amp', action='store_true',
                        help='禁用混合精度推理')
    parser.add_argument('--amp_dtype', type=str, choices=['float16', 'bfloat16'], default=None,
                        help='混合精度类型，默认读取配置并回退到 float16')
    parser.add_argument('--device_ids', type=str, default=None,
                        help='逗号分隔的 GPU ID 列表，例如 0,1,2,3')
    parser.add_argument('--disable_data_parallel', action='store_true',
                        help='禁用多卡 DataParallel')
    args = parser.parse_args()

    config = load_config(args.config)

    print("=" * 60)
    print("XsSAT Testing")
    print("=" * 60)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    device_ids = []
    if device.type == 'cuda':
        if args.device_ids:
            device_ids = [int(x.strip()) for x in args.device_ids.split(',') if x.strip()]
        else:
            device_ids = list(range(torch.cuda.device_count()))

        if not device_ids:
            raise ValueError("No CUDA device IDs available")

        torch.cuda.set_device(device_ids[0])
        device = torch.device(f'cuda:{device_ids[0]}')
        print(f"Primary GPU: {torch.cuda.get_device_name(device_ids[0])}")
        print(f"Visible GPUs: {device_ids}")

    amp_cfg = config.get('device', {})
    amp_enabled = (
        device.type == 'cuda'
        and amp_cfg.get('mixed_precision', True)
        and not args.disable_amp
    )
    amp_dtype_name = args.amp_dtype or amp_cfg.get('amp_dtype', 'float16')
    amp_dtype = _resolve_amp_dtype(amp_dtype_name)
    print(f"AMP: {'enabled' if amp_enabled else 'disabled'} ({amp_dtype_name})")

    # ====== 创建模型 ======
    print("\n[1/3] Creating model...")
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
    print(f"Total parameters: {total_params:,}")

    # ====== 加载检查点 ======
    print(f"\n[2/3] Loading checkpoint: {args.checkpoint}")
    if os.path.exists(args.checkpoint):
        checkpoint = _sanitize_checkpoint_metadata(
            torch.load(args.checkpoint, map_location='cpu')
        )
        load_result = model.load_state_dict(checkpoint['model_state_dict'], strict=False)
        epoch = checkpoint.get('epoch', '?')
        best_metric = checkpoint.get('best_metric', '?')
        print(f"  Loaded from epoch {epoch}, best_metric: {best_metric}")
        _print_key_summary('Missing keys', load_result.missing_keys)
        _print_key_summary('Unexpected keys', load_result.unexpected_keys)
    else:
        print(f"  WARNING: Checkpoint not found at {args.checkpoint}")
        print(f"  Running with random weights (for testing pipeline only)")

    model = model.to(device)
    if device.type == 'cuda' and len(device_ids) > 1 and not args.disable_data_parallel:
        model = torch.nn.DataParallel(
            model,
            device_ids=device_ids,
            output_device=device_ids[0],
        )
        print(f"  Using DataParallel across {len(device_ids)} GPUs")
    else:
        print("  Using single-device inference")
    model.eval()

    # ====== 加载测试数据 ======
    print(f"\n[3/3] Loading test data...")
    online_cfg = config.get('data', {}).get('online_generation', {})

    if args.test_dir or args.label_csv:
        if not args.test_dir or not args.label_csv:
            raise ValueError("--test_dir and --label_csv must be provided together")
        test_dataset = CNFCSVTestDataset(
            cnf_dir=args.test_dir,
            label_csv=args.label_csv,
            max_clauses=model_config['max_clauses'],
            max_vars=model_config['max_vars'],
        )
        _print_cnf_dataset_summary(test_dataset)
    elif args.test_path and os.path.exists(args.test_path):
        test_dataset = PackedSATDataset(args.test_path)
    else:
        print(f"  Generating {args.num_test} online test instances...")
        test_dataset = OnlineSATDataset(
            epoch_size=args.num_test,
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

    test_loader = create_dataloader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=config['data'].get('num_workers', 4),
        pin_memory=config['data'].get('pin_memory', True),
    )
    print(f"  Test samples: {len(test_dataset)}")

    # ====== 评估 ======
    print("\n" + "=" * 60)
    print("Starting evaluation...")
    print("=" * 60)

    results = evaluate(
        model,
        test_loader,
        device,
        args.threshold,
        amp_enabled=amp_enabled,
        amp_dtype=amp_dtype,
    )
    metrics = print_metrics(results, args.threshold)

    print("\nTesting completed!")


if __name__ == '__main__':
    main()
