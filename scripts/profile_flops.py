#!/usr/bin/env python3
"""
Profile XsSAT FLOPs on a real training batch or a synthetic batch.

The script reports:
- forward FLOPs
- train micro-batch FLOPs (forward + loss + backward)
- optimizer-step FLOPs
- effective training FLOPs under gradient accumulation
- approximate achieved TFLOPS from wall-clock timing

It uses torch.profiler(with_flops=True), so the totals only cover operators
for which PyTorch has registered FLOP formulas. Unsupported ops are undercounted.
"""

import os

os.environ["OMP_NUM_THREADS"] = "1"

import argparse
import random
import statistics
import sys
import time
from contextlib import nullcontext
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import torch
import yaml
from torch.cuda.amp import autocast
from torch.profiler import ProfilerActivity

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data import (  # noqa: E402
    MixedSATDataset,
    OnlineSATDataset,
    PackedSATDataset,
    SATDataset,
    create_dataloader,
    online_worker_init_fn,
)
from src.losses import MultiTaskLoss  # noqa: E402
from src.models import XsSAT  # noqa: E402


DEFAULT_TRAIN_PATHS = (
    "data/data_3_10/train.pt",
    "data/data_10_40/train.pt",
    "data/data_40/train.pt",
    "data/data_60/train.pt",
    "data/data_200/train.pt",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Profile XsSAT FLOPs")
    parser.add_argument(
        "--config",
        type=str,
        default="config/ablation/A0_full.yaml",
        help="Config file used to build the model and batch settings.",
    )
    parser.add_argument(
        "--train_path",
        type=str,
        default=None,
        help="Training data path. If omitted, the script tries common local datasets.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=None,
        help="Override config.training.batch_size for profiling.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=("auto", "cpu", "cuda"),
        help="Device used for profiling.",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=0,
        help="DataLoader workers used to fetch the profiling batch.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed. Defaults to config['seed'] when omitted.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="Epoch count used for total-training FLOPs estimation.",
    )
    parser.add_argument(
        "--steps_per_epoch",
        type=int,
        default=None,
        help="Manual override for steps per epoch in the total-training estimate.",
    )
    parser.add_argument(
        "--synthetic",
        action="store_true",
        help="Use a synthetic batch instead of loading data.",
    )
    parser.add_argument(
        "--synthetic_density",
        type=float,
        default=0.08,
        help="Literal density for synthetic VSM batches.",
    )
    parser.add_argument(
        "--warmup_steps",
        type=int,
        default=2,
        help="Warmup iterations before timing.",
    )
    parser.add_argument(
        "--timing_steps",
        type=int,
        default=5,
        help="Measured iterations for timing.",
    )
    parser.add_argument(
        "--top_ops",
        type=int,
        default=8,
        help="How many top operators by FLOPs to print.",
    )
    parser.add_argument(
        "--no_amp",
        action="store_true",
        help="Disable autocast even if the config enables mixed precision.",
    )
    return parser.parse_args()


def load_config(config_path: str) -> dict:
    with open(resolve_path(config_path), "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def resolve_path(path_str: str) -> Path:
    path = Path(path_str)
    if path.is_absolute():
        return path
    return ROOT / path


def pick_default_train_path() -> Optional[str]:
    for candidate in DEFAULT_TRAIN_PATHS:
        if (ROOT / candidate).exists():
            return candidate
    return None


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def choose_device(requested: str) -> torch.device:
    if requested == "cpu":
        return torch.device("cpu")
    if requested == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is not available.")
        return torch.device("cuda")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def choose_amp_dtype(config: dict) -> torch.dtype:
    amp_dtype = config.get("device", {}).get("amp_dtype", "float16")
    return torch.bfloat16 if amp_dtype == "bfloat16" else torch.float16


def amp_context(enabled: bool, amp_dtype: torch.dtype, device: torch.device):
    if enabled and device.type == "cuda":
        return autocast(enabled=True, dtype=amp_dtype)
    return nullcontext()


def build_model(config: dict) -> XsSAT:
    model_config = config["model"]
    ablation_cfg = config.get("ablation", {})

    return XsSAT(
        max_clauses=model_config["max_clauses"],
        max_vars=model_config["max_vars"],
        embed_dim=model_config["embed_dim"],
        num_layers=model_config["num_layers"],
        num_heads=model_config["num_heads"],
        ffn_ratio=model_config.get("ffn_ratio", 4),
        dropout=model_config["dropout"],
        tau_init=model_config.get("tau_init", 1.0),
        neg_samples=model_config.get("neg_samples", 256),
        use_gradient_checkpoint=model_config.get("gradient_checkpoint", False),
        use_multichannel_vsm=ablation_cfg.get("use_multichannel_vsm", True),
        use_negation=ablation_cfg.get("use_negation", True),
        attention_type=ablation_cfg.get("attention_type", "axial"),
        readout_type=ablation_cfg.get("readout_type", "semantic"),
        use_polarity_offset=ablation_cfg.get("use_polarity_offset", True),
        use_periodic_global_token=ablation_cfg.get("use_periodic_global_token", False),
        global_token_every_n_layers=ablation_cfg.get("global_token_every_n_layers", 2),
        global_token_writeback_scale=ablation_cfg.get("global_token_writeback_scale", 0.1),
        use_clause_literal_fusion=ablation_cfg.get("use_clause_literal_fusion", False),
        use_multiscale_clause_context=ablation_cfg.get("use_multiscale_clause_context", False),
        clause_hierarchy_levels=ablation_cfg.get("clause_hierarchy_levels", 2),
        clause_hierarchy_window=ablation_cfg.get("clause_hierarchy_window", 4),
        clause_context_prototypes=ablation_cfg.get("clause_context_prototypes", 4),
        detach_core_backbone=ablation_cfg.get("detach_core_backbone", False),
        use_polarity_pair_mixer=ablation_cfg.get("use_polarity_pair_mixer", False),
        pair_mixer_every_n_layers=ablation_cfg.get("pair_mixer_every_n_layers", 2),
        pair_mixer_writeback_scale=ablation_cfg.get("pair_mixer_writeback_scale", 0.1),
        hard_clause_topk_ratio=ablation_cfg.get("hard_clause_topk_ratio", 0.1),
        hard_clause_min_topk=ablation_cfg.get("hard_clause_min_topk", 8),
        use_structural_features=ablation_cfg.get("use_structural_features", False),
        clause_short_threshold=ablation_cfg.get("clause_short_threshold", 4),
        use_recurrent_axial=ablation_cfg.get("use_recurrent_axial", False),
        recurrent_steps=ablation_cfg.get("recurrent_steps", 4),
        recurrent_base_layers=ablation_cfg.get("recurrent_base_layers", 2),
    )


def build_criterion(config: dict) -> MultiTaskLoss:
    loss_cfg = config.get("loss", {})
    return MultiTaskLoss(
        alpha_core=loss_cfg.get("alpha_core", 0.2),
        alpha_neg=loss_cfg.get("alpha_neg", 0.1),
        sat_loss_config=loss_cfg.get("sat_loss_config"),
        clause_loss_config=loss_cfg.get("clause_loss_config"),
    )


def create_optimizer(model: torch.nn.Module, config: dict) -> torch.optim.Optimizer:
    train_cfg = config["training"]
    opt_cfg = train_cfg.get("optimizer", {})
    optimizer_name = opt_cfg.get("type", "adamw").lower()

    decay_params = []
    no_decay_params = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if "bias" in name or "norm" in name or "LayerNorm" in name:
            no_decay_params.append(param)
        else:
            decay_params.append(param)

    param_groups = [
        {"params": decay_params, "weight_decay": float(train_cfg.get("weight_decay", 0.01))},
        {"params": no_decay_params, "weight_decay": 0.0},
    ]

    common_kwargs = {
        "lr": float(train_cfg.get("learning_rate", 1e-4)),
        "betas": tuple(opt_cfg.get("betas", [0.9, 0.999])),
        "eps": float(opt_cfg.get("eps", 1e-8)),
    }

    if optimizer_name == "adamw":
        return torch.optim.AdamW(param_groups, **common_kwargs)
    if optimizer_name == "adam":
        return torch.optim.Adam(param_groups, **common_kwargs)
    raise ValueError(f"Unsupported optimizer for profiling: {optimizer_name}")


def build_real_batch(
    args: argparse.Namespace,
    config: dict,
) -> Tuple[Dict[str, torch.Tensor], Optional[int], Optional[int], str]:
    model_config = config["model"]
    data_cfg = config.get("data", {})
    online_cfg = data_cfg.get("online_generation", {})
    batch_size = args.batch_size or config["training"]["batch_size"]
    worker_init = None

    if online_cfg.get("enabled", False):
        online_dataset = OnlineSATDataset(
            epoch_size=online_cfg.get("epoch_size", 10000),
            max_clauses=model_config["max_clauses"],
            max_vars=model_config["max_vars"],
            min_vars=online_cfg.get("min_vars", 3),
            max_vars_gen=online_cfg.get("max_vars_gen", model_config["max_vars"]),
            k=online_cfg.get("k", 3),
            cv_ratio_range=tuple(online_cfg.get("cv_ratio_range", [3.5, 5.0])),
            init_size=online_cfg.get("init_size", 4),
            depth=online_cfg.get("depth", 2),
            bloom_choice=online_cfg.get("bloom_choice", 2),
            pad_k=online_cfg.get("pad_k", 3),
            sat_ratio=online_cfg.get("sat_ratio", 0.5),
            transform=None,
        )
        if online_cfg.get("mix_real", False):
            if not args.train_path:
                raise FileNotFoundError(
                    "Config enables mix_real, but no --train_path was provided."
                )
            real_dataset = PackedSATDataset(str(resolve_path(args.train_path)), transform=None)
            dataset = MixedSATDataset(
                real_dataset=real_dataset,
                online_dataset=online_dataset,
                real_ratio=online_cfg.get("real_ratio", 0.5),
            )
            source = "mixed(real+online)"
        else:
            dataset = online_dataset
            source = "online"
        worker_init = online_worker_init_fn
    else:
        train_path = args.train_path or pick_default_train_path()
        if not train_path:
            raise FileNotFoundError(
                "No train dataset found. Pass --train_path or use --synthetic."
            )
        train_path_resolved = resolve_path(train_path)
        if not train_path_resolved.exists():
            raise FileNotFoundError(
                f"Train dataset not found: {train_path_resolved}. "
                "Pass a valid --train_path or use --synthetic."
            )
        if train_path_resolved.suffix == ".pt":
            dataset = PackedSATDataset(str(train_path_resolved), transform=None)
            source = str(train_path_resolved.relative_to(ROOT))
        else:
            dataset = SATDataset(
                data_path=str(train_path_resolved),
                max_clauses=model_config["max_clauses"],
                max_vars=model_config["max_vars"],
                transform=None,
            )
            source = str(train_path_resolved.relative_to(ROOT))

    loader = create_dataloader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=False,
        worker_init_fn=worker_init,
    )
    batch = next(iter(loader))
    return batch, len(loader), len(dataset), source


def build_synthetic_batch(
    config: dict,
    batch_size: int,
    density: float,
) -> Dict[str, torch.Tensor]:
    model_config = config["model"]
    max_clauses = int(model_config["max_clauses"])
    max_vars = int(model_config["max_vars"])

    vsm = torch.zeros(batch_size, 2, max_clauses, max_vars, dtype=torch.float32)
    pos = torch.rand(batch_size, max_clauses, max_vars) < density
    neg = torch.rand(batch_size, max_clauses, max_vars) < density
    vsm[:, 0] = pos.float()
    vsm[:, 1] = neg.float()

    sat_label = torch.randint(0, 2, (batch_size, 1), dtype=torch.long)
    core_labels = torch.zeros(batch_size, max_clauses, dtype=torch.float32)
    unsat_rows = sat_label.squeeze(-1) == 0
    if unsat_rows.any():
        num_unsat = int(unsat_rows.sum().item())
        core_prob = torch.rand(num_unsat, max_clauses)
        core_labels[unsat_rows] = (core_prob < min(0.15, density * 2.0)).float()

    return {
        "vsm": vsm,
        "clause_mask": torch.ones(batch_size, max_clauses, dtype=torch.bool),
        "var_mask": torch.ones(batch_size, max_vars, dtype=torch.bool),
        "sat_label": sat_label,
        "core_labels": core_labels,
    }


def move_batch_to_device(
    batch: Dict[str, torch.Tensor],
    device: torch.device,
) -> Dict[str, torch.Tensor]:
    return {
        key: value.to(device) if torch.is_tensor(value) else value
        for key, value in batch.items()
    }


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def get_activities(device: torch.device) -> List[ProfilerActivity]:
    activities = [ProfilerActivity.CPU]
    if device.type == "cuda":
        activities.append(ProfilerActivity.CUDA)
    return activities


def total_flops_from_profiler(prof) -> int:
    return int(sum(int(getattr(event, "flops", 0) or 0) for event in prof.key_averages()))


def top_flop_ops(prof, limit: int) -> List[Tuple[str, int]]:
    ranked = [
        (event.key, int(getattr(event, "flops", 0) or 0))
        for event in prof.key_averages()
        if int(getattr(event, "flops", 0) or 0) > 0
    ]
    ranked.sort(key=lambda item: item[1], reverse=True)
    return ranked[:limit]


def profile_once(run_fn, device: torch.device) -> Tuple[int, List[Tuple[str, int]]]:
    synchronize(device)
    with torch.profiler.profile(
        activities=get_activities(device),
        record_shapes=True,
        with_flops=True,
    ) as prof:
        run_fn()
    synchronize(device)
    return total_flops_from_profiler(prof), top_flop_ops(prof, limit=64)


def benchmark(run_fn, device: torch.device, warmup_steps: int, timing_steps: int) -> Tuple[float, float]:
    for _ in range(max(warmup_steps, 0)):
        run_fn()
    synchronize(device)

    times = []
    for _ in range(max(timing_steps, 1)):
        synchronize(device)
        start = time.perf_counter()
        run_fn()
        synchronize(device)
        times.append(time.perf_counter() - start)

    return statistics.mean(times), statistics.median(times)


def run_forward(
    model: torch.nn.Module,
    batch: Dict[str, torch.Tensor],
    device: torch.device,
    use_amp: bool,
    amp_dtype: torch.dtype,
) -> None:
    model.eval()
    with torch.no_grad():
        with amp_context(use_amp, amp_dtype, device):
            model(batch["vsm"], batch.get("clause_mask"), batch.get("var_mask"))


def run_train_microbatch(
    model: torch.nn.Module,
    criterion: MultiTaskLoss,
    optimizer: torch.optim.Optimizer,
    batch: Dict[str, torch.Tensor],
    device: torch.device,
    use_amp: bool,
    amp_dtype: torch.dtype,
) -> None:
    model.train()
    optimizer.zero_grad(set_to_none=True)
    with amp_context(use_amp, amp_dtype, device):
        outputs = model(batch["vsm"], batch.get("clause_mask"), batch.get("var_mask"))
        loss, _ = criterion(outputs, batch)
    loss.backward()


def run_optimizer_step(optimizer: torch.optim.Optimizer) -> None:
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)


def prepare_grads(
    model: torch.nn.Module,
    criterion: MultiTaskLoss,
    optimizer: torch.optim.Optimizer,
    batch: Dict[str, torch.Tensor],
    device: torch.device,
    use_amp: bool,
    amp_dtype: torch.dtype,
) -> None:
    run_train_microbatch(model, criterion, optimizer, batch, device, use_amp, amp_dtype)
    synchronize(device)


def benchmark_optimizer_step(
    model: torch.nn.Module,
    criterion: MultiTaskLoss,
    optimizer: torch.optim.Optimizer,
    batch: Dict[str, torch.Tensor],
    device: torch.device,
    use_amp: bool,
    amp_dtype: torch.dtype,
    warmup_steps: int,
    timing_steps: int,
) -> Tuple[float, float]:
    for _ in range(max(warmup_steps, 0)):
        prepare_grads(model, criterion, optimizer, batch, device, use_amp, amp_dtype)
        synchronize(device)
        start = time.perf_counter()
        run_optimizer_step(optimizer)
        synchronize(device)
        _ = time.perf_counter() - start

    times = []
    for _ in range(max(timing_steps, 1)):
        prepare_grads(model, criterion, optimizer, batch, device, use_amp, amp_dtype)
        synchronize(device)
        start = time.perf_counter()
        run_optimizer_step(optimizer)
        synchronize(device)
        times.append(time.perf_counter() - start)
    return statistics.mean(times), statistics.median(times)


def format_flops(flops: float) -> str:
    units = ("FLOPs", "KFLOPs", "MFLOPs", "GFLOPs", "TFLOPs", "PFLOPs")
    value = float(flops)
    unit_idx = 0
    while value >= 1000.0 and unit_idx < len(units) - 1:
        value /= 1000.0
        unit_idx += 1
    return f"{value:.3f} {units[unit_idx]}"


def format_tflops(flops: float, seconds: float) -> str:
    if seconds <= 0:
        return "n/a"
    return f"{(flops / seconds) / 1e12:.3f} TFLOPS"


def summarize_batch(batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
    vsm = batch["vsm"]
    summary = {
        "batch_size": int(vsm.shape[0]),
        "channels": int(vsm.shape[1]),
        "clauses_dim": int(vsm.shape[2]),
        "vars_dim": int(vsm.shape[3]),
        "literal_density": float((vsm != 0).float().mean().item()),
    }
    clause_mask = batch.get("clause_mask")
    var_mask = batch.get("var_mask")
    if clause_mask is not None:
        summary["active_clauses_mean"] = float(clause_mask.float().sum(dim=1).mean().item())
        summary["active_clauses_max"] = int(clause_mask.sum(dim=1).max().item())
    if var_mask is not None:
        summary["active_vars_mean"] = float(var_mask.float().sum(dim=1).mean().item())
        summary["active_vars_max"] = int(var_mask.sum(dim=1).max().item())
    return summary


def print_top_ops(title: str, ops: Iterable[Tuple[str, int]], limit: int) -> None:
    print(title)
    for name, flops in list(ops)[:limit]:
        print(f"  - {name:<32} {format_flops(flops)}")


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    seed = args.seed if args.seed is not None else int(config.get("seed", 42))
    set_seed(seed)

    device = choose_device(args.device)
    use_amp = bool(config.get("device", {}).get("mixed_precision", True)) and not args.no_amp
    if device.type != "cuda":
        use_amp = False
    amp_dtype = choose_amp_dtype(config)

    model = build_model(config).to(device)
    criterion = build_criterion(config).to(device)
    optimizer = create_optimizer(model, config)

    batch_size = args.batch_size or int(config["training"]["batch_size"])
    if args.synthetic:
        batch = build_synthetic_batch(config, batch_size=batch_size, density=args.synthetic_density)
        steps_per_epoch = args.steps_per_epoch
        dataset_size = None
        batch_source = "synthetic"
    else:
        batch, loader_steps, dataset_size, batch_source = build_real_batch(args, config)
        steps_per_epoch = args.steps_per_epoch or loader_steps

    batch = move_batch_to_device(batch, device)
    batch_summary = summarize_batch(batch)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    grad_accum = int(config["training"].get("accumulation_steps", 1))
    epochs = args.epochs or int(config["training"].get("num_epochs", 1))

    forward_flops, forward_ops = profile_once(
        lambda: run_forward(model, batch, device, use_amp, amp_dtype),
        device,
    )
    forward_mean_s, forward_median_s = benchmark(
        lambda: run_forward(model, batch, device, use_amp, amp_dtype),
        device,
        args.warmup_steps,
        args.timing_steps,
    )

    train_microbatch_flops, train_ops = profile_once(
        lambda: run_train_microbatch(model, criterion, optimizer, batch, device, use_amp, amp_dtype),
        device,
    )
    optimizer.zero_grad(set_to_none=True)
    train_mean_s, train_median_s = benchmark(
        lambda: run_train_microbatch(model, criterion, optimizer, batch, device, use_amp, amp_dtype),
        device,
        args.warmup_steps,
        args.timing_steps,
    )
    optimizer.zero_grad(set_to_none=True)

    prepare_grads(model, criterion, optimizer, batch, device, use_amp, amp_dtype)
    optimizer_flops, optimizer_ops = profile_once(lambda: run_optimizer_step(optimizer), device)
    opt_mean_s, opt_median_s = benchmark_optimizer_step(
        model,
        criterion,
        optimizer,
        batch,
        device,
        use_amp,
        amp_dtype,
        args.warmup_steps,
        args.timing_steps,
    )

    effective_flops_per_microbatch = train_microbatch_flops + optimizer_flops / max(grad_accum, 1)
    effective_mean_s = train_mean_s + opt_mean_s / max(grad_accum, 1)
    effective_median_s = train_median_s + opt_median_s / max(grad_accum, 1)

    print("=" * 72)
    print("XsSAT FLOPs Profile")
    print("=" * 72)
    print(f"config:                 {args.config}")
    print(f"device:                 {device.type}")
    if device.type == "cuda":
        print(f"gpu:                    {torch.cuda.get_device_name(device)}")
    print(f"batch_source:           {batch_source}")
    print(f"mixed_precision:        {use_amp} ({str(amp_dtype).replace('torch.', '') if use_amp else 'disabled'})")
    print(f"gradient_accumulation:  {grad_accum}")
    print(f"total_params:           {total_params:,}")
    print(f"trainable_params:       {trainable_params:,}")
    if dataset_size is not None:
        print(f"dataset_size:           {dataset_size:,}")
    if steps_per_epoch is not None:
        print(f"steps_per_epoch:        {steps_per_epoch:,}")
    print("")
    print("Batch")
    print(f"  shape:                {tuple(batch['vsm'].shape)}")
    print(f"  active_clauses_mean:  {batch_summary.get('active_clauses_mean', 0.0):.2f}")
    print(f"  active_clauses_max:   {int(batch_summary.get('active_clauses_max', 0))}")
    print(f"  active_vars_mean:     {batch_summary.get('active_vars_mean', 0.0):.2f}")
    print(f"  active_vars_max:      {int(batch_summary.get('active_vars_max', 0))}")
    print(f"  literal_density:      {batch_summary['literal_density']:.6f}")
    print("")
    print("Profile")
    print(f"  forward_flops:                    {format_flops(forward_flops)}")
    print(f"  train_microbatch_flops:           {format_flops(train_microbatch_flops)}")
    print(f"  optimizer_step_flops:             {format_flops(optimizer_flops)}")
    print(f"  effective_train_flops/microbatch: {format_flops(effective_flops_per_microbatch)}")
    print(f"  effective_train_flops/update:     {format_flops(train_microbatch_flops * grad_accum + optimizer_flops)}")
    print("")
    print("Timing")
    print(f"  forward_mean:                     {forward_mean_s * 1000:.3f} ms")
    print(f"  forward_median:                   {forward_median_s * 1000:.3f} ms")
    print(f"  train_microbatch_mean:            {train_mean_s * 1000:.3f} ms")
    print(f"  train_microbatch_median:          {train_median_s * 1000:.3f} ms")
    print(f"  optimizer_step_mean:              {opt_mean_s * 1000:.3f} ms")
    print(f"  optimizer_step_median:            {opt_median_s * 1000:.3f} ms")
    print(f"  effective_train_mean:             {effective_mean_s * 1000:.3f} ms")
    print(f"  effective_train_median:           {effective_median_s * 1000:.3f} ms")
    print("")
    print("Achieved Throughput")
    print(f"  forward_mean:                     {format_tflops(forward_flops, forward_mean_s)}")
    print(f"  train_effective_mean:             {format_tflops(effective_flops_per_microbatch, effective_mean_s)}")
    print(f"  train_effective_median:           {format_tflops(effective_flops_per_microbatch, effective_median_s)}")

    if steps_per_epoch is not None:
        optimizer_updates_per_epoch = steps_per_epoch // max(grad_accum, 1)
        total_training_flops = (
            train_microbatch_flops * steps_per_epoch * epochs
            + optimizer_flops * optimizer_updates_per_epoch * epochs
        )
        print("")
        print("Total Training Estimate")
        print(f"  epochs:                           {epochs:,}")
        print(f"  optimizer_updates/epoch:          {optimizer_updates_per_epoch:,}")
        print(f"  total_training_flops:             {format_flops(total_training_flops)}")

    if args.top_ops > 0:
        print("")
        print_top_ops("Top forward ops by FLOPs", forward_ops, args.top_ops)
        print("")
        print_top_ops("Top train-microbatch ops by FLOPs", train_ops, args.top_ops)
        if optimizer_flops > 0:
            print("")
            print_top_ops("Top optimizer-step ops by FLOPs", optimizer_ops, args.top_ops)

    if optimizer_flops == 0 and opt_mean_s > 0:
        print("")
        print("note: optimizer.step() took measurable time, but torch.profiler reported")
        print("0 FLOPs for it. This is a profiler coverage limit, so total training FLOPs")
        print("from this script can slightly undercount optimizer math.")

    if forward_flops == 0 and train_microbatch_flops == 0:
        print("")
        print("warning: torch.profiler reported 0 FLOPs. This usually means the current")
        print("operator mix is unsupported by the profiler rather than the model doing no work.")


if __name__ == "__main__":
    main()
