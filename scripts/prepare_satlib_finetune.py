#!/usr/bin/env python3
"""
Prepare SATLIB uf/uuf benchmarks for fine-tuning with train.py.

The output layout is:
  output_dir/
    train/
      *.cnf
      *.label
    val/
      *.cnf
      *.label
    test/
      *.cnf
      *.label
"""

from __future__ import annotations

import argparse
import random
import shutil
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create SATLIB train/val/test splits with .label sidecars"
    )
    parser.add_argument("--uf-dir", type=Path, required=True)
    parser.add_argument("--uuf-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--test-ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--mode",
        choices=("symlink", "copy"),
        default="symlink",
        help="How to stage CNF files into split directories",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Remove an existing output directory before writing new splits",
    )
    return parser.parse_args()


def collect_cnfs(source_dir: Path, label: int) -> list[tuple[Path, int]]:
    if not source_dir.exists() or not source_dir.is_dir():
        raise ValueError(f"Invalid CNF directory: {source_dir}")

    cnf_files = sorted(source_dir.glob("*.cnf"))
    if not cnf_files:
        raise ValueError(f"No .cnf files found in: {source_dir}")

    return [(path, label) for path in cnf_files]


def split_entries(
    entries: list[tuple[Path, int]],
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
    seed: int,
) -> dict[str, list[tuple[Path, int]]]:
    total_ratio = train_ratio + val_ratio + test_ratio
    if abs(total_ratio - 1.0) > 1e-8:
        raise ValueError(
            f"Split ratios must sum to 1.0, got {total_ratio:.6f}"
        )

    rng = random.Random(seed)
    shuffled = list(entries)
    rng.shuffle(shuffled)

    total = len(shuffled)
    n_train = int(total * train_ratio)
    n_val = int(total * val_ratio)
    n_test = total - n_train - n_val

    return {
        "train": shuffled[:n_train],
        "val": shuffled[n_train:n_train + n_val],
        "test": shuffled[n_train + n_val:n_train + n_val + n_test],
    }


def stage_file(src: Path, dst: Path, mode: str) -> None:
    if mode == "copy":
        shutil.copy2(src, dst)
        return

    try:
        dst.symlink_to(src.resolve())
    except OSError:
        shutil.copy2(src, dst)


def write_split(split_dir: Path, entries: list[tuple[Path, int]], mode: str) -> None:
    split_dir.mkdir(parents=True, exist_ok=True)

    seen_names: set[str] = set()
    for src_path, label in entries:
        if src_path.name in seen_names:
            raise ValueError(f"Duplicate filename within split: {src_path.name}")
        seen_names.add(src_path.name)

        staged_cnf = split_dir / src_path.name
        stage_file(src_path, staged_cnf, mode=mode)

        label_path = staged_cnf.with_suffix(".label")
        label_path.write_text(f"{label}\n")


def main() -> None:
    args = parse_args()

    if args.output_dir.exists():
        if not args.overwrite:
            raise FileExistsError(
                f"Output directory already exists: {args.output_dir}. "
                "Use --overwrite to replace it."
            )
        shutil.rmtree(args.output_dir)

    sat_entries = collect_cnfs(args.uf_dir, 1)
    unsat_entries = collect_cnfs(args.uuf_dir, 0)

    sat_splits = split_entries(
        sat_entries,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        seed=args.seed,
    )
    unsat_splits = split_entries(
        unsat_entries,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        seed=args.seed,
    )

    summary: dict[str, dict[str, int]] = {}
    for split_name in ("train", "val", "test"):
        split_entries_combined = sat_splits[split_name] + unsat_splits[split_name]
        split_entries_combined.sort(key=lambda item: item[0].name)
        write_split(args.output_dir / split_name, split_entries_combined, mode=args.mode)
        summary[split_name] = {
            "total": len(split_entries_combined),
            "sat": len(sat_splits[split_name]),
            "unsat": len(unsat_splits[split_name]),
        }

    print(f"Wrote fine-tuning dataset to {args.output_dir}")
    for split_name in ("train", "val", "test"):
        split_summary = summary[split_name]
        print(
            f"{split_name}: total={split_summary['total']}, "
            f"SAT={split_summary['sat']}, UNSAT={split_summary['unsat']}"
        )


if __name__ == "__main__":
    main()
