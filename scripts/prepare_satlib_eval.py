#!/usr/bin/env python3
"""
Prepare SATLIB uf/uuf benchmark directories for GeoSATformer evaluation.

Example:
  python scripts/prepare_satlib_eval.py \
    --uf-dir /path/to/uf50-218 \
    --uuf-dir /path/to/uuf50-218 \
    --output-dir /path/to/satlib_eval
"""

from __future__ import annotations

import argparse
import csv
import shutil
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Stage SATLIB CNFs and generate labels.csv for test.py"
    )
    parser.add_argument(
        "--uf-dir",
        type=Path,
        required=True,
        help="Directory containing satisfiable SATLIB .cnf files",
    )
    parser.add_argument(
        "--uuf-dir",
        type=Path,
        required=True,
        help="Directory containing unsatisfiable SATLIB .cnf files",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Output directory that will contain cnf/ and labels.csv",
    )
    parser.add_argument(
        "--mode",
        choices=("symlink", "copy"),
        default="symlink",
        help="How to stage CNF files into output-dir/cnf",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace existing staged files and labels.csv if present",
    )
    return parser.parse_args()


def collect_cnfs(source_dir: Path, label: int) -> list[tuple[Path, int]]:
    if not source_dir.exists() or not source_dir.is_dir():
        raise ValueError(f"Invalid CNF directory: {source_dir}")

    cnf_files = sorted(source_dir.glob("*.cnf"))
    if not cnf_files:
        raise ValueError(f"No .cnf files found in: {source_dir}")

    return [(path, label) for path in cnf_files]


def stage_file(src: Path, dst: Path, mode: str, overwrite: bool) -> None:
    if dst.exists() or dst.is_symlink():
        if not overwrite:
            raise FileExistsError(
                f"Destination already exists: {dst}. Use --overwrite to replace it."
            )
        dst.unlink()

    if mode == "copy":
        shutil.copy2(src, dst)
        return

    try:
        dst.symlink_to(src.resolve())
    except OSError:
        shutil.copy2(src, dst)


def main() -> None:
    args = parse_args()

    entries = []
    entries.extend(collect_cnfs(args.uf_dir, 1))
    entries.extend(collect_cnfs(args.uuf_dir, 0))

    output_dir = args.output_dir
    cnf_dir = output_dir / "cnf"
    label_csv = output_dir / "labels.csv"

    output_dir.mkdir(parents=True, exist_ok=True)
    cnf_dir.mkdir(parents=True, exist_ok=True)

    seen_names: set[str] = set()
    rows: list[dict[str, object]] = []

    for src_path, label in entries:
        name = src_path.name
        if name in seen_names:
            raise ValueError(
                f"Duplicate CNF filename across inputs: {name}. "
                "Rename files or stage them separately."
            )
        seen_names.add(name)

        stage_file(
            src=src_path,
            dst=cnf_dir / name,
            mode=args.mode,
            overwrite=args.overwrite,
        )
        rows.append({"name": name, "satisfiability": label})

    if label_csv.exists() and not args.overwrite:
        raise FileExistsError(
            f"Output CSV already exists: {label_csv}. Use --overwrite to replace it."
        )

    with label_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["name", "satisfiability"])
        writer.writeheader()
        writer.writerows(sorted(rows, key=lambda row: str(row["name"])))

    sat_count = sum(1 for row in rows if row["satisfiability"] == 1)
    unsat_count = len(rows) - sat_count
    print(f"Wrote {label_csv}")
    print(f"Staged CNFs in {cnf_dir}")
    print(f"Total samples: {len(rows)}")
    print(f"SAT: {sat_count}")
    print(f"UNSAT: {unsat_count}")


if __name__ == "__main__":
    main()
