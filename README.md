# XsSAT

XsSAT is a matrix-based neural SAT predictor built around:

- multi-channel clause-variable input encoding
- structure-enhanced cell embeddings
- aligned negation supervision
- shared-step recurrent axial reasoning
- clause-centric hard-clause readout

The repository includes training and evaluation code, configuration files, data-preparation
scripts, ablation utilities, and the current paper draft.

## Repository Highlights

- **Main model:** [`src/models/XsSAT.py`](src/models/XsSAT.py)
- **Single-GPU training:** [`train.py`](train.py)
- **Distributed training (DDP):** [`train_ddp.py`](train_ddp.py)
- **Evaluation / testing:** [`test.py`](test.py)
- **Main config example:** [`config/sr10_40_vsm_plus.yaml`](config/sr10_40_vsm_plus.yaml)
- **Paper source:** [`Paper/XsSAT_final_candidate.tex`](Paper/XsSAT_final_candidate.tex)

## Model Overview

XsSAT follows the pipeline:

`CNF -> MC-VSM -> Cell Embedding -> Negation-Aware Encoding -> Recurrent Axial Backbone -> Hard-Clause Readout`

Key design ideas:

1. **Multi-channel variable-space matrix**
   Positive and negative literal occurrences are stored in separate channels instead of being
   compressed into a single `{-1, 0, +1}` signal.

2. **Structure-enhanced input**
   Clause statistics and variable statistics are injected before the main reasoning stack.

3. **Recurrent axial reasoning**
   The backbone reuses a small set of axial blocks across multiple rollout steps instead of
   relying on a deep stack of independent layers.

4. **Negation-aware supervision**
   Positive and negative states of the same variable are aligned during training.

5. **Hard-clause-centric readout**
   Final prediction emphasizes the hardest clauses instead of using simple global averaging.

## Repository Structure

```text
.
|-- src/
|   |-- data/          # datasets, online generation, VSM encoding
|   |-- losses/        # multi-task loss
|   |-- models/        # XsSAT model
|   `-- training/      # trainer and logging utilities
|-- config/            # training and ablation configs
|-- scripts/           # ablation, SATLIB prep, profiling, cluster scripts
|-- Paper/             # paper source, figures, and LaTeX assets
|-- train.py           # single-GPU training entrypoint
|-- train_ddp.py       # multi-GPU DDP training entrypoint
|-- test.py            # evaluation entrypoint
|-- gen_data_packed.py
|-- gen_data_packed_fast.py
`-- CURRENT_XSSAT_GUIDE.md
```

## Environment

Recommended:

- Python 3.10+
- PyTorch with CUDA support for training

Core Python dependencies used by the public code paths:

- `torch`
- `numpy`
- `pyyaml`
- `tqdm`
- `scikit-learn`
- `python-sat`
- `tensorboard` or `tensorboardX` for logging

Example installation:

```bash
pip install torch numpy pyyaml tqdm scikit-learn python-sat tensorboard
```

## Data

Large training data, checkpoints, and logs are **not** included in this public repository.

The code supports three main data workflows:

1. **Packed `.pt` datasets**
   Used by `train.py` / `test.py` via `--train_path`, `--val_path`, and `--test_path`.

2. **Online generation**
   Enabled through the `data.online_generation` section of a config file.

3. **CNF directory + CSV labels**
   Supported by `test.py` for benchmark-style evaluation such as SATLIB.

### Generate packed data

```bash
python gen_data_packed.py \
  --output_dir data/my_dataset \
  --num_train 10000 \
  --num_val 1000 \
  --num_test 1000
```

For a faster multiprocessing version:

```bash
python gen_data_packed_fast.py generate --num_train 10000 --num_workers 8
```

### Prepare SATLIB-style evaluation data

```bash
python scripts/prepare_satlib_eval.py \
  --uf-dir /path/to/uf50-218 \
  --uuf-dir /path/to/uuf50-218 \
  --output-dir /path/to/satlib_eval \
  --mode symlink
```

This creates:

- `output-dir/cnf/`
- `output-dir/labels.csv`

which can be consumed directly by `test.py`.

## Training

### Single-GPU training

```bash
python train.py \
  --config config/sr10_40_vsm_plus.yaml \
  --train_path /path/to/train.pt \
  --val_path /path/to/val.pt
```

Resume from a checkpoint:

```bash
python train.py \
  --config config/sr10_40_vsm_plus.yaml \
  --train_path /path/to/train.pt \
  --val_path /path/to/val.pt \
  --checkpoint /path/to/checkpoint.pt
```

### Multi-GPU training

```bash
torchrun --nproc_per_node=4 train_ddp.py \
  --config config/sr10_40_vsm_plus.yaml \
  --train_path /path/to/train.pt \
  --val_path /path/to/val.pt
```

The DDP entrypoint is checkpoint-compatible with `train.py`.

### Ablation runs

```bash
bash scripts/run_ablation.sh
```

The ablation configs are under [`config/ablation`](config/ablation).

## Evaluation

### Evaluate on a packed test set

```bash
python test.py \
  --config config/sr10_40_vsm_plus.yaml \
  --checkpoint /path/to/checkpoint_best.pt \
  --test_path /path/to/test.pt
```

### Evaluate on a CNF directory with labels

```bash
python test.py \
  --config config/sr10_40_vsm_plus.yaml \
  --checkpoint /path/to/checkpoint_best.pt \
  --test_dir /path/to/satlib_eval/cnf \
  --label_csv /path/to/satlib_eval/labels.csv
```

## Main Configuration

The main public reference config is:

- [`config/sr10_40_vsm_plus.yaml`](config/sr10_40_vsm_plus.yaml)

It enables:

- recurrent axial reasoning
- structural feature injection
- negation-aware supervision
- polarity pair mixer
- hard-clause readout

## Paper

Relevant paper files are under [`Paper/`](Paper):

- current LaTeX draft: [`Paper/XsSAT_final_candidate.tex`](Paper/XsSAT_final_candidate.tex)
- bibliography: [`Paper/references.bib`](Paper/references.bib)
- latest uploaded PDF snapshot: [`Paper/XsSAT.pdf`](Paper/XsSAT.pdf)

## Notes

- The repository still contains historical naming traces such as `GeoSATformer` and older
  experiment labels.
- The current mainline model for this repository is **XsSAT**.
- Large directories such as `data/`, `checkpoints/`, and `logs/` are intentionally excluded
  from version control.

## Citation

If you use this codebase, cite the accompanying XsSAT paper once the final bibliographic
information is available. Until then, the current draft can be found in [`Paper/`](Paper).
