#!/usr/bin/env bash
set -euo pipefail

NPROC_PER_NODE="${NPROC_PER_NODE:-6}"
OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export OMP_NUM_THREADS

CONFIG="${CONFIG:-config/satlib_sr40_60_vsm_plus_finetune.yaml}"
TRAIN_PATH="${TRAIN_PATH:-data/satlib_finetune_uf50_218/train}"
VAL_PATH="${VAL_PATH:-data/satlib_finetune_uf50_218/val}"
CHECKPOINT="${CHECKPOINT:-checkpoints/sr40_60_vsm_plus/checkpoint_best.pt}"

torchrun \
  --nproc_per_node="${NPROC_PER_NODE}" \
  train_ddp.py \
  --config "${CONFIG}" \
  --train_path "${TRAIN_PATH}" \
  --val_path "${VAL_PATH}" \
  --checkpoint "${CHECKPOINT}" \
  --weights_only
