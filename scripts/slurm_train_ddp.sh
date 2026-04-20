#!/usr/bin/env bash
#SBATCH --job-name=geosat-ddp
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G
#SBATCH --time=24:00:00
#SBATCH --output=slurm-%x-%j.out
#SBATCH --error=slurm-%x-%j.err

set -euo pipefail

# Edit these paths for your cluster account.
PROJECT_DIR="${PROJECT_DIR:-${SLURM_SUBMIT_DIR:-$PWD}}"
CONFIG_PATH="${CONFIG_PATH:-config/sr10_60.yaml}"
TRAIN_PATH="${TRAIN_PATH:-data_10_60/train.pt}"
VAL_PATH="${VAL_PATH:-data_10_60/val.pt}"
GPUS_PER_NODE="${GPUS_PER_NODE:-4}"

cd "${PROJECT_DIR}"
mkdir -p slurm_logs

# Load your cluster environment here.
# module purge
# module load cuda/12.1
# source ~/miniconda3/etc/profile.d/conda.sh
# conda activate geosat

export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"
export MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
export MASTER_PORT="${MASTER_PORT:-29500}"

echo "Job ID        : ${SLURM_JOB_ID:-N/A}"
echo "Host          : $(hostname)"
echo "Project dir   : ${PROJECT_DIR}"
echo "Config        : ${CONFIG_PATH}"
echo "Train data    : ${TRAIN_PATH}"
echo "Val data      : ${VAL_PATH}"
echo "GPUs per node : ${GPUS_PER_NODE}"

srun torchrun \
  --standalone \
  --nproc_per_node="${GPUS_PER_NODE}" \
  train_ddp.py \
  --config "${CONFIG_PATH}" \
  --train_path "${TRAIN_PATH}" \
  --val_path "${VAL_PATH}"
