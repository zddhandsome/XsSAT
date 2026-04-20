#!/usr/bin/env bash
#SBATCH --job-name=geosat-1gpu
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=slurm-%x-%j.out
#SBATCH --error=slurm-%x-%j.err

set -euo pipefail

# Edit these paths for your cluster account.
PROJECT_DIR="${PROJECT_DIR:-${SLURM_SUBMIT_DIR:-$PWD}}"
CONFIG_PATH="${CONFIG_PATH:-config/sr10_60.yaml}"
TRAIN_PATH="${TRAIN_PATH:-data_10_60/train.pt}"
VAL_PATH="${VAL_PATH:-data_10_60/val.pt}"

cd "${PROJECT_DIR}"
mkdir -p slurm_logs

# Load your cluster environment here.
# module purge
# module load cuda/12.1
# source ~/miniconda3/etc/profile.d/conda.sh
# conda activate geosat

export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"

echo "Job ID      : ${SLURM_JOB_ID:-N/A}"
echo "Host        : $(hostname)"
echo "Project dir : ${PROJECT_DIR}"
echo "Config      : ${CONFIG_PATH}"
echo "Train data  : ${TRAIN_PATH}"
echo "Val data    : ${VAL_PATH}"

srun python -u train.py \
  --config "${CONFIG_PATH}" \
  --train_path "${TRAIN_PATH}" \
  --val_path "${VAL_PATH}"
