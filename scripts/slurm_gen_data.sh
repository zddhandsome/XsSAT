#!/usr/bin/env bash
#SBATCH --job-name=geosat-gen-data
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=slurm-%x-%j.out
#SBATCH --error=slurm-%x-%j.err

set -euo pipefail

# Edit these paths for your cluster account.
PROJECT_DIR="${PROJECT_DIR:-${SLURM_SUBMIT_DIR:-$PWD}}"

cd "${PROJECT_DIR}"
mkdir -p slurm_logs

# Load your cluster environment here.
# module purge
# source ~/miniconda3/etc/profile.d/conda.sh
# conda activate geosat
# export PYTHON_BIN=$HOME/miniconda3/envs/geosat/bin/python

export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1
export NUM_WORKERS="${NUM_WORKERS:-${SLURM_CPUS_PER_TASK:-24}}"

echo "Job ID      : ${SLURM_JOB_ID:-N/A}"
echo "Host        : $(hostname)"
echo "Project dir : ${PROJECT_DIR}"
echo "Workers     : ${NUM_WORKERS}"

srun bash gen_data.sh
