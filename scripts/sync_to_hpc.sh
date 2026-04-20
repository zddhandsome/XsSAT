#!/usr/bin/env bash

set -euo pipefail

if [[ $# -lt 2 ]]; then
  cat <<'EOF'
Usage:
  ./scripts/sync_to_hpc.sh <user@cluster> <remote_project_dir> [--dry-run]

Examples:
  ./scripts/sync_to_hpc.sh alice@login.hpc.edu /home/alice/projects/GeoSATformer
  ./scripts/sync_to_hpc.sh alice@login.hpc.edu /scratch/alice/GeoSATformer --dry-run

Notes:
  - This script syncs the current project to the remote HPC server with rsync.
  - Logs, checkpoints, caches, and notebook checkpoints are excluded by default.
EOF
  exit 1
fi

REMOTE_HOST="$1"
REMOTE_DIR="$2"
DRY_RUN="${3:-}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
PROJECT_NAME="$(basename "${PROJECT_DIR}")"

RSYNC_ARGS=(
  -azhv
  --delete
  --exclude=.git/
  --exclude=__pycache__/
  --exclude=.ipynb_checkpoints/
  --exclude=.pytest_cache/
  --exclude=.mypy_cache/
  --exclude=.DS_Store
  --exclude=logs/
  --exclude=checkpoints/
  --exclude=slurm_logs/
  --exclude=*.pyc
)

if [[ "${DRY_RUN}" == "--dry-run" ]]; then
  RSYNC_ARGS+=(--dry-run)
fi

echo "Local project : ${PROJECT_DIR}"
echo "Remote target : ${REMOTE_HOST}:${REMOTE_DIR}"

ssh "${REMOTE_HOST}" "mkdir -p '${REMOTE_DIR}'"
rsync "${RSYNC_ARGS[@]}" "${PROJECT_DIR}/" "${REMOTE_HOST}:${REMOTE_DIR}/"

echo "Sync finished for ${PROJECT_NAME}"
