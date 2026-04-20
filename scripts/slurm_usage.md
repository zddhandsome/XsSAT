# GeoSATformer on Slurm

## 1. Sync project to the cluster

From your local machine:

```bash
./scripts/sync_to_hpc.sh <user@cluster> <remote_project_dir>
```

Example:

```bash
./scripts/sync_to_hpc.sh alice@login.hpc.edu /home/alice/GeoSATformer
```

## 2. Log in and prepare the environment

```bash
ssh <user@cluster>
cd <remote_project_dir>
```

Then edit the environment section in:

- `scripts/slurm_train_single.sh`
- `scripts/slurm_train_ddp.sh`

Typical cluster setup:

```bash
module purge
module load cuda/12.1
source ~/miniconda3/etc/profile.d/conda.sh
conda activate geosat
```

If you have not created the environment yet:

```bash
conda create -n geosat python=3.10 -y
conda activate geosat
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install numpy pyyaml tqdm scikit-learn tensorboard
```

## 3. Submit a single-GPU job

```bash
sbatch scripts/slurm_train_single.sh
```

## 4. Submit a multi-GPU DDP job

```bash
sbatch scripts/slurm_train_ddp.sh
```

This project's distributed entrypoint is `train_ddp.py`, which already expects `torchrun`.

## 5. Override paths without editing the file

```bash
sbatch --export=ALL,PROJECT_DIR=/home/alice/GeoSATformer,CONFIG_PATH=config/sr10_60.yaml,TRAIN_PATH=data_10_60/train.pt,VAL_PATH=data_10_60/val.pt scripts/slurm_train_single.sh
```

Multi-GPU:

```bash
sbatch --export=ALL,PROJECT_DIR=/home/alice/GeoSATformer,CONFIG_PATH=config/sr10_60.yaml,TRAIN_PATH=data_10_60/train.pt,VAL_PATH=data_10_60/val.pt,GPUS_PER_NODE=4 scripts/slurm_train_ddp.sh
```

## 6. Check job status

```bash
squeue -u $USER
tail -f slurm_logs/geosat-ddp-<jobid>.out
```
