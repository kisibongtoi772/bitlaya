#!/bin/bash
#SBATCH --account=a0152
#SBATCH --job-name=bitlaya-parity
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --partition=debug
#SBATCH --constraint=gpu
#SBATCH --cpus-per-task=32
#SBATCH --gpus-per-task=1
#SBATCH --mem=160G
#SBATCH --time=00:20:00
#SBATCH --output=slurm-bitlaya-parity-%j.out
#SBATCH --error=slurm-bitlaya-parity-%j.err
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-/users/manhnguyen/test/bitlayda/bitlaya_rabe3_24b}"
set -a; source config/experiment.env; set +a
source .venv/bin/activate
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=16 PYTHONUNBUFFERED=1
python src/check_parity_bf16.py
