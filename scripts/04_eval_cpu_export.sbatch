#!/bin/bash
#SBATCH --account=a0152
#SBATCH --job-name=bitlaya-cpu-eval
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --partition=debug
#SBATCH --constraint=gpu
#SBATCH --cpus-per-task=288
#SBATCH --gpus-per-task=1
#SBATCH --mem=450G
#SBATCH --time=00:30:00
#SBATCH --output=slurm-bitlaya-cpu-eval-%j.out
#SBATCH --error=slurm-bitlaya-cpu-eval-%j.err
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-/users/manhnguyen/test/bitlayda/bitlaya_rabe3_24b}"
set -a; source config/experiment.env; set +a
source .venv/bin/activate
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS=16

python src/evaluate_cpu_export.py   --model artifacts/export/bitlaya-rabe3-tq1_0.gguf   --workers 16   --threads 16   --split test
