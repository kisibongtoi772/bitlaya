#!/bin/bash
#SBATCH --account=a0152
#SBATCH --job-name=bitlaya-rabe-eval
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --partition=debug
#SBATCH --constraint=gpu
#SBATCH --cpus-per-task=64
#SBATCH --gpus-per-task=1
#SBATCH --mem=120G
#SBATCH --time=00:30:00
#SBATCH --output=slurm-bitlaya-rabe-eval-%j.out
#SBATCH --error=slurm-bitlaya-rabe-eval-%j.err
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-/users/manhnguyen/test/bitlayda/bitlaya_rabe3_24b}"
set -a; source config/experiment.env; set +a
source .venv/bin/activate
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1
python src/evaluate_gpu.py
cat artifacts/results/gpu_test_metrics.json
python src/diagnose_ag.py
echo 'GPU EVAL COMPLETE -> optional export: sbatch scripts/03_export_tq1.sbatch'
