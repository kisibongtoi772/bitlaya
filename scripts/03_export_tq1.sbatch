#!/bin/bash
#SBATCH --account=a0152
#SBATCH --job-name=bitlaya-export
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --partition=debug
#SBATCH --constraint=gpu
#SBATCH --cpus-per-task=64
#SBATCH --gpus-per-task=1
#SBATCH --mem=160G
#SBATCH --time=00:30:00
#SBATCH --output=slurm-bitlaya-export-%j.out
#SBATCH --error=slurm-bitlaya-export-%j.err
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-/users/manhnguyen/test/bitlayda/bitlaya_rabe3_24b}"
set -a; source config/experiment.env; set +a
source .venv/bin/activate
export PYTHONUNBUFFERED=1
mkdir -p artifacts/export
python third_party/llama.cpp/convert_hf_to_gguf.py artifacts/finetuned/backbone_bf16   --outfile artifacts/export/bitlaya-rabe3-tq1_0.gguf --outtype tq1_0
ls -lh artifacts/export/bitlaya-rabe3-tq1_0.gguf artifacts/finetuned/decision_head.safetensors
