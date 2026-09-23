#!/bin/bash
#SBATCH --account=a0152
#SBATCH --job-name=bitlaya-rabe-train
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --partition=debug
#SBATCH --constraint=gpu
#SBATCH --cpus-per-task=288
#SBATCH --gpus-per-task=4
#SBATCH --mem=450G
#SBATCH --time=01:30:00
#SBATCH --output=slurm-bitlaya-rabe-train-%j.out
#SBATCH --error=slurm-bitlaya-rabe-train-%j.err
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-/users/manhnguyen/test/bitlayda/bitlaya_rabe3_24b}"
set -a; source config/experiment.env; set +a
source .venv/bin/activate
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=8 PYTHONUNBUFFERED=1
export NCCL_ASYNC_ERROR_HANDLING=1
unset CC CXX CFLAGS CXXFLAGS CPPFLAGS LDFLAGS LD AR AS NM RANLIB STRIP OBJCOPY OBJDUMP READELF SIZE STRINGS host_alias HOST CMAKE_PREFIX_PATH CMAKE_ARGS
export CC=/usr/bin/gcc-14
export CXX=/usr/bin/g++-14
nvidia-smi
python -m torch.distributed.run --standalone --nproc_per_node=4 src/train.py
echo 'TRAIN COMPLETE -> Running Evaluation & Diagnostic...'
python src/evaluate_gpu.py
python src/diagnose_ag.py
echo 'FULL PIPELINE COMPLETE'
