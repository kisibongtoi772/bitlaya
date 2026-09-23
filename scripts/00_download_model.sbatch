#!/bin/bash
#SBATCH --account=a0152
#SBATCH --job-name=bitlaya-dl
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --partition=debug
#SBATCH --constraint=gpu
#SBATCH --cpus-per-task=64
#SBATCH --gpus-per-task=1
#SBATCH --mem=120G
#SBATCH --time=00:30:00
#SBATCH --output=slurm-bitlaya-dl-%j.out
#SBATCH --error=slurm-bitlaya-dl-%j.err
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-/users/manhnguyen/test/bitlayda/bitlaya_rabe3_24b}"
set -a; source config/experiment.env; set +a
if [ ! -d .venv ]; then python3 -m venv --system-site-packages .venv; fi
source .venv/bin/activate
python -m pip install -U pip wheel setuptools
python -m pip install -r requirements.txt
mkdir -p artifacts/base_repo third_party artifacts/results
python - <<'PY'
from huggingface_hub import snapshot_download
snapshot_download(repo_id="Rabe3/1-bit-embedding-general", allow_patterns=["bf16/*","README.md","LICENSE*"], local_dir="artifacts/base_repo")
PY
if [ ! -f artifacts/base_repo/bf16/model.safetensors ]; then echo 'missing bf16/model.safetensors'; exit 2; fi
if [ ! -d third_party/llama.cpp/.git ]; then git clone --depth 1 https://github.com/ggml-org/llama.cpp.git third_party/llama.cpp; fi
CC_BIN=$(command -v gcc-14 || command -v gcc-13 || command -v clang || command -v gcc || command -v cc)
CXX_BIN=$(command -v g++-14 || command -v g++-13 || command -v clang++ || command -v g++ || command -v c++)
unset CFLAGS CXXFLAGS CPPFLAGS LDFLAGS
cmake -S third_party/llama.cpp -B third_party/llama.cpp/build \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_C_COMPILER="$CC_BIN" \
  -DCMAKE_CXX_COMPILER="$CXX_BIN" \
  -DGGML_CUDA=OFF
cmake --build third_party/llama.cpp/build -j 64 --target llama-embedding
python - <<'PY'
from pathlib import Path
import os
p=Path(os.environ['DATA_DIR']).expanduser()
for s in ('train','val','test'):
 f=p/f'{s}.jsonl'
 if not f.exists(): raise SystemExit(f'Missing existing dataset: {f}')
print('MODEL + TOOLING + EXISTING DATA CHECK: OK')
PY
