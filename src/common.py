from __future__ import annotations
import json, os, random
from pathlib import Path
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
ART = ROOT / "artifacts"
BASE_REPO = ART / "base_repo"
BASE_MODEL = BASE_REPO / "bf16"
OUT = ART / "finetuned"
EXPORT = ART / "export"
RESULTS = ART / "results"
CPU_EMB = ART / "cpu_embeddings"
THIRD = ROOT / "third_party"
LLAMA_CPP = THIRD / "llama.cpp"
LLAMA_EMBED = LLAMA_CPP / "build" / "bin" / "llama-embedding"

def env_int(name: str, default: int) -> int:
    return int(os.getenv(name, str(default)))

def env_float(name: str, default: float) -> float:
    return float(os.getenv(name, str(default)))

def data_dir() -> Path:
    return Path(os.environ["DATA_DIR"]).expanduser().resolve()

def read_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)

def seed_all(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
