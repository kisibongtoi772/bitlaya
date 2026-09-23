from __future__ import annotations
import json, os
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader
from safetensors.torch import load_file
from common import data_dir, OUT, RESULTS, env_int
from data import DecisionDataset, make_collate
from decision_model import load_backbone, load_tokenizer, BitLayaDecisionModel

def main():
    cfg = json.loads((OUT / "head_config.json").read_text())
    tok = load_tokenizer(OUT / "backbone_bf16")
    bb = load_backbone(OUT / "backbone_bf16", training=False)
    model = BitLayaDecisionModel(bb, int(cfg["embed_dim"]))
    model.head.load_state_dict(load_file(str(OUT / "decision_head.safetensors")))
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device).eval()

    data_p = data_dir() if "DATA_DIR" in os.environ else Path("/users/manhnguyen/test/bitlayda/bitlaya_frozen_06b_sbatch/artifacts/data")
    ds = DecisionDataset(data_p / "test.jsonl")
    ag_rows = [r for r in ds.rows if r.get("task") == "ag_news"]
    print(f"Total AG News test samples: {len(ag_rows)}")

    class SubDataset(torch.utils.data.Dataset):
        def __init__(self, rows): self.rows = rows
        def __len__(self): return len(self.rows)
        def __getitem__(self, idx): return self.rows[idx]

    collate = make_collate(tok, env_int("MAX_LENGTH", 256))
    dl = DataLoader(SubDataset(ag_rows), batch_size=8, shuffle=False, collate_fn=collate)

    preds, trues = [], []
    with torch.no_grad():
        for b in dl:
            trues.extend(b["labels"].tolist())
            for k in ("input_ids", "attention_mask", "opt_mask", "option_mask"):
                b[k] = b[k].to(device)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=device.type=="cuda"):
                z = model(b["input_ids"], b["attention_mask"], b["opt_mask"], b["option_mask"])
            preds.extend(z.argmax(dim=-1).cpu().tolist())

    y_true = np.array(trues)
    y_pred = np.array(preds)
    classes = ag_rows[0]["options"]

    cm = np.zeros((len(classes), len(classes)), dtype=int)
    for t, p in zip(y_true, y_pred):
        cm[t, p] += 1

    print("\n=== AG NEWS CONFUSION MATRIX (Rows: True, Cols: Pred) ===")
    header = f"{'True \\ Pred':<12} | " + " | ".join(f"{c:>10}" for c in classes) + " | Total"
    print(header)
    print("-" * len(header))
    for i, c in enumerate(classes):
        row_str = " | ".join(f"{cm[i, j]:>10}" for j in range(len(classes)))
        print(f"{c:<12} | {row_str} | {cm[i].sum():>5}")

    print("\n=== PREDICTION DISTRIBUTION ===")
    pred_counts = np.bincount(y_pred, minlength=len(classes))
    for i, c in enumerate(classes):
        print(f"Class {i} ({c:<10}): predicted {pred_counts[i]:>4} times ({pred_counts[i]/len(y_pred)*100:>5.1f}%) | recall: {cm[i,i]/cm[i].sum()*100:>5.1f}%")

    overall_acc = (y_true == y_pred).mean() * 100
    print(f"\nOverall AG News Accuracy: {overall_acc:.2f}%\n")

if __name__ == "__main__":
    main()
