from __future__ import annotations
from pathlib import Path
import torch
from torch.utils.data import Dataset
from common import read_jsonl


def format_prompt_and_spans(row):
    q = str(row["question"]).strip()
    state = str(row["state"]).strip()
    prompt = f"Question: {q}\nContext: {state}\n\nOptions:\n"
    spans = []
    for i, opt in enumerate(row["options"]):
        opt_start = len(prompt) + len(f"Option {i}: ")
        opt_end = opt_start + len(str(opt))
        prompt += f"Option {i}: {opt}\n"
        spans.append((opt_start, opt_end))
    return prompt, spans


class DecisionDataset(Dataset):
    def __init__(self, path: Path):
        self.rows = list(read_jsonl(path))
        if not self.rows:
            raise RuntimeError(f"Empty dataset: {path}")

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        return self.rows[idx]


def make_collate(tokenizer, max_length: int):
    def collate(rows):
        prompts = []
        all_spans = []
        max_opts = max(len(r["options"]) for r in rows)
        for r in rows:
            p, sp = format_prompt_and_spans(r)
            prompts.append(p)
            all_spans.append(sp)

        toks = tokenizer(
            prompts,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_offsets_mapping=True,
            return_tensors="pt",
        )
        offsets = toks.pop("offset_mapping")
        B, seq_len = toks["input_ids"].shape
        opt_mask = torch.zeros((B, max_opts, seq_len), dtype=torch.float32)
        option_mask = torch.zeros((B, max_opts), dtype=torch.bool)

        for b in range(B):
            sample_offsets = offsets[b].tolist()
            num_opts = len(rows[b]["options"])
            option_mask[b, :num_opts] = True
            for j in range(num_opts):
                sc, ec = all_spans[b][j]
                tok_idx = [idx for idx,(s,e) in enumerate(sample_offsets) if max(s,sc) < min(e,ec) and s < e]
                if tok_idx:
                    w = 1.0 / len(tok_idx)
                    for t in tok_idx:
                        opt_mask[b,j,t] = w
                else:
                    last_valid = int(toks["attention_mask"][b].sum().item()) - 1
                    opt_mask[b,j,max(0,last_valid)] = 1.0

        labels = torch.tensor([int(r["label"]) for r in rows], dtype=torch.long)
        return {
            "input_ids": toks["input_ids"],
            "attention_mask": toks["attention_mask"],
            "opt_mask": opt_mask,
            "option_mask": option_mask,
            "labels": labels,
            "tasks": [str(r.get("task", "unknown")) for r in rows],
        }
    return collate
