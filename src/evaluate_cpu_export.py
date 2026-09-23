from __future__ import annotations
import argparse, json, resource, subprocess, time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
import numpy as np
import torch
from safetensors.torch import load_file
from tqdm import tqdm

from common import data_dir, OUT, RESULTS, LLAMA_EMBED, read_jsonl
from data import format_prompt_and_spans
from decision_model import load_tokenizer, SharedDecisionHead


def run_single_embedding(model_path: Path, prompt: str, threads: int) -> np.ndarray:
    cmd = [
        str(LLAMA_EMBED), "-m", str(model_path), "-p", prompt,
        "--embd-separator", "<|BITLAYA_UNIQUE_SEP|>",
        "--attention", "non-causal", "--pooling", "none",
        "--embd-normalize", "-1", "--embd-output-format", "array",
        "-t", str(threads), "-c", "4096", "-b", "2048", "-ub", "2048",
    ]
    res = subprocess.run(cmd, capture_output=True, text=True)
    if res.returncode != 0:
        raise RuntimeError(f"llama-embedding failed (code {res.returncode}): {res.stderr[-500:]}")
    s = res.stdout.strip()
    start_idx = s.find("[["); end_idx = s.rfind("]]")
    if start_idx == -1 or end_idx == -1:
        raise ValueError(f"No JSON array in llama-embedding output. stdout:\n{s[:300]}")
    return np.asarray(json.loads(s[start_idx:end_idx+2]), dtype=np.float32)


def process_sample(args_tuple):
    idx, row, prompt, spans, sample_offsets, model_path, threads = args_tuple
    num_opts = len(row["options"])
    t0 = time.time()
    h_cpu = run_single_embedding(model_path, prompt, threads)
    dt = time.time() - t0
    opt_vecs = []
    for j in range(num_opts):
        sc, ec = spans[j]
        tok_idx = [i for i,(s,e) in enumerate(sample_offsets) if max(s,sc) < min(e,ec) and s < e and i < len(h_cpu)]
        vec = h_cpu[tok_idx].mean(axis=0) if tok_idx else h_cpu[-1]
        vec = vec / max(1e-12, np.linalg.norm(vec))
        vec = vec[:768]
        vec = vec / max(1e-12, np.linalg.norm(vec))
        opt_vecs.append(vec)
    return idx, opt_vecs, dt


def main():
    parser = argparse.ArgumentParser(description="Full 2000-sample CPU Benchmark for BitLaya TQ1_0")
    parser.add_argument("--model", default="artifacts/export/bitlaya-rabe3-tq1_0.gguf")
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--threads", type=int, default=16)
    parser.add_argument("--split", default="test")
    parser.add_argument("--max-samples", type=int, default=0)
    args = parser.parse_args()

    model_path = Path(args.model)
    if not model_path.exists():
        raise FileNotFoundError(model_path)
    model_size_bytes = model_path.stat().st_size
    model_size_gb = model_size_bytes / (1024**3)

    cfg = json.loads((OUT / "head_config.json").read_text())
    tok = load_tokenizer(OUT / "backbone_bf16")
    embed_dim = int(cfg["embed_dim"])
    head = SharedDecisionHead(
        dim=embed_dim,
        layers=int(cfg["head_layers"]),
        nhead=int(cfg["head_nhead"]),
        ff=int(cfg["head_ff"]),
        dropout=float(cfg.get("head_dropout", 0.1)),
    )
    head.load_state_dict(load_file(str(OUT / "decision_head.safetensors")))
    head.eval()

    rows = list(read_jsonl(data_dir() / f"{args.split}.jsonl"))
    if args.max_samples > 0:
        rows = rows[:args.max_samples]

    work_items = []
    for idx, r in enumerate(rows):
        prompt, spans = format_prompt_and_spans(r)
        tok_out = tok([prompt], padding=False, truncation=True, max_length=384,
                      return_offsets_mapping=True, return_tensors="pt")
        work_items.append((idx, r, prompt, spans, tok_out["offset_mapping"][0].tolist(), model_path, args.threads))

    bench_start = time.time()
    results_by_idx = [None] * len(rows)
    latencies = []
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(process_sample, item): item[0] for item in work_items}
        for fut in tqdm(as_completed(futures), total=len(rows), desc="Benchmarking"):
            idx, opt_vecs, dt = fut.result()
            results_by_idx[idx] = opt_vecs
            latencies.append(dt)

    total_bench_time = time.time() - bench_start
    throughput = len(rows) / total_bench_time
    max_opts = max(len(r["options"]) for r in rows)
    X = np.zeros((len(rows), max_opts, embed_dim), dtype=np.float32)
    option_mask = np.zeros((len(rows), max_opts), dtype=bool)
    y_true = np.array([r["label"] for r in rows], dtype=np.int64)
    tasks = np.array([r.get("task", "unknown") for r in rows])

    for i, opt_vecs in enumerate(results_by_idx):
        X[i, :len(opt_vecs)] = np.array(opt_vecs, dtype=np.float32)
        option_mask[i, :len(opt_vecs)] = True

    with torch.no_grad():
        logits = head(torch.from_numpy(X), torch.from_numpy(option_mask)).numpy()

    y_pred = np.argmax(logits, axis=1)
    correct = y_pred == y_true
    overall_acc = float(np.mean(correct)) * 100
    task_results = {}
    for task in sorted(set(tasks)):
        m = tasks == task
        task_results[task] = {"n": int(m.sum()), "accuracy": float(np.mean(correct[m])) * 100}

    max_rss_kb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss + resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss
    summary = {
        "architecture": "BitLaya 2.4B TQ1_0 (Ternary 1.58-bit) + SharedDecisionHead",
        "checkpoint": "6_blocks_6_epochs",
        "split": args.split,
        "n_samples": len(rows),
        "overall_accuracy": round(overall_acc, 2),
        "delta_vs_gpu_overall": round(overall_acc - 74.80, 2),
        "per_task": {k: {"n": v["n"], "accuracy": round(v["accuracy"], 2)} for k,v in task_results.items()},
        "performance": {
            "total_wall_time_s": round(total_bench_time, 2),
            "throughput_samples_per_s": round(throughput, 2),
            "avg_latency_s": round(float(np.mean(latencies)), 3),
            "p50_latency_s": round(float(np.percentile(latencies, 50)), 3),
            "p90_latency_s": round(float(np.percentile(latencies, 90)), 3),
            "workers": args.workers,
            "threads_per_worker": args.threads,
        },
        "resource": {
            "model_size_gb": round(model_size_gb, 2),
            "model_size_mb": round(model_size_bytes / (1024**2), 1),
            "peak_ram_gb": round(max_rss_kb / (1024**2), 2),
        },
        "baselines_gpu_bf16": {"overall_accuracy": 74.80, "ag_news": 87.50, "emotion": 62.10},
    }

    RESULTS.mkdir(parents=True, exist_ok=True)
    out_file = RESULTS / f"cpu_export_{model_path.stem}_{args.split}_metrics.json"
    out_file.write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
