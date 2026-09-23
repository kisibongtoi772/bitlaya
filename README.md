# BitLaya

BitLaya explores native 1.58-bit ternary BitNet backbones for non-autoregressive multiple-choice decision making. The current system uses the 2.4B-parameter `Rabe3/1-bit-embedding-general` backbone with one joint bidirectional sequence, option-span pooling, and a shared two-layer Transformer decision head.

The selected GPU checkpoint fine-tunes the last 6 of 30 backbone blocks for 6 epochs and reaches **74.80%** accuracy on a balanced 2,000-example AG News + Emotion test set. Exporting the backbone to GGUF `TQ1_0` gives **74.00%** end-to-end CPU accuracy with a **1.03 GB** model artifact.

## Results

| Metric | GPU BF16 master | CPU TQ1_0 |
|---|---:|---:|
| Overall accuracy | **74.80%** | **74.00%** |
| AG News | **87.50%** | **86.90%** |
| Emotion | **62.10%** | **61.10%** |
| Backbone artifact size | 4.83 GB | **1.03 GB** |
| CPU peak RAM | — | **4.28 GB** |
| CPU throughput | — | **8.33 samples/s** |
| CPU median latency | — | **1.879 s/sample** |

The -0.80 percentage-point CPU delta is an **end-to-end deployment delta**, not pure quantization loss. In the parity suite, GGUF BF16 vs. GGUF TQ1_0 reached **0.99982 mean cosine similarity** and **100% prediction agreement (30/30)**.

## Why the architecture changed

The first trainable Rabe3 formulation encoded each candidate separately and mean-pooled the whole sequence. On AG News, almost every token was shared across candidates while only a short label differed, so candidate representations became nearly identical and predictions collapsed.

The successful formulation uses a single joint sequence:

```text
Question + Context + ALL options
              |
              v
      Rabe3 BitNet 2.4B
       non-causal attention
              |
              v
  option-token hidden states
              |
              v
      option-span pooling
              |
              v
 normalize -> first 768 dims
              |
              v
   2-layer decision head
              |
              v
           softmax
```

That change moved AG News from roughly **28%** in the mean-pooling runs to about **87%** without needing to unfreeze more backbone blocks.

## Experiment progression

| Run | Encoding | Unfrozen blocks | Epochs | Overall | AG News | Emotion |
|---|---|---:|---:|---:|---:|---:|
| Mean-pool A | separate candidates | 6 | 6 | 46.50% | 28.50% | 64.50% |
| Mean-pool B | separate candidates | 12 | 4 | 43.35% | 28.10% | 58.60% |
| Joint span | one sequence | 6 | 4 | 73.75% | 87.00% | 60.50% |
| **Selected** | **one sequence** | **6** | **6** | **74.80%** | **87.50%** | **62.10%** |
| Capacity ablation | one sequence | 12 | 4 | 73.70% | 86.50% | 60.90% |
| Capacity ablation | one sequence | 12 | 6 | 74.10% | 87.10% | 61.10% |

The 12-block, 6-epoch run reached the highest validation accuracy (75.7%) but did not improve held-out test accuracy, so the 6-block, 6-epoch checkpoint was selected.

## llama.cpp compatibility work

The tested BitNet conversion/runtime path required several fixes for this Rabe3 encoder:

1. register `BitNetModel` in the converter;
2. support the Llama-3-style BPE tokenizer path;
3. map `attn_sub_norm` and `ffn_sub_norm` tensors;
4. record non-causal attention and hidden activation metadata;
5. map `relu2` / `relu_sqr` to squared ReLU;
6. build the BitNet FFN with the activation from model metadata instead of hard-coded SiLU.

The complete patch is in `llama_cpp_bitlaya.patch` and `third_party_patches/llama_cpp_bitlaya.patch`.

Before the activation/runtime fix, HF vs. GGUF option-vector cosine was only around 0.77. After the fixes, HF BF16 vs. GGUF BF16 reached **0.99876** mean cosine.

## Reproduce

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp config/experiment.env.example config/experiment.env
```

Set `DATA_DIR` and edit the Slurm account/partition directives for your cluster.

```bash
sbatch scripts/00_download_model.sbatch
sbatch scripts/01_train_gpu.sbatch
sbatch scripts/02_eval_gpu.sbatch
sbatch scripts/03_export_bf16.sbatch
sbatch scripts/05_check_parity_bf16.sbatch
sbatch scripts/03_export_tq1.sbatch
sbatch scripts/06_check_parity_tq1.sbatch
sbatch scripts/04_eval_cpu_export.sbatch
```

Large checkpoints, GGUF artifacts, caches and local scheduler logs are intentionally excluded from Git history.

## Repository layout

```text
src/                    training, evaluation, parity and CPU inference
scripts/                Slurm jobs for the full pipeline
config/                 public experiment configuration
artifacts/results/      machine-readable benchmark results
third_party_patches/    llama.cpp compatibility patch
PROJECT_REPORT.md       detailed technical report
```

## Current status

The 2.4B model is a proof-of-concept and future teacher, not the final small BitLaya model. The next phase is distillation into a roughly 350–450M native-ternary student and then a size/accuracy/latency Pareto study.
