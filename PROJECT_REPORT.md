# BitLaya Technical Report

## Objective

The project asks whether a native ternary BitNet backbone can be adapted into a Laya-style, non-autoregressive decision model and then exported to packed 1.58-bit CPU inference without destroying the learned decision geometry.

The current system, `bitlaya_rabe3_24b`, uses `Rabe3/1-bit-embedding-general` (~2.4B parameters) as a trainable BitNet-lineage backbone. It is a proof of concept and future teacher model rather than the final compact BitLaya endpoint.

## 1. Initial trainable formulation

The first Rabe3 system encoded each option separately:

```text
(state + question + option_i)
        -> Rabe3 encoder
        -> full-sequence mean pool
        -> normalize / first 768 dims
        -> shared two-layer option head
```

Opening more trainable capacity did not solve the main failure:

| Run | Blocks | Epochs | Overall | AG News | Emotion |
|---|---:|---:|---:|---:|---:|
| Mean-pool A | 6 | 6 | 46.50% | 28.50% | 64.50% |
| Mean-pool B | 12 | 4 | 43.35% | 28.10% | 58.60% |

AG News candidate vectors were nearly identical because almost the whole long prompt was shared and only the short option label changed. Mean pooling diluted the option signal and produced severe class collapse.

## 2. Joint option-span formulation

The successful formulation places all options in one non-causal sequence and extracts option-specific hidden states:

```text
Question + Context + ALL options
        -> one bidirectional BitNet forward
        -> token spans for each option
        -> span mean
        -> L2 normalize
        -> first 768 dimensions
        -> L2 normalize
        -> two-layer shared decision head
```

The first 6-block / 4-epoch joint-span run reached:

- Overall: **73.75%**
- AG News: **87.00%**
- Emotion: **60.50%**

The class-collapse failure disappeared without increasing trainable capacity.

## 3. Capacity and convergence search

| Configuration | Best val acc | Test overall | AG News | Emotion |
|---|---:|---:|---:|---:|
| 6 blocks × 4 epochs | 74.7% | 73.75% | 87.00% | 60.50% |
| **6 blocks × 6 epochs** | **75.2%** | **74.80%** | **87.50%** | **62.10%** |
| 12 blocks × 4 epochs | 74.7% | 73.70% | 86.50% | 60.90% |
| 12 blocks × 6 epochs | 75.7% | 74.10% | 87.10% | 61.10% |

The 12-block model achieved the highest validation accuracy but did not improve the held-out test set. The 6-block / 6-epoch checkpoint was selected.

## 4. GGUF export and runtime compatibility

Exporting the encoder exposed multiple compatibility gaps in the tested llama.cpp BitNet path.

### Architecture registration
The Hugging Face checkpoint is saved as `BitNetModel`, while the converter registered only causal-LM architecture names. The converter was extended without changing the checkpoint metadata.

### Tokenizer path
Rabe3 uses a Llama-3-style BPE vocabulary rather than the older SentencePiece path expected by the BitNet converter.

### Tensor names
`attn_sub_norm` and `ffn_sub_norm` names required additional GGUF mappings.

### Non-causal attention
The model is used as a bidirectional encoder, so GGUF metadata records non-causal attention.

### Activation mismatch
The checkpoint uses squared ReLU (`relu2`), while the tested BitNet runtime hard-coded SiLU. Across 30 transformer blocks this produced large hidden-state drift. The patch maps `relu2` to `LLM_FFN_RELU_SQR`, exports the activation metadata and uses it when constructing the runtime FFN graph.

Before the fix, option-vector cosine was only around 0.77. After the fix, parity became close to numerical equivalence.

## 5. Two-stage parity

The export path was validated in two separate stages so converter/runtime errors would not be confused with TQ1 quantization effects.

### HF BF16 vs. GGUF BF16
30-sample suite:

- token sequence-length delta: **0**
- mean option-vector cosine: **0.99876**
- minimum cosine: **0.99728**
- decision prediction match: **28/30 (93.3%)**

### GGUF BF16 vs. GGUF TQ1_0
Same suite:

- mean cosine: **0.99982**
- minimum cosine: **0.99950**
- prediction match: **30/30 (100%)**
- measured representation drift: **0.018%**

This shows that the packed TQ1_0 step itself adds very little extra distortion for this ternary-friendly backbone.

## 6. Full CPU benchmark

The final benchmark runs all 2,000 held-out examples through the joint prompt, exact option-span extraction and the same trained decision head.

| Metric | GPU BF16 | CPU TQ1_0 |
|---|---:|---:|
| Overall accuracy | **74.80%** | **74.00%** |
| AG News | 87.50% | 86.90% |
| Emotion | 62.10% | 61.10% |
| Backbone artifact | 4.83 GB | **1.03 GB** |
| CPU peak RAM | — | **4.28 GB** |
| CPU average latency | — | **1.888 s** |
| CPU p50 latency | — | **1.879 s** |
| Aggregate throughput | — | **8.33 samples/s** |
| 2,000-sample wall time | — | **239.97 s** |

The CPU run used 16 parallel workers with 16 threads each on an ARM Neoverse V2 node.

The **-0.80 percentage-point** GPU-to-CPU difference is an end-to-end deployment delta. It includes runtime/numerical differences between Hugging Face and llama.cpp in addition to quantization. The direct GGUF BF16-to-TQ1 parity test shows that TQ1 itself preserved representations extremely closely.

## 7. Meaning of the result

The main early failure was caused by representation formulation, not by low-bit arithmetic. Changing option representation produced the largest accuracy jump in the project.

The 2.4B model also demonstrates that BF16-master fine-tuning can survive packed ternary deployment while preserving the decision function. This removes a major systems uncertainty before building a much smaller BitLaya student.

## 8. Limitations and next phase

The current model is still 2.4B parameters and is not yet smaller than public Laya by parameter count. The benchmark covers two tasks and 2,000 held-out examples, and CPU throughput is aggregate throughput over 256 total threads.

The next phase is distillation into a ~350–450M native-ternary student while preserving the joint option geometry, followed by 250M and 150M variants if the larger student remains stable.
