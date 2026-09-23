from __future__ import annotations
import json, subprocess, time
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from safetensors.torch import load_file

from common import data_dir, OUT, EXPORT, LLAMA_EMBED, read_jsonl
from data import format_prompt_and_spans
from decision_model import load_backbone, load_tokenizer, BitLayaDecisionModel


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    na = np.linalg.norm(a); nb = np.linalg.norm(b)
    if na < 1e-12 or nb < 1e-12: return 0.0
    return float(np.dot(a, b) / (na * nb))


def run_cpu_embedding(model_path: Path, prompt: str, threads: int = 16) -> np.ndarray:
    cmd = [str(LLAMA_EMBED), "-m", str(model_path), "-p", prompt,
           "--embd-separator", "<|BITLAYA_UNIQUE_SEP|>",
           "--attention", "non-causal", "--pooling", "none",
           "--embd-normalize", "-1", "--embd-output-format", "array",
           "-t", str(threads), "-c", "4096", "-b", "2048", "-ub", "2048"]
    res = subprocess.run(cmd, capture_output=True, text=True)
    if res.returncode != 0:
        raise RuntimeError(f"llama-embedding failed (code {res.returncode}): {res.stderr}")
    s = res.stdout.strip(); start_idx = s.find("[["); end_idx = s.rfind("]]")
    if start_idx == -1 or end_idx == -1:
        raise ValueError(f"No JSON array in llama-embedding output. stdout:\n{s[:300]}\nstderr:\n{res.stderr[:300]}")
    return np.asarray(json.loads(s[start_idx:end_idx+2]), dtype=np.float32)


def main():
    print("=" * 65); print("SANITY CHECK: GPU BF16 vs CPU GGUF BF16 PARITY"); print("=" * 65)
    gguf_path = EXPORT / "bitlaya-rabe3-bf16.gguf"
    if not gguf_path.exists(): raise FileNotFoundError(gguf_path)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    cfg = json.loads((OUT / "head_config.json").read_text())
    tok = load_tokenizer(OUT / "backbone_bf16")
    bb = load_backbone(OUT / "backbone_bf16", training=False)
    gpu_model = BitLayaDecisionModel(bb, int(cfg["embed_dim"]))
    gpu_model.head.load_state_dict(load_file(str(OUT / "decision_head.safetensors")))
    gpu_model.to(device).eval()
    test_rows = list(read_jsonl(data_dir() / "test.jsonl"))[:30]
    cos_sims=[]; token_len_deltas=[]; gpu_preds=[]; cpu_preds=[]
    for idx,r in enumerate(test_rows):
        prompt, spans = format_prompt_and_spans(r); num_opts=len(r["options"])
        tok_out = tok([prompt],padding=False,truncation=True,max_length=384,return_offsets_mapping=True,return_tensors="pt")
        sample_offsets=tok_out["offset_mapping"][0].tolist(); seq_len_hf=len(sample_offsets)
        input_ids=tok_out["input_ids"].to(device); attention_mask=tok_out["attention_mask"].to(device)
        opt_mask_gpu=torch.zeros((1,num_opts,seq_len_hf),dtype=torch.float32,device=device)
        option_mask_gpu=torch.ones((1,num_opts),dtype=torch.bool,device=device)
        for j in range(num_opts):
            sc,ec=spans[j]
            tok_idx=[i for i,(s,e) in enumerate(sample_offsets) if max(s,sc)<min(e,ec) and s<e]
            if tok_idx: opt_mask_gpu[0,j,tok_idx]=1.0/len(tok_idx)
            else: opt_mask_gpu[0,j,-1]=1.0
        with torch.no_grad(), torch.autocast(device_type="cuda",dtype=torch.bfloat16,enabled=device.type=="cuda"):
            out=gpu_model.backbone(input_ids=input_ids,attention_mask=attention_mask,use_cache=False,return_dict=True)
            h_gpu=out.last_hidden_state
            opt_reps_gpu=torch.bmm(opt_mask_gpu.to(h_gpu.dtype),h_gpu)
            opt_reps_gpu=F.normalize(opt_reps_gpu,p=2,dim=-1)[...,:768]
            opt_reps_gpu=F.normalize(opt_reps_gpu,p=2,dim=-1)
            logits_gpu=gpu_model.head(opt_reps_gpu,option_mask_gpu)
            pred_gpu=int(logits_gpu[0,:num_opts].argmax().item())
        t0=time.time(); h_cpu=run_cpu_embedding(gguf_path,prompt,threads=16); dt=time.time()-t0
        seq_len_gguf=len(h_cpu); token_len_deltas.append(seq_len_gguf-seq_len_hf)
        bos_offset=1 if seq_len_gguf==seq_len_hf+1 else 0
        cpu_opt_vecs=[]
        for j in range(num_opts):
            sc,ec=spans[j]
            tok_idx=[i+bos_offset for i,(s,e) in enumerate(sample_offsets) if max(s,sc)<min(e,ec) and s<e]
            vec=h_cpu[tok_idx].mean(axis=0) if tok_idx else h_cpu[-1]
            vec=vec/max(1e-12,np.linalg.norm(vec)); vec=vec[:768]; vec=vec/max(1e-12,np.linalg.norm(vec))
            cpu_opt_vecs.append(vec)
        gpu_np=opt_reps_gpu[0,:num_opts].float().cpu().numpy()
        for j in range(num_opts): cos_sims.append(cosine(gpu_np[j],cpu_opt_vecs[j]))
        with torch.no_grad():
            cpu_tensor=torch.from_numpy(np.array(cpu_opt_vecs,dtype=np.float32)).unsqueeze(0)
            logits_cpu=gpu_model.head(cpu_tensor.to(device),option_mask_gpu)
            pred_cpu=int(logits_cpu[0,:num_opts].argmax().item())
        gpu_preds.append(pred_gpu); cpu_preds.append(pred_cpu)
        print(f"Sample {idx+1:02d}: opts={num_opts}, HF_len={seq_len_hf}, GGUF_len={seq_len_gguf}, opt_cos={np.mean(cos_sims[-num_opts:]):.4f}, GPU_pred={pred_gpu}, CPU_pred={pred_cpu}, {'OK' if pred_gpu==pred_cpu else 'DIFF'} ({dt:.2f}s)",flush=True)
    mean_cos=float(np.mean(cos_sims)); pred_match=float(np.mean(np.array(gpu_preds)==np.array(cpu_preds)))*100.0
    print(f"Option Vector Cosine Similarity: mean = {mean_cos:.5f} (min = {np.min(cos_sims):.5f}, max = {np.max(cos_sims):.5f})")
    print(f"Sequence Length Delta (GGUF - HF): {set(token_len_deltas)}")
    print(f"Decision Head Prediction Match: {pred_match:.1f}%")
    print("PASS" if mean_cos>=0.98 and pred_match>=90.0 else "FAIL")

if __name__=="__main__": main()
