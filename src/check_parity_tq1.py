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


def cosine(a,b):
    na=np.linalg.norm(a); nb=np.linalg.norm(b)
    return 0.0 if na<1e-12 or nb<1e-12 else float(np.dot(a,b)/(na*nb))


def run_cpu_embedding(model_path,prompt,threads=16):
    cmd=[str(LLAMA_EMBED),"-m",str(model_path),"-p",prompt,"--embd-separator","<|BITLAYA_UNIQUE_SEP|>",
         "--attention","non-causal","--pooling","none","--embd-normalize","-1","--embd-output-format","array",
         "-t",str(threads),"-c","4096","-b","2048","-ub","2048"]
    res=subprocess.run(cmd,capture_output=True,text=True)
    if res.returncode!=0: raise RuntimeError(res.stderr)
    s=res.stdout.strip(); a=s.find("[["); b=s.rfind("]]")
    if a==-1 or b==-1: raise ValueError("No JSON array in llama-embedding output")
    return np.asarray(json.loads(s[a:b+2]),dtype=np.float32)


def extract_option_vectors(h_cpu,sample_offsets,spans,num_opts,seq_len_hf):
    bos_offset=1 if len(h_cpu)==seq_len_hf+1 else 0
    out=[]
    for j in range(num_opts):
        sc,ec=spans[j]
        tok_idx=[i+bos_offset for i,(s,e) in enumerate(sample_offsets) if max(s,sc)<min(e,ec) and s<e]
        vec=h_cpu[tok_idx].mean(axis=0) if tok_idx else h_cpu[-1]
        vec=vec/max(1e-12,np.linalg.norm(vec)); vec=vec[:768]; vec=vec/max(1e-12,np.linalg.norm(vec)); out.append(vec)
    return out


def main():
    bf16_path=EXPORT/"bitlaya-rabe3-bf16.gguf"; tq1_path=EXPORT/"bitlaya-rabe3-tq1_0.gguf"
    if not tq1_path.exists(): raise FileNotFoundError(tq1_path)
    device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    cfg=json.loads((OUT/"head_config.json").read_text())
    tok=load_tokenizer(OUT/"backbone_bf16"); bb=load_backbone(OUT/"backbone_bf16",training=False)
    gpu_model=BitLayaDecisionModel(bb,int(cfg["embed_dim"]))
    gpu_model.head.load_state_dict(load_file(str(OUT/"decision_head.safetensors"))); gpu_model.to(device).eval()
    rows=list(read_jsonl(data_dir()/"test.jsonl"))[:30]
    cos_bt=[]; cos_gt=[]; gpu_preds=[]; bf16_preds=[]; tq1_preds=[]
    for idx,r in enumerate(rows):
        prompt,spans=format_prompt_and_spans(r); num_opts=len(r["options"])
        tok_out=tok([prompt],padding=False,truncation=True,max_length=384,return_offsets_mapping=True,return_tensors="pt")
        offsets=tok_out["offset_mapping"][0].tolist(); seq_len=len(offsets)
        ids=tok_out["input_ids"].to(device); attn=tok_out["attention_mask"].to(device)
        mask=torch.zeros((1,num_opts,seq_len),dtype=torch.float32,device=device); om=torch.ones((1,num_opts),dtype=torch.bool,device=device)
        for j,(sc,ec) in enumerate(spans):
            idxs=[i for i,(s,e) in enumerate(offsets) if max(s,sc)<min(e,ec) and s<e]
            mask[0,j,idxs if idxs else [-1]]=1.0/(len(idxs) if idxs else 1)
        head_dtype=next(gpu_model.head.parameters()).dtype
        with torch.no_grad(), torch.autocast(device_type="cuda",dtype=torch.bfloat16,enabled=device.type=="cuda"):
            h=gpu_model.backbone(input_ids=ids,attention_mask=attn,use_cache=False,return_dict=True).last_hidden_state
            reps=F.normalize(torch.bmm(mask.to(h.dtype),h),p=2,dim=-1)
            if gpu_model.embed_dim<reps.shape[-1]: reps=F.normalize(reps[...,:gpu_model.embed_dim],p=2,dim=-1)
            pred_gpu=int(gpu_model.head(reps.to(dtype=head_dtype),om)[0,:num_opts].argmax().item())
        vb=extract_option_vectors(run_cpu_embedding(bf16_path,prompt),offsets,spans,num_opts,seq_len)
        vt=extract_option_vectors(run_cpu_embedding(tq1_path,prompt),offsets,spans,num_opts,seq_len)
        with torch.no_grad():
            pb=int(gpu_model.head(torch.from_numpy(np.array(vb,dtype=np.float32)).unsqueeze(0).to(device=device,dtype=head_dtype),om)[0,:num_opts].argmax().item())
            pt=int(gpu_model.head(torch.from_numpy(np.array(vt,dtype=np.float32)).unsqueeze(0).to(device=device,dtype=head_dtype),om)[0,:num_opts].argmax().item())
        gpu_preds.append(pred_gpu); bf16_preds.append(pb); tq1_preds.append(pt)
        g=reps[0,:num_opts].float().cpu().numpy()
        for j in range(num_opts): cos_bt.append(cosine(vb[j],vt[j])); cos_gt.append(cosine(g[j],vt[j]))
        print(f"Sample {idx+1:02d}: cos(BF16,TQ1)={np.mean(cos_bt[-num_opts:]):.4f}, GPU_p={pred_gpu}, BF16_p={pb}, TQ1_p={pt}",flush=True)
    print(f"BF16 vs TQ1_0 Cosine Similarity: mean = {np.mean(cos_bt):.5f} (min = {np.min(cos_bt):.5f}, max = {np.max(cos_bt):.5f})")
    print(f"GPU vs TQ1_0 Cosine Similarity: mean = {np.mean(cos_gt):.5f}")
    print(f"BF16 vs TQ1_0 Prediction Match: {100*np.mean(np.array(bf16_preds)==np.array(tq1_preds)):.1f}%")
    print(f"GPU vs TQ1_0 Prediction Match: {100*np.mean(np.array(gpu_preds)==np.array(tq1_preds)):.1f}%")

if __name__=="__main__": main()
