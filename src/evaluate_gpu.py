from __future__ import annotations
import json, math
import numpy as np
import torch
from torch import nn
from safetensors.torch import load_file
from torch.utils.data import DataLoader
from common import data_dir, OUT, RESULTS, env_int
from data import DecisionDataset, make_collate
from decision_model import load_backbone, load_tokenizer, BitLayaDecisionModel


def softmax(z):
    z=z-z.max(axis=1,keepdims=True); e=np.exp(z); return e/e.sum(axis=1,keepdims=True)
def ece(p,y,bins=15):
    conf=p.max(1); pred=p.argmax(1); ok=(pred==y); edges=np.linspace(0,1,bins+1); out=0.0
    for a,b in zip(edges[:-1],edges[1:]):
        s=(conf>a)&(conf<=b)
        if s.any(): out += s.mean()*abs(ok[s].mean()-conf[s].mean())
    return float(out)
def metrics(logits,y):
    p=softmax(logits); one=np.zeros_like(p); one[np.arange(len(y)),y]=1
    return {"accuracy":float((p.argmax(1)==y).mean()),"nll":float(-np.mean(np.log(np.clip(p[np.arange(len(y)),y],1e-12,1)))),"brier":float(np.mean(np.sum((p-one)**2,axis=1))),"ece_15":ece(p,y)}

def main():
    cfg=json.loads((OUT/"head_config.json").read_text())
    tok=load_tokenizer(OUT/"backbone_bf16")
    bb=load_backbone(OUT/"backbone_bf16",training=False)
    model=BitLayaDecisionModel(bb,int(cfg["embed_dim"]))
    model.head.load_state_dict(load_file(str(OUT/"decision_head.safetensors")))
    device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu"); model.to(device).eval()
    ds=DecisionDataset(data_dir()/"test.jsonl"); collate=make_collate(tok,env_int("MAX_LENGTH",256))
    maxo=max(len(r["options"]) for r in ds.rows)
    dl=DataLoader(ds,batch_size=env_int("PER_DEVICE_BATCH",4),shuffle=False,collate_fn=collate,num_workers=0)
    logits=[]; ys=[]; tasks=[]
    with torch.no_grad():
        for b in dl:
            tasks.extend(b["tasks"]); y=b["labels"].numpy(); ys.append(y)
            for k in ("input_ids","attention_mask","opt_mask","option_mask"): b[k]=b[k].to(device)
            with torch.autocast(device_type="cuda",dtype=torch.bfloat16,enabled=device.type=="cuda"):
                z=model(b["input_ids"],b["attention_mask"],b["opt_mask"],b["option_mask"])
            zn=z.float().cpu().numpy()
            if zn.shape[1] < maxo:
                padded=np.full((zn.shape[0], maxo), -1e4, dtype=np.float32)
                padded[:, :zn.shape[1]]=zn
                zn=padded
            logits.append(zn)
    z=np.concatenate(logits); y=np.concatenate(ys); tasks=np.asarray(tasks)
    result={"architecture":"Rabe3 bidirectional BitNet 2.4B BF16 master/ternary-forward + Laya-style 2-layer option head","overall":metrics(z,y),"per_task":{}}
    for t in sorted(set(tasks.tolist())):
        m=tasks==t; result["per_task"][t]={"n":int(m.sum()),**metrics(z[m],y[m])}
    RESULTS.mkdir(parents=True,exist_ok=True); (RESULTS/"gpu_test_metrics.json").write_text(json.dumps(result,indent=2),encoding="utf-8")
    print(json.dumps(result,indent=2))
if __name__=="__main__": main()
