from __future__ import annotations
import json, math, os, time
from pathlib import Path
import torch
import torch.distributed as dist
from torch import nn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
from safetensors.torch import save_file
from common import data_dir, OUT, RESULTS, env_int, env_float, seed_all
from data import DecisionDataset, make_collate
from decision_model import load_backbone, load_tokenizer, BitLayaDecisionModel, configure_trainable_backbone


def init_dist():
    dist.init_process_group("nccl")
    rank = dist.get_rank(); local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    return rank, local_rank, dist.get_world_size()


def move(batch, device):
    return {k: (v.to(device, non_blocking=True) if torch.is_tensor(v) else v) for k,v in batch.items()}


def reduce_pair(a: float, b: float, device):
    t = torch.tensor([a,b], dtype=torch.float64, device=device)
    dist.all_reduce(t, op=dist.ReduceOp.SUM)
    return t[0].item(), t[1].item()


def evaluate(model, loader, device):
    model.eval(); correct = 0.0; total = 0.0; loss_sum = 0.0
    ce = nn.CrossEntropyLoss(reduction="sum")
    with torch.no_grad():
        for batch in loader:
            batch = move(batch, device)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                z = model(batch["input_ids"], batch["attention_mask"], batch["opt_mask"], batch["option_mask"])
            y = batch["labels"]
            loss_sum += ce(z, y).item(); correct += (z.argmax(-1)==y).sum().item(); total += y.numel()
    correct, total = reduce_pair(correct, total, device)
    loss_sum, _ = reduce_pair(loss_sum, 0.0, device)
    model.train()
    return correct/total, loss_sum/total


def save_best(module, tokenizer, meta):
    OUT.mkdir(parents=True, exist_ok=True)
    bb = OUT / "backbone_bf16"
    module.backbone.save_pretrained(bb, safe_serialization=True)
    tokenizer.save_pretrained(bb)
    head_tensors = {k:v.detach().cpu().contiguous() for k,v in module.head.state_dict().items()}
    save_file(head_tensors, str(OUT/"decision_head.safetensors"))
    (OUT/"head_config.json").write_text(json.dumps({
        "embed_dim": module.embed_dim,
        "head_layers": env_int("HEAD_LAYERS",2),
        "head_nhead": env_int("HEAD_NHEAD",8),
        "head_ff": env_int("HEAD_FF",2048),
        "head_dropout": env_float("HEAD_DROPOUT",0.1),
        **meta,
    }, indent=2), encoding="utf-8")


def main():
    rank, local_rank, world = init_dist(); device = torch.device("cuda", local_rank)
    seed_all(env_int("SEED",42) + rank)
    ddir = data_dir()
    for split in ("train","val","test"):
        if not (ddir/f"{split}.jsonl").exists():
            raise SystemExit(f"Missing existing data: {ddir}/{split}.jsonl")

    tokenizer = load_tokenizer()
    backbone = load_backbone(training=True)
    train_info = configure_trainable_backbone(backbone, env_int("UNFREEZE_LAST_N",6))
    module = BitLayaDecisionModel(backbone, env_int("EMBED_DIM",768)).to(device)
    if rank == 0:
        print(json.dumps({"backbone_train":train_info,"embed_dim":module.embed_dim,"world_size":world},indent=2),flush=True)

    train_ds = DecisionDataset(ddir/"train.jsonl"); val_ds = DecisionDataset(ddir/"val.jsonl")
    collate = make_collate(tokenizer, env_int("MAX_LENGTH",256))
    tr_sampler = DistributedSampler(train_ds, num_replicas=world, rank=rank, shuffle=True, seed=env_int("SEED",42))
    va_sampler = DistributedSampler(val_ds, num_replicas=world, rank=rank, shuffle=False)
    tr = DataLoader(train_ds, batch_size=env_int("PER_DEVICE_BATCH",4), sampler=tr_sampler, collate_fn=collate, num_workers=env_int("NUM_WORKERS",0), pin_memory=True)
    va = DataLoader(val_ds, batch_size=env_int("PER_DEVICE_BATCH",4), sampler=va_sampler, collate_fn=collate, num_workers=0, pin_memory=True)

    bb_params = [p for p in module.backbone.parameters() if p.requires_grad]
    groups = [{"params": module.head.parameters(), "lr": env_float("HEAD_LR",2e-4)}]
    if bb_params:
        groups.append({"params": bb_params, "lr": env_float("BACKBONE_LR",2e-5)})
    opt = torch.optim.AdamW(groups, weight_decay=env_float("WEIGHT_DECAY",0.01), fused=True)

    grad_acc = env_int("GRAD_ACCUM",2); epochs = env_int("EPOCHS",2)
    updates_per_epoch = math.ceil(len(tr)/grad_acc); total_updates = max(1, updates_per_epoch*epochs)
    warm = max(1, int(total_updates*env_float("WARMUP_RATIO",0.05)))
    def lr_lambda(step):
        if step < warm: return max(1e-4, step/max(1,warm))
        p=(step-warm)/max(1,total_updates-warm)
        return 0.5*(1+math.cos(math.pi*min(1.0,p)))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)

    ddp = DDP(module, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=False)
    ce = nn.CrossEntropyLoss(); best=-1.0; global_update=0; hist=[]
    opt.zero_grad(set_to_none=True)
    start=time.time()
    for epoch in range(1,epochs+1):
        tr_sampler.set_epoch(epoch); ddp.train(); running=0.0; seen=0
        for step,batch in enumerate(tr,1):
            batch=move(batch,device)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                z=ddp(batch["input_ids"],batch["attention_mask"],batch["opt_mask"],batch["option_mask"])
                loss=ce(z,batch["labels"])/grad_acc
            loss.backward(); running += loss.item()*grad_acc*batch["labels"].numel(); seen += batch["labels"].numel()
            if step%grad_acc==0 or step==len(tr):
                torch.nn.utils.clip_grad_norm_(ddp.parameters(), env_float("MAX_GRAD_NORM",1.0))
                opt.step(); sched.step(); opt.zero_grad(set_to_none=True); global_update+=1
                if rank==0 and global_update%10==0:
                    print(json.dumps({"epoch":epoch,"update":global_update,"total_updates":total_updates,"loss":running/max(1,seen),"elapsed_min":(time.time()-start)/60}),flush=True)
        val_acc,val_nll=evaluate(ddp,va,device)
        rec={"epoch":epoch,"val_accuracy":val_acc,"val_nll":val_nll,"elapsed_min":(time.time()-start)/60}
        hist.append(rec)
        if rank==0:
            print(json.dumps(rec),flush=True)
            if val_acc>best:
                best=val_acc
                save_best(ddp.module,tokenizer,{"best_val_accuracy":best,"backbone_train":train_info})
        dist.barrier()
    if rank==0:
        RESULTS.mkdir(parents=True,exist_ok=True)
        (RESULTS/"train_history.json").write_text(json.dumps(hist,indent=2),encoding="utf-8")
        print(f"TRAIN COMPLETE best_val_accuracy={best:.6f}",flush=True)
    dist.destroy_process_group()

if __name__=="__main__": main()
