from __future__ import annotations
import math
from typing import Any
import torch
from torch import nn
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer
from common import BASE_MODEL, env_int, env_float


def _bidirectional_mask(*args, **kwargs):
    """Replacement for Transformers' BitNet causal-mask builder.

    Returns an additive full-attention mask with only padding keys blocked.
    This keeps the Rabe3/LLM2Vec representation path non-causal during GPU fine-tuning.
    """
    inputs_embeds = kwargs.get("inputs_embeds")
    attention_mask = kwargs.get("attention_mask")
    if inputs_embeds is None:
        for x in args:
            if torch.is_tensor(x) and x.ndim == 3:
                inputs_embeds = x
                break
    if inputs_embeds is None:
        raise RuntimeError("Could not locate inputs_embeds while building bidirectional mask")
    bsz, q_len = inputs_embeds.shape[:2]
    dtype, device = inputs_embeds.dtype, inputs_embeds.device
    if attention_mask is None:
        return torch.zeros((bsz, 1, q_len, q_len), dtype=dtype, device=device)
    if attention_mask.ndim == 4:
        return attention_mask.to(device=device, dtype=dtype)
    key_len = attention_mask.shape[-1]
    minv = torch.finfo(dtype).min
    blocked = (1.0 - attention_mask.to(dtype=dtype, device=device))[:, None, None, :] * minv
    return blocked.expand(bsz, 1, q_len, key_len)


def install_bidirectional_bitnet_patch() -> None:
    try:
        import transformers.models.bitnet.modeling_bitnet as mb
    except Exception as exc:
        raise RuntimeError("Transformers BitNet implementation is unavailable") from exc
    if not hasattr(mb, "create_causal_mask"):
        raise RuntimeError("Unsupported transformers version: BitNet create_causal_mask not found")
    mb.create_causal_mask = _bidirectional_mask


def find_layers(model: nn.Module) -> list[nn.Module]:
    candidates = ["layers", "model.layers", "transformer.h", "encoder.layer", "backbone.layers"]
    for path in candidates:
        cur: Any = model
        ok = True
        for part in path.split("."):
            if not hasattr(cur, part):
                ok = False
                break
            cur = getattr(cur, part)
        if ok and isinstance(cur, (nn.ModuleList, list, tuple)) and len(cur) > 0:
            return list(cur)
    found = []
    for name, mod in model.named_modules():
        if isinstance(mod, nn.ModuleList) and len(mod) > len(found):
            found = list(mod)
    if not found:
        raise RuntimeError("Could not identify transformer layers for selective unfreezing")
    return found


def configure_trainable_backbone(model: nn.Module, last_n: int) -> dict:
    for p in model.parameters():
        p.requires_grad = False
    layers = find_layers(model)
    if last_n < 0:
        for p in model.parameters():
            p.requires_grad = True
        mode = "full"
    elif last_n == 0:
        mode = "frozen"
    else:
        n = min(last_n, len(layers))
        for layer in layers[-n:]:
            for p in layer.parameters():
                p.requires_grad = True
        for name, mod in model.named_modules():
            lname = name.lower()
            if lname.endswith("norm") or lname.endswith("final_layernorm"):
                if not any(True for _ in mod.children()):
                    for p in mod.parameters(recurse=False):
                        p.requires_grad = True
        mode = f"last_{n}"
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    return {"mode": mode, "layers": len(layers), "trainable": trainable, "total": total}


class SharedDecisionHead(nn.Module):
    def __init__(self, dim: int, layers: int, nhead: int, ff: int, dropout: float):
        super().__init__()
        enc = nn.TransformerEncoderLayer(
            d_model=dim, nhead=nhead, dim_feedforward=ff, dropout=dropout,
            activation="gelu", batch_first=True, norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(enc, num_layers=layers, norm=nn.LayerNorm(dim))
        self.scorer = nn.Linear(dim, 1)

    def forward(self, x: torch.Tensor, option_mask: torch.Tensor) -> torch.Tensor:
        h = self.encoder(x, src_key_padding_mask=~option_mask)
        z = self.scorer(h).squeeze(-1)
        return z.masked_fill(~option_mask, -1e4)


def mean_pool(last_hidden: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    m = attention_mask.unsqueeze(-1).to(last_hidden.dtype)
    return (last_hidden * m).sum(1) / m.sum(1).clamp_min(1.0)


class BitLayaDecisionModel(nn.Module):
    def __init__(self, backbone: nn.Module, embed_dim: int):
        super().__init__()
        self.backbone = backbone
        self.embed_dim = embed_dim
        hidden = int(getattr(backbone.config, "hidden_size"))
        if embed_dim <= 0 or embed_dim > hidden:
            self.embed_dim = hidden
        self.head = SharedDecisionHead(
            dim=self.embed_dim,
            layers=env_int("HEAD_LAYERS", 2),
            nhead=env_int("HEAD_NHEAD", 8),
            ff=env_int("HEAD_FF", 2048),
            dropout=env_float("HEAD_DROPOUT", 0.1),
        )

    def forward(self, input_ids, attention_mask, opt_mask, option_mask):
        out = self.backbone(input_ids=input_ids, attention_mask=attention_mask, use_cache=False, return_dict=True)
        h = out.last_hidden_state
        opt_reps = torch.bmm(opt_mask.to(h.dtype), h)
        opt_reps = F.normalize(opt_reps, p=2, dim=-1)
        if self.embed_dim < opt_reps.shape[-1]:
            opt_reps = opt_reps[..., : self.embed_dim]
            opt_reps = F.normalize(opt_reps, p=2, dim=-1)
        return self.head(opt_reps, option_mask)


def load_tokenizer(path=BASE_MODEL):
    tok = AutoTokenizer.from_pretrained(path, local_files_only=True, use_fast=True)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    return tok


def load_backbone(path=BASE_MODEL, training: bool = True):
    install_bidirectional_bitnet_patch()
    model = AutoModel.from_pretrained(
        path,
        local_files_only=True,
        torch_dtype=torch.bfloat16,
        attn_implementation="eager",
    )
    model.config.use_cache = False
    for mod in model.modules():
        if hasattr(mod, "is_causal"):
            try:
                mod.is_causal = False
            except Exception:
                pass
    if training:
        model.train()
    else:
        model.eval()
    return model
