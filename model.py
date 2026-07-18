"""GPT Nano — minGPT's smallest config, with switchable improvements.

Baseline matches minGPT gpt-nano: learned positional embeddings, LayerNorm,
GELU MLP, dropout 0.1, no weight tying. Each improvement is a config flag so
the autoresearch loop can toggle them independently.
"""
import math
from dataclasses import dataclass

import torch
import torch.nn as nn
from torch.nn import functional as F


@dataclass
class GPTConfig:
    block_size: int = 128
    vocab_size: int = 65
    n_layer: int = 3
    n_head: int = 3
    n_embd: int = 48
    dropout: float = 0.1
    bias: bool = True
    # --- improvement flags (all off = baseline) ---
    weight_tying: bool = False
    rmsnorm: bool = False
    swiglu: bool = False
    rotary: bool = False


class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        norm = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return norm * self.weight


def make_norm(config):
    if config.rmsnorm:
        return RMSNorm(config.n_embd)
    return nn.LayerNorm(config.n_embd, bias=config.bias)


class RotaryCache:
    """Precomputed cos/sin tables for RoPE, cached per device."""

    def __init__(self, head_dim, max_seq_len, base=10000.0):
        inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2).float() / head_dim))
        t = torch.arange(max_seq_len).float()
        freqs = torch.outer(t, inv_freq)  # (T, head_dim/2)
        self.cos = freqs.cos()
        self.sin = freqs.sin()

    def to(self, device):
        self.cos = self.cos.to(device)
        self.sin = self.sin.to(device)
        return self


def apply_rotary(x, cos, sin):
    # x: (B, nh, T, hd)
    T = x.size(2)
    cos, sin = cos[:T].unsqueeze(0).unsqueeze(0), sin[:T].unsqueeze(0).unsqueeze(0)
    x1, x2 = x[..., 0::2], x[..., 1::2]
    out = torch.empty_like(x)
    out[..., 0::2] = x1 * cos - x2 * sin
    out[..., 1::2] = x1 * sin + x2 * cos
    return out


class CausalSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        self.c_proj.RESIDUAL_PROJ = True
        self.attn_dropout = config.dropout
        self.resid_dropout = nn.Dropout(config.dropout)
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.rotary = config.rotary
        if config.rotary:
            self.rope = RotaryCache(config.n_embd // config.n_head, config.block_size)

    def forward(self, x):
        B, T, C = x.size()
        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
        hd = C // self.n_head
        q = q.view(B, T, self.n_head, hd).transpose(1, 2)
        k = k.view(B, T, self.n_head, hd).transpose(1, 2)
        v = v.view(B, T, self.n_head, hd).transpose(1, 2)
        if self.rotary:
            if self.rope.cos.device != x.device:
                self.rope.to(x.device)
            q = apply_rotary(q, self.rope.cos, self.rope.sin)
            k = apply_rotary(k, self.rope.cos, self.rope.sin)
        y = F.scaled_dot_product_attention(
            q, k, v, is_causal=True,
            dropout_p=self.attn_dropout if self.training else 0.0,
        )
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.resid_dropout(self.c_proj(y))


class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.swiglu = config.swiglu
        if config.swiglu:
            # 2/3 * 4 * n_embd keeps parameter count equal to the GELU MLP
            hidden = int(2 * 4 * config.n_embd / 3)
            self.w1 = nn.Linear(config.n_embd, hidden, bias=config.bias)
            self.w3 = nn.Linear(config.n_embd, hidden, bias=config.bias)
            self.w2 = nn.Linear(hidden, config.n_embd, bias=config.bias)
            self.w2.RESIDUAL_PROJ = True
        else:
            self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd, bias=config.bias)
            self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd, bias=config.bias)
            self.c_proj.RESIDUAL_PROJ = True
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        if self.swiglu:
            x = self.w2(F.silu(self.w1(x)) * self.w3(x))
        else:
            x = self.c_proj(F.gelu(self.c_fc(x)))
        return self.dropout(x)


class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln_1 = make_norm(config)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = make_norm(config)
        self.mlp = MLP(config)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class GPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.wte = nn.Embedding(config.vocab_size, config.n_embd)
        self.wpe = None if config.rotary else nn.Embedding(config.block_size, config.n_embd)
        self.drop = nn.Dropout(config.dropout)
        self.blocks = nn.ModuleList([Block(config) for _ in range(config.n_layer)])
        self.ln_f = make_norm(config)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        if config.weight_tying:
            self.lm_head.weight = self.wte.weight
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            std = 0.02
            if getattr(module, "RESIDUAL_PROJ", False):
                std = 0.02 / math.sqrt(2 * self.config.n_layer)
            torch.nn.init.normal_(module.weight, mean=0.0, std=std)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def num_params(self):
        # parameters() deduplicates tied weights, so this is correct under tying
        return sum(p.numel() for p in self.parameters())

    def forward(self, idx, targets=None):
        B, T = idx.size()
        x = self.wte(idx)
        if self.wpe is not None:
            pos = torch.arange(T, device=idx.device)
            x = x + self.wpe(pos)
        x = self.drop(x)
        for block in self.blocks:
            x = block(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss

    def configure_optimizer(self, lr, weight_decay, betas):
        decay, no_decay = [], []
        for p in self.parameters():
            if not p.requires_grad:
                continue
            (decay if p.dim() >= 2 else no_decay).append(p)
        groups = [
            {"params": decay, "weight_decay": weight_decay},
            {"params": no_decay, "weight_decay": 0.0},
        ]
        return torch.optim.AdamW(groups, lr=lr, betas=betas)
