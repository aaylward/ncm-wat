"""Transformer block shared by the VQ tokenizer and the generator: pre-LN, QK-norm, optional RoPE,
boolean attention masks (True = may attend) and optional prepended key/value cache."""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def rope_cache(positions: torch.Tensor, head_dim: int, base: float) -> tuple[torch.Tensor, torch.Tensor]:
    half = head_dim // 2
    inv = 1.0 / (base ** (torch.arange(half, dtype=torch.float32) / half))
    f = positions.float()[:, None] * inv[None, :]
    return torch.cat([f.cos(), f.cos()], -1), torch.cat([f.sin(), f.sin()], -1)


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    half = x.shape[-1] // 2
    rot = torch.cat([-x[..., half:], x[..., :half]], -1)
    return x * cos.to(x.dtype) + rot * sin.to(x.dtype)


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x):
        xf = x.float()
        return (xf * xf.pow(2).mean(-1, keepdim=True).add(self.eps).rsqrt()).to(x.dtype) * self.weight


class Attention(nn.Module):
    def __init__(self, dim: int, n_head: int, dropout: float):
        super().__init__()
        self.n_head, self.dropout = n_head, dropout
        self.qkv = nn.Linear(dim, 3 * dim)
        self.proj = nn.Linear(dim, dim)
        self.q_norm = RMSNorm(dim // n_head)
        self.k_norm = RMSNorm(dim // n_head)
        self.drop = nn.Dropout(dropout)

    def forward(self, x, mask=None, rope=None, past=None):
        B, T, C = x.shape
        q, k, v = self.qkv(x).view(B, T, 3, self.n_head, C // self.n_head).permute(2, 0, 3, 1, 4)
        q, k = self.q_norm(q), self.k_norm(k)
        if rope is not None:
            q, k = apply_rope(q, *rope), apply_rope(k, *rope)
        kv = (k, v)
        if past is not None:
            k, v = torch.cat([past[0], k], 2), torch.cat([past[1], v], 2)
        y = F.scaled_dot_product_attention(
            q, k, v, attn_mask=mask, dropout_p=self.dropout if self.training else 0.0)
        return self.drop(self.proj(y.transpose(1, 2).reshape(B, T, C))), kv


class Block(nn.Module):
    def __init__(self, dim: int, n_head: int, dropout: float):
        super().__init__()
        self.ln1 = nn.LayerNorm(dim)
        self.attn = Attention(dim, n_head, dropout)
        self.ln2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(nn.Linear(dim, 4 * dim), nn.GELU(), nn.Linear(4 * dim, dim), nn.Dropout(dropout))

    def forward(self, x, mask=None, rope=None, past=None):
        a, kv = self.attn(self.ln1(x), mask, rope, past)
        x = x + a
        return x + self.mlp(self.ln2(x)), kv


def init_weights(module: nn.Module, n_layer: int) -> None:
    for name, p in module.named_parameters():
        if p.dim() < 2:
            continue
        std = 0.02 / (2 * n_layer) ** 0.5 if name.endswith(("proj.weight", "mlp.2.weight")) else 0.02
        nn.init.normal_(p, 0.0, std)
