"""Stage 2: next-concept generator (upstream models/next_concept.py, query_level0 variant).

Sequence = [prefix tokens (P)] + [level 0 | level 1 | ... | level n-1] code positions.
Prefix positions attend to the (non-pad) prefix only. Level k positions attend to the prefix and
to their own level (block-diagonal). Level 0's inputs are learned queries; level k>0's inputs are
the frozen tokenizer's partial reconstruction after levels < k, pooled to levels[k] positions.
Heads are tied to each level's codebook (negative squared distance, learned temperature).

Deviation from upstream: the prefix has its own token embedding, and RoPE places a level-k
position j at its token-space centre after the prefix (P + (j + 0.5) * block_len / l_k - 0.5)
instead of restarting each level at 0, because our prefix is left-padded so its newest token
always sits at P - 1.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import GenConfig, VQConfig
from .nn import Block, init_weights, rope_cache
from .text import PAD
from .vq import MultiscaleVQ


def sample(logits, temperature: float = 1.0, top_k: int = 0, top_p: float = 1.0):
    if temperature <= 0:
        return logits.argmax(-1)
    shape = logits.shape[:-1]
    lg = logits.reshape(-1, logits.shape[-1]).float() / temperature
    if top_k > 0:
        kth = torch.topk(lg, min(top_k, lg.shape[-1]), -1).values[:, -1:]
        lg = lg.masked_fill(lg < kth, float("-inf"))
    if top_p < 1.0:
        srt, order = lg.sort(-1, descending=True)
        p = srt.softmax(-1)
        srt = srt.masked_fill(p.cumsum(-1) - p > top_p, float("-inf"))
        lg = torch.full_like(lg, float("-inf")).scatter(-1, order, srt)
    return torch.multinomial(lg.softmax(-1), 1).view(shape)


class Generator(nn.Module):
    def __init__(self, g: GenConfig, v: VQConfig, vocab_size: int, codebook: torch.Tensor):
        super().__init__()
        self.P, self.levels = g.prefix_len, list(v.levels)
        n, E, D = len(self.levels), g.n_embd, v.embed_dim
        self.tok = nn.Embedding(vocab_size, E)
        self.inp = nn.Linear(D, E)
        self.lvl = nn.Embedding(n, E)
        self.l0_queries = nn.Parameter(torch.randn(self.levels[0], D) * D ** -0.5)
        self.blocks = nn.ModuleList(Block(E, g.n_head, g.dropout) for _ in range(g.n_layer))
        self.ln_f = nn.LayerNorm(E)
        self.head_proj = nn.ModuleList(nn.Linear(E, D, bias=False) for _ in range(n))
        self.head_log_inv_t = nn.Parameter(torch.zeros(n))
        init_weights(self.blocks, g.n_layer)
        for m in (self.tok, self.inp, self.lvl, *self.head_proj):
            nn.init.normal_(m.weight, 0.0, 0.02)

        cb = codebook.detach().clone().float()
        self.register_buffer("codebook", cb, persistent=False)
        self.register_buffer("cb_sq", cb.pow(2).sum(-1), persistent=False)
        self.register_buffer("cb_scale", cb.pow(2).sum(-1).mean(-1).clamp_min(1e-12), persistent=False)

        self.offsets = [sum(self.levels[:k]) for k in range(n)]
        level_id = torch.cat([torch.full((l,), k) for k, l in enumerate(self.levels)])
        self.register_buffer("tgt_mask", level_id[:, None] == level_id[None, :], persistent=False)
        self.register_buffer("level_id", level_id, persistent=False)
        B = v.block_len
        pos = torch.cat([torch.arange(self.P, dtype=torch.float32)] + [
            self.P + (torch.arange(l, dtype=torch.float32) + 0.5) * (B / l) - 0.5 for l in self.levels])
        cos, sin = rope_cache(pos, E // g.n_head, g.rope_base)
        self.register_buffer("rope_cos", cos, persistent=False)
        self.register_buffer("rope_sin", sin, persistent=False)

    @property
    def T(self) -> int:
        return sum(self.levels)

    def _logits(self, h, k):
        z = self.head_proj[k](h).float()
        neg_d2 = (2 * z @ self.codebook[k].t() - self.cb_sq[k]) / self.cb_scale[k]
        return neg_d2 * self.head_log_inv_t[k].exp()

    def forward(self, prefix_ids, inputs):
        """Teacher-forced logits [B, T, K]. inputs = MultiscaleVQ.teacher_inputs(codes)."""
        B, P = prefix_ids.shape
        q0 = self.l0_queries.expand(B, -1, -1).to(inputs.dtype)
        tgt = self.inp(torch.cat([q0, inputs], 1)) + self.lvl(self.level_id)
        x = torch.cat([self.tok(prefix_ids), tgt], 1)
        S = P + self.T
        keep = prefix_ids != PAD
        mask = torch.zeros(B, 1, S, S, dtype=torch.bool, device=x.device)
        mask[:, 0, :, :P] = keep[:, None, :]
        mask[:, 0, P:, P:] = self.tgt_mask
        rope = (self.rope_cos, self.rope_sin)
        for b in self.blocks:
            x, _ = b(x, mask, rope)
        h = self.ln_f(x[:, P:])
        return torch.cat([self._logits(h[:, o:o + l], k)
                          for k, (o, l) in enumerate(zip(self.offsets, self.levels))], 1)

    @torch.no_grad()
    def generate(self, prefix_ids, vq: MultiscaleVQ, temperature: float = 1.0, top_k: int = 0,
                 top_p: float = 1.0, teacher_codes=None, return_logits: bool = False):
        """One forward pass per level; the prefix is encoded once and its keys/values reused."""
        B, P = prefix_ids.shape
        keep = prefix_ids != PAD
        x = self.tok(prefix_ids)
        pmask = keep[:, None, None, :].expand(B, 1, P, P)
        rope_p = (self.rope_cos[:P], self.rope_sin[:P])
        cache = []
        for b in self.blocks:
            x, kv = b(x, pmask, rope_p)
            cache.append(kv)
        zhat, nxt = vq.empty(B, prefix_ids.device), None
        codes, all_logits = [], []
        for k, (o, l) in enumerate(zip(self.offsets, self.levels)):
            xin = self.l0_queries.expand(B, -1, -1) if k == 0 else nxt
            h = self.inp(xin.to(self.inp.weight.dtype)) + self.lvl.weight[k]
            mask = torch.cat([keep[:, None, None, :].expand(B, 1, l, P),
                              torch.ones(B, 1, l, l, dtype=torch.bool, device=h.device)], -1)
            rope = (self.rope_cos[P + o:P + o + l], self.rope_sin[P + o:P + o + l])
            for b, kv in zip(self.blocks, cache):
                h, _ = b(h, mask, rope, past=kv)
            lg = self._logits(self.ln_f(h), k)
            idx = teacher_codes[:, o:o + l] if teacher_codes is not None else sample(lg, temperature, top_k, top_p)
            codes.append(idx)
            all_logits.append(lg)
            zhat, nxt = vq.step(k, zhat, idx)
        out = torch.cat(codes, 1)
        return (out, torch.cat(all_logits, 1)) if return_logits else out


def code_loss(logits, codes):
    return F.cross_entropy(logits.reshape(-1, logits.shape[-1]).float(), codes.reshape(-1))
