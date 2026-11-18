"""Stage 1: multiscale residual VQ tokenizer.

A block of `block_len` tokens is embedded to z [B, L, D] and quantized coarse-to-fine: level i
area-pools the remaining residual to `levels[i]` positions, snaps each to the nearest code of its
own EMA-updated codebook, upsamples back to L, and subtracts. A shallow decoder maps the summed
reconstruction back to token logits. Mirrors upstream models/quant.py (EMA codebooks, Phi residual
convs, learned level-0 sampler with straight-through, fine-level dropout) without the research knobs.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import VQConfig
from .nn import Block


def _pool(x, size):  # [B, L, D] -> [B, size, D]
    return F.interpolate(x.transpose(1, 2), size=size, mode="area").transpose(1, 2)


class Phi(nn.Module):
    """Residual conv applied after upsampling a level: (1 - r) * h + r * conv(h)."""

    def __init__(self, dim: int, ratio: float):
        super().__init__()
        self.conv = nn.Conv1d(dim, dim, 3, padding=1)
        self.ratio = ratio

    def forward(self, h):
        t = h.transpose(1, 2)
        return (t * (1 - self.ratio) + self.conv(t) * self.ratio).transpose(1, 2)


class MultiscaleVQ(nn.Module):
    def __init__(self, cfg: VQConfig):
        super().__init__()
        self.levels = list(cfg.levels)
        self.L, D, K, n = cfg.block_len, cfg.embed_dim, cfg.codebook_size, len(cfg.levels)
        self.commitment, self.decay = cfg.commitment, cfg.ema_decay
        self.fine_drop, self.restart = cfg.fine_drop, cfg.restart_threshold
        self.register_buffer("codebook", torch.randn(n, K, D))
        # EMA counts start at 1 so unused codes keep their scale instead of blowing up.
        self.register_buffer("ema_count", torch.ones(n, K))
        self.register_buffer("ema_sum", self.codebook.clone())
        self.register_buffer("initialized", torch.zeros(n, dtype=torch.bool))
        self.phi = nn.ModuleList(Phi(D, cfg.quant_resi) if cfg.quant_resi > 0 else nn.Identity()
                                 for _ in range(n))
        self.down0 = self.up0 = None
        if cfg.learned_l0 and n > 1:
            s = self.L // self.levels[0]
            self.down0 = nn.Conv1d(D, D, s, stride=s)
            self.up0 = nn.ConvTranspose1d(D, D, s, stride=s)
            with torch.no_grad():  # start as average-pool / nearest-repeat
                eye = torch.arange(D)
                for conv, val in ((self.down0, 1.0 / s), (self.up0, 1.0)):
                    conv.weight.zero_()
                    conv.weight[eye, eye, :] = val
                    conv.bias.zero_()

    @property
    def n_levels(self) -> int:
        return len(self.levels)

    def _down(self, z, i):
        if i == self.n_levels - 1:
            return z
        if i == 0 and self.down0 is not None:
            return self.down0(z.transpose(1, 2)).transpose(1, 2)
        return _pool(z, self.levels[i])

    def _up(self, q, i):
        if i == self.n_levels - 1:
            u = q
        elif i == 0 and self.up0 is not None:
            u = self.up0(q.transpose(1, 2)).transpose(1, 2)
        else:
            u = F.interpolate(q.transpose(1, 2), size=self.L, mode="linear").transpose(1, 2)
        return self.phi[i](u)

    def _nearest(self, x, i):
        e = self.codebook[i]
        d = x.pow(2).sum(1, keepdim=True) + e.pow(2).sum(1) - 2 * x @ e.t()
        return d.argmin(1)

    @torch.no_grad()
    def _update(self, i, x, idx):
        K = self.codebook.shape[1]
        cnt = torch.bincount(idx, minlength=K).float()
        tot = torch.zeros_like(self.ema_sum[i]).index_add_(0, idx, x)
        self.ema_count[i].mul_(self.decay).add_(cnt, alpha=1 - self.decay)
        self.ema_sum[i].mul_(self.decay).add_(tot, alpha=1 - self.decay)
        n = self.ema_count[i].sum()
        smoothed = (self.ema_count[i] + 1e-5) / (n + K * 1e-5) * n
        self.codebook[i] = self.ema_sum[i] / smoothed[:, None]
        if self.restart > 0:
            dead = (self.ema_count[i] < self.restart).nonzero().squeeze(1)
            if dead.numel():
                new = x[torch.randint(len(x), (dead.numel(),), device=x.device)]
                self.codebook[i, dead] = new
                self.ema_sum[i, dead] = new
                self.ema_count[i, dead] = 1.0

    @torch.no_grad()
    def _maybe_init(self, i, x):
        if self.initialized[i]:
            return
        K = self.codebook.shape[1]
        pick = x[torch.randint(len(x), (K,), device=x.device)]
        self.codebook[i] = pick + 0.01 * torch.randn_like(pick)
        self.ema_sum[i] = self.codebook[i]
        self.ema_count[i] = 1.0
        self.initialized[i] = True

    def forward(self, z):
        """Training/eval quantization. Returns (straight-through z_hat, vq loss, codes)."""
        with torch.autocast(z.device.type, enabled=False):
            z = z.float()
            B, L, D = z.shape
            rest, zhat = z.detach().clone(), torch.zeros_like(z)
            loss, codes, cum = 0.0, [], []
            for i, l in enumerate(self.levels):
                x = self._down(rest, i)
                flat = x.reshape(-1, D)
                if self.training:
                    self._maybe_init(i, flat.detach())
                idx = self._nearest(flat.detach(), i)
                if self.training:
                    self._update(i, flat.detach(), idx)
                q = self.codebook[i][idx].view(B, l, D)
                if i == 0 and self.down0 is not None:
                    q = x + (q - x).detach()  # straight-through so the level-0 sampler learns
                u = self._up(q, i)
                zhat, rest = zhat + u, rest - u
                loss = loss + self.commitment * F.mse_loss(zhat.detach(), z) + F.mse_loss(zhat, z.detach())
                codes.append(idx.view(B, l))
                cum.append(zhat)
            out = zhat
            if self.training and self.fine_drop > 0 and self.n_levels > 1 and torch.rand(()) < self.fine_drop:
                out = cum[int(torch.randint(0, self.n_levels - 1, ()))]
            return z + (out - z).detach(), loss / self.n_levels, torch.cat(codes, 1)

    @torch.no_grad()
    def encode(self, z) -> torch.Tensor:
        with torch.autocast(z.device.type, enabled=False):
            z = z.float()
            B, L, D = z.shape
            rest, codes = z.clone(), []
            for i, l in enumerate(self.levels):
                idx = self._nearest(self._down(rest, i).reshape(-1, D), i)
                rest = rest - self._up(self.codebook[i][idx].view(B, l, D), i)
                codes.append(idx.view(B, l))
            return torch.cat(codes, 1)

    def split(self, codes):
        return torch.split(codes, self.levels, dim=1)

    @torch.no_grad()
    def step(self, i, zhat, idx_i):
        """Add level i's codes to z_hat; return (z_hat, input for level i+1 or None)."""
        with torch.autocast(zhat.device.type, enabled=False):
            zhat = zhat + self._up(self.codebook[i][idx_i], i)
            nxt = _pool(zhat, self.levels[i + 1]) if i + 1 < self.n_levels else None
            return zhat, nxt

    def empty(self, batch: int, device) -> torch.Tensor:
        return torch.zeros(batch, self.L, self.codebook.shape[2], device=device)

    @torch.no_grad()
    def decode(self, codes, cumulative: bool = False):
        """codes -> z_hat (or the list of z_hat after each level)."""
        zhat, outs = self.empty(codes.shape[0], codes.device), []
        for i, idx in enumerate(self.split(codes)):
            zhat, _ = self.step(i, zhat, idx)
            outs.append(zhat)
        return outs if cumulative else zhat

    @torch.no_grad()
    def teacher_inputs(self, codes) -> torch.Tensor:
        """Generator inputs for levels 1..n-1 under teacher forcing: [B, sum(levels[1:]), D]."""
        zhat, ins = self.empty(codes.shape[0], codes.device), []
        for i, idx in enumerate(self.split(codes)[:-1]):
            zhat, nxt = self.step(i, zhat, idx)
            ins.append(nxt)
        return torch.cat(ins, 1)


class VQTokenizer(nn.Module):
    """Token block <-> multiscale codes."""

    def __init__(self, cfg: VQConfig, vocab_size: int):
        super().__init__()
        D = cfg.embed_dim
        self.cfg = cfg
        self.wte = nn.Embedding(vocab_size, D)
        self.wpe = nn.Embedding(cfg.block_len, D)
        self.drop = nn.Dropout(cfg.dropout)
        self.enc = nn.ModuleList(Block(D, cfg.n_head, cfg.dropout) for _ in range(cfg.enc_layers))
        self.enc_ln = nn.LayerNorm(D) if cfg.enc_layers else nn.Identity()
        self.norm = nn.GroupNorm(cfg.norm_groups, D, affine=False)
        self.pq_drop = nn.Dropout(cfg.pre_quant_dropout)
        self.vq = MultiscaleVQ(cfg)
        self.dec = nn.ModuleList(Block(D, cfg.n_head, cfg.dropout) for _ in range(cfg.dec_layers))
        self.dec_ln = nn.LayerNorm(D)
        self.head = nn.Linear(D, vocab_size, bias=False)

    def encode_z(self, ids):
        x = self.drop(self.wte(ids) + self.wpe(torch.arange(ids.shape[1], device=ids.device)))
        for b in self.enc:
            x, _ = b(x)
        x = self.enc_ln(x)
        x = self.norm(x.transpose(1, 2)).transpose(1, 2)
        return self.pq_drop(x)

    def decode_z(self, z):
        for b in self.dec:
            z, _ = b(z)
        return self.head(self.dec_ln(z))

    def forward(self, ids):
        zq, vq_loss, codes = self.vq(self.encode_z(ids))
        return self.decode_z(zq), vq_loss, codes

    @torch.no_grad()
    def encode(self, ids) -> torch.Tensor:
        return self.vq.encode(self.encode_z(ids))

    @torch.no_grad()
    def decode(self, codes) -> torch.Tensor:
        """codes -> token ids [B, block_len]."""
        return self.decode_z(self.vq.decode(codes)).argmax(-1)

    @torch.no_grad()
    def decode_levels(self, codes) -> list[torch.Tensor]:
        """Token ids decoded from the reconstruction after each level (coarse-to-fine view)."""
        return [self.decode_z(z).argmax(-1) for z in self.vq.decode(codes, cumulative=True)]
