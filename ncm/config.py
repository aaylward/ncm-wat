"""Run configuration. A size preset gives every field; a TOML file may override any of them.

The resolved config is written to <run>/config.json and is the single source of truth
for every later stage (generator training, inference, serving).
"""
from __future__ import annotations

import dataclasses
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path

if sys.version_info >= (3, 11):
    import tomllib
else:  # pragma: no cover
    import tomli as tomllib


@dataclass
class VQConfig:
    """Stage 1: multiscale residual VQ tokenizer (upstream models/vqvae.py + quant.py)."""
    block_len: int = 64                 # tokens produced per generator call
    levels: list[int] = field(default_factory=lambda: [4, 8, 12, 16, 24, 32, 48, 64])
    codebook_size: int = 2048
    embed_dim: int = 256
    n_head: int = 4
    enc_layers: int = 0                 # upstream default: encoder is embedding + position only
    dec_layers: int = 1
    dropout: float = 0.1
    pre_quant_dropout: float = 0.0
    norm_groups: int = 8
    commitment: float = 0.25
    ema_decay: float = 0.99
    quant_resi: float = 0.5             # blend ratio of the per-level residual conv (Phi)
    fine_drop: float = 0.5              # prob. the decoder sees a truncated (coarse) z_hat
    restart_threshold: float = 0.01     # EMA usage below which a code is re-seeded; 0 disables
    learned_l0: bool = True             # learned down/up sampler for level 0

    def validate(self) -> None:
        lv = self.levels
        if lv[-1] != self.block_len:
            raise ValueError(f"levels[-1]={lv[-1]} must equal block_len={self.block_len}")
        if any(a >= b for a, b in zip(lv, lv[1:])):
            raise ValueError(f"levels must be strictly increasing: {lv}")
        if self.learned_l0 and self.block_len % lv[0]:
            raise ValueError("learned_l0 needs block_len divisible by levels[0]")
        if self.embed_dim % self.norm_groups or self.embed_dim % self.n_head:
            raise ValueError("embed_dim must be divisible by norm_groups and n_head")


@dataclass
class GenConfig:
    """Stage 2: next-concept generator (upstream models/next_concept.py)."""
    prefix_len: int = 512
    n_layer: int = 6
    n_head: int = 6
    n_embd: int = 384
    dropout: float = 0.1
    rope_base: float = 10000.0

    def validate(self) -> None:
        if self.n_embd % self.n_head or (self.n_embd // self.n_head) % 2:
            raise ValueError("n_embd/n_head must be an even integer")


@dataclass
class TrainConfig:
    steps: int = 20000
    batch_size: int = 32
    lr: float = 3e-4
    min_lr_frac: float = 0.1
    warmup: int = 500
    weight_decay: float = 0.1
    grad_clip: float = 1.0
    log_every: int = 50
    eval_every: int = 1000
    eval_batches: int = 20
    save_every: int = 1000
    seed: int = 0


@dataclass
class ChatConfig:
    include_system: bool = False        # small models do better without long client system prompts
    tool_result_chars: int = 2000       # tool output is clipped (head + tail) to this many chars


@dataclass
class Config:
    vq: VQConfig = field(default_factory=VQConfig)
    gen: GenConfig = field(default_factory=GenConfig)
    train_vq: TrainConfig = field(default_factory=TrainConfig)
    train_gen: TrainConfig = field(default_factory=TrainConfig)
    chat: ChatConfig = field(default_factory=ChatConfig)

    def validate(self) -> "Config":
        self.vq.validate()
        self.gen.validate()
        return self

    def to_dict(self) -> dict:
        return dataclasses.asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "Config":
        cfg = cls()
        apply_overrides(cfg, d)
        return cfg.validate()

    def save(self, path: Path) -> None:
        Path(path).write_text(json.dumps(self.to_dict(), indent=2) + "\n")

    @classmethod
    def load(cls, path: Path) -> "Config":
        return cls.from_dict(json.loads(Path(path).read_text()))


def apply_overrides(cfg: Config, d: dict) -> None:
    sections = {f.name for f in dataclasses.fields(cfg)}
    for name, values in d.items():
        if name not in sections:
            raise KeyError(f"unknown config section {name!r}")
        section = getattr(cfg, name)
        known = {f.name for f in dataclasses.fields(section)}
        for k, v in values.items():
            if k not in known:
                raise KeyError(f"unknown key {name}.{k}")
            setattr(section, k, list(v) if isinstance(v, list) else v)


def _train(steps, bs, lr, warmup, eval_every, save_every):
    return TrainConfig(steps=steps, batch_size=bs, lr=lr, warmup=warmup,
                       eval_every=eval_every, save_every=save_every)


# tiny: tests and CPU smoke runs. small: hours on one consumer GPU. base: ~paper scale (12L/768).
SIZES: dict[str, dict] = {
    "tiny": dict(
        vq=VQConfig(block_len=16, levels=[2, 4, 8, 16], codebook_size=64, embed_dim=32,
                    n_head=2, norm_groups=4, dropout=0.0, fine_drop=0.2),
        gen=GenConfig(prefix_len=32, n_layer=2, n_head=2, n_embd=64, dropout=0.0),
        train_vq=_train(300, 16, 3e-3, 20, 100, 300),
        train_gen=_train(300, 16, 3e-3, 20, 100, 300),
    ),
    "small": dict(
        vq=VQConfig(),
        gen=GenConfig(),
        train_vq=_train(10000, 64, 1e-3, 500, 1000, 1000),
        train_gen=_train(30000, 32, 3e-4, 1000, 1000, 2000),
    ),
    "base": dict(
        vq=VQConfig(block_len=128, levels=[4, 8, 16, 24, 32, 48, 64, 80, 96, 112, 128],
                    codebook_size=4096, embed_dim=384, n_head=6),
        gen=GenConfig(prefix_len=1024, n_layer=12, n_head=12, n_embd=768),
        train_vq=_train(20000, 64, 1e-3, 1000, 2000, 2000),
        train_gen=_train(100000, 32, 3e-4, 2000, 2000, 5000),
    ),
}


def make_config(size: str = "small", toml_path: str | Path | None = None) -> Config:
    if size not in SIZES:
        raise KeyError(f"unknown size {size!r}; choose from {sorted(SIZES)}")
    cfg = Config(**{k: dataclasses.replace(v) for k, v in SIZES[size].items()})
    if toml_path:
        d = tomllib.loads(Path(toml_path).read_text())
        d.pop("size", None)
        apply_overrides(cfg, d)
    return cfg.validate()
