"""Training loops.

Stage 1 (`train_vq`) fits the multiscale VQ tokenizer on target blocks (reconstruction CE + VQ loss).
Stage 2 (`train_gen`) freezes it, encodes each target block to codes, and trains the generator with
teacher forcing (cross-entropy over every code position of every level).

Run directory layout: config.json, bpe.json, vq.pt, gen.pt, log.jsonl. Checkpoints hold
model + optimizer + step, so re-running the same command resumes.
"""
from __future__ import annotations

import contextlib
import dataclasses
import filecmp
import json
import math
import shutil
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from .config import ChatConfig, Config, TrainConfig, make_config
from .data import Corpus
from .generator import Generator, code_loss
from .text import BPE
from .vq import VQTokenizer


def pick_device(name: str = "auto") -> torch.device:
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def autocast(device: torch.device):
    if device.type == "cuda" and torch.cuda.is_bf16_supported():
        return torch.autocast("cuda", dtype=torch.bfloat16)
    return contextlib.nullcontext()


def lr_at(step: int, t: TrainConfig) -> float:
    if step < t.warmup:
        return t.lr * (step + 1) / t.warmup
    frac = min(1.0, (step - t.warmup) / max(1, t.steps - t.warmup))
    return t.lr * (t.min_lr_frac + (1 - t.min_lr_frac) * 0.5 * (1 + math.cos(math.pi * frac)))


def make_optimizer(model: torch.nn.Module, t: TrainConfig) -> torch.optim.Optimizer:
    params = [p for p in model.parameters() if p.requires_grad]
    decay = [p for p in params if p.dim() >= 2]
    no_decay = [p for p in params if p.dim() < 2]
    return torch.optim.AdamW([{"params": decay, "weight_decay": t.weight_decay},
                              {"params": no_decay, "weight_decay": 0.0}], lr=t.lr, betas=(0.9, 0.95))


class Batches:
    def __init__(self, corpus: Corpus, block_len: int, prefix_len: int, seed: int):
        self.corpus, self.B, self.P = corpus, block_len, prefix_len
        self.index = corpus.index(block_len)
        if not len(self.index):
            raise ValueError("corpus has no target blocks")
        self.rng = np.random.default_rng(seed)

    def sample(self, n: int, device):
        rows = self.index[self.rng.integers(0, len(self.index), n)]
        pre, tgt = self.corpus.batch(rows, self.B, self.P)
        return torch.from_numpy(pre).to(device), torch.from_numpy(tgt).to(device)

    def fixed(self, n_batches: int, bs: int, device):
        rng = np.random.default_rng(1234)
        for _ in range(n_batches):
            rows = self.index[rng.integers(0, len(self.index), bs)]
            pre, tgt = self.corpus.batch(rows, self.B, self.P)
            yield torch.from_numpy(pre).to(device), torch.from_numpy(tgt).to(device)


class Logger:
    def __init__(self, run: Path, stage: str, echo):
        self.path, self.stage, self.echo = run / "log.jsonl", stage, echo

    def __call__(self, **rec):
        rec = {"stage": self.stage, **{k: round(v, 5) if isinstance(v, float) else v for k, v in rec.items()}}
        with open(self.path, "a") as f:
            f.write(json.dumps(rec) + "\n")
        self.echo(" ".join(f"{k}={v}" for k, v in rec.items()))


def _loop(model, opt, step_fn, eval_fn, t: TrainConfig, ckpt: Path, start: int, log: Logger):
    t0 = time.time()
    for step in range(start, t.steps):
        for g in opt.param_groups:
            g["lr"] = lr_at(step, t)
        model.train()
        loss, metrics = step_fn()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        gn = torch.nn.utils.clip_grad_norm_(model.parameters(), t.grad_clip)
        opt.step()
        done = step + 1
        if done % t.log_every == 0 or done == t.steps:
            log(step=done, loss=loss.item(), grad_norm=float(gn), lr=lr_at(step, t),
                sec=round(time.time() - t0, 1), **metrics)
        if done % t.eval_every == 0 or done == t.steps:
            model.eval()
            log(step=done, **{"val_" + k: v for k, v in eval_fn().items()})
        if done % t.save_every == 0 or done == t.steps:
            save_checkpoint(ckpt, model, opt, done)


def save_checkpoint(path: Path, model, opt, step: int) -> None:
    tmp = path.with_suffix(".tmp")
    torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "step": step}, tmp)
    tmp.replace(path)


def _resume(path: Path, model, opt) -> int:
    if not path.exists():
        return 0
    ck = torch.load(path, map_location="cpu", weights_only=False)
    model.load_state_dict(ck["model"])
    opt.load_state_dict(ck["opt"])
    return int(ck["step"])


def _val_batches(data: Path, cfg: Config, train: "Batches") -> "Batches":
    val = Corpus.load(data / "val")
    if not len(val.index(cfg.vq.block_len)):
        return train  # no held-out split: eval numbers are train numbers
    return Batches(val, cfg.vq.block_len, cfg.gen.prefix_len, 0)


def _apply_train_overrides(t: TrainConfig, overrides: dict | None) -> None:
    for k, v in (overrides or {}).items():
        if v is not None:
            setattr(t, k, v)


def prepare_run(run: Path, data: Path, size: str = "small", toml: str | None = None) -> Config:
    run.mkdir(parents=True, exist_ok=True)
    if (run / "config.json").exists():
        cfg = Config.load(run / "config.json")
    else:
        cfg = make_config(size, toml)
        cfg.chat = ChatConfig(**_meta(data).get("chat", {}))  # the template the data was rendered with
        cfg.save(run / "config.json")
    _check_data(run, data, cfg)
    return cfg


def _meta(data: Path) -> dict:
    p = data / "meta.json"
    return json.loads(p.read_text()) if p.exists() else {}


def _check_data(run: Path, data: Path, cfg: Config) -> None:
    """A run and a dataset must share the BPE vocab and chat template, or tokens/prompts drift."""
    src, dst = data / "bpe.json", run / "bpe.json"
    if not dst.exists():
        shutil.copy(src, dst)
    elif not filecmp.cmp(src, dst, shallow=False):
        raise ValueError(f"{data} was tokenized with a different BPE than {run}; "
                         f"rebuild it with `ncm data build --bpe {dst}`")
    meta = _meta(data)
    if cfg.gen.prefix_len > meta.get("max_context", cfg.gen.prefix_len):
        raise ValueError(f"prefix_len {cfg.gen.prefix_len} exceeds the {meta['max_context']} context tokens "
                         f"{data} kept; rebuild it with a larger --max-context")
    chat = meta.get("chat")
    if chat is not None and chat != dataclasses.asdict(cfg.chat):
        raise ValueError(f"{data} was rendered with chat config {chat}, run uses {dataclasses.asdict(cfg.chat)}")


def train_vq(run: Path, data: Path, size: str = "small", toml: str | None = None, device: str = "auto",
             overrides: dict | None = None, echo=print) -> Path:
    run, data = Path(run), Path(data)
    cfg = prepare_run(run, data, size, toml)
    _apply_train_overrides(cfg.train_vq, overrides)
    cfg.save(run / "config.json")
    t, dev = cfg.train_vq, pick_device(device)
    torch.manual_seed(t.seed)
    vocab = BPE.load(run / "bpe.json").vocab_size
    model = VQTokenizer(cfg.vq, vocab).to(dev)
    opt = make_optimizer(model, t)
    ckpt = run / "vq.pt"
    start = _resume(ckpt, model, opt)
    train = Batches(Corpus.load(data / "train"), cfg.vq.block_len, cfg.gen.prefix_len, t.seed)
    val = _val_batches(data, cfg, train)
    log = Logger(run, "vq", echo)
    log(event="start", device=str(dev), step=start, params=sum(p.numel() for p in model.parameters()),
        train_blocks=len(train.index))

    def step_fn():
        _, tgt = train.sample(t.batch_size, dev)
        with autocast(dev):
            logits, vq_loss, _ = model(tgt)
            rec = F.cross_entropy(logits.float().reshape(-1, logits.shape[-1]), tgt.reshape(-1))
        return rec + vq_loss, {"rec": rec.item(), "vq": vq_loss.detach().item()}

    @torch.no_grad()
    def eval_fn():
        tok = exact = n = 0
        used = [set() for _ in cfg.vq.levels]
        for _, tgt in val.fixed(t.eval_batches, t.batch_size, dev):
            codes = model.encode(tgt)
            rec = model.decode(codes)
            tok += (rec == tgt).float().mean().item()
            exact += (rec == tgt).all(1).float().mean().item()
            n += 1
            for i, c in enumerate(model.vq.split(codes)):
                used[i].update(c.unique().tolist())
        return {"token_acc": tok / n, "block_exact": exact / n,
                "code_use": [round(len(u) / cfg.vq.codebook_size, 3) for u in used]}

    _loop(model, opt, step_fn, eval_fn, t, ckpt, start, log)
    return ckpt


def load_vq(run: Path, device) -> tuple[Config, VQTokenizer, BPE]:
    cfg = Config.load(run / "config.json")
    bpe = BPE.load(run / "bpe.json")
    vq = VQTokenizer(cfg.vq, bpe.vocab_size)
    vq.load_state_dict(torch.load(run / "vq.pt", map_location="cpu", weights_only=False)["model"])
    vq.to(device).eval().requires_grad_(False)
    return cfg, vq, bpe


def train_gen(run: Path, data: Path, init_from: str | None = None, device: str = "auto",
              overrides: dict | None = None, echo=print) -> Path:
    run, data = Path(run), Path(data)
    if init_from:
        base = Path(init_from)
        run.mkdir(parents=True, exist_ok=True)
        for f in ("config.json", "bpe.json", "vq.pt"):
            if not (run / f).exists():
                shutil.copy(base / f, run / f)
    if not (run / "vq.pt").exists():
        raise FileNotFoundError(f"{run}/vq.pt missing: run `ncm train vq` first (or pass --init-from)")
    dev = pick_device(device)
    cfg, vq, bpe = load_vq(run, dev)
    _check_data(run, data, cfg)
    _apply_train_overrides(cfg.train_gen, overrides)
    cfg.save(run / "config.json")
    t = cfg.train_gen
    torch.manual_seed(t.seed)
    gen = Generator(cfg.gen, cfg.vq, bpe.vocab_size, vq.vq.codebook).to(dev)
    opt = make_optimizer(gen, t)
    ckpt = run / "gen.pt"
    start = 0
    if init_from and not ckpt.exists():
        gen.load_state_dict(torch.load(Path(init_from) / "gen.pt", map_location="cpu", weights_only=False)["model"])
    else:
        start = _resume(ckpt, gen, opt)
    train = Batches(Corpus.load(data / "train"), cfg.vq.block_len, cfg.gen.prefix_len, t.seed)
    val = _val_batches(data, cfg, train)
    log = Logger(run, "gen", echo)
    log(event="start", device=str(dev), step=start, params=sum(p.numel() for p in gen.parameters()),
        train_blocks=len(train.index), positions=cfg.gen.prefix_len + gen.T)

    def forward(pre, tgt):
        codes = vq.encode(tgt)
        with autocast(dev):
            logits = gen(pre, vq.vq.teacher_inputs(codes))
        return logits, codes

    def step_fn():
        logits, codes = forward(*train.sample(t.batch_size, dev))
        loss = code_loss(logits, codes)
        return loss, {"code_acc": (logits.argmax(-1) == codes).float().mean().item()}

    @torch.no_grad()
    def eval_fn():
        loss = acc = n = 0
        per_level = np.zeros(len(cfg.vq.levels))
        for pre, tgt in val.fixed(t.eval_batches, t.batch_size, dev):
            logits, codes = forward(pre, tgt)
            loss += code_loss(logits, codes).item()
            hit = (logits.argmax(-1) == codes).float()
            acc += hit.mean().item()
            per_level += np.array([h.mean().item() for h in torch.split(hit, cfg.vq.levels, 1)])
            n += 1
        return {"loss": loss / n, "code_acc": acc / n, "level_acc": [round(float(x), 3) for x in per_level / n]}

    _loop(gen, opt, step_fn, eval_fn, t, ckpt, start, log)
    return ckpt
