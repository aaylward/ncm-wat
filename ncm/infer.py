"""Inference: block-by-block generation and the chat wrapper.

Each generator call emits one block of `block_len` tokens in len(levels) parallel passes. Longer
outputs append the block to the conditioning window and generate the next one, until a stop token.
"""
from __future__ import annotations

from pathlib import Path

import torch

from . import chat
from .api import Completion
from .generator import Generator
from .text import BOS, STOP_IDS, BPE, build_prefix
from .train import load_vq, pick_device


class NCM:
    def __init__(self, run: str | Path, device: str = "auto"):
        run = Path(run)
        self.device = pick_device(device)
        self.cfg, self.vq, self.bpe = load_vq(run, self.device)
        self.gen = Generator(self.cfg.gen, self.cfg.vq, self.bpe.vocab_size, self.vq.vq.codebook)
        self.gen.load_state_dict(torch.load(run / "gen.pt", map_location="cpu", weights_only=False)["model"])
        self.gen.to(self.device).eval()

    @property
    def block_len(self) -> int:
        return self.cfg.vq.block_len

    def _block(self, header, body, temperature, top_k, top_p):
        prefix = torch.tensor([build_prefix(header, body, self.cfg.gen.prefix_len)], device=self.device)
        codes = self.gen.generate(prefix, self.vq.vq, temperature=temperature, top_k=top_k, top_p=top_p)
        return codes

    @torch.no_grad()
    def generate_ids(self, header: list[int], body: list[int], max_new_tokens: int = 256,
                     temperature: float = 0.7, top_k: int = 0, top_p: float = 0.95,
                     seed: int | None = None) -> tuple[list[int], bool]:
        """Returns (new tokens without the stop token, whether a stop token was produced)."""
        if seed is not None:
            torch.manual_seed(seed)
        out: list[int] = []
        while len(out) < max_new_tokens:
            codes = self._block(header, list(body) + out, temperature, top_k, top_p)
            for t in self.vq.decode(codes)[0].tolist():
                if t in STOP_IDS:
                    return out, True
                out.append(t)
                if len(out) >= max_new_tokens:
                    break
        return out, False

    @torch.no_grad()
    def levels(self, header: list[int], body: list[int], temperature: float = 0.7, top_k: int = 0,
               top_p: float = 0.95) -> list[str]:
        """The first block decoded after each level: a coarse-to-fine view of one generation."""
        codes = self._block(header, body, temperature, top_k, top_p)
        return [self.bpe.decode(x[0].tolist()) for x in self.vq.decode_levels(codes)]

    def complete_text(self, prompt: str, header: str = "", **kw) -> str:
        """Raw continuation. With a file path header ("<|file|>src/App.java\\n") it is code completion."""
        h = [BOS] + (self.bpe.encode(header) if header else [])
        ids, _ = self.generate_ids(h, self.bpe.encode(prompt), **kw)
        return self.bpe.decode(ids)


class NCMChatModel:
    """ChatModel backed by a local NCM run."""

    def __init__(self, ncm: NCM, name: str = "ncm", max_tokens: int = 1024,
                 temperature: float = 0.7, top_p: float = 0.95):
        self.ncm, self.name = ncm, name
        self.max_tokens, self.temperature, self.top_p = max_tokens, temperature, top_p

    def complete(self, messages, tools=None, *, max_tokens=None, temperature=None, top_p=None) -> Completion:
        header_text, body_text = chat.prompt_parts(messages, tools, self.ncm.cfg.chat)
        bpe: BPE = self.ncm.bpe
        header, body = bpe.encode(header_text), bpe.encode(body_text)
        limit = min(max_tokens or self.max_tokens, self.max_tokens)
        ids, stopped = self.ncm.generate_ids(
            header, body, limit,
            temperature=self.temperature if temperature is None else temperature,
            top_p=self.top_p if top_p is None else top_p)
        msg = chat.parse_completion(bpe.decode(ids))
        finish = "tool_calls" if msg.get("tool_calls") else ("stop" if stopped else "length")
        return Completion(msg, finish, len(header) + len(body), len(ids))


def load_chat_model(run: str | Path, name: str | None = None, device: str = "auto", **kw) -> NCMChatModel:
    return NCMChatModel(NCM(run, device), name or Path(run).name, **kw)

