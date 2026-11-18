"""Byte-level BPE pre-tokenizer with a fixed set of special tokens, plus prefix construction.

Special tokens are registered first, so their ids are stable across every trained vocab.
"""
from __future__ import annotations

from pathlib import Path
from typing import Iterable, Sequence

SPECIAL_TOKENS = (
    "<|pad|>", "<|bos|>", "<|eos|>", "<|file|>", "<|system|>", "<|tools|>",
    "<|user|>", "<|assistant|>", "<|tool|>", "<|call|>", "<|end|>",
)
PAD, BOS, EOS, FILE, SYSTEM, TOOLS, USER, ASSISTANT, TOOL, CALL, END = range(len(SPECIAL_TOKENS))
# Generation stops at any of these. PAD only appears after the end of a sequence.
STOP_IDS = frozenset({PAD, EOS, END})


class BPE:
    def __init__(self, tok):
        self.tok = tok
        for i, s in enumerate(SPECIAL_TOKENS):
            if tok.token_to_id(s) != i:
                raise ValueError(f"special token {s} must have id {i}")

    @classmethod
    def train(cls, texts: Iterable[str], vocab_size: int) -> "BPE":
        from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers
        tok = Tokenizer(models.BPE())
        tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
        tok.decoder = decoders.ByteLevel()
        trainer = trainers.BpeTrainer(
            vocab_size=vocab_size, special_tokens=list(SPECIAL_TOKENS),
            initial_alphabet=pre_tokenizers.ByteLevel.alphabet(), show_progress=False)
        tok.train_from_iterator(texts, trainer)
        return cls(tok)

    @classmethod
    def load(cls, path: str | Path) -> "BPE":
        from tokenizers import Tokenizer
        return cls(Tokenizer.from_file(str(path)))

    def save(self, path: str | Path) -> None:
        self.tok.save(str(path))

    @property
    def vocab_size(self) -> int:
        return self.tok.get_vocab_size()

    def encode(self, text: str) -> list[int]:
        return self.tok.encode(text, add_special_tokens=False).ids

    def encode_batch(self, texts: Sequence[str]) -> list[list[int]]:
        return [e.ids for e in self.tok.encode_batch(list(texts), add_special_tokens=False)]

    def decode(self, ids: Iterable[int]) -> str:
        return self.tok.decode(list(ids), skip_special_tokens=False)


def build_prefix(header: Sequence[int], body: Sequence[int], prefix_len: int) -> list[int]:
    """Left-padded conditioning window: the header (pinned, at most half the window) followed by
    the most recent tokens of the body. Shared by training and inference so they cannot drift."""
    header = list(header[: prefix_len // 2])
    keep = prefix_len - len(header)
    tail = list(body[len(body) - keep:]) if len(body) > keep else list(body)
    ids = header + tail
    return [PAD] * (prefix_len - len(ids)) + ids
