"""The pluggable seam: anything with `complete(messages, tools) -> Completion` can drive the agent
or sit behind the OpenAI-compatible server. Messages and tools use the OpenAI chat format."""
from __future__ import annotations

import json
import os
import urllib.request
from dataclasses import dataclass
from typing import Protocol


@dataclass
class Completion:
    message: dict                    # {"role": "assistant", "content": str | None, "tool_calls"?: [...]}
    finish_reason: str = "stop"      # stop | tool_calls | length
    prompt_tokens: int = 0
    completion_tokens: int = 0


class ChatModel(Protocol):
    name: str

    def complete(self, messages: list[dict], tools: list[dict] | None = None, *,
                 max_tokens: int | None = None, temperature: float | None = None,
                 top_p: float | None = None) -> Completion: ...


class OpenAIChatModel:
    """Any OpenAI-compatible /chat/completions endpoint: `ncm serve`, vLLM, Ollama, OpenRouter, OpenAI..."""

    def __init__(self, base_url: str, model: str, api_key: str | None = None,
                 api_key_env: str = "OPENAI_API_KEY", timeout: float = 600.0):
        self.base_url, self.name, self.timeout = base_url.rstrip("/"), model, timeout
        self.api_key = api_key or os.environ.get(api_key_env, "")

    def complete(self, messages, tools=None, *, max_tokens=None, temperature=None, top_p=None) -> Completion:
        body = {"model": self.name, "messages": messages}
        for k, v in (("tools", tools or None), ("max_tokens", max_tokens), ("temperature", temperature), ("top_p", top_p)):
            if v is not None:
                body[k] = v
        req = urllib.request.Request(
            self.base_url + "/chat/completions", data=json.dumps(body).encode(),
            headers={"Content-Type": "application/json", **({"Authorization": f"Bearer {self.api_key}"} if self.api_key else {})})
        with urllib.request.urlopen(req, timeout=self.timeout) as r:
            data = json.loads(r.read())
        choice = data["choices"][0]
        msg = {k: v for k, v in choice["message"].items() if k in ("role", "content", "tool_calls")}
        usage = data.get("usage") or {}
        return Completion(msg, choice.get("finish_reason") or "stop",
                          usage.get("prompt_tokens", 0), usage.get("completion_tokens", 0))
