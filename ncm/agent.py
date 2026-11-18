"""Minimal agent loop: ask the model, run its tool calls, feed results back, repeat until it answers
without tools. Any ChatModel plugs in (local NCM, `ncm serve`, or a hosted model). With `record`,
each finished episode is appended as a trajectory that `ncm data build --sessions jsonl:` ingests."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Callable

from .api import ChatModel
from .tools import Toolbox

DEFAULT_SYSTEM = ("You are a coding agent for backend and distributed systems work. Inspect the "
                  "workspace with tools before changing it, make focused edits, run the build or tests "
                  "when you can, and finish with a short summary of what changed.")


class Agent:
    def __init__(self, model: ChatModel, tools: Toolbox, system: str | None = DEFAULT_SYSTEM,
                 max_steps: int = 30, record: str | Path | None = None,
                 on_event: Callable[[str, dict], None] | None = None):
        self.model, self.tools, self.max_steps = model, tools, max_steps
        self.record = Path(record) if record else None
        self.on_event = on_event or (lambda kind, data: None)
        self.messages: list[dict] = [{"role": "system", "content": system}] if system else []

    def run(self, task: str) -> str | None:
        """Run one user turn to completion; returns the final assistant text."""
        self.messages.append({"role": "user", "content": task})
        text = None
        for _ in range(self.max_steps):
            c = self.model.complete(self.messages, self.tools.schemas)
            msg = {"role": "assistant", "content": c.message.get("content")}
            if c.message.get("tool_calls"):
                msg["tool_calls"] = c.message["tool_calls"]
            self.messages.append(msg)
            text = msg["content"]
            self.on_event("assistant", msg)
            if not msg.get("tool_calls"):
                break
            for tc in msg["tool_calls"]:
                fn = tc["function"]
                result = self.tools.call(fn["name"], fn.get("arguments") or "{}")
                self.messages.append({"role": "tool", "tool_call_id": tc["id"], "content": result})
                self.on_event("tool", {"name": fn["name"], "arguments": fn.get("arguments"), "result": result})
        else:
            self.on_event("limit", {"max_steps": self.max_steps})
        return text

    def save(self) -> None:
        """Append the whole conversation to the record file (call once per session)."""
        if self.record and any(m["role"] == "assistant" for m in self.messages):
            self.record.parent.mkdir(parents=True, exist_ok=True)
            with open(self.record, "a") as f:
                f.write(json.dumps({"messages": self.messages, "tools": self.tools.schemas}) + "\n")
