"""Chat template: OpenAI-format messages <-> model text. Used by data building, inference and
serving, so the format the model is trained on is exactly the format it is prompted with.

    header:  <|bos|><|tools|>bash edit read<|end|>[<|system|>...<|end|>]
    body:    <|user|>...<|end|><|assistant|>text<|call|>{"name":..,"arguments":{..}}<|end|>
             <|tool|>result<|end|> ... <|assistant|>
    target:  text<|call|>{...}<|end|>
"""
from __future__ import annotations

import json
import re
import uuid

from .config import ChatConfig

S_BOS, S_TOOLS, S_SYSTEM, S_USER, S_ASSISTANT, S_TOOL, S_CALL, S_END = (
    "<|bos|>", "<|tools|>", "<|system|>", "<|user|>", "<|assistant|>", "<|tool|>", "<|call|>", "<|end|>")
_SPECIAL_RE = re.compile(r"<\|(pad|bos|eos|file|system|tools|user|assistant|tool|call|end)\|>")


def text_of(content) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    return "".join(p.get("text", "") for p in content if isinstance(p, dict) and p.get("type") == "text")


def sanitize(text: str) -> str:
    """Neutralize literal special-token strings inside user/tool content."""
    return _SPECIAL_RE.sub(lambda m: f"<| {m.group(1)} |>", text)


def clip(text: str, limit: int) -> str:
    if limit <= 0 or len(text) <= limit:
        return text
    head = limit * 2 // 3
    tail = limit - head
    return f"{text[:head]}\n...[{len(text) - limit} chars omitted]...\n{text[-tail:]}"


def tool_names(tools) -> list[str]:
    names = set()
    for t in tools or []:
        fn = t.get("function", t)
        if fn.get("name"):
            names.add(fn["name"])
    return sorted(names)


def render_header(tools, system: str | None, cfg: ChatConfig) -> str:
    out = S_BOS
    names = tool_names(tools)
    if names:
        out += S_TOOLS + " ".join(names) + S_END
    if cfg.include_system and system:
        out += S_SYSTEM + sanitize(system) + S_END
    return out


def _arguments(raw):
    if isinstance(raw, str):
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            return raw
    return raw


def render_assistant(msg: dict) -> str:
    out = sanitize(text_of(msg.get("content")))
    for tc in msg.get("tool_calls") or []:
        fn = tc["function"]
        call = {"name": fn["name"], "arguments": _arguments(fn.get("arguments", "{}"))}
        out += S_CALL + json.dumps(call, ensure_ascii=False, separators=(",", ":"))
    return out


def render_turns(messages: list[dict], cfg: ChatConfig) -> str:
    parts = []
    for m in messages:
        role = m.get("role")
        if role == "user":
            parts.append(S_USER + sanitize(text_of(m.get("content"))) + S_END)
        elif role == "assistant":
            parts.append(S_ASSISTANT + render_assistant(m) + S_END)
        elif role == "tool":
            parts.append(S_TOOL + sanitize(clip(text_of(m.get("content")), cfg.tool_result_chars)) + S_END)
    return "".join(parts)


def system_text(messages: list[dict]) -> str | None:
    texts = [text_of(m.get("content")) for m in messages if m.get("role") in ("system", "developer")]
    return "\n\n".join(texts) if texts else None


def prompt_parts(messages: list[dict], tools, cfg: ChatConfig) -> tuple[str, str]:
    """(header, body) for generating the next assistant turn."""
    return render_header(tools, system_text(messages), cfg), render_turns(messages, cfg) + S_ASSISTANT


def training_examples(messages: list[dict], tools, cfg: ChatConfig,
                      allowed_tools: set[str] | None = None) -> list[tuple[str, str, str]]:
    """One (header, context, completion) per assistant turn. Turns that call a tool outside
    `allowed_tools` stay in later contexts but are not used as targets."""
    header = render_header(tools, system_text(messages), cfg)
    out = []
    for i, m in enumerate(messages):
        if m.get("role") != "assistant":
            continue
        calls = [tc["function"]["name"] for tc in m.get("tool_calls") or []]
        if allowed_tools is not None and any(c not in allowed_tools for c in calls):
            continue
        completion = render_assistant(m)
        if not completion.strip():
            continue
        context = render_turns(messages[:i], cfg) + S_ASSISTANT
        out.append((header, context, completion + S_END))
    return out


def parse_completion(text: str) -> dict:
    """Model text (already cut at the stop token) -> OpenAI assistant message."""
    pieces = text.split(S_CALL)
    content = pieces[0]
    calls = []
    for raw in pieces[1:]:
        try:
            obj = json.loads(raw)
            name, args = obj["name"], obj.get("arguments", {})
            if not isinstance(name, str):
                raise TypeError("name")
        except (json.JSONDecodeError, KeyError, TypeError):
            content += S_CALL + raw  # keep malformed calls visible rather than dropping them
            continue
        calls.append({
            "id": "call_" + uuid.uuid4().hex[:12],
            "type": "function",
            "function": {"name": name,
                         "arguments": args if isinstance(args, str) else json.dumps(args, ensure_ascii=False)},
        })
    msg = {"role": "assistant", "content": content.strip() or None}
    if calls:
        msg["tool_calls"] = calls
    return msg
