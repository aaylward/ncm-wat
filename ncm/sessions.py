"""Turn agent sessions into training trajectories: {"messages": [OpenAI format], "tools": [...]}.

Sources: this harness's --record files (already in that format), pi session files
(~/.pi/agent/sessions/**.jsonl) and Claude Code session files (~/.claude/projects/**.jsonl).
Tool calls are mapped onto the pi-compatible vocabulary in ncm.tools; calls that cannot be mapped
keep their original name, so those turns stay in context but are never used as targets.
Secrets are scrubbed with regexes -- a best effort, not a guarantee.
"""
from __future__ import annotations

import json
import re
import uuid
from pathlib import Path
from typing import Iterator

from .chat import text_of
from .tools import TOOL_SCHEMAS

_SECRETS = [
    re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----.*?-----END [A-Z ]*PRIVATE KEY-----", re.S),
    re.compile(r"\b(?:AKIA|ASIA)[0-9A-Z]{16}\b"),
    re.compile(r"\b(?:ghp|gho|ghu|ghs|ghr)_[A-Za-z0-9]{30,}\b|\bgithub_pat_[A-Za-z0-9_]{30,}\b"),
    re.compile(r"\bxox[abprs]-[A-Za-z0-9-]{10,}\b"),
    re.compile(r"\bsk-(?:ant-|proj-)?[A-Za-z0-9_-]{20,}\b"),
    re.compile(r"\b(?:sk|rk|pk)_(?:live|test)_[A-Za-z0-9]{16,}\b"),
    re.compile(r"\bAIza[0-9A-Za-z_-]{35}\b"),
    re.compile(r"\beyJ[A-Za-z0-9_-]{10,}\.eyJ[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}\b"),
    re.compile(r"(?:postgres|postgresql|mysql|mongodb(?:\+srv)?|redis|amqp)://[^:\s/]+:[^@\s]+@"),
]
_ASSIGN = re.compile(
    r"""(?i)\b([A-Z0-9_]*(?:api[_-]?key|secret|token|passw(?:or)?d|credential)[A-Z0-9_]*)(\s*[:=]\s*)(["']?)([^\s"']{8,})\3""")


def scrub(text: str) -> str:
    for rx in _SECRETS:
        text = rx.sub("<REDACTED>", text)
    return _ASSIGN.sub(lambda m: f"{m.group(1)}{m.group(2)}{m.group(3)}<REDACTED>{m.group(3)}", text)


def scrub_obj(obj):
    if isinstance(obj, str):
        return scrub(obj)
    if isinstance(obj, list):
        return [scrub_obj(x) for x in obj]
    if isinstance(obj, dict):
        return {k: scrub_obj(v) for k, v in obj.items()}
    return obj


def _rel(path, cwd):
    if isinstance(path, str) and cwd and path.startswith(cwd.rstrip("/") + "/"):
        return path[len(cwd.rstrip("/")) + 1:]
    return path


def _call(name, args, call_id=None):
    return {"id": call_id or "call_" + uuid.uuid4().hex[:12], "type": "function",
            "function": {"name": name, "arguments": json.dumps(args, ensure_ascii=False)}}


def _drop_none(d):
    return {k: v for k, v in d.items() if v is not None}


def map_claude_tool(name: str, inp: dict, cwd: str | None) -> tuple[str, dict]:
    """Claude Code tool call -> pi-compatible call. Unknown tools pass through unchanged."""
    p = lambda k="file_path": _rel(inp.get(k), cwd)  # noqa: E731
    if name == "Read":
        return "read", _drop_none({"path": p(), "offset": inp.get("offset"), "limit": inp.get("limit")})
    if name == "Write":
        return "write", {"path": p(), "content": inp.get("content", "")}
    if name == "Edit" and not inp.get("replace_all"):
        return "edit", {"path": p(), "edits": [{"oldText": inp.get("old_string", ""), "newText": inp.get("new_string", "")}]}
    if name == "MultiEdit" and not any(e.get("replace_all") for e in inp.get("edits", [])):
        return "edit", {"path": p(), "edits": [{"oldText": e.get("old_string", ""), "newText": e.get("new_string", "")}
                                               for e in inp.get("edits", [])]}
    if name == "Bash":
        t = inp.get("timeout")
        return "bash", _drop_none({"command": inp.get("command", ""), "timeout": t / 1000 if t else None})
    if name == "Grep":
        return "grep", _drop_none({"pattern": inp.get("pattern"), "path": p("path"), "glob": inp.get("glob"),
                                   "ignoreCase": inp.get("-i"), "context": inp.get("-C"), "limit": inp.get("head_limit")})
    if name == "Glob":
        return "find", _drop_none({"pattern": inp.get("pattern"), "path": p("path")})
    if name == "LS":
        return "ls", _drop_none({"path": p("path")})
    return name, inp


_REMINDER = re.compile(r"<system-reminder>.*?</system-reminder>", re.S)
_NOISE_PREFIXES = ("<command-name>", "<command-message>", "<local-command-", "Caveat:", "[Request interrupted")


def _user_text(content) -> str:
    text = _REMINDER.sub("", text_of(content)).strip()
    return "" if text.startswith(_NOISE_PREFIXES) else text


def _chain(rows: dict, leaf_id, parent_key: str) -> list[dict]:
    out, seen = [], set()
    while leaf_id and leaf_id in rows and leaf_id not in seen:
        seen.add(leaf_id)
        out.append(rows[leaf_id])
        leaf_id = rows[leaf_id].get(parent_key)
    return out[::-1]


def _read_jsonl(path: Path) -> list[dict]:
    rows = []
    for line in Path(path).read_text(errors="replace").splitlines():
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return rows


def load_claude_session(path: Path) -> dict | None:
    rows = [r for r in _read_jsonl(path) if r.get("uuid")]  # attachments etc. are links in the chain
    convo = [r for r in rows if r.get("type") in ("user", "assistant") and not r.get("isSidechain")]
    if not convo:
        return None
    chain = _chain({r["uuid"]: r for r in rows}, convo[-1]["uuid"], "parentUuid")
    messages, cur_id = [], None
    for r in chain:
        att = r.get("attachment") if r.get("type") == "attachment" else None
        if isinstance(att, dict) and att.get("type") == "queued_command" and isinstance(att.get("prompt"), str):
            text = _user_text(att["prompt"])  # a message the user typed while the agent was working
            if text:
                messages.append({"role": "user", "content": text})
            cur_id = None
            continue
        if r.get("type") not in ("user", "assistant") or r.get("isSidechain") or r.get("isMeta"):
            continue
        msg, cwd = r.get("message") or {}, r.get("cwd")
        content = msg.get("content")
        if r["type"] == "assistant":
            if msg.get("id") != cur_id or not messages or messages[-1]["role"] != "assistant":
                messages.append({"role": "assistant", "content": ""})
                cur_id = msg.get("id")
            a = messages[-1]
            for b in content or []:
                if b.get("type") == "text":
                    a["content"] = (a["content"] + "\n\n" + b["text"]).strip()
                elif b.get("type") == "tool_use":
                    name, args = map_claude_tool(b["name"], b.get("input") or {}, cwd)
                    a.setdefault("tool_calls", []).append(_call(name, args, b.get("id")))
            continue
        cur_id = None
        if isinstance(content, str):
            text = _user_text(content)
            if text:
                messages.append({"role": "user", "content": text})
            continue
        texts = []
        for b in content or []:
            if b.get("type") == "tool_result":
                c = b.get("content")
                messages.append({"role": "tool", "tool_call_id": b.get("tool_use_id"),
                                 "content": c if isinstance(c, str) else text_of(c)})
            elif b.get("type") == "text":
                texts.append(b.get("text", ""))
        text = _user_text("\n".join(texts))
        if text:
            messages.append({"role": "user", "content": text})
    return _finish(messages)


def load_pi_session(path: Path) -> dict | None:
    rows = _read_jsonl(path)
    cwd = next((r.get("cwd") for r in rows if r.get("type") == "session"), None)
    entries = {r["id"]: r for r in rows if r.get("id") and r.get("type") != "session"}
    leaf = next((r["id"] for r in reversed(rows) if r.get("type") == "message" and r.get("id")), None)
    messages = []
    for r in _chain(entries, leaf, "parentId"):
        m = r.get("message") if r.get("type") == "message" else None
        role = (m or {}).get("role")
        if role == "user":
            text = _user_text(m.get("content"))
            if text:
                messages.append({"role": "user", "content": text})
        elif role == "assistant":
            a = {"role": "assistant", "content": ""}
            for b in m.get("content") or []:
                if b.get("type") == "text":
                    a["content"] = (a["content"] + "\n\n" + b["text"]).strip()
                elif b.get("type") == "toolCall":
                    args = dict(b.get("arguments") or {})
                    if "path" in args:
                        args["path"] = _rel(args["path"], cwd)
                    a.setdefault("tool_calls", []).append(_call(b["name"], args, b.get("id")))
            messages.append(a)
        elif role == "toolResult":
            messages.append({"role": "tool", "tool_call_id": m.get("toolCallId"), "content": text_of(m.get("content"))})
    return _finish(messages)


def _finish(messages: list[dict]) -> dict | None:
    if not any(m["role"] == "assistant" for m in messages):
        return None
    for m in messages:
        if m["role"] == "assistant" and not m["content"]:
            m["content"] = None
    return {"messages": scrub_obj(messages), "tools": TOOL_SCHEMAS}


def load_trajectory_file(path: Path) -> Iterator[dict]:
    for r in _read_jsonl(path):
        if isinstance(r.get("messages"), list):
            yield {"messages": scrub_obj(r["messages"]), "tools": r.get("tools") or TOOL_SCHEMAS}


def _files(root: Path) -> list[Path]:
    root = Path(root).expanduser()
    return [root] if root.is_file() else sorted(root.rglob("*.jsonl"))


def load_sessions(kind: str, root: Path) -> Iterator[dict]:
    loader = {"claude": load_claude_session, "pi": load_pi_session}.get(kind)
    for f in _files(root):
        if kind == "jsonl":
            yield from load_trajectory_file(f)
            continue
        if loader is None:
            raise ValueError(f"unknown session kind {kind!r}")
        if kind == "claude" and "subagents" in f.parts:
            continue
        t = loader(f)
        if t:
            yield t
