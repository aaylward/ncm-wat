"""Agent tools. Names and argument shapes match pi's built-ins (read, write, edit, ls, bash,
grep, find) so trajectories from pi, Claude Code (after mapping) and this harness are one format."""
from __future__ import annotations

import fnmatch
import json
import os
import re
import subprocess
from pathlib import Path
from typing import Callable

MAX_OUTPUT = 50_000
SKIP_DIRS = {".git", "node_modules", "target", "build", "dist", ".gradle", ".idea", "__pycache__", ".venv", "venv"}


def _fn(name, description, props, required):
    return {"type": "function", "function": {
        "name": name, "description": description,
        "parameters": {"type": "object", "properties": props, "required": required}}}


S, N, B = {"type": "string"}, {"type": "number"}, {"type": "boolean"}
TOOL_SCHEMAS = [
    _fn("read", "Read a file. offset is 1-indexed.", {"path": S, "offset": N, "limit": N}, ["path"]),
    _fn("write", "Create or overwrite a file.", {"path": S, "content": S}, ["path", "content"]),
    _fn("edit", "Replace exact, unique text spans in a file.", {"path": S, "edits": {
        "type": "array", "items": {"type": "object", "properties": {"oldText": S, "newText": S},
                                   "required": ["oldText", "newText"]}}}, ["path", "edits"]),
    _fn("ls", "List a directory.", {"path": S, "limit": N}, []),
    _fn("bash", "Run a shell command in the workspace.", {"command": S, "timeout": N}, ["command"]),
    _fn("grep", "Search file contents with a regex.", {"pattern": S, "path": S, "glob": S, "ignoreCase": B,
                                                       "literal": B, "context": N, "limit": N}, ["pattern"]),
    _fn("find", "Find files by glob pattern.", {"pattern": S, "path": S, "limit": N}, ["pattern"]),
]
TOOL_NAMES = frozenset(t["function"]["name"] for t in TOOL_SCHEMAS)


class ToolError(Exception):
    pass


class Toolbox:
    """File tools confined to `workspace`. bash is not sandboxed: it runs only if allowed and,
    when `confirm` is given, only if confirm(command) returns True."""

    def __init__(self, workspace: str | Path, allow_bash: bool = False,
                 confirm: Callable[[str], bool] | None = None, bash_timeout: float = 120.0):
        self.root = Path(workspace).resolve()
        self.allow_bash, self.confirm, self.bash_timeout = allow_bash, confirm, bash_timeout

    schemas = TOOL_SCHEMAS

    def call(self, name: str, arguments) -> str:
        try:
            args = json.loads(arguments) if isinstance(arguments, str) else dict(arguments or {})
            if not isinstance(args, dict):
                raise ToolError("arguments must be a JSON object")
            if name not in TOOL_NAMES:
                raise ToolError(f"unknown tool {name!r}")
            out = getattr(self, "_" + name)(**args)
        except (ToolError, OSError, re.error, subprocess.SubprocessError) as e:
            out = f"Error: {e}"
        except (TypeError, KeyError, ValueError) as e:  # includes invalid JSON
            out = f"Error: bad arguments for {name}: {e!r}"
        return out if len(out) <= MAX_OUTPUT else out[:MAX_OUTPUT] + "\n...[truncated]"

    def path(self, p: str | None) -> Path:
        q = Path(p or ".")
        q = (q if q.is_absolute() else self.root / q).resolve()
        if q != self.root and self.root not in q.parents:
            raise ToolError(f"path outside workspace: {p}")
        return q

    def _rel(self, p: Path) -> str:
        return str(p.relative_to(self.root)) or "."

    def _read(self, path, offset=None, limit=None):
        lines = self.path(path).read_text(errors="replace").splitlines(keepends=True)
        start = max(int(offset or 1), 1) - 1
        end = start + int(limit) if limit else len(lines)
        return "".join(lines[start:end])

    def _write(self, path, content):
        p = self.path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(content)
        return f"Wrote {len(content.encode())} bytes to {self._rel(p)}"

    def _edit(self, path, edits):
        p = self.path(path)
        text = p.read_text()
        spans = []
        for e in edits:
            old, new = e["oldText"], e["newText"]
            n = text.count(old) if old else 0
            if n != 1:
                raise ToolError(f"oldText must occur exactly once in {path} (found {n}): {old[:80]!r}")
            i = text.index(old)
            spans.append((i, i + len(old), new))
        spans.sort()
        if any(a[1] > b[0] for a, b in zip(spans, spans[1:])):
            raise ToolError("edits overlap")
        for a, b, new in reversed(spans):
            text = text[:a] + new + text[b:]
        p.write_text(text)
        return f"Applied {len(spans)} edit(s) to {self._rel(p)}"

    def _ls(self, path=None, limit=500):
        p = self.path(path)
        names = sorted(c.name + ("/" if c.is_dir() else "") for c in p.iterdir())
        return "\n".join(names[: int(limit)]) or "(empty)"

    def _walk(self, base: Path):
        for dirpath, dirnames, filenames in os.walk(base):
            dirnames[:] = sorted(d for d in dirnames if d not in SKIP_DIRS)
            for f in sorted(filenames):
                yield Path(dirpath) / f

    def _grep(self, pattern, path=None, glob=None, ignoreCase=False, literal=False, context=0, limit=100):
        rx = re.compile(re.escape(pattern) if literal else pattern, re.I if ignoreCase else 0)
        base, out, ctx = self.path(path), [], int(context or 0)
        files = [base] if base.is_file() else self._walk(base)
        for f in files:
            rel = self._rel(f)
            if glob and not (fnmatch.fnmatch(rel, glob) or fnmatch.fnmatch(f.name, glob)):
                continue
            try:
                lines = f.read_text().splitlines()
            except (UnicodeDecodeError, OSError):
                continue
            for i, line in enumerate(lines):
                if rx.search(line):
                    for j in range(max(0, i - ctx), min(len(lines), i + ctx + 1)):
                        out.append(f"{rel}:{j + 1}:{lines[j]}")
                    if len(out) >= int(limit):
                        return "\n".join(out[: int(limit)])
        return "\n".join(out) or "No matches"

    def _find(self, pattern, path=None, limit=1000):
        base, out = self.path(path), []
        for f in self._walk(base):
            rel = str(f.relative_to(base))
            if fnmatch.fnmatch(rel, pattern) or fnmatch.fnmatch(f.name, pattern):
                out.append(self._rel(f))
                if len(out) >= int(limit):
                    break
        return "\n".join(out) or "No files found"

    def _bash(self, command, timeout=None):
        if not self.allow_bash:
            raise ToolError("bash is disabled in this harness (start it with --allow-bash)")
        if self.confirm is not None and not self.confirm(command):
            raise ToolError("command rejected by user")
        r = subprocess.run(command, shell=True, cwd=self.root, capture_output=True, text=True,
                           timeout=float(timeout or self.bash_timeout))
        return f"{r.stdout}{r.stderr}\n[exit code {r.returncode}]".lstrip("\n")
