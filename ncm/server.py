"""OpenAI-compatible HTTP server (stdlib only) in front of any ChatModel.

GET /v1/models, POST /v1/chat/completions with or without `stream`. NCM has no token streaming,
so a streamed response arrives as a few chunks once the whole completion is ready.
"""
from __future__ import annotations

import json
import threading
import time
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from .api import ChatModel, Completion


def _chunks(cid: str, model: str, c: Completion, include_usage: bool):
    base = {"id": cid, "object": "chat.completion.chunk", "created": int(time.time()), "model": model}

    def chunk(delta, finish=None, **extra):
        return {**base, "choices": [{"index": 0, "delta": delta, "finish_reason": finish}], **extra}

    yield chunk({"role": "assistant", "content": ""})
    if c.message.get("content"):
        yield chunk({"content": c.message["content"]})
    for i, tc in enumerate(c.message.get("tool_calls") or []):
        yield chunk({"tool_calls": [{"index": i, "id": tc["id"], "type": "function",
                                     "function": {"name": tc["function"]["name"],
                                                  "arguments": tc["function"]["arguments"]}}]})
    yield chunk({}, c.finish_reason)
    if include_usage:
        yield {**base, "choices": [], "usage": _usage(c)}


def _usage(c: Completion) -> dict:
    return {"prompt_tokens": c.prompt_tokens, "completion_tokens": c.completion_tokens,
            "total_tokens": c.prompt_tokens + c.completion_tokens}


def make_server(model: ChatModel, host: str = "127.0.0.1", port: int = 8000, log=print) -> ThreadingHTTPServer:
    lock = threading.Lock()  # one generation at a time; the model is not re-entrant

    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, fmt, *args):
            log(f"{self.address_string()} {fmt % args}")

        def _json(self, code: int, obj):
            body = json.dumps(obj).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _error(self, code, msg):
            self._json(code, {"error": {"message": msg, "type": "invalid_request_error"}})

        def do_GET(self):
            if self.path.rstrip("/") in ("/v1/models", "/models"):
                return self._json(200, {"object": "list", "data": [
                    {"id": model.name, "object": "model", "created": 0, "owned_by": "ncm"}]})
            if self.path.rstrip("/") in ("", "/health"):
                return self._json(200, {"status": "ok", "model": model.name})
            self._error(404, f"no route {self.path}")

        def do_POST(self):
            if self.path.rstrip("/") not in ("/v1/chat/completions", "/chat/completions"):
                return self._error(404, f"no route {self.path}")
            try:
                req = json.loads(self.rfile.read(int(self.headers.get("Content-Length") or 0)) or b"{}")
                messages = req["messages"]
                if not isinstance(messages, list):
                    raise TypeError("messages must be a list")
            except (json.JSONDecodeError, KeyError, TypeError, ValueError) as e:
                return self._error(400, f"bad request: {e}")
            with lock:
                c = model.complete(messages, req.get("tools"),
                                   max_tokens=req.get("max_completion_tokens") or req.get("max_tokens"),
                                   temperature=req.get("temperature"), top_p=req.get("top_p"))
            cid = "chatcmpl-" + uuid.uuid4().hex[:24]
            if not req.get("stream"):
                return self._json(200, {
                    "id": cid, "object": "chat.completion", "created": int(time.time()), "model": model.name,
                    "choices": [{"index": 0, "message": c.message, "finish_reason": c.finish_reason}],
                    "usage": _usage(c)})
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Cache-Control", "no-cache")
            self.send_header("Connection", "close")
            self.end_headers()
            include_usage = bool((req.get("stream_options") or {}).get("include_usage"))
            for ch in _chunks(cid, model.name, c, include_usage):
                self.wfile.write(f"data: {json.dumps(ch)}\n\n".encode())
            self.wfile.write(b"data: [DONE]\n\n")
            self.wfile.flush()
            self.close_connection = True

    return ThreadingHTTPServer((host, port), Handler)
