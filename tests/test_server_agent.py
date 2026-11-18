import json
import threading
import urllib.error
import urllib.request

import pytest

from ncm import chat
from ncm.agent import Agent
from ncm.api import Completion, OpenAIChatModel
from ncm.cli import main
from ncm.config import ChatConfig
from ncm.server import make_server
from ncm.sessions import load_sessions
from ncm.tools import TOOL_NAMES, Toolbox


def call(name, **args):
    return {"id": f"call_{name}", "type": "function", "function": {"name": name, "arguments": json.dumps(args)}}


class Scripted:
    """ChatModel that replays canned assistant messages and records what it was sent."""
    name = "scripted"

    def __init__(self, *messages):
        self.script, self.seen = list(messages), []

    def complete(self, messages, tools=None, **_):
        self.seen.append((json.loads(json.dumps(messages)), tools))
        msg = self.script.pop(0)
        return Completion(msg, "tool_calls" if msg.get("tool_calls") else "stop", 11, 7)


@pytest.fixture
def serve():
    servers = []

    def start(model):
        srv = make_server(model, port=0, log=lambda *_: None)
        threading.Thread(target=srv.serve_forever, daemon=True).start()
        servers.append(srv)
        return f"http://127.0.0.1:{srv.server_port}/v1"
    yield start
    for s in servers:
        s.shutdown()
        s.server_close()


def post(url, body):
    req = urllib.request.Request(url + "/chat/completions", json.dumps(body).encode(),
                                 {"Content-Type": "application/json"})
    with urllib.request.urlopen(req) as r:
        return r.headers.get("Content-Type"), r.read().decode()


TOOL_MSG = {"role": "assistant", "content": "Reading.", "tool_calls": [call("read", path="A.java")]}


def test_non_streaming_response_shape(serve):
    url = serve(Scripted(TOOL_MSG))
    _, body = post(url, {"model": "x", "messages": [{"role": "user", "content": "hi"}]})
    d = json.loads(body)
    assert d["object"] == "chat.completion" and d["model"] == "scripted"
    assert d["choices"][0]["message"] == TOOL_MSG and d["choices"][0]["finish_reason"] == "tool_calls"
    assert d["usage"] == {"prompt_tokens": 11, "completion_tokens": 7, "total_tokens": 18}


def test_streaming_response_reassembles(serve):
    url = serve(Scripted(TOOL_MSG))
    ctype, body = post(url, {"messages": [], "stream": True, "stream_options": {"include_usage": True}})
    assert ctype == "text/event-stream"
    events = [line[6:] for line in body.splitlines() if line.startswith("data: ")]
    assert events[-1] == "[DONE]"
    chunks = [json.loads(e) for e in events[:-1]]
    deltas = [c["choices"][0]["delta"] for c in chunks if c["choices"]]
    assert "".join(d.get("content") or "" for d in deltas) == "Reading."
    tcs = [tc for d in deltas for tc in d.get("tool_calls", [])]
    assert tcs[0]["index"] == 0 and tcs[0]["function"]["name"] == "read"
    assert [c["choices"][0]["finish_reason"] for c in chunks if c["choices"]][-1] == "tool_calls"
    assert chunks[-1]["usage"]["total_tokens"] == 18


def test_client_server_roundtrip_and_errors(serve):
    model = Scripted(TOOL_MSG)
    url = serve(model)
    c = OpenAIChatModel(url, "x", api_key="k").complete([{"role": "user", "content": "hi"}], [{"type": "function", "function": {"name": "read"}}])
    assert c.message == TOOL_MSG and c.finish_reason == "tool_calls" and c.prompt_tokens == 11
    assert model.seen[0][1] == [{"type": "function", "function": {"name": "read"}}]
    with urllib.request.urlopen(url + "/models") as r:
        assert json.loads(r.read())["data"][0]["id"] == "scripted"
    with pytest.raises(urllib.error.HTTPError) as e:
        post(url, {"no": "messages"})
    assert e.value.code == 400


def test_agent_runs_tools_feeds_results_and_records(tmp_path):
    ws, rec = tmp_path / "ws", tmp_path / "rec.jsonl"
    ws.mkdir()
    model = Scripted(
        {"role": "assistant", "content": None, "tool_calls": [call("write", path="Health.java", content="class Health {}\n")]},
        {"role": "assistant", "content": None, "tool_calls": [call("read", path="Health.java"), call("bash", command="ls")]},
        {"role": "assistant", "content": "Added Health.java."},
    )
    agent = Agent(model, Toolbox(ws), record=rec)
    assert agent.run("add a health class") == "Added Health.java."
    agent.save()
    assert (ws / "Health.java").read_text() == "class Health {}\n"
    sent, tools = model.seen[2]
    assert [m["role"] for m in sent] == ["system", "user", "assistant", "tool", "assistant", "tool", "tool"]
    assert sent[5]["content"] == "class Health {}\n" and "disabled" in sent[6]["content"]
    assert {t["function"]["name"] for t in tools} == TOOL_NAMES
    traj = list(load_sessions("jsonl", rec))
    assert len(traj) == 1
    ex = chat.training_examples(traj[0]["messages"], traj[0]["tools"], ChatConfig(), allowed_tools=set(TOOL_NAMES))
    assert len(ex) == 3 and ex[-1][2] == "Added Health.java.<|end|>"


def test_agent_stops_at_step_limit(tmp_path):
    model = Scripted(*[{"role": "assistant", "content": None, "tool_calls": [call("ls")]}] * 3)
    events = []
    Agent(model, Toolbox(tmp_path), max_steps=3, on_event=lambda k, d: events.append(k)).run("loop")
    assert events.count("assistant") == 3 and events[-1] == "limit"


def test_cli_agent_against_server_and_pi_config(serve, tmp_path, capsys):
    url = serve(Scripted({"role": "assistant", "content": None, "tool_calls": [call("ls")]},
                         {"role": "assistant", "content": "Empty workspace."}))
    main(["agent", "--openai", url, "--model", "m", "--workspace", str(tmp_path), "--record",
          str(tmp_path / "r.jsonl"), "what is here?"])
    out = capsys.readouterr().out
    assert "-> ls" in out and "Empty workspace." in out and (tmp_path / "r.jsonl").exists()
    main(["pi-config", "--url", url, "--model", "ncm-java"])
    cfg = json.loads(capsys.readouterr().out)
    assert cfg["providers"]["ncm"]["models"][0]["id"] == "ncm-java"
