import json

from ncm import chat
from ncm.config import ChatConfig
from ncm.sessions import load_claude_session, load_pi_session, load_sessions, scrub
from ncm.tools import TOOL_NAMES

CWD = "/home/dev/shop"


def _w(path, rows):
    path.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    return path


def claude_rows():
    """Mirrors the real layout: one row per content block, rows of one API message share
    message.id, attachments sit inside the parentUuid chain, sidechains and branches exist."""
    def row(t, uid, parent, content, mid=None, **kw):
        msg = {"role": t, "content": content}
        if mid:
            msg["id"] = mid
        return {"type": t, "uuid": uid, "parentUuid": parent, "isSidechain": False, "cwd": CWD, "message": msg, **kw}
    return [
        row("user", "u1", None, "<system-reminder>ctx</system-reminder>Add a /health route"),
        {"type": "attachment", "uuid": "at1", "parentUuid": "u1", "attachment": {"type": "environment"}},
        row("assistant", "a1", "at1", [{"type": "thinking", "thinking": "hmm"}], mid="m1"),
        row("assistant", "a2", "a1", [{"type": "text", "text": "Looking."}], mid="m1"),
        row("assistant", "a3", "a2", [{"type": "tool_use", "id": "t1", "name": "Read",
                                       "input": {"file_path": CWD + "/src/App.java"}}], mid="m1"),
        row("user", "u2", "a3", [{"type": "tool_result", "tool_use_id": "t1", "content": "class App {}"}]),
        row("assistant", "dead", "u2", [{"type": "text", "text": "abandoned branch"}], mid="mx"),
        row("assistant", "a4", "u2", [{"type": "tool_use", "id": "t2", "name": "TodoWrite", "input": {"todos": []}}], mid="m2"),
        row("user", "u3", "a4", [{"type": "tool_result", "tool_use_id": "t2", "content": [{"type": "text", "text": "ok"}]}]),
        {"type": "attachment", "uuid": "q1", "parentUuid": "u3",
         "attachment": {"type": "queued_command", "prompt": "use port 8081"}},
        row("assistant", "a5", "q1", [{"type": "tool_use", "id": "t3", "name": "Edit", "input": {
            "file_path": CWD + "/src/App.java", "old_string": "{}", "new_string": "{ }"}}], mid="m3"),
        row("user", "u4", "a5", [{"type": "tool_result", "tool_use_id": "t3", "content": "export KEY=sk-ant-abcdefghijklmnopqrstuvwxyz"}]),
        {**row("assistant", "side", "u4", [{"type": "text", "text": "subagent"}], mid="ms"), "isSidechain": True},
        row("assistant", "a6", "u4", [{"type": "text", "text": "Done."}], mid="m4"),
    ]


def test_claude_session_follows_leaf_chain_merges_blocks_and_maps_tools(tmp_path):
    t = load_claude_session(_w(tmp_path / "s.jsonl", claude_rows()))
    m = t["messages"]
    assert [x["role"] for x in m] == ["user", "assistant", "tool", "assistant", "tool", "user", "assistant", "tool", "assistant"]
    assert m[0]["content"] == "Add a /health route" and m[5]["content"] == "use port 8081"
    assert m[1]["content"] == "Looking."
    read = m[1]["tool_calls"][0]
    assert read["id"] == "t1" and read["function"]["name"] == "read"
    assert json.loads(read["function"]["arguments"]) == {"path": "src/App.java"}
    edit = json.loads(m[6]["tool_calls"][0]["function"]["arguments"])
    assert edit == {"path": "src/App.java", "edits": [{"oldText": "{}", "newText": "{ }"}]}
    assert "sk-ant" not in m[7]["content"] and "<REDACTED>" in m[7]["content"]
    assert all("abandoned" not in (x.get("content") or "") and "subagent" not in (x.get("content") or "") for x in m)


def test_unmapped_tool_turns_are_context_not_targets(tmp_path):
    t = load_claude_session(_w(tmp_path / "s.jsonl", claude_rows()))
    ex = chat.training_examples(t["messages"], t["tools"], ChatConfig(), allowed_tools=set(TOOL_NAMES))
    targets = [comp for _, _, comp in ex]
    assert len(targets) == 3 and not any("TodoWrite" in c for c in targets)
    assert "TodoWrite" in ex[-1][1]  # still visible in the later context


def test_pi_session_uses_leaf_branch(tmp_path):
    rows = [
        {"type": "session", "version": 3, "id": "s", "cwd": CWD},
        {"type": "message", "id": "1", "parentId": None, "message": {"role": "user", "content": "fix the build"}},
        {"type": "message", "id": "2", "parentId": "1", "message": {"role": "assistant", "content": [
            {"type": "thinking", "thinking": "x"},
            {"type": "toolCall", "id": "c1", "name": "bash", "arguments": {"command": "make"}}]}},
        {"type": "message", "id": "3", "parentId": "2", "message": {"role": "toolResult", "toolCallId": "c1",
                                                                    "toolName": "bash", "content": [{"type": "text", "text": "error"}]}},
        {"type": "message", "id": "4", "parentId": "3", "message": {"role": "assistant", "content": [{"type": "text", "text": "old"}]}},
        {"type": "model_change", "id": "5", "parentId": "3", "provider": "x", "modelId": "y"},
        {"type": "message", "id": "6", "parentId": "5", "message": {"role": "assistant", "content": [
            {"type": "toolCall", "id": "c2", "name": "write", "arguments": {"path": CWD + "/Makefile", "content": "all:\n"}}]}},
    ]
    t = load_pi_session(_w(tmp_path / "p.jsonl", rows))
    m = t["messages"]
    assert [x["role"] for x in m] == ["user", "assistant", "tool", "assistant"]
    assert json.loads(m[3]["tool_calls"][0]["function"]["arguments"])["path"] == "Makefile"
    assert list(load_sessions("pi", tmp_path))[0]["messages"] == m


def test_scrub_redacts_common_secret_shapes():
    s = scrub("AKIAABCDEFGHIJKLMNOP ghp_" + "a" * 36 + " password = 'hunter2hunter2' "
              "postgres://u:pw@db/x -----BEGIN RSA PRIVATE KEY-----\nabc\n-----END RSA PRIVATE KEY----- keep me")
    assert "AKIA" not in s and "ghp_" not in s and "hunter2" not in s and "pw@" not in s and "abc" not in s
    assert s.endswith("keep me") and "password = '<REDACTED>'" in s
