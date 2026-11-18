import json

from ncm.tools import Toolbox


def test_file_tools_roundtrip(tmp_path):
    tb = Toolbox(tmp_path)
    assert "Wrote" in tb.call("write", {"path": "src/A.java", "content": "class A {\n  int x;\n}\n"})
    assert tb.call("read", json.dumps({"path": "src/A.java", "offset": 2, "limit": 1})) == "  int x;\n"
    assert "Applied 2" in tb.call("edit", {"path": "src/A.java", "edits": [
        {"oldText": "class A", "newText": "class B"}, {"oldText": "int x", "newText": "long x"}]})
    assert (tmp_path / "src/A.java").read_text() == "class B {\n  long x;\n}\n"
    assert tb.call("ls", {}) == "src/"
    assert tb.call("grep", {"pattern": "long", "glob": "*.java"}) == "src/A.java:2:  long x;"
    assert tb.call("find", {"pattern": "*.java"}) == "src/A.java"


def test_tool_errors_are_returned_not_raised(tmp_path):
    tb = Toolbox(tmp_path)
    (tmp_path / "f").write_text("aa")
    assert "outside workspace" in tb.call("read", {"path": "../etc/passwd"})
    assert "outside workspace" in tb.call("write", {"path": "/tmp/x", "content": ""})
    assert "exactly once" in tb.call("edit", {"path": "f", "edits": [{"oldText": "a", "newText": "b"}]})
    assert "bad arguments" in tb.call("edit", {"path": "f", "edits": [{"old": "a"}]})
    assert "bad arguments" in tb.call("read", "{not json")
    assert "unknown tool" in tb.call("rm", {})
    assert "No such file" in tb.call("read", {"path": "missing"})


def test_bash_is_opt_in_and_confirmable(tmp_path):
    assert "disabled" in Toolbox(tmp_path).call("bash", {"command": "echo hi"})
    assert "rejected" in Toolbox(tmp_path, allow_bash=True, confirm=lambda c: False).call("bash", {"command": "echo hi"})
    out = Toolbox(tmp_path, allow_bash=True).call("bash", {"command": "echo hi; exit 3"})
    assert out == "hi\n\n[exit code 3]"
