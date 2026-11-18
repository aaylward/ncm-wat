"""pi (badlogic/pi-mono coding agent) driving `ncm serve` through its custom-provider config.
Skipped unless a `pi` binary is on PATH (or NCM_PI_BIN points at one)."""
import json
import os
import shutil
import subprocess
import threading

import pytest

from ncm.cli import main
from ncm.server import make_server

from .test_server_agent import Scripted, call

PI = os.environ.get("NCM_PI_BIN") or shutil.which("pi")
pytestmark = pytest.mark.skipif(not PI, reason="pi not installed")


def test_pi_runs_a_tool_loop_against_ncm_server(tmp_path, capsys):
    ws = tmp_path / "ws"
    ws.mkdir()
    (ws / "App.java").write_text("class App {}\n")
    model = Scripted({"role": "assistant", "content": None, "tool_calls": [call("read", path="App.java")]},
                     {"role": "assistant", "content": "App.java declares an empty class."})
    srv = make_server(model, port=0, log=lambda *_: None)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    try:
        agent_dir = tmp_path / "pi"
        agent_dir.mkdir()
        main(["pi-config", "--url", f"http://127.0.0.1:{srv.server_port}/v1", "--model", "ncm-test"])
        (agent_dir / "models.json").write_text(capsys.readouterr().out)
        r = subprocess.run([PI, "--model", "ncm/ncm-test", "--no-session", "-p", "what is in App.java?"],
                           cwd=ws, capture_output=True, text=True, timeout=120, stdin=subprocess.DEVNULL,
                           env={**os.environ, "PI_CODING_AGENT_DIR": str(agent_dir)})
    finally:
        srv.shutdown()
        srv.server_close()
    assert r.returncode == 0, r.stderr
    assert "App.java declares an empty class." in r.stdout
    second, tools = model.seen[1]
    assert {"bash", "edit", "read", "write"} <= {t["function"]["name"] for t in tools}
    tool_msg = next(m for m in second if m["role"] == "tool")
    assert "class App {}" in json.dumps(tool_msg["content"])
