"""Train a tiny model end to end on CPU and check it can reproduce what it was trained on:
multi-block code completion and an agent tool call through the chat path."""
import json

import pytest

from ncm.data import build_dataset
from ncm.infer import NCM, NCMChatModel
from ncm.tools import TOOL_SCHEMAS
from ncm.train import train_gen, train_vq

from .conftest import SAMPLE_CODE

pytestmark = pytest.mark.slow

TRAJ = {"messages": [
    {"role": "user", "content": "add a health route"},
    {"role": "assistant", "content": None, "tool_calls": [
        {"id": "a", "type": "function", "function": {"name": "read", "arguments": json.dumps({"path": "app.ts"})}}]},
    {"role": "tool", "tool_call_id": "a", "content": SAMPLE_CODE[2]},
    {"role": "assistant", "content": "It already has one."},
], "tools": TOOL_SCHEMAS}
QUIET = dict(log_every=10**6, eval_every=10**6)


@pytest.fixture(scope="module")
def trained(tmp_path_factory):
    root = tmp_path_factory.mktemp("e2e")
    for i, (name, text) in enumerate(zip(["Greeting.java", "main.go", "app.ts"], SAMPLE_CODE)):
        (root / "src" / f"r{i}").mkdir(parents=True)
        (root / "src" / f"r{i}" / name).write_text(text)
    build_dataset(root / "ds", [str(root / "src")], ["java", "go", "typescript"], trajectories=[TRAJ],
                  vocab_size=300, val_frac=0.0, chat_repeat=3, log=lambda *_: None)
    train_vq(root / "run", root / "ds", size="tiny", overrides=dict(steps=1000, **QUIET), echo=lambda *_: None)
    train_gen(root / "run", root / "ds", overrides=dict(steps=800, **QUIET), echo=lambda *_: None)
    return root, NCM(root / "run")


def test_greedy_completion_reproduces_training_file_across_blocks(trained):
    _, m = trained
    out = m.complete_text("", header="<|file|>r1/main.go\n", max_new_tokens=400, temperature=0)
    assert len(m.bpe.encode(SAMPLE_CODE[1])) > 3 * m.block_len  # really multi-block
    assert out == SAMPLE_CODE[1]


def test_chat_model_emits_trained_tool_call_then_answer(trained):
    _, m = trained
    cm = NCMChatModel(m, temperature=0)
    c = cm.complete(TRAJ["messages"][:1], TOOL_SCHEMAS)
    assert c.finish_reason == "tool_calls"
    fn = c.message["tool_calls"][0]["function"]
    assert fn["name"] == "read" and json.loads(fn["arguments"]) == {"path": "app.ts"}
    c = cm.complete(TRAJ["messages"][:3], TOOL_SCHEMAS)
    assert c.message == {"role": "assistant", "content": "It already has one."} and c.finish_reason == "stop"


def test_levels_view_and_max_tokens(trained):
    _, m = trained
    views = m.levels([1], [])
    assert len(views) == len(m.cfg.vq.levels)
    ids, stopped = m.generate_ids([1], [], max_new_tokens=5, temperature=0)
    assert len(ids) == 5 and not stopped


def test_resume_and_finetune_from_base_run(trained):
    root, _ = trained
    logs = []
    train_gen(root / "run", root / "ds", overrides=dict(steps=810, **QUIET), echo=logs.append)
    assert "step=800" in logs[0] and "step=810" in logs[-1]  # resumed, not restarted
    logs = []
    train_gen(root / "ft", root / "ds", init_from=str(root / "run"), overrides=dict(steps=5, **QUIET), echo=logs.append)
    assert "step=0" in logs[0] and (root / "ft" / "gen.pt").exists() and (root / "ft" / "vq.pt").exists()
