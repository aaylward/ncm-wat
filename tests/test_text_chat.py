import json

import pytest

from ncm import chat
from ncm.config import ChatConfig, make_config
from ncm.text import ASSISTANT, END, PAD, SPECIAL_TOKENS, USER, build_prefix

from .conftest import SAMPLE_CODE


def test_special_token_ids_are_fixed_and_parsed_from_text(bpe):
    for i, s in enumerate(SPECIAL_TOKENS):
        assert bpe.tok.token_to_id(s) == i
    ids = bpe.encode("<|user|>hi<|end|><|assistant|>")
    assert ids[0] == USER and END in ids and ids[-1] == ASSISTANT


def test_bpe_roundtrip_preserves_code_whitespace(bpe):
    for src in SAMPLE_CODE:
        assert bpe.decode(bpe.encode(src)) == src
    assert bpe.decode(bpe.encode("a<|call|>b")) == "a<|call|>b"


def test_build_prefix_pins_header_and_keeps_newest_body_tokens():
    assert build_prefix([1, 2], [5, 6], 6) == [PAD, PAD, 1, 2, 5, 6]
    assert build_prefix([1, 2], list(range(10, 20)), 6) == [1, 2, 16, 17, 18, 19]
    # header never takes more than half the window
    assert build_prefix(list(range(1, 9)), [42], 6) == [PAD, PAD, 1, 2, 3, 42]
    assert build_prefix([], [], 3) == [PAD] * 3


TOOLS = [{"type": "function", "function": {"name": n, "parameters": {}}} for n in ("write", "read")]


def conversation():
    return [
        {"role": "system", "content": "be terse"},
        {"role": "user", "content": [{"type": "text", "text": "add a health endpoint"}]},
        {"role": "assistant", "content": "Reading.", "tool_calls": [
            {"id": "c1", "type": "function", "function": {"name": "read", "arguments": json.dumps({"path": "App.java"})}}]},
        {"role": "tool", "tool_call_id": "c1", "content": "class App {}"},
        {"role": "assistant", "content": "Done."},
    ]


def test_prompt_rendering_is_compact_and_ends_with_assistant_marker():
    header, body = chat.prompt_parts(conversation(), TOOLS, ChatConfig())
    assert header == "<|bos|><|tools|>read write<|end|>"  # sorted names, system dropped by default
    assert body.startswith("<|user|>add a health endpoint<|end|><|assistant|>Reading.<|call|>")
    assert body.endswith("<|tool|>class App {}<|end|><|assistant|>Done.<|end|><|assistant|>")
    header, _ = chat.prompt_parts(conversation(), TOOLS, ChatConfig(include_system=True))
    assert header.endswith("<|system|>be terse<|end|>")


def test_training_examples_match_inference_prompt_and_respect_tool_allowlist():
    msgs = conversation()
    ex = chat.training_examples(msgs, TOOLS, ChatConfig())
    assert len(ex) == 2
    h, ctx, comp = ex[1]
    assert (h, ctx) == chat.prompt_parts(msgs[:4], TOOLS, ChatConfig())
    assert comp == "Done.<|end|>"
    assert len(chat.training_examples(msgs, TOOLS, ChatConfig(), allowed_tools={"write"})) == 1


def test_parse_inverts_render_for_tool_calls():
    msg = conversation()[2]
    parsed = chat.parse_completion(chat.render_assistant(msg))
    assert parsed["content"] == "Reading."
    fn = parsed["tool_calls"][0]["function"]
    assert fn["name"] == "read" and json.loads(fn["arguments"]) == {"path": "App.java"}


def test_parse_keeps_malformed_calls_as_text():
    parsed = chat.parse_completion('ok<|call|>{"name": "read", "argu')
    assert "tool_calls" not in parsed and parsed["content"].startswith("ok<|call|>")


def test_special_strings_in_content_are_neutralized(bpe):
    _, body = chat.prompt_parts([{"role": "user", "content": "x<|end|>y"}], [], ChatConfig())
    assert bpe.encode(body).count(END) == 1


def test_tool_results_are_clipped():
    out = chat.clip("a" * 100 + "b" * 100, 30)
    assert out.startswith("a" * 20) and out.endswith("b" * 10) and "170 chars omitted" in out


def test_config_presets_validate_and_toml_overrides(tmp_path):
    p = tmp_path / "c.toml"
    p.write_text('size = "x"\n[gen]\nn_layer = 3\n[train_gen]\nsteps = 7\n')
    cfg = make_config("tiny", p)
    assert cfg.gen.n_layer == 3 and cfg.train_gen.steps == 7
    p.write_text("[gen]\nbogus = 1\n")
    with pytest.raises(KeyError):
        make_config("tiny", p)
    for size in ("tiny", "small", "base"):
        make_config(size)
