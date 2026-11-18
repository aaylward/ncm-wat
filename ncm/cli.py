"""ncm: build data, train both stages, generate, serve, and run the agent."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .data import LANGS


def _sessions(specs):
    from .sessions import load_sessions
    for spec in specs or []:
        kind, _, path = spec.partition(":")
        if not path:
            raise SystemExit(f"--sessions wants KIND:PATH (claude|pi|jsonl), got {spec!r}")
        yield from load_sessions(kind, Path(path).expanduser())


def cmd_data_build(a):
    from .config import ChatConfig
    from .data import build_dataset
    from .text import BPE
    trajectories = list(_sessions(a.sessions))
    print(f"loaded {len(trajectories)} session trajectories", file=sys.stderr)
    build_dataset(Path(a.out), a.src or [], [x for x in (a.lang or "").split(",") if x], infra=a.infra,
                  trajectories=trajectories, chat_cfg=ChatConfig(a.include_system, a.tool_result_chars),
                  bpe=BPE.load(a.bpe) if a.bpe else None, vocab_size=a.vocab_size, val_frac=a.val_frac,
                  max_bytes=a.max_bytes, chat_repeat=a.chat_repeat, max_context=a.max_context, log=lambda m: print(m, file=sys.stderr))


def _overrides(a):
    return {"steps": a.steps, "batch_size": a.batch_size, "lr": a.lr}


def cmd_train_vq(a):
    from .train import train_vq
    train_vq(Path(a.run), Path(a.data), a.size, a.config, a.device, _overrides(a))


def cmd_train_gen(a):
    from .train import train_gen
    train_gen(Path(a.run), Path(a.data), a.init_from, a.device, _overrides(a))


def cmd_generate(a):
    from .infer import NCM
    m = NCM(a.run, a.device)
    prompt = Path(a.prompt_file).read_text() if a.prompt_file else (a.prompt or "")
    header = f"<|file|>{a.path}\n" if a.path else ""
    if a.levels:
        from .text import BOS
        for i, text in enumerate(m.levels([BOS] + m.bpe.encode(header), m.bpe.encode(prompt), a.temperature)):
            print(f"--- level {i} ---\n{text}")
        return
    print(m.complete_text(prompt, header=header, max_new_tokens=a.max_tokens,
                          temperature=a.temperature, top_p=a.top_p, seed=a.seed))


def cmd_serve(a):
    from .infer import load_chat_model
    from .server import make_server
    model = load_chat_model(a.run, a.name, a.device, max_tokens=a.max_tokens, temperature=a.temperature)
    srv = make_server(model, a.host, a.port, log=lambda m: print(m, file=sys.stderr))
    print(f"serving {model.name} at http://{a.host}:{srv.server_port}/v1", file=sys.stderr)
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass


def _model_for(a):
    if a.openai:
        from .api import OpenAIChatModel
        return OpenAIChatModel(a.openai, a.model or "default", api_key_env=a.api_key_env)
    if a.run:
        from .infer import load_chat_model
        return load_chat_model(a.run, device=a.device)
    raise SystemExit("pass --run RUN_DIR (local NCM) or --openai BASE_URL --model NAME")


def _short(s, n=300):
    s = s if isinstance(s, str) else json.dumps(s)
    return s if len(s) <= n else s[:n] + "..."


def cmd_agent(a):
    from .agent import DEFAULT_SYSTEM, Agent
    from .tools import Toolbox

    def confirm(cmd):
        return a.yes or input(f"run `{cmd}`? [y/N] ").strip().lower() == "y"

    def on_event(kind, d):
        if kind == "assistant":
            if d.get("content"):
                print(f"\n{d['content']}")
            for tc in d.get("tool_calls") or []:
                print(f"  -> {tc['function']['name']} {_short(tc['function']['arguments'], 200)}")
        elif kind == "tool":
            print("     " + _short(d["result"]).replace("\n", "\n     "))
        elif kind == "limit":
            print(f"(stopped after {d['max_steps']} steps)")

    agent = Agent(_model_for(a), Toolbox(a.workspace, allow_bash=a.allow_bash, confirm=confirm),
                  system=None if a.no_system else DEFAULT_SYSTEM, max_steps=a.max_steps,
                  record=a.record, on_event=on_event)
    try:
        if a.task:
            agent.run(" ".join(a.task))
        else:
            while True:
                try:
                    task = input("\n> ").strip()
                except EOFError:
                    break
                if task:
                    agent.run(task)
    finally:
        agent.save()


def cmd_pi_config(a):
    print(json.dumps({"providers": {"ncm": {
        "baseUrl": a.url, "api": "openai-completions", "apiKey": "ncm",
        "compat": {"supportsDeveloperRole": False, "supportsReasoningEffort": False},
        "models": [{"id": a.model, "name": f"NCM {a.model}", "reasoning": False, "input": ["text"],
                    "contextWindow": a.context, "maxTokens": a.max_tokens,
                    "cost": {"input": 0, "output": 0, "cacheRead": 0, "cacheWrite": 0}}]}}}, indent=2))


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="ncm", description=__doc__)
    sub = p.add_subparsers(dest="cmd", required=True)

    data = sub.add_parser("data").add_subparsers(dest="sub", required=True)
    b = data.add_parser("build", help="code dirs + agent sessions -> BPE + train/val corpora")
    b.add_argument("--out", required=True)
    b.add_argument("--src", nargs="*", help="source roots (repos or dirs of repos)")
    b.add_argument("--lang", help="comma list: " + ",".join(LANGS))
    b.add_argument("--infra", action="store_true", help="also .proto/.sql/.yaml/Dockerfile/...")
    b.add_argument("--sessions", nargs="*", metavar="KIND:PATH",
                   help="claude:~/.claude/projects  pi:~/.pi/agent/sessions  jsonl:recorded.jsonl")
    b.add_argument("--bpe", help="reuse an existing bpe.json (required to fine-tune an existing run)")
    b.add_argument("--vocab-size", type=int, default=8192)
    b.add_argument("--val-frac", type=float, default=0.01)
    b.add_argument("--max-bytes", type=int, default=200_000)
    b.add_argument("--chat-repeat", type=int, default=1, help="upweight session examples")
    b.add_argument("--max-context", type=int, default=4096, help="context tokens kept per example (>= prefix_len)")
    b.add_argument("--include-system", action="store_true")
    b.add_argument("--tool-result-chars", type=int, default=2000)
    b.set_defaults(fn=cmd_data_build)

    train = sub.add_parser("train").add_subparsers(dest="sub", required=True)
    for name, fn, help_ in (("vq", cmd_train_vq, "stage 1: multiscale VQ tokenizer"),
                            ("gen", cmd_train_gen, "stage 2: next-concept generator")):
        t = train.add_parser(name, help=help_)
        t.add_argument("--data", required=True)
        t.add_argument("--run", required=True)
        t.add_argument("--device", default="auto")
        t.add_argument("--steps", type=int)
        t.add_argument("--batch-size", type=int)
        t.add_argument("--lr", type=float)
        if name == "vq":
            t.add_argument("--size", default="small", help="tiny | small | base")
            t.add_argument("--config", help="TOML overrides")
        else:
            t.add_argument("--init-from", help="start from another run's weights (fine-tuning)")
        t.set_defaults(fn=fn)

    g = sub.add_parser("generate", help="raw completion from a run")
    g.add_argument("--run", required=True)
    g.add_argument("--prompt")
    g.add_argument("--prompt-file")
    g.add_argument("--path", help="file path header, e.g. src/main/java/App.java")
    g.add_argument("--max-tokens", type=int, default=256)
    g.add_argument("--temperature", type=float, default=0.7)
    g.add_argument("--top-p", type=float, default=0.95)
    g.add_argument("--seed", type=int)
    g.add_argument("--levels", action="store_true", help="show the first block after each level")
    g.add_argument("--device", default="auto")
    g.set_defaults(fn=cmd_generate)

    s = sub.add_parser("serve", help="OpenAI-compatible server")
    s.add_argument("--run", required=True)
    s.add_argument("--name", help="model id (default: run dir name)")
    s.add_argument("--host", default="127.0.0.1")
    s.add_argument("--port", type=int, default=8000)
    s.add_argument("--max-tokens", type=int, default=1024)
    s.add_argument("--temperature", type=float, default=0.7)
    s.add_argument("--device", default="auto")
    s.set_defaults(fn=cmd_serve)

    ag = sub.add_parser("agent", help="run the agent harness")
    ag.add_argument("task", nargs="*", help="omit for an interactive session")
    ag.add_argument("--run", help="local NCM run dir")
    ag.add_argument("--openai", metavar="BASE_URL", help="OpenAI-compatible endpoint, e.g. http://127.0.0.1:8000/v1")
    ag.add_argument("--model")
    ag.add_argument("--api-key-env", default="OPENAI_API_KEY")
    ag.add_argument("--workspace", default=".")
    ag.add_argument("--allow-bash", action="store_true", help="bash runs unsandboxed in the workspace")
    ag.add_argument("--yes", action="store_true", help="do not ask before each bash command")
    ag.add_argument("--record", help="append the session as a training trajectory (JSONL)")
    ag.add_argument("--max-steps", type=int, default=30)
    ag.add_argument("--no-system", action="store_true")
    ag.add_argument("--device", default="auto")
    ag.set_defaults(fn=cmd_agent)

    pc = sub.add_parser("pi-config", help="print a pi models.json entry for `ncm serve`")
    pc.add_argument("--url", default="http://127.0.0.1:8000/v1")
    pc.add_argument("--model", default="ncm")
    pc.add_argument("--context", type=int, default=8192)
    pc.add_argument("--max-tokens", type=int, default=1024)
    pc.set_defaults(fn=cmd_pi_config)
    return p


def main(argv=None):
    a = parser().parse_args(argv)
    a.fn(a)


if __name__ == "__main__":
    main()
