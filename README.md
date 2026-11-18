# ncm-wat

Train [Next-Concept Models](https://github.com/HazyResearch/ncm) (NCM) on backend/distributed code,
serve them over an OpenAI-compatible API, and drive them from an agent harness, either the one here or [pi](https://github.com/badlogic/pi-mono).

## Read this first

- **Upstream ships no training code.** `HazyResearch/ncm` (MIT, Oct 2026) has the model and
  evaluation scripts only. Both training stages here are written from scratch, following upstream's
  model code (`models/quant.py`, `models/next_concept.py`). Deviations are listed under
  *Architecture*.
- **NCM emits fixed-size blocks.** One generator call produces `block_len` tokens (64 for `small`, 128 for `base`) in
  `len(levels)` parallel passes. Longer outputs are generated block by block, so the "fixed number of
  passes" speed claim holds per block, not per response. There is no token streaming.
- **A home-trained 10–100M model will not be a capable agent by itself.** Pretraining on code
  gives you completion. Agent behaviour (tool calls, task structure) comes from fine-tuning on agent
  trajectories. Below, those come from sessions you record with a strong model. Treat this as
  a research rig, not a Copilot replacement.

## Install

```bash
pip install -e '.[dev]'          # torch, numpy, tokenizers; Python >= 3.10
```

For a CUDA GPU, install the matching torch build first. `--device auto` picks cuda, then mps, then cpu.

## Pipeline

```bash
# 1. corpus: source trees -> BPE + train/val token corpora
git clone --depth 1 https://github.com/spring-projects/spring-petclinic repos/spring-petclinic
ncm data build --out data/java --src repos --lang java,kotlin --infra
```

```bash
# 2. stage 1: multiscale residual VQ tokenizer (codes per block, coarse -> fine)
ncm train vq --data data/java --run runs/java --size small
```

```bash
# 3. stage 2: next-concept generator, frozen tokenizer, teacher forcing over all levels
ncm train gen --data data/java --run runs/java
```

```bash
# 4. use it
ncm generate --run runs/java --path src/main/java/com/acme/OrderController.java --prompt "package com.acme;"
```

```bash
ncm generate --run runs/java --levels --prompt "public class "   # watch one block refine level by level
```

```bash
ncm serve --run runs/java --port 8000                             # OpenAI-compatible /v1/chat/completions
```

```bash
ncm agent --run runs/java --workspace ~/src/service "add a /health endpoint"
```

Re-running a train command resumes from its checkpoint. Per-step logs go to `runs/<name>/log.jsonl`.
Stage 1 eval reports `token_acc`, `block_exact` and per-level `code_use`; codebook collapse shows up there.
Stage 2 eval reports per-level code accuracy.

Languages (`--lang`): java, kotlin, scala, go, rust, typescript, javascript, python, csharp, elixir,
erlang, ruby, php. `--infra` adds .proto/.sql/.graphql/.yaml/.tf/.sh/Dockerfile/Makefile. Vendored,
generated (`DO NOT EDIT`, `@generated`), minified and duplicate files are skipped.

Sizes (`--size`, override any field with `--config overrides.toml`; see `ncm/config.py`):

| size  | block_len | levels | prefix_len | generator         | notes |
|-------|-----------|--------|------------|-------------------|-------|
| tiny  | 16        | 4      | 32         | 2L / 64d, ~0.6M   | tests and CPU smoke runs |
| small | 64        | 8      | 512        | 6L / 384d, ~15M   | ~3.7 s/step at batch 8 on a 4-core CPU; use a GPU |
| base  | 128       | 11     | 1024       | 12L / 768d, ~95M  | roughly the upstream LM1B/OWT scale |

Parameter counts assume an 8192-token BPE vocab. GPU throughput has not been measured.

## Agent sessions as training data

Agent trajectories are what teach tool use. Three sources feed one format:

```bash
# record sessions with a strong model through the same harness (any OpenAI-compatible endpoint)
ncm agent --openai https://api.openai.com/v1 --model <model> --workspace ~/src/service --allow-bash --record traj/run1.jsonl "..."
```

```bash
# fine-tune an existing run: reuse its BPE, then start the generator from its weights
ncm data build --out data/java-agent --bpe runs/java/bpe.json --src repos --lang java \
  --sessions jsonl:traj/ claude:~/.claude/projects pi:~/.pi/agent/sessions --chat-repeat 4
```

```bash
ncm train gen --data data/java-agent --run runs/java-agent --init-from runs/java
```

- **Tool vocabulary.** The harness's tools match pi's built-ins: `read write edit ls bash grep find`.
  Claude Code calls are mapped onto them (`Read→read`, `Edit/MultiEdit→edit`, `Bash→bash`,
  `Grep→grep`, `Glob→find`, `LS→ls`), and absolute paths inside the session's cwd become relative.
  Turns that call anything else (TodoWrite, Task, MCP tools, `Edit` with `replace_all`) stay in
  the context of later turns but are never training targets.
- **Claude Code import** follows the `parentUuid` chain from the last message, so abandoned branches and
  sidechains (subagents) are dropped. It merges the per-block rows of each API message, strips
  `<system-reminder>`, and keeps messages you typed mid-turn.
- **Secrets.** Every imported or recorded message passes a regex scrubber (cloud keys, GitHub/Slack/OpenAI/
  Anthropic/Stripe tokens, JWTs, private keys, DB URLs, `*_KEY=`/`password:` assignments). It is a
  best-effort filter. Look at what you ingest.
- **Terms.** Anthropic's and OpenAI's terms restrict using model outputs to build competing models.
  Decide for yourself whether your use is allowed before training on Claude Code sessions or on
  trajectories recorded from hosted models.
- Per example, only the last `--max-context` (4096) context tokens are stored, so long sessions don't
  blow up storage. `prefix_len` must not exceed it.

## pi

`ncm serve` speaks the OpenAI Chat Completions API, streamed or not, so pi can use it as a custom
provider:

```bash
ncm serve --run runs/java-agent --name ncm-java --port 8000 &
mkdir -p ~/.pi/agent && ncm pi-config --url http://127.0.0.1:8000/v1 --model ncm-java > ~/.pi/agent/models.json
pi --model ncm/ncm-java --tools read,bash,edit,write,grep,find,ls
```

- Pass all seven `--tools`. The prompt header lists tool names, and training data from this harness
  or from Claude Code has all seven. pi's default set is four.
- pi's ~2.5k-character system prompt is dropped (`chat.include_system = false`). Reasons: it would take
  a large share of a 512–1024-token prefix, and imported sessions don't contain it anyway. Set
  `--include-system` at `data build` time if your trajectories do carry the prompt you'll serve with.
- `tests/test_pi.py` runs pi 0.73.1 through a read → answer tool loop against the server
  (skipped unless `pi` is on PATH or `NCM_PI_BIN` is set).

The harness is pluggable in both directions. `ncm agent` takes any `ChatModel`: an in-process NCM
(`--run`) or any OpenAI-compatible endpoint (`--openai`). `ncm serve` puts any `ChatModel`
behind HTTP. See `ncm/api.py`.

## Architecture

```
text --BPE--> tokens --(block_len)--> VQ tokenizer --> codes per level: l0 < l1 < ... < block_len
                                                      (each level: pool residual, nearest code, upsample, subtract)
prefix tokens (left-padded, header pinned) + level-k inputs --> generator --> level-k codes
   level-k input = tokenizer's partial reconstruction after levels < k, pooled to l_k positions
   attention: prefix<->prefix; level k -> prefix + level k (block-diagonal)
generation: encode prefix once (KV cache), then one pass per level; decode codes -> tokens
```

Kept from upstream: EMA codebooks, per-level residual convs (Phi), a learned level-0 down/up
sampler with straight-through, fine-level dropout, the shallow token-level decoder, learned
level-0 queries (`query_level0`), codebook-tied distance heads with learned temperature, and QK-norm + RoPE.

Deviations:
- The generator has its own prefix token embedding.
- Prefixes are left-padded. RoPE places level-k position j at its token-space centre after the
  prefix, instead of restarting each level at 0.
- EMA counts start at 1, codebooks are seeded from the first batch, and dead codes are re-seeded
  (`restart_threshold`).
- There is no CFG, frequency penalty, flex-attention or T5 context encoder.

Chat template (`ncm/chat.py`) is shared by data building, inference and serving:

```
<|bos|><|tools|>bash edit find grep ls read write<|end|>
<|user|>add a health route<|end|><|assistant|><|call|>{"name":"read","arguments":{"path":"App.java"}}<|end|>
<|tool|>class App {}<|end|><|assistant|>   <- generation starts here, stops at <|end|>
```

## Tests

```bash
pytest            # ~30 s on CPU; includes tiny end-to-end training (-m "not slow" to skip)
```

Notable checks:
- Cached level-by-level generation reproduces the teacher-forced training logits exactly.
- Block-diagonal isolation between levels.
- A tiny model trained in-test reproduces a multi-block file exactly and emits a trained tool call through the chat path.
- Server streaming/non-streaming shapes, and the client↔server round trip.
- Agent tool loop and recording.
- pi, Claude Code and recorded-session import; secret scrubbing.
