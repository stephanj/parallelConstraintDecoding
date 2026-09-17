# Parallel Constrained Decoding

Two implementations of *parallel constrained decoding* — structured extraction
that fills a whole JSON schema of booleans/enums in a couple of forward passes
instead of generating it token by token — runnable and measurable on this
Apple Silicon machine:

- **`python/`** — a local port of the engine published at
  [harshatheg/Qwen-2.5-1B-RLCD](https://huggingface.co/harshatheg/Qwen-2.5-1B-RLCD)
  (MLX). Despite the name, that Hugging Face repo contains **no model weights
  and no fine-tune**: it is source code that runs stock
  `mlx-community/Qwen2.5-1.5B-Instruct-4bit` with a custom decoding strategy.
  See [NOTICE](NOTICE) for what was vendored and what was changed.
- **`java/`** — a from-scratch Java implementation of the same technique on
  llama.cpp through the JDK Foreign Function & Memory API. **It is the fastest
  of the three engines measured here**, and it ships with a local web app
  (`java -jar … serve`) that races it against token-by-token generation,
  lets you build your own scenarios, and charts the benchmarks.

## Benchmark summary (M1 Ultra, macOS 26.6, GPU otherwise idle, median after warm-up)

| Preset | Fields | Autoregressive JSON (Python, MLX) | Python parallel (MLX) | **Java native, cold** | Java native, warm |
|---|---|---|---|---|---|
| FinTech fraud / AML | 28 | 2388 ms | 337 ms | **266 ms** (1.27× vs Python) | 130 ms |
| Code security triage | 28 | 2473 ms | 344 ms | **277 ms** (1.24×) | 131 ms |
| Incident triage | 28 | 2360 ms | 361 ms | **269 ms** (1.34×) | 131 ms |
| 255-choice tariff | 4 | 476 ms | 170 ms | **158 ms** (1.08×) | 67 ms |

"Cold" is the head-to-head: same prompt tokens, full prefill every call, same
timing boundary as the Python engine. "Warm" keeps the schema catalog's KV
cells between calls (single-pass mode) — a production optimization the Python
engine does not have. Details and caveats in the sections below.

## The technique

Structured extraction with bounded field values (booleans, enums) doesn't need
autoregressive token-by-token JSON generation. Instead:

1. **Single prefill** — the context + a compact one-line-per-field schema
   description is run once through the model into an MLX KV cache.
2. **KV-cache broadcast** — the cache's `keys`/`values` are `mx.repeat`ed along
   the batch axis, once per schema field (`M` fields → batch of `M`).
3. **One batched forward pass** — each batch row gets a suffix like
   `  "risk_tier": "` and the logits at the last suffix token are read.
4. **Sub-vocabulary logit slicing** — softmax is taken only over the token IDs
   that begin each allowed choice, giving a calibrated probability per choice.
5. **Programmatic assembly** — the JSON is built from the winners, so it is
   always syntactically valid and never omits or hallucinates keys.

Implementation: `python/core/engine_mlx.py::run_parallel_generation`,
`python/core/schema.py::StructuredSchema.compile_parallel_metadata`.

## Python engine (MLX port)

### Setup

Requires an Apple Silicon Mac and Python ≥ 3.10 (system Python here is 3.9,
so a dedicated venv is used).

```bash
uv venv --python 3.12 python/.venv
uv pip install --python python/.venv/bin/python -e python/
```

### Run

Benchmark parallel-constrained vs. autoregressive JSON on all presets:

```bash
python/.venv/bin/python -m core.benchmark
# or a subset:
python/.venv/bin/python -m core.benchmark --presets presets/fintech_fraud.json
```

Minimal API usage (define a schema, extract, inspect confidences):

```bash
python/.venv/bin/python python/example.py
```

```python
from core.schema import StructuredSchema
from core.engine import run_parallel_generation

schema = StructuredSchema({
    "priority": {"type": "enum", "choices": ["P0_CRITICAL", "P1_HIGH", "P2_NORMAL", "P3_LOW"],
                 "description": "Urgency tier based on customer business impact"},
    "requires_escalation": {"type": "boolean",
                            "description": "Whether an on-call engineer must be notified"},
})
result = run_parallel_generation("Production DB at 100% CPU, checkout failing for 40% of users.", schema)
result["parsed_json"]      # {"priority": {"value": ..., "prob": ...}, ...}
result["field_telemetry"]  # ranked choices + probabilities per field
```

### Results (mlx 0.32.2 / mlx-lm 0.31.3)

With the choice-listing prompt fix (see below):

| Preset | Fields | Autoregressive | Parallel | Speedup | Naive schema-valid? |
|---|---|---|---|---|---|
| FinTech fraud / AML | 28 | 2388 ms (288 tok) | 362 ms | 6.6× | no |
| Code security triage | 28 | 2473 ms (299 tok) | 365 ms | 6.8× | no |
| Incident triage | 28 | 2360 ms (283 tok) | 376 ms | 6.3× | no |
| 255-choice tariff | 4 | 476 ms (53 tok) | 181 ms | 2.6× | yes |

With the upstream prompt (no choices listed) the parallel numbers were
325 / 312 / 327 / 99 ms (8.1× / 8.2× / 7.4× / 5.7×) — the fix costs ~40 ms of
prefill on the 28-field presets and ~80 ms on the tariff preset, whose prefill
now carries a 20-choice sample of the 255 codes.

The latency speedup reproduces. The parallel output is schema-valid by
construction every time; the naive baseline dropped keys or produced values
outside the enum on three of four presets.

### Local fix: the parallel prompt now lists the allowed choices

Upstream, the parallel prefill contained only `"field": description` lines —
the model was scoring first tokens of choices it had never been shown. This
port changes `StructuredSchema.to_parallel_schema_str()` to append the
allowed values per field, e.g.

```
  "risk_tier": Calculated risk tier [LOW | MEDIUM | HIGH | CRITICAL]
  "is_fraudulent": Whether transaction is fraudulent [true | false]
```

using the same truncation rule as the naive prompt (all choices if ≤ 50,
otherwise the first 20 plus a count), so both engines see the same
information. Effect on the fintech preset, parallel engine before → after:

| Field | Upstream prompt | With choices | Naive baseline |
|---|---|---|---|
| `velocity_score` | NORMAL | EXTREME_BURST | EXTREME_BURST |
| `chargeback_probability` | LOW | VERY_HIGH | VERY_HIGH |
| `counterparty_jurisdiction_risk` | TIER_1_LOW | TIER_3_HIGH | TIER_3_HIGH |
| `sanctions_screening_risk` | HIGH_CONFIDENCE_MATCH | POTENTIAL_MATCH | POTENTIAL_MATCH |
| `loss_prevention_priority` | LOW_PRIORITY | LOW_PRIORITY (0.52) | IMMEDIATE_P0 |

Agreement with the naive baseline went from 12/28 to 16/28 fields; most
remaining differences are booleans where the parallel engine answers `true`
(block card, freeze banking, escalate to FIU) for what is plainly a fraud
case and the naive baseline answers `false`. `example.py` now returns
`P0_CRITICAL` (0.66) / `INFRASTRUCTURE` (0.92) instead of `P2_NORMAL` /
`SECURITY`.

### Caveats you should know before trusting the numbers

These are properties of the upstream code, kept as-is in this port so that
what you measure is what the author published.

- **Accuracy is not benchmarked, only latency and syntax validity.** There is
  no ground truth in the presets; the field-level comparison above is against
  the naive baseline of the same 1.5B model, not against labels. Some parallel
  answers still look wrong (`loss_prevention_priority: LOW_PRIORITY` at 0.52
  on an obvious fraud case).
- **Multi-token choices that share a first token hit a fallback path**
  (`has_collisions` in `engine_mlx.py`): an unconstrained 4-step greedy decode
  followed by fuzzy string matching. Its reported confidence is *fabricated*
  — `max(min(prod_of_token_probs, 0.9999), 0.75)` — so a prob of exactly
  `0.75` is a strong signal a field came from this path.
  `sequential_forward_passes` stays hard-coded to `1` even though extra passes
  were run. Fields that take this path per preset (verified by inspecting
  `compile_parallel_metadata(...)["has_collisions"]` with the Qwen tokenizer):
  - `fintech_fraud`: `counterparty_jurisdiction_risk`
  - `code_security`: `secondary_cwe`
  - `support_triage`: `target_resolution_hours`, `assigned_agent_tier`
  - `high_cardinality_255`: `customs_category` — **the 255-choice field
    itself**, so the headline high-cardinality row is not exercising
    constrained logit slicing at all.
- **Only the first token of each choice is scored** in the fast path; suffixes
  are right-padded and the pad tokens sit in the KV cache during the collision
  continuation.
- `presets/fintech_fraud.json` upstream has its dollar amounts eaten
  (`Transaction Amount: ,850.00 USD`); left untouched for fidelity.

The Java engine below replaces the collision fallback with proper multi-token
constrained scoring.

## Why not an HTTP client (langchain4j)?

An earlier Java attempt went through langchain4j against a local
OpenAI-compatible `mlx_lm.server`: one request per field fired concurrently
with `max_tokens=1` and `top_logprobs`, scoring the allowed choices from the
returned logprobs. It produced schema-valid output but ran at **0.6–0.8× the
speed of the naive baseline** (≈4 s for 28 fields), because `mlx_lm.server`
has a fixed cost of ~120–140 ms per request regardless of size and its request
batching barely dents it — thirty requests cost more than one 290-token
generation. A client library cannot touch the KV cache or read logits directly,
which is exactly where the technique's speed comes from, so that path was
dropped in favour of the native engine below.

## Java native engine (libllama via FFM) — faster than the Python engine

### Grammar-constrained autoregressive baseline

The web race and web benchmarks now use the same loaded GGUF model with a
llama.cpp grammar sampler followed by greedy selection. The grammar enforces
all fields in schema order, boolean types and exact enum strings, with no extra
keys. Output is compact JSON. `NaiveJsonEngine` remains available as the original
prompt-only baseline; the historical measurements below have not been rerun.

Run the new baseline alongside the parallel CLI benchmark:

```bash
cd java
mvn -q package
PCD_GGUF=/absolute/path/to/model.gguf java -jar target/pcd-benchmark.jar --grammar --runs 3
```

The grammar supports this project's flat boolean/enum presets, not arbitrary
JSON Schema. It follows llama.cpp's [GBNF grammar format](https://github.com/ggml-org/llama.cpp/blob/master/grammars/README.md).
Elapsed time includes prompt tokenization, grammar construction, prefill, sampling,
and streaming callbacks; model loading and final validation are excluded.
The 700-token limit can still truncate a result (`completed=false`), and grammar
constraints do not ensure factual correctness. API responses identify the new
baseline as `mode: "grammar_constrained_autoregressive"` and report actual decode
calls in `forwardPasses`, including prefill and the final end-token evaluation.

For native integration tests, set `PCD_TEST_GGUF` to the model path when running
`mvn test`. Without that variable, native integration tests are skipped.

The Java engine (`java/src/main/java/pcd/nativeengine/`) drives **llama.cpp
directly** through the JDK Foreign Function & Memory API (JDK 22+, no JNI) and
implements the technique step for step:

| Python engine (MLX) | Java native (`pcd.nativeengine`) |
|---|---|
| prefill prompt into KV cache | same: one `llama_decode` of the ChatML prompt into sequence 0 (Q8_0 GGUF, flash attention, Metal) |
| `mx.repeat` the cache M times | `llama_memory_seq_cp` with `kv_unified=true` — the prefix cells are *tagged* with each field's sequence, zero copy (≈1 ms for 28 fields) |
| one batched forward pass over padded suffixes | one `llama_decode` over all (unpadded) suffixes, logits only at each field's last token |
| softmax over candidate first tokens, per-candidate `float(mx.array)` reads | softmax over candidate first tokens, read straight from the logits buffer |
| collisions: unconstrained 4-token greedy decode per field, fabricated ≥0.75 confidence | collisions: **batched token-tree levels** — one decode per level for all still-ambiguous fields, sub-vocabulary softmax at every level, confidence = product of per-level probabilities |
| programmatic JSON assembly | same |

Optional `--single-pass` mode tags the prefix tokens with every sequence inside
the *same* batch as the suffixes, so prefill + broadcast + suffix evaluation is
a single forward pass. Optional warm mode keeps the schema catalog's KV cells
between calls and prefills only the context (a production optimization the
Python engine does not have; reported separately, not as the head-to-head).

### Web app: race, scenarios, benchmarks

```bash
cd java && mvn -q package
java -jar target/pcd-benchmark.jar serve          # then open http://localhost:8000
```

Three views, one page, no build step (JDK `HttpServer` + vanilla JS, Chart.js
from a CDN):

- **Race** — opens on the Devoxx CFP routing scenario; pick another (or paste
  your own text, e.g. an email) and press *Run the race*. Both engines run on the same input and are replayed on one
  time ruler: the parallel result lands as a block of fields with a confidence
  bar each; the token-by-token baseline streams its JSON as it is generated.
  The verdict line between the ruler and the panes gives the speedup and the
  number of forward passes each needed; the baseline's JSON is pretty-printed
  as it streams, and its schema violations (if any) are listed under it. For
  scenarios with a sample bank, both panes say per field whether they agree
  with the human filing (✓ same as the CFP / ✗ CFP filed it as …).
- **Presets** — create and edit scenarios in the browser: title, example input,
  and a table of fields (`true / false` or a list of allowed values, each with
  a one-line description). Saved to `presets/*.json`, the same files the CLI
  benchmark reads. *Save and run the race* jumps straight to the demo.
- **Benchmarks** — runs every scenario through both engines, keeps the runs in
  `results/`, and shows a latency table, a token-by-token vs parallel chart
  and the parallel engine's per-phase breakdown.

`?preset=<id>` and `?autorun` in the URL pre-select a scenario and start the
race on load, which is handy when presenting. `--port` changes the port.

**Model selection.** The header dropdown lists every `*.gguf` in `models/`
**and every model of a local Ollama install**, and switches the loaded model
in place (~0.5 s for a 3 GB file, ~2 s for 8B; runs in flight finish first).
Benchmark runs record which model produced them. Both
`qwen2.5-1.5b-instruct-q8_0` and `qwen2.5-3b-instruct-q8_0` are worth having:
the 3B is roughly 2× slower per pass and noticeably better at judgement calls
such as audience level.

**Ollama models.** Ollama's store (`~/.ollama/models`, or `$OLLAMA_MODELS`)
is a directory of GGUF blobs addressed by manifests, so the engine can load
them directly — nothing talks to Ollama itself and it doesn't need to be
running. They appear as `ollama:<name>:<tag>` (source group "Ollama"). Each
model's prompts are rendered with the chat template embedded in its GGUF
(`llama_chat_apply_template`), so Llama 3, Qwen, Gemma … each get their own
format; files without a template fall back to ChatML. Embedding models and
vision projectors are filtered out by reading `general.architecture` from the
GGUF header. Measured: `ollama:llama3.1:latest` (8B) runs the spam scenario
in 747 ms / 2 passes vs 2.2 s / 53 passes for the grammar baseline.
Caveat: some Ollama blobs are Ollama-specific conversions that upstream
llama.cpp rejects (`gemma4:*`, `qwen3-vl:*` here: "wrong number of tensors");
the dropdown reports the error and keeps the previous model loaded.

**Same prompt for both engines.** In the web app the parallel engine is
prefilled with exactly the prompt the grammar-constrained baseline gets
(schema with descriptions and allowed values, plus the same instruction),
so the only difference between the two panes is the decoding. The CLI
benchmark keeps the Python port's one-line catalog prompt
(`CompiledSchema.PromptStyle.CATALOG`) so its token counts stay comparable
with the Python numbers above.

**Why the two engines can disagree.** Scored against the CFP's own filing on
the 100 cached Devoxx talks (`python3 java/tools/eval/devoxx_eval.py`, server
running), with the 1.5B model and the shared prompt: track 44 vs 47,
audience level 22 vs 27 (parallel vs baseline). The
investigation behind those numbers:

- The disagreement is *not* the decoding. Scoring choices by their first token
  or by the summed probability of every token that starts them gives
  identical decisions on all 100 talks.
- It was the prompt: the two engines originally saw different prompts, and a
  1.5B model's preferences move a lot with wording. Hence the shared prompt.
- Only `track` is really inferable from an abstract (8 classes, majority
  28%): both engines land around 45–50% on the 1.5B and ~53% on the 3B.
  `audience_level` (~50% `INTERMEDIATE`) is the speaker's form choice, so
  "accuracy" there mostly measures majority-class bias. The session format
  was removed from the scenario for the same reason: any talk can be
  submitted in any format.

Two scenarios ship next to the four upstream presets:

- **Spam & phishing triage** — 8 fields (safe to deliver, category,
  impersonated brand, sender authenticated, asks for credentials, urgency
  pressure, risk level, recommended action). On the sample phishing email all
  8 are decided in 2 forward passes (~130–200 ms) where the baseline needs 73
  (~530 ms).
- **Devoxx CFP talk routing** — routes a submitted talk to one of the eight
  real Devoxx Belgium 2026 tracks (pulled from the public CFP API at
  `dvbe26.cfp.dev`), plus session format, audience level, main technology and
  three booleans (AI central, live coding, laptop needed). The preset carries a
  bank of the **100 most-favourited real talks**, each with the track, format
  and level the CFP actually filed it under. The Race view picks a random talk
  when the scenario is selected, moves to the next one on every run (or via
  *Next talk*), shows the CFP's filing under each routed value — green where
  the engine agrees, amber where it doesn't — and the verdict line counts the
  agreement for both engines. Typical: 7 fields in 2 passes, ~140–170 ms, vs
  ~70 tokens / ~530 ms. Refresh the bank from the live CFP with
  `java -jar target/pcd-benchmark.jar devoxx-samples [--event dvbe26] [--count 100]`.

Any preset can carry such a `samples` bank (`[{label, context, expected}]`);
the editor leaves it untouched when you save.

The token-by-token baseline in the app uses `GrammarJsonEngine` on the same
libllama runtime and identical weights. A grammar enforces the schema while
the model generates compact JSON one token at a time. Historical measurements
above used the original prompt-only baseline and have not been rerun.

### CLI benchmark

Requires JDK 22+ (25 used here), Maven, and llama.cpp's shared library.

```bash
brew install llama.cpp          # provides libllama.dylib + libggml (ggml is a separate formula; 0.4.1 / ggml 0.24.0 used here)
curl -L -o models/qwen2.5-1.5b-instruct-q8_0.gguf \
  https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct-GGUF/resolve/main/qwen2.5-1.5b-instruct-q8_0.gguf
cd java && mvn -q package        # compiles + runs the unit tests
java -jar target/pcd-benchmark.jar bench                    # all presets, cold + warm rows
java -jar target/pcd-benchmark.jar bench -v --single-pass ../presets/fintech_fraud.json
```

Flags: `-v` per-field values/probabilities, `--runs N` (median of N, default 5),
`--single-pass`, `--pad` (Python-style right-padded suffixes). Environment:
`PCD_GGUF` (model path), `PCD_LLAMA_LIB_DIR` (colon-separated; default `/opt/homebrew/opt/ggml/lib:/opt/homebrew/opt/llama.cpp/lib`).
The FFM bindings in `java/src/main/java/llama/` are generated by
`java/tools/gen-bindings.sh` (jextract 22, downloaded into `java/tools/`).
**Re-run it after every `brew upgrade llama.cpp`**: the bindings bake in the
struct layouts of `llama_context_params` / `llama_model_params`, and those
change between releases (0.4.1 added seven fields to the context params).

### Results (M1 Ultra, GPU otherwise idle, median after warm-up)

Timing boundary is the same on both sides: tokenize + prefill + passes +
assembly; model load and per-schema token compilation excluded. Prompt token
counts are identical (901 / 983 / 959 / 567).

| Preset | Fields | Python parallel (MLX 4-bit) | Java native cold | Speedup | Java warm | Java warm, single-pass |
|---|---|---|---|---|---|---|
| FinTech fraud / AML | 28 | 337 ms | **266 ms** | 1.27× | 143 ms | 130 ms |
| Code security triage | 28 | 344 ms | **277 ms** | 1.24× | 140 ms | 131 ms |
| Incident triage | 28 | 361 ms | **269 ms** | 1.34× | 140 ms | 131 ms |
| 255-choice tariff | 4 | 170 ms | **158 ms** | 1.08× | 86 ms | 67 ms |

Per-phase breakdown of a cold fintech run (28 fields, 901 prompt tokens):
tokenize 1 ms · prefill 182 ms · broadcast 1 ms · suffix pass 61 ms ·
token-tree 20 ms (3 levels).

Where the cold win comes from (fintech): prefill is a tie (182 ms vs 174 —
Q8_0 llama.cpp prefill measured at ~5,000 tok/s, MLX ≈5,100); the suffix pass
is 61 ms vs 95; and collision handling plus host overhead is 21 ms vs ~70.

Measurement notes:
- An idle `mlx_lm.server` or an Ollama runner holding a large model on the GPU
  roughly doubles *both* engines' times; the numbers above were taken with
  nothing else on the GPU.
- The numbers are unchanged by the JDK and by llama.cpp `b8330` → `0.4.1`
  (prefill 5,035 → 5,010 tok/s). Essentially all of the time is Metal GPU work
  inside libllama; the JVM executes well under 1 ms per run.

#### JDK 25 vs JDK 27 (same jar, llama.cpp 0.4.1 / ggml 0.24.0, median of 7)

| Preset | JDK 25.0.2 cold | JDK 27 cold | JDK 25.0.2 warm, single-pass | JDK 27 warm, single-pass |
|---|---|---|---|---|
| FinTech fraud / AML | 266.3 ms | 266.0 ms | 129.2 ms | 129.2 ms |
| Code security triage | 277.9 ms | 276.6 ms | 130.5 ms | 129.8 ms |
| Incident triage | 270.6 ms | 269.3 ms | 131.8 ms | 131.0 ms |
| 255-choice tariff | 158.6 ms | 157.2 ms | 66.9 ms | 67.2 ms |

Every difference is within run-to-run noise (±2 ms).

### Caveats

- **Different quantization.** Python runs `mlx-community/Qwen2.5-1.5B-Instruct-4bit`;
  the Java engine runs the Q8_0 GGUF (same speed as Q4_0 here because prefill
  is compute-bound, and closer to full precision). Field-level agreement with
  the Python engine: 20/28, 24/28, 25/28, 2/4 per preset — disagreements are
  low-margin fields, e.g. `is_fraudulent` flips to `false` at 0.71 with the
  Q4_0 GGUF and is `true` at 0.57 with Q8_0.
- Prompt tokenization is llama.cpp's (matches the HF token counts exactly on
  all presets); suffixes and choice remainders are tokenized separately, as
  in Python, so candidate first tokens can differ from in-context tokenization.
- Multi-level fields (the collision cases) report a *product* of per-level
  calibrated probabilities, which is naturally lower than a single-token
  probability (e.g. `counterparty_jurisdiction_risk` → `TIER_3_HIGH` at 0.46
  after 4 levels).

## Hosting it (Clever Cloud / any Docker host)

`Dockerfile` builds a self-contained CPU image (llama.cpp compiled at the
pinned tag, JRE 25, the Qwen 2.5 1.5B and Qwen 3.5 0.8B models baked in) that
serves the web app on `0.0.0.0:8080` in read-only demo mode. See
[deploy/clever-cloud.md](deploy/clever-cloud.md) for the Clever Cloud steps,
the environment knobs (`PORT`, `PCD_BIND`, `PCD_READ_ONLY`, `PCD_GGUF`) and
what to expect from CPU inference.

## Layout

```
presets/            scenario JSONs (context + schema): four upstream ones + spam_email + devoxx_cfp
results/            benchmark runs saved by the web app (gitignored)
python/
  pyproject.toml    mlx, mlx-lm, numpy
  example.py        quickstart
  core/
    engine.py         entry point (MLX only)
    engine_mlx.py     parallel constrained engine + autoregressive baseline
    schema.py         FieldDefinition / StructuredSchema, candidate-token compilation
    prompt_builder.py prompts for the naive baseline
    benchmark.py      python -m core.benchmark
java/
  pom.xml            Java 22+, Jackson, JUnit; shaded jar target/pcd-benchmark.jar
  tools/gen-bindings.sh  regenerates the FFM bindings with jextract
  tools/eval/devoxx_eval.py  scores both engines against the CFP filing on the cached talks
  src/main/java/pcd/
    Main.java              entry point: `serve` (web app), `bench` (CLI), `devoxx-samples` (refresh talk bank)
    DevoxxSamples.java     fetches tracks + top talks from the Devoxx CFP API into presets/devoxx_cfp.json
    NativeBenchmark.java   CLI benchmark
    Preset.java            preset model, validation, JSON (de)serialization
    ModelCatalog.java      models/*.gguf + Ollama store → loadable models (GGUF header reader)
    Prompts.java           same prompt text as python/core (catalog + naive baseline)
  src/main/java/pcd/nativeengine/
    LlamaRuntime.java          FFM wrapper: model/context, tokenize, batched decode, logits, seq ops
    CompiledSchema.java        per-schema suffix + choice token metadata (compiled once)
    NativeParallelEngine.java  prefill → broadcast → batched suffix pass → token-tree levels
    NaiveJsonEngine.java       token-by-token baseline (greedy decode, streamed)
  src/main/java/pcd/web/
    Server.java                JDK HttpServer: REST + SSE endpoints, static files
    EngineService.java         single locked runtime shared by all requests
    Stores.java                presets/ and results/ on disk
  src/main/resources/web/      index.html, app.js, style.css
  src/main/java/llama/         generated libllama bindings (jextract)
models/                        GGUF weights (gitignored)
```

## License

Apache 2.0 — see [LICENSE](LICENSE) and [NOTICE](NOTICE).
