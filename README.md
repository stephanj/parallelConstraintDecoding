# Parallel Constrained Decoding

Fill a whole JSON schema of booleans and enums in **two forward passes**
instead of generating it token by token — with calibrated confidence per
field and output that is schema-valid by construction.

[![Demo: Parallel Constrained Decoding — the web app racing the parallel engine against token-by-token generation](https://img.youtube.com/vi/C4Vjbf_HvYY/maxresdefault.jpg)](https://www.youtube.com/watch?v=C4Vjbf_HvYY)

*Watch the demo (YouTube, Devoxx): the web app races the parallel engine
against grammar-constrained token-by-token generation on the same model.*

This repository contains:

- **`java/`** — a Java implementation of the technique that drives
  [llama.cpp](https://github.com/ggml-org/llama.cpp) directly through the JDK
  Foreign Function & Memory API (no JNI). It is the fastest engine measured
  here, and it ships with a local **web app** that races it against a
  grammar-constrained token-by-token baseline on the same model, lets you
  build your own extraction scenarios, and charts benchmarks. It can load any
  GGUF, including the models of a local Ollama install.
- **`python/`** — a port of the original engine published at
  [harshatheg/Qwen-2.5-1B-RLCD](https://huggingface.co/harshatheg/Qwen-2.5-1B-RLCD)
  (MLX, Apple Silicon only), kept for reference and comparison. Despite the
  name, that Hugging Face repo contains no weights and no fine-tune — it is
  source code running stock `Qwen2.5-1.5B-Instruct` with a custom decoding
  strategy. See [NOTICE](NOTICE) for what was vendored and changed.
- **`presets/`** — extraction scenarios shared by both engines (fraud
  triage, code-security audit, support triage, a 255-choice tariff router,
  spam/phishing triage, and Devoxx CFP talk routing with a bank of 100 real
  talks).

## Benchmark summary (Apple M1 Ultra, GPU otherwise idle, median after warm-up)

| Preset | Fields | Token-by-token JSON (Python, MLX) | Python parallel (MLX) | **Java native, cold** | Java native, warm |
|---|---|---|---|---|---|
| FinTech fraud / AML | 28 | 2388 ms | 337 ms | **266 ms** (1.27× vs Python) | 130 ms |
| Code security triage | 28 | 2473 ms | 344 ms | **277 ms** (1.24×) | 131 ms |
| Incident triage | 28 | 2360 ms | 361 ms | **269 ms** (1.34×) | 131 ms |
| 255-choice tariff | 4 | 476 ms | 170 ms | **158 ms** (1.08×) | 67 ms |

"Cold" is the head-to-head: same prompt tokens, full prefill every call, same
timing boundary as the Python engine. "Warm" keeps the schema's KV cells
between calls (single-pass mode) — a production optimization the Python
engine does not have. Full details, per-phase breakdowns and caveats are in
[Java engine details](#java-engine-details) below.

---

## Install on a local machine

### What you need

| | Required | Notes |
|---|---|---|
| OS / hardware | macOS on Apple Silicon (recommended) or Linux x86-64 / arm64 | On a Mac the engine runs on the GPU through Metal (sub-second races). On Linux it runs on the CPU: expect seconds per race, same 2-passes-vs-50 story. |
| JDK | 22 or newer (25 and 27 tested, identical speed) | e.g. `brew install openjdk` |
| Maven | 3.9+ | `brew install maven` |
| llama.cpp shared library | `v0.4.1` (the tag the Java bindings were generated against) | `brew install llama.cpp` on macOS; build from source on Linux (below) |
| A GGUF model | any decoder LLM | Qwen2.5-1.5B-Instruct Q8_0 is the default (1.9 GB) |
| Optional | Ollama | its already-downloaded models appear in the model dropdown |
| Optional | Python ≥ 3.10 + `uv` | only for the Python reference port (Apple Silicon only) |

### Steps (macOS)

```bash
git clone https://github.com/stephanj/parallelConstraintDecoding.git
cd parallelConstraintDecoding

# 1. llama.cpp (libllama + libggml; Homebrew installs ggml as a separate formula)
brew install llama.cpp

# 2. a model (the default one; any GGUF works, see "Models" below)
mkdir -p models
curl -L -o models/qwen2.5-1.5b-instruct-q8_0.gguf \
  https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct-GGUF/resolve/main/qwen2.5-1.5b-instruct-q8_0.gguf

# 3. build (compiles and runs the unit tests, produces java/target/pcd-benchmark.jar)
cd java && mvn -q package && cd ..

# 4. run the web app, then open http://localhost:8000
java -jar java/target/pcd-benchmark.jar serve
```

The first race on a scenario includes a one-off warm-up (Metal compiles
kernels for that schema's batch shapes); numbers from the second run on are
representative.

### Steps (Linux)

Same as above, except llama.cpp is built from source at the pinned tag and
the engine is told where the libraries are:

```bash
git clone --depth 1 --branch v0.4.1 https://github.com/ggml-org/llama.cpp.git /tmp/llama.cpp
cmake -S /tmp/llama.cpp -B /tmp/llama.cpp/build -DCMAKE_BUILD_TYPE=Release -DBUILD_SHARED_LIBS=ON \
      -DGGML_NATIVE=ON -DLLAMA_CURL=OFF -DLLAMA_BUILD_TESTS=OFF -DLLAMA_BUILD_EXAMPLES=OFF \
      -DLLAMA_BUILD_TOOLS=OFF -DLLAMA_BUILD_SERVER=OFF
cmake --build /tmp/llama.cpp/build --target llama ggml-cpu -j
mkdir -p ~/llama-lib && find /tmp/llama.cpp/build -name 'lib*.so*' -exec cp -a {} ~/llama-lib/ \;

export PCD_LLAMA_LIB_DIR=~/llama-lib LD_LIBRARY_PATH=~/llama-lib
java -jar java/target/pcd-benchmark.jar serve
```

The CPU backend is built with OpenMP; install `libgomp1` (Debian/Ubuntu) if
the JVM reports `libgomp.so.1: cannot open shared object file`. The engine
uses all available cores.

### Models

- Put any `*.gguf` in `models/`; the web app's header dropdown lists them
  all and switches in place. `PCD_GGUF=/path/to/model.gguf` picks the one
  loaded at start (default `models/qwen2.5-1.5b-instruct-q8_0.gguf`).
- Models of a local **Ollama** install are listed too, as `ollama:<name>:<tag>`
  (Ollama's store is a directory of GGUF blobs; nothing talks to Ollama and it
  needn't be running; `$OLLAMA_MODELS` is honoured). Some Ollama blobs are
  Ollama-specific conversions upstream llama.cpp cannot load (`gemma4:*`,
  `qwen3-vl:*` at the time of writing) — the dropdown reports the error and
  keeps the previous model.
- Every prompt is rendered with the chat template embedded in the GGUF, so
  Qwen, Llama 3, Gemma … each get their own format (ChatML fallback when a
  file has none).
- Useful companions to the default: `Qwen2.5-3B-Instruct` Q8_0 (2× slower per
  pass, better judgement) and `Qwen3.5-0.8B` Q8_0 from `ggml-org` (smaller,
  but its hybrid attention is *slower* in llama.cpp than the 1.5B).

### Environment and options

| Variable / flag | Default | Meaning |
|---|---|---|
| `PCD_GGUF` | `models/qwen2.5-1.5b-instruct-q8_0.gguf` | model loaded at start |
| `PCD_LLAMA_LIB_DIR` | `/opt/homebrew/opt/ggml/lib:/opt/homebrew/opt/llama.cpp/lib:/opt/llama/lib:/usr/local/lib` | where `libllama` / `libggml*` live (colon-separated) |
| `PORT` / `--port` | `8000` | web app port |
| `PCD_BIND` / `--bind` | `127.0.0.1` | bind address (`0.0.0.0` to expose on a network) |
| `PCD_READ_ONLY` / `--read-only` | `false` | demo mode for a shared instance: scenarios, model and stored benchmark runs cannot be changed; races still run |
| `PCD_SEQUENTIAL_RACE` | `false` | show the two engines one after the other, each timed live, instead of replayed on one timeline (the honest display on a shared CPU) |

### Troubleshooting

- **`ZipFile invalid LOC header`** in the browser: the jar was rebuilt while
  the server was running (the JVM maps the jar). Stop the server, build,
  start again.
- **Wrong or crashing behaviour right after `brew upgrade llama.cpp`**: the
  FFM bindings bake in llama.cpp's struct layouts. Re-run
  `java/tools/gen-bindings.sh` (downloads jextract 22 into `java/tools/` on
  first use) and rebuild.
- **Slow numbers on a Mac**: anything else holding the GPU (an idle
  `mlx_lm.server`, an Ollama runner with a large model) roughly doubles both
  engines' times.
- Native integration tests run only when `PCD_TEST_GGUF` points at a model
  (`PCD_TEST_GGUF=… mvn test`); otherwise they are skipped.

---

## The web app

```bash
java -jar java/target/pcd-benchmark.jar serve          # http://localhost:8000
```

One page, three views, no build step (JDK `HttpServer` + vanilla JS; Chart.js
is vendored so it works offline).

- **Race** — opens on the Devoxx CFP routing scenario with a random talk from
  its bank; pick another scenario, or paste your own text (an email, a ticket,
  an abstract), and press *Run the race*. Both engines run on the same input
  and the same prompt. The parallel result lands as a block of fields with a
  confidence bar each (hover for the runner-up probabilities); the baseline
  streams its JSON, pretty-printed, as it is generated. The verdict line gives
  the speedup and the forward passes each needed. For scenarios with a sample
  bank, both panes state per field whether they agree with the human filing
  ("✓ same as the CFP" / "✗ CFP filed it as …"). By default the two are
  replayed on one time ruler; `PCD_SEQUENTIAL_RACE=true` runs them visibly one
  after the other, each timed live.
- **Presets** — create and edit scenarios in the browser: title, example
  input, and a table of fields (`true / false` or a list of allowed values,
  each with a one-line description). Saved to `presets/*.json`, the same files
  the CLI benchmark reads. *Save and run the race* jumps straight to the demo.
  Any preset may carry a `samples` bank (`[{label, context, expected}]`); the
  editor leaves it untouched.
- **Benchmarks** — runs every scenario through both engines, stores runs in
  `results/`, shows a latency table, a token-by-token vs parallel chart and
  the parallel engine's per-phase breakdown. Runs record which model made them.

`?preset=<id>` and `?autorun` in the URL pre-select a scenario and start a
race on load — handy when presenting. For a live talk on a Mac, expose the
local server to the audience with a tunnel (e.g. `cloudflared tunnel --url
http://localhost:8000`) and start it with `PCD_READ_ONLY=true`.

### The opponent: a grammar-constrained baseline

The token-by-token pane uses `GrammarJsonEngine`: the same loaded model
generates compact JSON one token at a time under a llama.cpp GBNF grammar that
enforces every field in schema order, boolean types and exact enum strings,
with no extra keys. So both panes are schema-valid by construction and the
race isolates *decoding strategy*: one forward pass per token versus two
passes for the whole schema. (`NaiveJsonEngine`, the original prompt-only
baseline that can drop keys or invent values, is still in the code base; the
Python-comparison numbers above were measured against it.)

### Scenarios that ship

- **Spam & phishing triage** — 8 fields (safe to deliver, category,
  impersonated brand, sender authenticated, asks for credentials, urgency
  pressure, risk level, recommended action). On the sample phish all 8 are
  decided in 2 forward passes (~130–200 ms) where the baseline needs ~54
  (~530 ms).
- **Devoxx CFP talk routing** — routes a submitted talk to one of the eight
  real Devoxx Belgium 2026 tracks (from the public CFP API at
  `dvbe26.cfp.dev`), plus audience level, main technology and three booleans
  (AI central, live coding, laptop needed). The preset carries the **100
  most-favourited real talks** with the track and level the CFP filed them
  under; each run moves to the next talk (or use *Next talk*). Refresh the
  bank with `java -jar target/pcd-benchmark.jar devoxx-samples [--event dvbe26] [--count 100]`.
  Session format is deliberately not a field: any talk can be submitted in
  any format, so it isn't inferable from an abstract.
- The four presets of the upstream project (fraud/AML, code security, support
  triage, 255-choice tariff), unchanged.

### On accuracy, and why the two engines can disagree

Scored against the CFP's own filing on the 100 cached Devoxx talks
(`python3 java/tools/eval/devoxx_eval.py` with the server running), 1.5B
model, shared prompt: track 44 vs 47, audience level 22 vs 27 (parallel vs
baseline). What the investigation behind those numbers found:

- The disagreement is *not* the decoding: scoring choices by their first
  token or by the summed probability of every token that starts them gives
  identical decisions on all 100 talks.
- It was the prompt. The two engines originally saw different prompts, and a
  1.5B model's preferences move a lot with wording. Hence they now share one
  (the CLI benchmark keeps the Python port's catalog prompt for comparable
  token counts — `CompiledSchema.PromptStyle`).
- Only `track` is really inferable from an abstract (8 classes, majority
  28%): both engines land around 45–50% on the 1.5B, ~53% on the 3B.
  `audience_level` is the speaker's form choice, so "accuracy" there mostly
  measures majority-class bias. Low confidence on a field is the useful
  signal: it is where a human should look.

---

## The technique

Structured extraction with bounded field values (booleans, enums) doesn't
need autoregressive token-by-token JSON generation:

1. **Prefill once.** The schema (field descriptions and allowed values) and the
   text go through the model a single time and stay in the KV cache.
2. **Broadcast.** The cached prefix is made visible to one sequence per field
   — in llama.cpp with `kv_unified=true` this is a zero-copy tag on the cells.
3. **One batched pass.** Every sequence gets its own `"field": "` suffix; the
   logits at each suffix's last token are read from the same forward pass.
4. **Score only the allowed values.** Softmax over the candidate tokens gives
   a calibrated probability per choice; nothing outside the schema can win.
5. **Resolve shared prefixes level by level.** Choices that start with the
   same token (`TIER_1_LOW` / `TIER_3_HIGH`) are separated with one more
   batched pass per level for the still-ambiguous fields only; confidence is
   the product of the per-level probabilities.
6. **Assemble.** The JSON is built from the winners: always valid, never a
   missing or invented key.

---

## Java engine details

`java/src/main/java/pcd/nativeengine/` implements the steps above on
libllama through FFM (JDK 22+):

| Python engine (MLX) | Java native (`pcd.nativeengine`) |
|---|---|
| prefill prompt into KV cache | one `llama_decode` of the templated prompt into sequence 0 (flash attention, Metal on macOS) |
| `mx.repeat` the cache M times | `llama_memory_seq_cp` under `kv_unified=true`: cells tagged with each field's sequence, zero copy (≈1 ms for 28 fields) |
| one batched pass over padded suffixes | one `llama_decode` over all (unpadded) suffixes, logits only at each field's last token |
| softmax over candidate first tokens, per-candidate `float(mx.array)` reads | softmax over candidate first tokens, read straight from the logits buffer |
| collisions: unconstrained 4-token greedy decode per field, fabricated ≥0.75 confidence | **batched token-tree levels**: one decode per level for all still-ambiguous fields, sub-vocabulary softmax at every level |
| programmatic JSON assembly | same |

Optional `--single-pass` mode tags the prefix tokens with every sequence
inside the *same* batch as the suffixes (prefill + broadcast + suffix pass in
one forward pass). Optional warm mode keeps the schema's KV cells between calls
and prefills only the context.

### CLI benchmark

```bash
cd java && mvn -q package
java -jar target/pcd-benchmark.jar bench                    # all presets, cold + warm rows
java -jar target/pcd-benchmark.jar bench -v --single-pass ../presets/fintech_fraud.json
java -jar target/pcd-benchmark.jar bench --grammar --runs 3 # also time the grammar baseline
```

Flags: `-v` per-field values/probabilities, `--runs N` (median of N, default
5), `--single-pass`, `--pad` (Python-style right-padded suffixes),
`--grammar`. `PCD_GGUF` selects the model. The FFM bindings in
`java/src/main/java/llama/` are generated by `java/tools/gen-bindings.sh`
(jextract 22); re-run it after upgrading llama.cpp.

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
token-tree 20 ms (3 levels). Where the cold win comes from: prefill is a tie
(182 ms vs 174 — llama.cpp Q8_0 prefill ≈5,000 tok/s, MLX ≈5,100); the suffix
pass is 61 ms vs 95; collision handling plus host overhead is 21 ms vs ~70.

The numbers are unchanged by the JDK (25.0.2 vs 27, every difference within
±2 ms) and by llama.cpp `b8330` → `0.4.1`: essentially all the time is GPU
work inside libllama, the JVM executes well under 1 ms per run.

| Preset | JDK 25.0.2 cold | JDK 27 cold | JDK 25.0.2 warm, single-pass | JDK 27 warm, single-pass |
|---|---|---|---|---|
| FinTech fraud / AML | 266.3 ms | 266.0 ms | 129.2 ms | 129.2 ms |
| Code security triage | 277.9 ms | 276.6 ms | 130.5 ms | 129.8 ms |
| Incident triage | 270.6 ms | 269.3 ms | 131.8 ms | 131.0 ms |
| 255-choice tariff | 158.6 ms | 157.2 ms | 66.9 ms | 67.2 ms |

Other models measured with the web app (spam scenario, parallel vs grammar
baseline): `ollama:llama3.1:latest` (8B) 747 ms / 2 passes vs 2.2 s / 53;
`Qwen3.5-0.8B` Q8 151 ms / 4 passes vs 1.39 s / 54. On a 20-vCPU Linux box
without a GPU the 1.5B runs the same scenario in ~1.1 s / 2 passes vs ~8.5 s
/ 54.

### Caveats

- **Different quantization** from the Python port: Python runs the MLX 4-bit
  model, the Java engine a Q8_0 GGUF (same speed here because prefill is
  compute-bound, closer to full precision). Field-level agreement with the
  Python engine on the upstream presets: 20/28, 24/28, 25/28, 2/4 —
  disagreements are low-margin fields.
- Prompt tokenization is llama.cpp's (matches the HF token counts on all
  presets); suffixes and choice remainders are tokenized separately, as in
  Python, so candidate first tokens can differ from in-context tokenization.
- Multi-level fields report a *product* of per-level probabilities, which is
  naturally lower than a single-token probability.

### Why not an HTTP client (langchain4j)?

An earlier Java attempt went through langchain4j against a local
OpenAI-compatible `mlx_lm.server`: one request per field fired concurrently
with `max_tokens=1` and `top_logprobs`. It produced schema-valid output but ran
at 0.6–0.8× the speed of the naive baseline (≈4 s for 28 fields): the server
has a fixed ~120–140 ms cost per request and thirty requests cost more than
one 290-token generation. A client library cannot touch the KV cache or read
logits directly, which is exactly where the technique's speed comes from.

---

## Python reference port (MLX, Apple Silicon only)

Kept so the Java numbers can be compared with the original engine.

```bash
uv venv --python 3.12 python/.venv
uv pip install --python python/.venv/bin/python -e python/
python/.venv/bin/python -m core.benchmark                 # all presets
python/.venv/bin/python python/example.py                 # quickstart
```

The first run downloads `mlx-community/Qwen2.5-1.5B-Instruct-4bit` (~1 GB).

Measured (mlx 0.32.2 / mlx-lm 0.31.3), parallel vs the prompt-only
autoregressive baseline: 362 vs 2388 ms (6.6×), 365 vs 2473 (6.8×), 376 vs
2360 (6.3×), 181 vs 476 (2.6×) on the four upstream presets; the baseline
dropped keys or produced out-of-enum values on three of four.

Local changes to the upstream code (see [NOTICE](NOTICE)): the PyTorch/Spaces
backend router was removed, and the parallel prompt now lists each field's
allowed choices — upstream scored first tokens of values the model had never
been shown, and listing them moved several fields from wrong to right
(agreement with the baseline on the fintech preset 12/28 → 16/28).

Caveats inherited from upstream and left as-is for fidelity: multi-token
choices that share a first token go through an unconstrained 4-token greedy
fallback whose confidence is *fabricated* (`max(min(p, 0.9999), 0.75)` — a
prob of exactly 0.75 is the tell) while `sequential_forward_passes` stays 1;
this affects `counterparty_jurisdiction_risk`, `secondary_cwe`,
`target_resolution_hours`, `assigned_agent_tier` and the 255-choice
`customs_category` itself. Only the first token of each choice is scored, and
`presets/fintech_fraud.json` has its dollar amounts missing upstream. The Java
engine replaces the fallback with the batched token tree.

---

## Layout

```
presets/            scenario JSONs (context + schema [+ samples]); shared by both engines
results/            benchmark runs saved by the web app (gitignored)
models/             GGUF weights (gitignored)
deploy/api-design.md  notes for a possible public API (proposal, not implemented)
python/             MLX reference port (core/, example.py, pyproject.toml)
java/
  pom.xml                     Java 22+, Jackson, JUnit; shaded jar target/pcd-benchmark.jar
  tools/gen-bindings.sh       regenerates the FFM bindings with jextract
  tools/eval/devoxx_eval.py   scores both engines against the CFP filing on the cached talks
  src/main/java/pcd/
    Main.java                 entry point: serve | bench | devoxx-samples
    NativeBenchmark.java      CLI benchmark
    Preset.java               scenario model, validation, JSON (de)serialization, sample banks
    ModelCatalog.java         models/*.gguf + Ollama store → loadable models (GGUF header reader)
    Prompts.java              schema catalog and baseline prompt text
    DevoxxSamples.java        fetches tracks + top talks from the Devoxx CFP API
  src/main/java/pcd/nativeengine/
    LlamaRuntime.java         FFM wrapper: model/context, chat template, tokenize, batched decode, logits, KV ops
    CompiledSchema.java       per-schema suffix + choice token metadata (compiled once)
    NativeParallelEngine.java prefill → broadcast → batched suffix pass → token-tree levels
    GrammarJsonEngine.java    grammar-constrained token-by-token baseline (JsonGrammar.java builds the GBNF)
    NaiveJsonEngine.java      prompt-only token-by-token baseline
  src/main/java/pcd/web/
    Server.java               JDK HttpServer: REST + SSE endpoints, static files, read-only guard
    EngineService.java        one locked runtime shared by all requests; model switching
    Stores.java               presets/ and results/ on disk
  src/main/resources/web/     index.html, app.js, style.css, vendor/chart.umd.min.js
  src/main/java/llama/        generated libllama bindings (jextract)
```

## License

Apache 2.0 — see [LICENSE](LICENSE) and [NOTICE](NOTICE).
