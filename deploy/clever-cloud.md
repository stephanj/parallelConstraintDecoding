# Hosting the demo on Clever Cloud

The web app runs as a **Docker** application on Clever Cloud. The image compiles
llama.cpp (CPU backend) from the tag the Java bindings were generated against
and bakes the Qwen2.5-1.5B Q8 model into the image, so a deployed instance has
no external dependencies. Inference is CPU-only there. Measured in the image on a
20-vCPU arm64 Docker VM: the parallel engine decides the 8-field spam scenario
in **~1.1 s / 2 passes**, the grammar baseline needs **~8.5 s / 54 passes** —
the 2-passes-vs-50 story and the schema guarantees are unchanged, the
sub-second numbers of an Apple Silicon Mac are not. More vCPUs = faster
prefill; the x86-64 build targets AVX2/FMA (x86-64-v3).

## What the image does

- `Dockerfile` (repo root), four stages: build `libllama`/`libggml` at
  `v0.4.1` (AVX2 CPU backend on x86-64), download the two
  GGUFs, `mvn package`, assemble a JRE 25 runtime image (~3 GB).
- Two models are baked in and selectable from the header — `qwen2.5-1.5b-instruct-q8_0`
  (default) and `Qwen3.5-0.8B-Q8_0` — **except that read-only mode fixes the
  model**; set `PCD_GGUF` to change which one an instance serves.
- Runs `java -jar pcd-benchmark.jar serve` with `PCD_BIND=0.0.0.0`, `PORT=8080`
  and **`PCD_READ_ONLY=true`**: on a public URL, scenarios can be viewed and
  raced (including pasted text) but not edited, the model cannot be switched,
  and no benchmark runs are written. Unset `PCD_READ_ONLY` only behind
  authentication.
- The Ollama catalog is simply empty in the cloud; the header shows the baked
  model as fixed.

## Deploy with clever-tools

```bash
npm i -g clever-tools && clever login

# once, from the repo root
clever create --type docker parallel-constrained-decoding
clever env set CC_DOCKER_EXPOSED_HTTP_PORT 8080
clever scale --flavor L          # >= 4 GB RAM: 1.9 GB model + KV cache + JVM; more vCPUs = faster prefill
clever scale --build-flavor XL   # compiling llama.cpp + a 1.9 GB download needs room

git push clever master           # or link the GitHub repo in the console
clever open
```

Redeploys rebuild the image; the llama.cpp and model stages are cached by the
build host between deploys unless the `Dockerfile` args change.

## Knobs

| Variable | Default in image | Meaning |
|---|---|---|
| `PORT` | `8080` | listening port (Clever Cloud expects 8080) |
| `PCD_BIND` | `0.0.0.0` | bind address |
| `PCD_READ_ONLY` | `true` | demo mode; set to `false` only behind auth |
| `PCD_GGUF` | `/app/models/<file>` | model to load |
| `PCD_LLAMA_LIB_DIR` | `/opt/llama/lib` | where `libllama.so` / `libggml*.so` live |
| build arg `GGUF_URL` / `GGUF_FILE` | Qwen2.5-1.5B Q8_0 | default model (Q4 quants are ~2× faster on CPU) |
| build arg `GGUF2_URL` / `GGUF2_FILE` | Qwen3.5-0.8B Q8_0 | second baked model |
| build arg `LLAMA_TAG` | `v0.4.1` | must match the tag `java/tools/gen-bindings.sh` was run against |

## Local check

```bash
docker build -t pcd .
docker run --rm -p 8080:8080 pcd
open http://localhost:8080
```

## For a live talk

Keep the Mac as the demo machine (Metal, sub-second races) and expose it with a
tunnel for the audience, e.g. `cloudflared tunnel --url http://localhost:8000`,
started with `PCD_READ_ONLY=true java -jar java/target/pcd-benchmark.jar serve`.
Use the Clever Cloud instance as the always-on "try it yourself" page.
