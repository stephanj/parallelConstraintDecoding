# Hosting the demo with Docker

The web app ships as a self-contained **Docker** image: llama.cpp (CPU
backend) compiled from the tag the Java bindings were generated against, JRE 25,
and the Qwen2.5-1.5B and Qwen3.5-0.8B Q8 models baked in, so a container has no
external dependencies. It runs on any container host (a VPS, Fly.io, Render,
Kubernetes, a PaaS with a Docker runtime, …).

Inference is CPU-only in the image. Measured on a 20-vCPU arm64 Docker VM:
the parallel engine decides the 8-field spam scenario in **~1.1 s / 2 passes**,
the grammar baseline needs **~8.5 s / 54 passes**; on a smaller x86-64 cloud
instance expect ~2.7 s vs ~3.8 s. The 2-passes-vs-50 story and the schema
guarantees are unchanged; the sub-second numbers of an Apple Silicon Mac are
not. More vCPUs = faster prefill; the x86-64 build targets AVX2/FMA (x86-64-v3).

## What the image does

- `Dockerfile` (repo root), four stages: build `libllama`/`libggml` at
  `v0.4.1`, download the two GGUFs, `mvn package`, assemble a JRE 25 runtime
  image (~3 GB).
- Runs `java -jar pcd-benchmark.jar serve` with `PCD_BIND=0.0.0.0`,
  `PORT=8080`, **`PCD_READ_ONLY=true`** (on a public URL scenarios can be
  viewed and raced — including pasted text — but not edited, the model cannot
  be switched, and no benchmark runs are written; unset only behind
  authentication) and **`PCD_SEQUENTIAL_RACE=true`** (the two engines are shown
  one after the other, each timed live, which is the honest presentation on a
  shared CPU).
- Two models are baked in; `PCD_GGUF` selects the one an instance serves
  (read-only mode fixes it).
- The Ollama catalog is simply empty in a container.

## Build and run

```bash
docker build -t pcd .
docker run --rm -p 8080:8080 pcd                 # http://localhost:8080
docker run --rm -p 8080:8080 -e PCD_GGUF=/app/models/Qwen3.5-0.8B-Q8_0.gguf pcd
```

The host must expose port 8080 (or set `PORT`). Give the container at least
4 GB of RAM (1.9 GB model + KV cache + JVM); more CPU cores mean faster prefill.

## Knobs

| Variable | Default in image | Meaning |
|---|---|---|
| `PORT` | `8080` | listening port |
| `PCD_BIND` | `0.0.0.0` | bind address |
| `PCD_READ_ONLY` | `true` | demo mode; set to `false` only behind auth |
| `PCD_SEQUENTIAL_RACE` | `true` | show the engines one after the other, each timed live (a laptop defaults to replaying both on one timeline) |
| `PCD_GGUF` | `/app/models/qwen2.5-1.5b-instruct-q8_0.gguf` | model to load |
| `PCD_LLAMA_LIB_DIR` | `/opt/llama/lib` | where `libllama.so` / `libggml*.so` live |
| build arg `GGUF_URL` / `GGUF_FILE` | Qwen2.5-1.5B Q8_0 | default model (Q4 quants are ~2× faster on CPU) |
| build arg `GGUF2_URL` / `GGUF2_FILE` | Qwen3.5-0.8B Q8_0 | second baked model |
| build arg `LLAMA_TAG` | `v0.4.1` | must match the tag `java/tools/gen-bindings.sh` was run against |

## Notes for PaaS builders

Compiling llama.cpp with every x86 SIMD variant (`GGML_CPU_ALL_VARIANTS`) was
too heavy for a hosted build instance (aborted after minutes of AVX-512
compilation and thousands of warnings); the Dockerfile therefore builds a
single AVX2/FMA target with warnings silenced and at most 8 parallel jobs,
which takes about four minutes.

## For a live talk

Keep the Mac as the demo machine (Metal, sub-second races) and expose it with a
tunnel for the audience, e.g. `cloudflared tunnel --url http://localhost:8000`,
started with `PCD_READ_ONLY=true java -jar java/target/pcd-benchmark.jar serve`.
Use the hosted container as the always-on "try it yourself" page.
