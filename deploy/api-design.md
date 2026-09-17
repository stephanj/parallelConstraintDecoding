# API investigation

Investigated 2026-09-17 against the current working tree, including the in-progress Docker/public-demo changes. This is a proposal, not an implemented API. Findings are based on source inspection; no server was listening on localhost:8000, so no live inference or capacity measurements were made.

## Recommendation

Expose a synchronous `POST /v1/extract` backed by the existing Java native engine. Keep a fixed model per service instance, accept inline schemas initially, and add immutable registered schemas for repeated workloads later. Retain the current UI endpoints separately for demo compatibility.

The HTTP boundary surrounds the entire extraction: one request invokes native KV sharing and batched field scoring inside the server. This preserves the technique; it does not repeat the earlier experiment of sending one remote LLM request per field.

No inference rewrite or new web framework is required for a first release. The existing JDK HttpServer can handle the transport if validation, admission control, and operational handling are added explicitly. A framework migration can be evaluated separately if maintaining routing, authentication, and documentation becomes burdensome.

## What already exists

`java/src/main/java/pcd/web/Server.java` registers:

| Endpoint | Current role |
| --- | --- |
| `/api/run/parallel` | JSON extraction response; full inline preset or preset ID |
| `/api/run/race` | SSE demo: parallel result, then grammar baseline |
| `/api/status` | Model name, preset count, read-only flag |
| `/api/models` | List models; POST switches the shared model when writable |
| `/api/presets` | Read/write stored scenarios |
| `/api/benchmark`, `/api/results` | Benchmark execution and stored results |

This request is supported by the current source, without adding an API:

```sh
curl http://localhost:8000/api/run/parallel \
  -H 'Content-Type: application/json' \
  -d '{
    "context": "Production database is unavailable; checkout is failing.",
    "schema": {
      "priority": {
        "type": "enum",
        "choices": ["HIGH", "NORMAL", "LOW"],
        "description": "Urgency based on customer impact"
      },
      "requires_escalation": {
        "type": "boolean",
        "description": "Whether an on-call engineer should be notified"
      }
    }
  }'
```

It returns `json`, `fields`, `elapsedMs`, `phases`, `promptTokens`, `treeLevels`, and `forwardPasses`. The server fills in an ad-hoc ID/title. Inline schemas are not saved. Important existing behavior: `{ "presetId": "spam_email", "context": "new text" }` ignores the supplied context and runs the preset's stored example; callers must currently send the full schema with their new context.

## Proposed v1 contract

```http
POST /v1/extract
Authorization: Bearer <api-key>
Content-Type: application/json
```

```json
{
  "input": "Production database is unavailable; checkout is failing.",
  "schema": {
    "priority": {
      "type": "enum",
      "choices": ["HIGH", "NORMAL", "LOW"],
      "description": "Urgency based on customer impact"
    },
    "requires_escalation": {
      "type": "boolean",
      "description": "Whether an on-call engineer should be notified"
    }
  },
  "include_scores": true
}
```

Illustrative response only; values and timings below are not measurements:

```json
{
  "request_id": "req_example",
  "model": "qwen2.5-1.5b-instruct-q8_0",
  "engine_version": "pcd-v1",
  "output": {
    "priority": "HIGH",
    "requires_escalation": true
  },
  "scores": {
    "priority": { "selected_path_score": 0.83, "decision_levels": 1 },
    "requires_escalation": { "selected_path_score": 0.91, "decision_levels": 1 }
  },
  "usage": { "prompt_tokens": 180, "forward_passes": 2 },
  "timing": { "queue_ms": 0.5, "setup_ms": 0.8, "inference_ms": 150.0, "total_ms": 152.0 }
}
```

Contract decisions:

- `input` is required and must be a non-empty string. No demo text fallback.
- `schema` uses the existing flat boolean/enum format, explicitly named as a PCD schema. It is not general JSON Schema. Reject arrays, nested objects, free text, numbers, nullable/optional fields, unknown keywords, and cross-field rules. If JSON Schema interoperability is later needed, add a strict subset adapter that rejects unsupported features.
- Preserve exact enum strings; reject duplicates/empty strings rather than silently trimming or deduplicating as `Preset.fromJson` currently does. Validate types without Jackson string coercion.
- Every successful response contains exactly the requested fields with allowed typed values. This is a structural guarantee, not a guarantee that a classification is correct or mutually consistent with other fields. Clients can include an explicit `UNKNOWN` enum option when appropriate.
- Scores are opt-in. They describe constrained token decisions, not the probability a classification is correct. The current tree search follows greedy token branches and stops once one candidate remains; it does not score every complete candidate sequence. Its `probs` map can assign the same branch mass to multiple choices and is not generally a normalized distribution over labels. Do not expose it as one.
- Include the model identity and a version covering prompt/decoder behavior in every result, captured from the worker that executed the request. Internally record a model artifact digest for reproducibility.
- Define `total_ms` as server processing time through response assembly, including admission/queueing, schema setup and any warm-up; it excludes network delivery. Keep inference time separate. Omit synthetic output-token counts: the output JSON is assembled, not autoregressively generated.
- Later add `schema_id` plus an immutable `schema_version`, mutually exclusive with inline `schema`; always require new `input`. Registration should not store example texts or samples from presets.

Additional initial endpoints: `GET /health/live`, `GET /health/ready`, and a read-only `GET /v1/capabilities` reporting the fixed model, supported field types, and enforced limits. Publish an OpenAPI document and curl/Python/Java examples. Token streaming adds little for this engine; return a single JSON result. Defer bulk asynchronous jobs until workload evidence justifies them.

## Changes required before external use

| Area | Evidence in current implementation | Required change |
| --- | --- | --- |
| Admission control | `EngineService` serializes work with a fair lock; virtual threads can accumulate waiting requests | Bounded FIFO queue, one inference worker initially, queue expiry and overload rejection |
| Capacity | Runtime defaults: 8,192 KV context cells, 64 sequences, 2,048 tokens per decode batch | At most 63 fields under these defaults; validate actual token/batch/KV budgets before decoding |
| Prompt length | Parallel prefill is one `decode(batch)` | The current full rendered prefix must fit 2,048 tokens, even though the configured context is 8,192; chunked prefill is a separate enhancement |
| Branch capacity | All field suffixes are decoded in one batch; each branch consumes additional KV cells | Bound aggregate suffix tokens and reserve KV space/positions for collision continuation; context text alone is insufficient to calculate capacity |
| Schema limits | Enum count capped at 255; no field count or description/choice length caps | Add body byte limit and structural limits, then model-specific token validation; publish effective limits |
| Cache | Unbounded `LinkedHashMap` of compiled schemas | Bounded cache by entry count/token footprint; key includes ordered schema, model and prompt version |
| Native memory | `LlamaRuntime.tokenize()` allocates input bytes in the runtime-lifetime arena on every call | Use a per-call arena or reusable buffer so input allocations are released; audit metadata allocations and run a sustained memory test |
| First request | New schema runs an extra full warm-up before measured inference | Account for it in latency; prewarm known schemas or benchmark whether unconditional warm-up is worthwhile in service mode |
| Decoder completion | After 24 continuation levels, `forceResolve()` selects a remaining choice | Expose resolution status internally; reject unsupported schemas or return a structured resolution error rather than silently claiming full resolution |
| HTTP contract | Extraction lacks method checks; JSON casts/coercions can become 500s; contexts match URL prefixes | Exact route checks, POST-only extraction, media-type checks, strict DTO validation and stable error codes |
| Service isolation | Demo races and benchmarks share the extraction runtime; model switching is global | Dedicated API mode/process with fixed model; demo/admin routes disabled or separately authenticated |
| Shutdown | `stop(0)` followed by unguarded runtime close | Stop admissions, expire queued work, drain current inference, then release native memory |

Large enums have another accuracy concern: `Prompts.truncatedChoices()` only lists the first 20 when there are more than 50 choices. All candidates can still be selected, but they are not all described in the prompt. For a stable API, explicitly choose and version this behavior; preferably list all admitted choices and reject schemas exceeding the prompt budget. Escape quotes/control characters when constructing prompts, and evaluate how literal model special tokens in user text are handled.

Suggested initial configurable policy limits, to validate with tests: 256 KiB request body, 63 fields, 255 choices per enum, bounded field descriptions/choice strings, and a short bounded queue. These are ceilings, not a promise that every combination fits: token and KV checks can reject a smaller request. Set queue length and deadlines from measured deployment latency, especially on CPU.

Use errors shaped as `{ "error": { "code": "invalid_schema", "message": "...", "request_id": "..." } }`. Distinguish malformed JSON (400), authentication failure (401), unknown schema (404), method (405), body too large (413), media type (415), unsupported schema/token budget (422), per-client quota (429), unavailable/full worker queue (503), and unexpected engine failure (500). Supply `Retry-After` for retryable overload and avoid exposing native stack traces or local paths.

Terminate TLS at the deployment ingress, enforce API keys and per-client quotas, and keep input/output text out of default logs. `PCD_READ_ONLY=true` only prevents selected mutations: it does not authenticate inference or limit its cost. On deadline/disconnect, remove queued work; an in-progress native decode may finish before cooperative cancellation can take effect. Do not free or reuse its context prematurely. After a native failure, reset or recreate the context and restore readiness only when safe.

## Performance and scaling

The current parallelism is across fields of one extraction. Extra HTTP threads do not create extra inference capacity. Start with:

```text
Client -> TLS/auth/quota -> bounded queue -> one Java inference worker -> JSON
```

Scale with independent worker processes/replicas and fixed models after measuring hardware utilization. Each current runtime owns both model and context, so simply pooling runtimes duplicates model ownership and KV storage. Sharing weights across multiple contexts would require an explicit native lifecycle refactor and a throughput/memory benchmark. More workers on an already saturated CPU/GPU can worsen latency.

The existing web path caches compiled schema metadata but calls `run(..., false)`: it does not reuse the system KV prefix. Do not simply enable prefix reuse on every cached engine. All those engines share one mutable context, and their per-engine `cachedSystemTokens` state does not establish that their prefix is still resident after another schema or baseline runs. Add worker-level cache ownership/invalidation first; consider schema-affinity scheduling only after measuring fairness and throughput.

Do not derive an SLA from the README's Mac medians. Measure end-to-end p50/p95/p99, throughput, queue wait/rejection, schema cache misses and resident memory on the actual CPU or GPU host. Include new versus repeated schemas, short versus long input, colliding labels and high-cardinality enums, and concurrency of 1/2/4/8 clients. Separately evaluate labeled task accuracy and score calibration; valid output does not establish either.

## Implementation sequence

1. Separate transport-neutral inference results from the UI serializer in `EngineService`; add a thin v1 adapter and strict request validation. Keep `NativeParallelEngine` as the execution core.
2. Fix native temporary allocation lifetime, add token/sequence/KV guards, and make unresolved token-tree results explicit. Add bounded scheduling, cache eviction, readiness, and safe shutdown.
3. Add auth/quota integration, exact HTTP/error behavior, server timing, OpenAPI, and runnable client examples. Deploy in API-only mode with a fixed model.
4. Verify HTTP contract and overload/timeout behavior using a fake inference backend; run native tests for boundaries, collision resolution, alternating schemas, recovery, and sustained memory usage. Load-test on target hardware and publish measured limits.
5. Add registered schemas, safe warm-prefix reuse, replicas, or asynchronous batches when observed workload needs them.

## References

- Local sources: `Server.java`, `EngineService.java`, `Preset.java`, `Prompts.java`, `nativeengine/{LlamaRuntime,CompiledSchema,NativeParallelEngine}.java`, root `Dockerfile`, and `deploy/clever-cloud.md`.
- [JDK 25 HttpServer documentation](https://docs.oracle.com/en/java/javase/25/docs/api/jdk.httpserver/com/sun/net/httpserver/HttpServer.html): context routing uses longest-prefix matching, so strict endpoint matching must be added in handlers; shutdown delay is bounded.
- [llama.cpp native API](https://github.com/ggml-org/llama.cpp/blob/master/include/llama.h): upstream reference for native integration. Implementation must remain matched to the repository's pinned llama.cpp build and generated FFM layouts; upstream master is not an ABI compatibility promise.
