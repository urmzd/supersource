# CLI roles

<!-- chapter: ml/08-tinyllm/p10-serving/00-your-first-endpoint.md
     chapter: ai-platform-engineering/12-gateway/00-streaming-proxy.md -->

The course never builds or runs your binaries by guessing. You declare each entry point once in `system.toml` under `[entry]` (schema: [`config/system.schema.json`](../config/system.schema.json)) as an argv list, and milestones call it by its **role**. A milestone step written `{tinyllm} generate --prompt hi` runs your `[entry].tinyllm` argv followed by `generate --prompt hi`.

```toml
[entry]
tinyllm = ["uv", "run", "--project", "python", "python", "-m", "tinyllm"]
engine  = ["rust/target/release/tl-serve", "--model-dir", "{model_dir}", "--port", "{port}", "--health-port", "{health_port}"]
gateway = ["go", "run", "./go/cmd/gateway", "--port", "{port}", "--health-port", "{health_port}", "--upstream", "http://127.0.0.1:{engine.port}"]
```

A missing role fails the milestone with `milestone <MS-ID> needs [entry].<role> in system.toml`. Placeholders such as `{port}` are filled by the milestone runner (ports are allocated per run, never fixed on your machine).

This page fixes the verbs, flags, and outputs that milestones depend on. Your CLI may have more verbs and flags; the ones here must behave as written. Later modules add verbs and flags; the ones below stay valid.

## Rules for every role

- **Exit codes.** `0` on success; `2` for a usage error (unknown verb or flag); any other non-zero code for a failure, with the reason on stderr.
- **Machine output.** When a verb's output is marked *final JSON line*, the **last line of stdout** is one JSON object on one line. Earlier lines are free (progress, streamed text). Matchers read only that line, so they never need your tokenizer.
- **Native library.** Roles that load `libtinyllm` find it through `TINYLLM_LIB` when it is set (the harness sets it to its own build), else through your own default path.
- **Telemetry.** Servers read the standard OTel variables: `OTEL_EXPORTER_OTLP_ENDPOINT` (base URL; unset means export nothing) and `OTEL_SERVICE_NAME` (in Kubernetes your charts set it to `<system>-engine`, `<system>-gateway`, ...).
- **Shutdown.** Servers exit `0` on SIGTERM.

## Roles

| Role | Language (typical) | First used | Contract |
|---|---|---|---|
| `tinyllm` | Python | Pass 1 (L0.0) | [below](#tinyllm) |
| `engine` | Rust | Pass 1 (L10.0); `--config` from L10.5 | [below](#engine) |
| `gateway` | Go | Pass 1 (gw.00); `--config` from gw.01 | [below](#gateway) |
| `corpus` | Python | Pass 3 (data.*) | `{corpus} <verb> --spec ... --progress ...` under `spec/subprocess-activity.md` |
| `tl-tok` | Rust | Pass 3 (L1.5) | `encode`, `bench`; defined by L1.5 |
| `durable` | Go | Pass 8 (dur.02) | `--data {data} --port {grpc_port}`; `--test-clock` enables `/debug/clock` |
| `worker` | Go | Pass 8 (dur.04) | `--queue {queue} --durable <addr>`; `--test-activities` registers the course test activity `tl.test.Append` (appends its idempotency key to the `effects` sink) and the course test workflows used by replay and kill-loop tests |
| `ctl` | Go | Pass 8 (dur.11) | the umbrella CLI: `train`, `eval`, `release`, `data build`, defined by their modules |
| `loadgen` | Go | Pass 7 (load.01) | defined by load.01 and load.02 |
| `agent` | Go | Pass 10 (ag.*) | defined by the agent modules |

## `tinyllm`

The Python CLI, `python -m tinyllm` in your uv project. In Pass 1 it has four verbs.

### `train bigram`

```
{tinyllm} train bigram --data <file> --out <dir> [--alpha <float>]
```

Fits the byte bigram of L0.0 by counting. `--data` is a file whose raw bytes are the token ids (the byte tokenizer, [formats/tokenizer.md](../formats/tokenizer.md)). `--alpha` is the add-one smoothing constant, default `1.0`. Writes a model directory:

- `<dir>/config.json`: `{"tl_arch": "bigram", "tl_tokenizer": "bytes", "vocab_size": 256, "tl_format": 1}` ([formats/config.schema.json](../formats/config.schema.json))
- `<dir>/model.safetensors`: `bigram.weight`, F32 `[256, 256]`, metadata `{"format": "tinyllm"}` ([formats/safetensors.md](../formats/safetensors.md))

Final JSON line: `{"out": "<dir>", "tokens": <int>, "nll": <float>}`, where `nll` is the mean negative log-likelihood in nats per byte of the fitted model on `--data`.

### `generate`

```
{tinyllm} generate --model <dir> --prompt <text> [--max-tokens <n>] [--greedy | --temperature <t>] [--seed <s>]
```

Continues `--prompt` (non-empty) with exactly `--max-tokens` new tokens (default 16; the byte tokenizer has no end-of-sequence token). `--greedy` is `--temperature 0`: the highest logit, ties to the lowest id. Otherwise tokens are sampled at `--temperature` (default 1.0) from a generator seeded with `--seed` (default 0), so the same seed gives the same ids.

Final JSON line: `{"ids": [<int>, ...], "text": "<string>"}`. `ids` holds the **generated ids only**, not the prompt's; `text` is `decode(ids)` with replacement. The `tokens-equal` matcher compares `ids` with an expected list.

### `logits`

```
{tinyllm} logits --model <dir> --prompt <text> [--prefix-ids <id,id,...>]
```

Teacher forcing: runs the model on the prompt's ids followed by `--prefix-ids` (comma-separated, may be empty) and prints the next-token logits, before temperature.

Final JSON line: `{"logits": [<float>, ...]}`, `vocab_size` values. The `tokens-equal` matcher's near-tie rule calls this verb: it takes `generate`'s argv, replaces the verb with `logits`, and appends `--prefix-ids` with the expected ids up to the first divergence.

### `info --native`

```
{tinyllm} info --native
```

Loads `libtinyllm` through your ctypes loader (rt.01) and calls `tl_abi_version()`.

Final JSON line: `{"abi_version": <int>, "lib": "<path loaded>"}`. Exits non-zero, with the reason on stderr, when the library is missing or its ABI version differs from the one the loader was written for.

## `engine`

The model server: OpenAI subset over HTTP and SSE ([openapi/openai-subset.v0.yaml](../openapi/openai-subset.v0.yaml)).

### Tracer form (L10.0 to L10.4)

```
{engine} --model-dir <dir> --port <n> --health-port <n>
```

The tracer engine is std-only Rust and has no TOML parser, so it takes flags.

- `--model-dir`: a model directory as written by `train bigram`. Contract v0 serves `tl_arch = bigram` with `tl_tokenizer = bytes`; every later engine keeps serving it, so the tracer smoke stays green for the whole course.
- `--port`: serves `POST /v1/completions` (API v0) on `0.0.0.0:<port>`.
- `--health-port`: serves `GET /healthz` on `0.0.0.0:<port>` (`9464` in Kubernetes). The engine must also answer `GET /healthz` on `--port`: the tracer gateway's `/readyz` probes `<upstream>/healthz` on the API port.
- Telemetry: reads `traceparent` and, when `OTEL_EXPORTER_OTLP_ENDPOINT` is set, posts one SERVER span `POST /v1/completions` per request as OTLP/HTTP JSON to `<endpoint>/v1/traces`, as the child of the incoming context.

### Config form (from L10.5)

```
{engine} --config <runtime.toml>
```

Reads `[engine]` from `runtime.toml` (schema `config/runtime.schema.json`, arriving with L10.5), with `TL_ENGINE__<KEY>` overrides. The runner fills `http_listen`, `grpc_listen`, `kv_listen`, and `health_listen` with allocated ports.

## `gateway`

The front door: API keys, then SSE passed through without buffering.

### Tracer form (gw.00)

```
{gateway} --port <n> --health-port <n> --upstream <engine base URL>
```

- `--port`: serves the gateway tier of API v0 on `0.0.0.0:<port>`.
- `--health-port`: serves `GET /healthz` (200 while up) and `GET /readyz` (200 once `<upstream>/healthz` answers, else 503).
- `--upstream`: the engine's base URL without `/v1`, for example `http://127.0.0.1:{engine.port}` (or the engine Service in Kubernetes).
- **Key.** Accepts `Authorization: Bearer <key>` where `<key>` is the value of the environment variable that `[endpoints].api_key_env` names (default `TL_API_KEY`). The harness passes its environment to every service it starts and uses the same variable as its client key. A missing or different key is `401` with `code: invalid_api_key`, before any upstream call.
- Propagates `traceparent` (as the child of its own `gateway.proxy` CLIENT span) and `X-Request-Id` to the engine.

### Config form (from gw.01)

```
{gateway} --config <runtime.toml>
```

Reads `[gateway]` from `runtime.toml` with `TL_GATEWAY__<KEY>` overrides; keys and routes come from the files it names.
