# Part 10: Serving

The Rust engine: an OpenAI-compatible HTTP server that streams completions from a checkpoint through your C kernels. Pass 1 has one chapter here: a std-only HTTP/1.1 and SSE server (no async runtime, no dependencies) serving the byte bigram, with one hand-written OTLP span per request. Pass 7 rebuilds it on tokio and hyper with continuous batching, chunked prefill, disaggregation, speculative decoding, and tool calls, behind the same API.

**Course passes**: 1 (L10.0, gate [MS-P1](../../../paths/course-p01-tracer/milestone.md)), 7 (L10.1 to L10.9, load.01, load.02, gate MS-P7).

**Before you start**: the [Rust](../../../software-craftsmanship/12-language-and-tool-primers/04-rust.md) and [HTTP and SSE](../../../software-craftsmanship/12-language-and-tool-primers/05-http-and-sse.md) primers; the API contract [`openai-subset.v0.yaml`](../../../course/contracts/openapi/openai-subset.v0.yaml) and the engine role in [`spec/cli-roles.md`](../../../course/contracts/spec/cli-roles.md).

<!-- ss:chapters -->
| # | Module | Chapter | Kind | Pass |
|---|---|---|---|---|
| 1 | `L10.0` | [Your first endpoint: a std-only Rust HTTP/1.1 + SSE server over the C matmul](00-your-first-endpoint.md) | build | 1 |
<!-- /ss:chapters -->
