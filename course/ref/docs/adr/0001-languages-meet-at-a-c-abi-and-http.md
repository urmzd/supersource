# ADR-0001: Languages meet at a C ABI and over HTTP

## Status

Accepted (2026-10-08)

## Context

The tracer system uses four languages, each for one job: Python trains the
byte bigram, C holds the matmul kernel, Rust serves completions, and Go fronts
the engine with an API-key gateway. Every pair of languages that talks needs a
boundary, and every boundary is a place where two programs must agree on
types, memory ownership, errors, and versions.

Two kinds of crossing exist. Python and Rust call C in the same process, many
times per token, so a crossing must cost nanoseconds and must not copy
weights. The gateway and the engine run as separate processes, later in
separate pods, and are deployed, restarted, and scaled on their own, so a
crash in one must not take the other down.

The course grades each side against a written contract: `tinyllm.h` for C,
`openai-subset.v0.yaml` for HTTP. Whatever we pick must be testable from both
sides without the other side running.

## Decision

We will use exactly two kinds of boundary:

1. **In process: the C ABI of `tinyllm.h`.** C exports plain functions over
   raw pointers with explicit dimensions, returns a positive `tl_status`, and
   reports details through `tl_last_error()`. Python calls it through ctypes;
   Rust calls it through `extern "C"` declarations in `tl-sys`. The caller owns
   every buffer it passes; C never frees memory it did not allocate.
2. **Between processes: HTTP/1.1 with JSON and Server-Sent Events**, as fixed
   by `openai-subset.v0.yaml`. The gateway and the engine share nothing but
   that wire format, the `traceparent` header, and the health endpoints.

No other crossing is allowed: no shared memory between processes, no
language-specific RPC, no embedding one runtime in another.

## Consequences

- Good: one C library serves both Python and Rust, so the matmul is written
  and tested once, and both callers see the same numbers.
- Good: the C ABI is stable across compiler versions and needs no code
  generator; ctypes and `extern "C"` read it directly.
- Good: HTTP lets us test the engine with curl and the conformance suite
  before the gateway exists, and swap either side without touching the other.
- Bad: the C ABI carries no type safety. A wrong dimension or a freed buffer
  is undefined behaviour, so every C unit runs under ASan and UBSan in tests,
  and the bindings check shapes before they call.
- Bad: errors cross the ABI as integers plus a thread-local string, which is
  easy to ignore. Every binding turns a non-zero status into an exception or
  a `Result` at the call site.
- Bad: HTTP plus JSON costs a parse per request and per streamed token. That
  is noise next to model latency for the tracer, and the internal gRPC
  control plane arrives only where it pays (Pass 7).

## Alternatives considered

| Option | Why we did not choose it |
|---|---|
| Python C extension modules (the CPython C API) | ties C to one Python version and one caller; Rust could not reuse it |
| A Rust core with PyO3 and a C shim | puts the kernels behind Rust's ABI, which is not stable; C callers would need a second layer |
| gRPC between gateway and engine | needs code generation and HTTP/2 in a std-only tracer; curl and SSE cover the tracer, and gRPC stays internal for later passes |
| One process with every language embedded | a crash anywhere kills the whole system, and the parts cannot be deployed or scaled apart |
