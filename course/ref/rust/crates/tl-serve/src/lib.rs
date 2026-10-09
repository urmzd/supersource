//! tl-serve: the engine's HTTP front.
//!
//! L10.0 writes `http` (v0): a std-only HTTP/1.1 + SSE server for API v0
//! (contracts/openapi/openai-subset.v0.yaml) that serves the byte bigram
//! through the C matmul. L10.5 takes `http` over with tokio and hyper and
//! adds `sse`, `openai`, and `template`; each arrives as one more `pub mod`
//! line here.
//!
//! The binary, src/main.rs, is yours: it parses `--model-dir --port
//! --health-port` (spec/cli-roles.md), loads the model, binds the two
//! listeners, and calls `http::serve`.

pub mod http;
