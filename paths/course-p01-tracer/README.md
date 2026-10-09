# Course Pass 1: The Tracer

A thin system through every layer, before any layer is deep. You learn C, Rust, HTTP and SSE, Go, and containers as you need them, and build: the C ABI and a naive matmul, a byte-level bigram fitted by counting whose logits go through your matmul by ctypes, a std-only Rust engine that streams completions from your checkpoint, a Go gateway that checks a key and streams without buffering, images and Helm charts on kind, one trace across gateway and engine in Jaeger, your first incident drill with its runbook, and your first architecture decision record.

**Part of**: [the course](../course/). About 5 weeks at 10 to 12 hours a week. **Needs**: [Pass 0](../course-p00-setup/) and its gate MS-P0.

**Gate**: [MS-P1](milestone.md), every layer yours and running end to end.

The stages, their chapters, and what "done" means for each are in [`path.tsv`](path.tsv). Every stage after a primer depends on the stages before it: the engine serves the checkpoint your CLI wrote, through the matmul you wrote.

```bash
practice/bin/ss learn course-p01-tracer         # stages and progress
practice/bin/ss learn course-p01-tracer next    # read the next unfinished stage
```

Pass 1 needs a local kind cluster for stages 10 to 13 (`ss doctor --pass 1`). Everything before stage 10 runs as local processes.
