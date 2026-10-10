# Course build handoff

## Status update (2026-10-10, Claude subagent): read this first

Steps B and D are done; step E is mostly done; step F is partly green; step C (B14) is not started. Pick up from here:

- **Step B is done.** No ctypes, PyO3, tl-sys, or Rust-to-C calls remain in active code. DESIGN.md is aligned (D7, D8, D14, D39, D40, the M09.x rows, parity rows, harness snippet). `c/ABI.md` documents the standalone C interfaces; the optional C modules are `kind = "side"`; ds.08 (Rust Bloom) is now side too, since no Rust module can call it without FFI.
- **Step D is done.** Links are clean; `ss export` vendors ASSETS.tsv and every data contract, sets fixture env for Go and Rust, and leaves course-tree practice checks (`# ss-export: course-tree`) out of the native run. The SHARED items of open-items.txt are applied (contracts 0.3.0, schema and spec fixes, chapter back-pointers, harness: newline-only stubber, SS_MUTATION_JOBS, partial contract checks for taken-over units, .pyi constants as declared names, self proofs satisfy requires, go.mod contracts require dropped, ulp parity equality, SLO window durations, tdd --ref-deps, FlakyHTTP poll_interval, solve sync for proofs, sanitizer self-tests fixed on Linux and macOS). modules.tsv and paths are current. CI: harness-unit job, 16-shard `ss verify course --changed origin/main --shard I/N`, 120-minute e2e and kind jobs.
- **Reference system:** the engine role runs the v1 server (`tl-serve --config {config}`, template `course/ref/entry/config/engine.runtime.toml`, tiny-llama copied by [build], served as `tracer`); it exports OTLP request spans, records L10.7 metrics, routes L10.9 tools, and exits 0 on SIGTERM. The gateway entry serves /metrics and JSON logs. New `go/cmd/durable` and `go/cmd/worker` mains (dep.06). Still missing: the `ctl`, `agent`, and `loadgen` entry points and `[entry]`/`[services]` rows for durable and worker, so MS-durable, MS-agent, MS-gateway (usage verb), and MS-L10's loadgen steps cannot run in e2e yet.
- **Call sites (verify check 9):** given to M02.2, M09.3, M01.2, L6.7, data.08, gw.03, gw.05, gw.06, L10.7, L10.9. **Still failing check 9, documented:** L8.3 (`course/deviations/L8.3.md`) and load.02 (`course/deviations/load.02.md`).
- **Per-module verify on CI (16 shards, Linux):** the last full round failed only load.02 and L8.3 (check 9) and dep.06 (fixed since). Platform fixes along the way: batch-invariant candle runner (L10.1), Python 3.11 mutants (L8.1), gcc float-cast-overflow sanitizer (M09.6), glibc bench include order, deterministic Go tests, regenerated mutants.
- **e2e:** the assembled reference learner now has learner tests with red and green tdd journals, ref solve answers, and sampled mutation grades (SS_MUTATION_SAMPLE=1, falling back to the full grade when an estimate misses the bar). Conformance runs the API versions milestones grade (v0, v1). `commit_all` can no longer commit into an outer repo (it did twice in this session; those commits were undone). Last local run (a machine at load 100 from another session) hit the 3600 s check timeout after 198 of 245 modules with 2 failures, both since fixed.
- **Step E adversarial pass:** planted learner bugs in M09.2 (no max shift, NaN masked rows, plain sum), L0.3 (padding dilutes the mean), ds.05 (stale probe distance), ds.09 (off-by-one ring search): all caught by the course tests.
- **CI at fe5a958:** Lint, Test, Build Exercises, Predict, Book PDF, Build site, harness self-tests, and 14 of 16 verify shards pass; shards 1 and 5 fail only check 9 of load.02 and L8.3. Reference learner e2e: assembly ok; MS-L0 to MS-L9 and MS-L11 `--smoke` pass; `ss check --all --ci` hits its 3600 s limit on the runner (exit 124); MS-L10 fails its 4 tool conformance cases because the tool-schema prompt (455 to 474 byte tokens) exceeds tiny-llama-2l's 256-position context; MS-P0's "CI is green" step re-runs check --all and hits 1800 s; the job is cancelled at 120 min. Next fixes: make check --all reuse fresh verdicts (or cache mutation grades across CI runs), give the tool cases a compact schema or a longer-context PR model, then MS-P0 and the conformance and export steps after them. MS-P1 on kind: check --all times out the same way, MS-P1 `--ref-deps` hits 1800 s, MS-P7, P8, P9, durable, durable-ha, and prod exit 5 for missing services and entry points (ctl, loadgen, agent; durable and worker are not in system.toml yet).
- **Step F so far:** `ss lint`, `ss lint --links`, `ss learn --verify`, `ss verify predict`, `site pnpm build`, harness self-tests, and `ss parity --ref` pass locally. `ss verify course` fails only check 9 of L8.3 and load.02. See the PR #19 summary comment for the latest CI and e2e output.
- **Not done:** step C (B14 multimodal); MS-* pass-gate `--smoke` runs beyond what e2e covers; the open harness features in open-items.txt (logits-close matcher, chaos steps, Tempo traces, drill workloads, disagg suite); OpenAPI v2 on the engine (craft.14 wiring).
- **Runner notes:** give every agent its own scratch dir inside the worktree (`.scratchpad/w/<name>`) and its own runner script there; agents sharing the session scratchpad overwrote each other's runners. Another project's session may load the machine; long verifies then need longer timeouts.

You are taking over the build of the supersource course from a Claude Code session. Finish everything below, in order. This file is the brief; `course/DESIGN.md` is the spec.

Repo: this git worktree (`.worktrees/course`, branch `feat/course`, PR urmzd/supersource#19). Work only inside it.

## 1. What the course is (read first)

One course where the learner builds one system end to end, module by module, and owns all of it. Read `course/DESIGN.md` sections 1, 2, 3, 5, 6, 9. Built modules are the exemplars: study `course/modules/L0.0.toml`, `L10.5.toml`, `gw.04.toml`, their references in `course/ref/`, tests in `course/tests/`, mutants in `course/mutants/`, and their chapters (path in each registry file's `chapter` field). The harness is `practice/bin/ss` plus the Python package in `course/harness` (`practice/README.md` documents the commands). Deviations already taken are in `course/DEVIATIONS.md`; append new ones.

## 2. Status

| Area | State |
|---|---|
| Design | `course/DESIGN.md` Draft v2, 307 modules in 13 batches |
| B1 harness + tracer | Done and verified |
| Foundation for B2-B13 (contracts, consolidation, harness) | Done |
| Module groups | 35 of 48 done (B2 to B9, B10-3). 240 registry files exist |
| Remaining groups | 13, listed in `course/handoff/remaining-groups.md`. B10-1, B10-2, B11-1, B11-2, B12-1, B12-2 were interrupted mid-run: their partial files are on disk, continue them rather than starting over |
| Open items from finished groups | 442 lines in `course/handoff/open-items.txt` (deferred verifications, incomplete ids, and SHARED changes other groups requested). Integration resolves them |
| B14 multimodal | Designed in `course/design/B14-multimodal.md`, not built, not yet merged into DESIGN.md |

## 3. Decisions made AFTER DESIGN.md was written (they override it)

1. **No FFI anywhere.** No ctypes, no PyO3, no Rust calling C (`extern "C"`, `tl-sys`). Languages meet only at process and file boundaries: HTTP, gRPC, subprocess, and files (safetensors, tokenizer.json, fixtures). Reason from the user: FFI is hard and teaches nothing meaningful.
2. **Custom kernels are optional depth, not core.** L9.1 to L9.6 (C kernels), ds.01 to ds.04 and rt.02 to rt.04 (C data structures and runtime), M09.5 and M09.6 stay as optional, skippable, standalone C modules: each has its own C test binary and is parity-checked against fixture files generated by the Python reference. No engine backend switch (that would be FFI). No pass gate or core milestone may depend on them. No Triton or CUDA part.
3. **The Rust engine runs on candle** (`candle-core`, `candle-nn`; Metal on macOS, CPU in CI), not on course C kernels. `candle-transformers` is forbidden so the learner still writes every layer. Rust's purpose is the production serving layer: scheduler, continuous batching, KV block manager, prefix cache, HTTP/SSE, tokenizer.
4. **Python and Rust exchange data through files only.** Model weights go Python to Rust via safetensors. The Rust tokenizer (L1.5) serves the engine; Python keeps its own BPE; parity is checked through shared fixture files.
5. **Four languages:** Python, Rust, Go, and C (optional kernels only).
6. **B14 Q12 resolved:** Python never decodes images through Rust. Fixtures ship pre-decoded `.npy` pixels; Python uses Pillow for everyday decoding; the engine decodes with the Rust `image` crate and one test checks it against the `.npy` pixels within a small tolerance.

## 4. Work to do, in order

### Step A: finish the 13 remaining module groups
Per `remaining-groups.md`. For each module: registry toml, reference with markers, course tests in exemplar style (each test says why it exists), mutants, fixtures, solve sets for `S-*`, milestone definitions for `MS-*`, and a six-beat chapter at its chapter home. Apply the decisions in section 3 (for example the Go durable and agent code must not depend on any FFI). Prove each: `ss start <ID>` gives a stub that compiles and fails, the reference passes, `ss verify course <ids>` passes, `ss lint` is clean.

### Step B: restructure batch (applies section 3 to already built modules)
Modules that touch FFI today: `ds.04 L0.0 L10.1 L1.5 L10.6 L10.0 L10.4 L10.8 L9.1 L9.5 M03.1 L9.2 L9.6 L9.3 L8.3 lang.03 L9.7 M09.6 L9.4 M09.5 rt.03 rt.01` (find them with `grep -rlE "ctypes|pyo3|PyO3|extern \"C\"|tl-sys|cdll" course/modules`).
- B1 tracer: the Python bigram (L0.0) uses numpy, no C matmul. Retire rt.01 (C ABI + ctypes). M03.1 becomes a math chapter with a Python matmul; its C matmul becomes an optional standalone exercise.
- L1.5: drop the PyO3 binding; parity with Python BPE through fixture files.
- L8.3: paged KV cache in pure Python.
- L9.1 to L9.6, ds.01 to ds.04, rt.02 to rt.04, M09.5, M09.6: mark optional, standalone C test binaries, parity via fixture files from the Python reference. Retire L9.7.
- L10.x: engine on candle; retire tl-sys and every Rust-to-C call; keep scheduler, batching, KV block manager, prefix cache, HTTP/SSE, disaggregation, metrics, spec decoding, tool calls.
- Contracts: shrink `tinyllm.h` to what the optional C modules need; remove PyO3 stubs; update `course/DESIGN.md` (sections 1.5 decisions, 2, 4, 5, 7, 9) so the spec matches.
- Harness: remove the PyO3 farm build, the C overlay's two-build ctypes path, and symbol-drift checks that only existed for FFI. Keep the C test path for the optional modules.
- Pass gates and milestones (MS-P6 and later): must not require optional modules.

### Step C: merge and build B14
Merge `course/design/B14-multimodal.md` into `course/DESIGN.md` (with decision 6 above and its optional C kernels standalone, no `kernels = "c"` switch). Use these defaults for its open questions: Pass 12 is core but after v1.0.0; raise the committed-fixture cap to 56 MiB; accept no cross-request batching in the encoder; WAV-only audio input for v1. Verify the licenses (SmolVLM-256M-Instruct, whisper-tiny, PD12M, LibriSpeech, Common Voice) before committing any of their assets; if a license is unclear, use a synthetic stand-in and record it. Then build all B14 modules.

### Step D: integrate
Apply every SHARED change in `open-items.txt` that is consistent with the design and section 3. Regenerate `course/modules.tsv`. Fill `paths/course-p02` through `course-p12` and the role paths with the check column. Link chapters from track index READMEs. CI jobs per DESIGN 5.14 in `.github/workflows`. sr.yaml artifacts. Final STUDY-PLAN.md, CS-CURRICULUM.md, README.md.

### Step E: adversarial verification and fixes
Per layer (math; early ML; late ML; Rust; Go; ops and practices; multimodal): act as a learner and implement a few modules from the chapter alone; plant wrong implementations and confirm the tests catch them; check chapters for the six-beat template, correct math, first-principles completeness, and no em dashes. Fix every real defect.

### Step F: final gate (all must pass, report real output)
```bash
practice/bin/ss lint && practice/bin/ss lint --links && practice/bin/ss learn --verify
practice/bin/ss verify course
practice/bin/ss verify course --e2e
(cd course/harness && uv run pytest -q)
PATH=~/.local/share/fnm/aliases/default/bin:$PATH practice/bin/ss verify predict
(cd site && pnpm build)   # if site/ exists on this branch
practice/bin/ss milestone <each pass gate MS-P0 .. MS-P12> --smoke
```
kind is NOT installed and must not be installed: kind-only tiers run through `--smoke` and are reported as blocked.

## 5. Runner rules (mandatory: the machine crashed under load before)

- Run mutants one at a time, never in parallel.
- Start every subprocess you launch from scripts (builds, test runs, mutant runs, servers) with `start_new_session=True`.
- Kill the whole process group (`os.killpg`) if its total RSS goes above 4 GB or it runs longer than 120 s; report it as a failure, do not retry in a loop.
- Before each batch of runs: delete `.course-verify` under your `SS_SCRATCH`, and delete stale `$TMPDIR/tmp*` entries (not modified in 30 minutes).
- Before each batch of runs: check free disk with `df -g .`; while it is under 30 GB, wait instead of running. Delete `.scratchpad/w/<name>` dirs once their work is finished.
- If you run work in parallel, give each worker its own `SS_SCRATCH`, `SS_COURSE_HOME`, `SS_CACHE` under `.scratchpad/w/<name>` and run at most 6 at once.

## 6. Rules for commits and content

- Prose has no em dashes. Chapters follow DESIGN 6.2.
- Never publish anything: no release assets, no package publishes. Downloads only from official sources with permissive licenses.
- Fixtures must not contain strings shaped like real secrets (GitHub push protection rejected fake Slack tokens once; the generator now uses `xoxb-EXAMPLE-...`). Keep fake keys obviously fake.
- Commit in logical checkpoints with conventional commit messages, no AI attribution lines, and push to `origin feat/course` (PR #19) after each step (A to F). Do not commit `uv.lock` changes that only bump the project version, or anything under `.scratchpad/`.
- Report honestly: if a check fails, say so with the output.

## 7. Done means

Every core module in DESIGN.md (including B14) is registered, has a reference, tests, mutants, and a chapter, and passes `ss verify course`; optional modules pass their standalone checks; no FFI remains on any path; the final gate passes except kind-only tiers; everything is pushed to PR #19 with a summary comment listing what passed and what is blocked.
