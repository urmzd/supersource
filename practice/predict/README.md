# Predict-Then-Run

A harness for the one drill that actually finds holes in your mental model of a
language: **write down what you think the code prints, then run it.** The diff
between your prediction and reality is the only signal here. Nothing in this
directory grades style, teaches syntax, or rewards effort.

The rest of [Polyglot Practice](../README.md) asks you to *build* things. This
asks you to *predict* things, which is a different and much faster way to find
out that you were wrong about `defer`, about integer overflow, or about what
`this` is bound to.

## Why prediction and not reading

Reading a language spec produces recognition ("yes, that looks right"), which is
not the same as recall. Committing to an exact expected output before running
forces the prediction into a form that can be falsified. When it is wrong you get
a precise, memorable correction instead of a vague feeling that you should reread
the chapter.

Two rules make this work:

1. **Commit before you run.** A prediction you edit after seeing the output is
   worth nothing.
2. **Log every miss.** A miss you do not write down is a miss you will repeat.
   `LEDGER.md` is where they go.

## The harness

`bin/lr` is a single bash script with no dependencies beyond the language
toolchains themselves.

| Command | What it does |
|---------|--------------|
| `lr list [lang]` | Show snippets and whether you have predicted them |
| `lr show <lang> <id>` | Print a snippet's source |
| `lr predict <lang> <id>` | Show the source, then open your prediction file in `$EDITOR` |
| `lr check <lang> [id]` | Run for real and diff your prediction against actual output |
| `lr score [lang]` | Tally passed, missed, and not attempted |
| `lr reveal <lang> <id>` | Just run it, burning the exercise |
| `lr reset <lang> <id>` | Delete your prediction and start over |

`<id>` is the numeric prefix of a snippet, so `lr check go 03` is enough.

Your predictions are written to `<lang>/predictions/<snippet>.txt`. They are
yours, they are not committed, and the harness never looks at them until you run
`lr check`.

## Where snippets live

Each language keeps its snippets in whatever layout its toolchain expects, so
that `go run`, `cargo run`, and friends work with no wrapper.

| Language | Snippet path | How the harness runs it |
|----------|--------------|-------------------------|
| Go | `go/predict/NN-name/main.go` | `go run ./predict/NN-name` |
| Rust | `rust/predict/src/bin/NN-name.rs` | `cargo run --bin NN-name` |
| Python | `python/predict/NN-name.py` | `uv run python predict/NN-name.py` |
| TypeScript | `ts/predict/NN-name.ts` | `node predict/NN-name.ts` (native type stripping, Node 22+) |

Every snippet must be **deterministic**. Output that varies between runs (map
iteration order, goroutine interleaving, timestamps, addresses) makes prediction
impossible and is a bug in the snippet, not a lesson.

## Status: baseline

This commit is the starting point on purpose. The harness is complete and the
language scaffolds are in place, but **there are no snippets yet**, so:

```bash
$ bin/lr list go
go
$ bin/lr score
go      0 passed  0 missed  0 not attempted
rust    0 passed  0 missed  0 not attempted
python  0 passed  0 missed  0 not attempted
ts      0 passed  0 missed  0 not attempted
```

Everything that follows (the snippet corpus, the determinism check, and the
memory and speed budgets) is layered on top of this baseline so each addition can
be read as its own diff.
