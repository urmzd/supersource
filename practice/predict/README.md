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
| `lr verify [lang]` | Maintainer check: every snippet builds, runs, and prints the same thing three times in a row |

`<id>` is the numeric prefix of a snippet, so `lr check go 03` is enough.

Your predictions are written to `<lang>/predictions/<snippet>.txt`. They are
yours, they are not committed, and the harness never looks at them until you run
`lr check`.

## A session

```bash
cd practice/predict
bin/lr list go                 # what is here, and what you have attempted
bin/lr predict go 03           # read the source, write the exact expected output
bin/lr check go 03             # PASS, or a diff of prediction against reality
bin/lr score                   # where you stand across all four languages
```

A miss prints a unified diff and one instruction: record it in
[`LEDGER.md`](LEDGER.md). Six snippets is roughly a 40-minute session, and going
slower is better, because the value is entirely in the minute you spend deciding
what you think happens.

`lr check` runs the real toolchain. First contact with Rust or Go pays for a
compile, so `lr verify <lang>` (below) is a reasonable way to warm the cache
before a session.

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
impossible and is a bug in the snippet, not a lesson. `lr verify` is what
enforces that.

Node runs the TypeScript files by stripping the types, not by checking them.
Some snippets are deliberately unsound, because being unsound at runtime is the
lesson, so do not expect `tsc --noEmit` to be happy with all of them.

## The corpus

Twenty-four snippets, six per language, each aimed at a belief that is common,
load-bearing, and wrong.

| # | Go | Rust |
|---|-----|------|
| 01 | `loop-var-capture` (Go 1.22 changed the answer) | `shadowing-and-blocks` |
| 02 | `nil-interface` (typed nil is not nil) | `drop-order` (including `let _ =`) |
| 03 | `slice-aliasing` (when append shares, when it copies) | `iterator-laziness` |
| 04 | `defer-order` (LIFO, arg evaluation, named returns) | `integer-overflow` (wrap, saturate, `as`) |
| 05 | `strings-and-runes` (bytes vs runes vs characters) | `string-vs-str` (byte-indexed slicing) |
| 06 | `channels-and-close` (closed channels never block) | `move-and-rc` (watching the refcount) |

| # | Python | TypeScript |
|---|--------|------------|
| 01 | `mutable-default-arg` (evaluated once, at `def`) | `number-precision` (everything is a double) |
| 02 | `late-binding-closures` | `array-sort-and-holes` (lexicographic by default) |
| 03 | `identity-vs-equality` (`is` asks the allocator) | `equality-coercion` |
| 04 | `aliasing-and-copies` (`[[0]*3]*3`) | `this-binding` |
| 05 | `generators-and-exhaustion` (what `zip` eats) | `types-are-erased` |
| 06 | `class-attrs-and-mro` (reads walk, writes do not) | `async-ordering` (microtasks before timers) |

The cross-language pairings are the real payload. Aliasing shows up as Go slice
capacity, Python list multiplication, and Rust move semantics; the same missing
model of value versus reference produces all three. Overflow is a Rust API
choice, a TypeScript silent precision loss, and in Go a compile error you never
see. Predicting them within an hour of each other is what makes the connection
stick.

## Validating the corpus

`lr check` validates *you*. `lr verify` validates the *snippets*, and it is the
one that runs in CI:

```bash
$ bin/lr verify go
OK    go/01-loop-var-capture (4 lines, stable over 3 runs)
OK    go/02-nil-interface (7 lines, stable over 3 runs)
...
```

For each snippet it builds the language's targets, then runs the program three
times and requires that it exits zero, prints something, and prints the identical
something every time. A snippet that fails any of those is unusable as an
exercise no matter how good the lesson inside it is.

It deliberately prints no snippet output, only line counts, so you can run it on
a fresh checkout without spoiling a single exercise.
