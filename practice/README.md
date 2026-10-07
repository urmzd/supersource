# Practice

The rest of this repository is the **learning path**: tracks you read, in
order, with a textbook behind each topic. This directory is the **practice
path**. Nothing here teaches you anything. Everything here tells you whether
you actually know it.

The two are not interchangeable and they fail differently. Reading produces
recognition, which feels like knowledge and is not. Practice produces a verdict,
which is uncomfortable and is the only thing that transfers.

```bash
practice/bin/ss where          # what exists, and where it lives
practice/bin/ss list           # every exercise, and which ones you have started
```

## Three kinds of exercise

They differ in what you produce, and that is the only thing they differ in. One
CLI drives all three with the same verbs.

| Kind | You produce | It is wrong when |
|------|-------------|------------------|
| [`predict`](predict/) | The exact output you expect from a snippet | Your prediction differs from what ran |
| [`build`](build/) | An implementation from scratch | The exercise's own assertions fail |
| [`reattempt`](reattempt/) | A second solution to something you already solved | The original's assertions fail |

`predict` finds holes fastest, because committing to an output is cheap and the
correction is precise. `build` finds different holes, the ones that only appear
when you have to make every decision yourself. `reattempt` is the honest one:
code you wrote before, handed back with the solution removed, which is how you
find out whether you learned it or just finished it.

## The loop

```bash
ss start build rust 01     # clone it into .scratchpad/, stubbed, and open it
ss check build rust 01     # run it; the exit code is the verdict
ss diff  build rust 01     # your attempt against the reference
```

`ss start` never touches the repository. It copies the exercise into
`.scratchpad/` at the repo root, which is gitignored in its entirety, and
strips out the parts you are supposed to write. Your attempts are yours: they
are never committed, and nothing in the repo changes as you work.

Add `practice/bin` to your `PATH` and the `practice/bin/` prefix goes away.

| Command | Does |
|---------|------|
| `ss start <kind> <lang> <id>` | Clone the exercise into `.scratchpad/` and open it |
| `ss check <kind> <lang> [id]` | Run your attempt. Exit code is the verdict |
| `ss diff <kind> <lang> <id>` | Your attempt against the reference |
| `ss list [kind] [lang]` | What exists, and what you have started |
| `ss score [kind] [lang]` | Passing, failing, not attempted |
| `ss show <kind> <lang> <id>` | Print the reference. Spoils the exercise |
| `ss reveal <kind> <lang> <id>` | Run the reference. Also spoils it |
| `ss reset <kind> <lang> <id>` | Throw your attempt away and start cold |
| `ss where` | Paths and exercise counts |
| `ss verify [kind] [lang]` | Maintainer check, described below |
| `ss bench [lang] [--assert]` | Cost of each predict snippet against `predict/budgets.tsv` |

`<id>` is the numeric prefix, so `ss check build c 02` is enough. Languages
accept the spelling you would expect: `ts` and `typescript` both work.

## How an exercise is stored

One committed source of truth per exercise, never a reference and a separate
stub that drift apart. The reference marks the regions you are meant to write,
and `ss start` replaces each marked region with a `TODO` on the way into your
scratchpad.

That means the same file is the thing you diff against, the thing CI runs, and
the thing your stub is generated from. Tests, headers, and scaffolding carry no
markers, so they arrive intact and you cannot accidentally edit a test into
passing.

Every `build` exercise is self-testing: one entry point, its own assertions,
nonzero exit on the first failure. No test framework in any of the nine
languages, which is why `ss check` is the same command everywhere. The
[case studies](../case-studies/) already worked this way and CI already depended
on it.

## Build exercises

Ten per language, ordered by concept rather than by difficulty, so read the
stars and not the numbers.

Standard library only unless an exercise says otherwise. The point is to write
the thing, not to find the crate that already did.

### Core

The languages this curriculum actually targets. `ss verify build` with no
argument covers exactly these, and so does CI.

| Language | Focus | References |
|----------|-------|-----------|
| [Go](build/cloud/go/) | Concurrency, channels, interfaces | 0/10 |
| [Rust](build/systems/rust/) | Ownership, lifetimes, zero-cost abstractions | 0/10 |
| [Python](build/general/python/) | Data model, metaprogramming, asyncio | 0/10 |
| [TypeScript](build/general/typescript/) | Type narrowing, generics, runtime safety | 0/10 |
| [C](build/systems/c/) | Manual memory, pointers, undefined behaviour | 10/10 |
| [C++](build/systems/cpp/) | RAII, templates, move semantics | 10/10 |

### Optional

Kept because the exercises are worth doing, not because the languages are in
use here. Every command works on them by name, but they are left out of the
bare fan-out so a toolchain you have not installed is never a failure in a run
you did not ask for.

| Language | Focus | Toolchain |
|----------|-------|-----------|
| [Zig](build/systems/zig/) | Comptime, explicit allocators, C interop | `brew install zig` |
| [Scala](build/cloud/scala/) | FP plus OOP, type system, given instances | `brew install scala-cli` |
| [Java](build/cloud/java/) | JVM internals, virtual threads, generics | `mise` plus a `.mise.toml` pin |

On a chezmoi-managed machine all three come from the `install_alt_langs` flag,
which is off by default.

```bash
ss verify build              # core only
ss verify build zig          # by name, and skips cleanly if zig is absent
ss where                     # which languages are core, optional, or missing a toolchain
```

**Reference implementations are being filled in incrementally.** `ss list build`
shows what is ready right now; an exercise with no directory yet is a table row
in its language README and nothing more.

## Predict exercises

Twenty-four snippets across Go, Rust, Python, and TypeScript, six each, every
one aimed at a belief that is common, load-bearing, and wrong. Full detail in
[`predict/README.md`](predict/README.md), including why the cross-language
pairings are the real payload.

Two rules make it work: commit before you run, and log every miss in
[`predict/LEDGER.md`](predict/LEDGER.md). A prediction edited after seeing the
output is worth nothing, and a miss you do not write down is one you will
repeat.

## Verifying the exercises

`ss check` validates you. `ss verify` validates the exercises, and it is what
runs in CI:

| Kind | Verified by |
|------|-------------|
| `predict` | Every snippet builds, exits zero, prints something, and prints the identical something three runs running |
| `build` | Every reference passes its own assertions and carries markers so it can be stubbed |
| `reattempt` | Every manifest row points at a file that exists, carries markers, and still runs clean |

It prints no exercise output, only shapes and counts, so it is safe to run on a
fresh checkout without spoiling a single exercise.

```bash
just run-predict           # ss verify predict
just run-build             # ss verify build, for every installed toolchain
just run-predict-bench     # ss bench --assert
```

## Picking something

Start with `predict`. It costs nothing to set up, the corpus is complete, and
the diff between what you expected and what happened tells you which `build`
exercise is worth your time. Then use the patterns in your ledger to choose the
language: three entries about aliasing in three languages is not three misses,
it is one missing model of value versus reference semantics, and that is a
`build` exercise waiting to happen.
