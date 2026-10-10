# The Course: Build Your Own LLM System

You build one system, end to end, in four languages: a byte-level language model trained in Python on your own numerics, kernels in C behind a stable ABI, a Rust inference engine that streams completions over HTTP, a Go gateway in front of it, and the containers, charts, traces, and runbooks that keep it running on Kubernetes. Every layer is yours. The course grades it with tests you can read, milestones that run your entry points, and drills that break your deployment on purpose.

**For**: engineers who want to own an LLM system from the matrix multiply to the pager, not just call one.

**How it works**: each chapter teaches one module from first principles (the math, a worked example by hand, the interface, the pitfalls), then you write the code in your own repo and `ss check <ID>` grades it. A pass ends at a milestone that runs your whole system as it stands: after every pass it runs end to end ([the spiral](SYSTEM.md#the-spiral)).

## Start

```bash
practice/bin/ss doctor                         # the tools pass 0 and pass 1 need
practice/bin/ss course init --name forge       # your repo (default .scratchpad/course; SS_COURSE_HOME moves it)
practice/bin/ss learn course                   # every stage and your progress
practice/bin/ss learn course next              # read the next unfinished stage
practice/bin/ss start lang.01                  # stub a module into your repo
practice/bin/ss check lang.01                  # grade it
```

`ss learn course --done <stage>` marks a stage done only when its check passes (`[x]`), or with `--force` (`[~]`). The fifth column of [`path.tsv`](path.tsv) names each stage's check.

## Passes

| Pass | Path | What your system does after it | Gate |
|---|---|---|---|
| P0 | [course-p00-setup](../course-p00-setup/) | an empty repo whose CI lints commits, runs native tests, and runs `ss check --all --ci` | MS-P0 |
| P1 | [course-p01-tracer](../course-p01-tracer/) | a byte bigram trained by your CLI, served by your Rust engine, behind your Go gateway, on kind, with one trace and one runbook | MS-P1 |

Passes 2 to 11 (autograd, tokenizers and data, sequence models, the transformer, inference and kernels, the serving platform, durable workflows, the capstone training run, agents, operations) arrive batch by batch; their pass paths are added to [`path.tsv`](path.tsv) as they land.

## Reading order

- [SYSTEM.md](SYSTEM.md): the components, their languages, the contracts between them, and how the system grows pass by pass. Read it first.
- Each pass path's README says what that pass builds; its `milestone.md` says what the gate runs.
- Contracts live in [`course/contracts/`](../../course/contracts/) and are vendored into your repo's `contracts/` by `ss course init`.
