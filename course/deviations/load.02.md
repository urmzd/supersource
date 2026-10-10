# load.02 deviations

- Verify check 9 (call site) still fails: `used_by` is empty, and no registered module calls `go/loadgen/compare`. Every user of the gate goes through the learner's `{loadgen} compare` entry point (D16), which does not count as a call site:
  - `dep.05`'s `perf-gate` job runs `go/bin/loadgen compare <base> head.json --metric ttft_p95 --max-regress 5%`. Its check reads the workflow statically and calls no load.02 code, so load.02 stays in `dep.05`'s `reading` (B95-04). Moving it to `deps` would only satisfy the mirror rule on paper.
  - `ops.07` (drill) and `craft.16` (practice, docs) run the same command under `git bisect run`.
  - No library module needs an unpaired two-sample gate over load reports. `ag.12`'s A/B experiments are paired and use a sign-permutation test over eval scores. `dur.12`'s canary gate reads PromQL burn rates, not load reports.
- What unblocks it: a library unit owned by a later Go module that takes two `loadgen.Report`s and acts on the verdict, for example a `dur.12` canary activity that runs the load generator against the stable and canary pools and calls `compare.Compare`. The module then lists load.02 in `deps`, and load.02's `used_by` gains it. Adding that unit is a design change for its owner, not a wiring fix, so it is not done here.
