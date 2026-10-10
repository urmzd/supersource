## ds.08: optional Rust Bloom filter

`ds.08` is now `kind = "side"`. Its only call site was `data.03`, through the
PyO3 class `tinyllm_rs.Bloom`. Under the no-FFI design `data.03` is Python
with its own `_Bloom` screen (see `data.03.md`), and no Rust module on the
serving path needs a Bloom filter (D21 dropped the prefix Bloom in
heartbeats; it lives on as `sq.prefix-bloom`). A `build` module needs a real
call site (P1), so the module is optional, like the standalone C data
structures. The Rust filter and the Python screen still meet over files: the
`bloom` parity suite compares both with the golden bit arrays of
`formats/bloom.md`. The course tests, mutants, and the R4 learner-test rung
are unchanged.

Open: `MS-corpus` and `MS-P3` still list `ds.08` in `requires`, so an
optional module gates those milestones. Dropping it from both lists (and
moving its `milestone` to one that exercises it) is a shared milestone edit.
