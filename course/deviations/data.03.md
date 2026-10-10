## data.03: Bloom mutants target the local Python screen

The dedup stage now uses its private `_Bloom` implementation. The empty-input
and document-count sizing mutants were retargeted to that class's constructor,
so they still exercise the intended sizing boundaries without a Rust binding.
