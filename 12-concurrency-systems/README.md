# 12-concurrency-systems

## Summary
- Contains a C-based miner simulation in `miner/` and a test harness in `tests/`.
- The miner uses pthreads to handle concurrent input and processing.

## Key takeaways
- Thread-safe queues and condition variables coordinate producer/consumer behavior.
- Deterministic outputs are verified by stripping nondeterministic lines in tests.

## How to run
- Build: `make -C 12-concurrency-systems/miner`
- Run a single test: `cd 12-concurrency-systems && ./tests/test.sh 00`
- Run all tests: loop over `tests/test.*.cfg` and call `./tests/test.sh <id>`
