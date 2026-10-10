# tests

## Summary
- Test harness and fixtures for the miner.
- Inputs live in `test.<id>.in`, expected outputs in `test.<id>.expected`, and test metadata in
  `test.<id>.cfg` / `test.<id>.desc`.

## Key takeaways
- The harness filters nondeterministic lines (e.g., `Thread` and `Received`) before comparing.
- RANDOM tests validate concurrency by requiring output differences across runs.

## How to run
- From `12-concurrency-systems`: `./tests/test.sh 00`
- For all tests: iterate over `tests/test.*.cfg` and call `./tests/test.sh <id>`
