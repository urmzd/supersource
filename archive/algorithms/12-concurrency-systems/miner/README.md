# miner

## Summary
- C implementation of a threaded miner simulation.
- Core modules: `reader.*` (input thread), `event.*` (event queue), `memory_pool.*`, `transactions.*`,
  `block.*`, `nonce.*`, `siggen.*`, `main.c`.

## Key takeaways
- The reader thread ingests events and feeds a queue protected by a mutex/condition.
- Mining logic composes blocks, transactions, and signatures from events.

## How to run
- Build: `make` (from this folder).
- Execute: `./miner` and provide input on stdin (use the test inputs in `../tests`).
