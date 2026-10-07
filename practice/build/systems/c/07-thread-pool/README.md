# C 07: Thread pool

**Concepts:** pthreads, condition variables, bounded work queue
**Difficulty:** ⭐⭐⭐⭐

A fixed set of worker threads pulling from a bounded queue, with backpressure on
the producer and a clean shutdown that finishes what it accepted.

## The contract

`pool.h` declares it, `main.c` tests it, you write the marked regions of
`pool.c`. The struct, `pool_new`, and `pool_completed` are given; you write the
worker loop, `pool_submit`, `pool_wait`, and `pool_destroy`.

| Function | Does |
|----------|------|
| `pool_submit(p, fn, arg)` | Enqueue, blocking while the queue is full |
| `pool_wait(p)` | Block until everything submitted so far has finished |
| `pool_destroy(p)` | Stop accepting, drain the backlog, join, free |

This exercise ships a `run.sh` because it needs `-pthread`, which the harness's
default C compile line does not pass.

## Determinism

The order tasks run in is genuinely nondeterministic and nothing asserts on it.
What is deterministic, and what the tests check, is that every submitted task
runs *exactly once*. Most tests give each task its own slot to write, so a lost
or duplicated run is visible with no locking in the test itself. The whole suite
runs three times, because a race that appears one run in ten is still a bug.

## What to notice

**Wait in a `while`, never an `if`.** `pthread_cond_wait` can return without the
condition holding: spuriously, or because several workers were woken for one
signal and only the first to reacquire the lock got the task. Re-testing the
predicate after waking is not defensive programming, it is the required usage,
and an `if` here produces a pool that works until it is loaded.

**Run the task outside the lock.** Holding the mutex while calling `t.fn(t.arg)`
serialises every worker down to one, which is the most common way to write a
thread pool that is slower than a plain loop and still passes all its tests. The
lock protects the queue, not the work.

**`pool_wait` has to account for in-flight tasks, not just an empty queue.** A
task that has been dequeued is no longer counted by the queue and has not
finished either. That is what the `active` counter is for, and why the idle
signal fires only when `count == 0 && active == 0`.
`test_wait_covers_in_flight_work` uses tasks slow enough that a queue-only wait
returns early and sees an incomplete total.

**Broadcast on shutdown, do not signal.** Every worker is parked on `not_empty`,
and every one of them must wake to notice the shutdown flag and exit. A signal
wakes exactly one, leaving the rest blocked forever and `pthread_join` hanging.
This is the classic thread-pool deadlock and it only shows up with more than one
worker.

**Shutdown means "no new work", not "abandon the backlog".** A `pool_submit`
that returned 0 is a promise the task will run. `pool_destroy` must therefore
drain the queue before workers exit, which is why the worker's exit test is
`count == 0 && shutting_down` rather than `shutting_down` alone.

**The bounded queue is a design choice with a name.** Blocking the producer when
the queue is full is backpressure: the system slows the source rather than
growing an unbounded backlog until memory runs out. An unbounded queue makes
`submit` never block and turns a sustained overload into an OOM kill instead of
a slowdown.

## Verifying it yourself

Passing once means little for concurrent code. Run it under ThreadSanitizer,
which finds data races the tests cannot:

```bash
cc -std=c11 -Wall -Wextra -pthread -fsanitize=thread -O1 -o /tmp/pool ./*.c && /tmp/pool
```

## Extending it

Give each worker its own deque and let idle workers steal from the back of a
busy one's. Work stealing is what Go's scheduler, Rayon, and Java's
`ForkJoinPool` all do, and the reason is contention: one shared queue means every
worker touches the same cache line on every task, which stops scaling long
before the cores run out.
