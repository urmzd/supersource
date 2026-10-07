# C++ 05: Thread-safe queue

**Concepts:** `std::mutex`, `std::condition_variable`, move semantics
**Difficulty:** ⭐⭐⭐

A bounded blocking queue: producers wait when it is full, consumers wait when it
is empty, and closing it wakes everyone.

This exercise ships a `run.sh` because it needs `-pthread`.

## The contract

| Member | Does |
|--------|------|
| `push(v)` | Block until there is room. False if closed |
| `try_push(v)` | Never blocks. False if full or closed |
| `pop()` | Block until an element exists. `nullopt` once closed *and* drained |
| `try_pop()` | Never blocks |
| `close()` | Wake every waiter, refuse new pushes, keep delivering the backlog |

## What to notice

**RAII is the whole difference from the C version.** There is no path through
this file where a lock is acquired and not released, including the paths where
an exception is thrown, because `unique_lock`'s destructor runs during stack
unwinding. Compare with the C thread pool, where every early return has to
remember to unlock, and one that forgets deadlocks the process.

**Use the predicate form of `wait`.** `cv.wait(lock, pred)` is exactly
`while (!pred()) cv.wait(lock);` and the loop is mandatory: a bare `wait` can
return spuriously, and several waiters can be woken by one `notify` with only
the first to reacquire the mutex finding the condition true. Writing `if`
instead of `while` produces a queue that works under light load and corrupts
under real load.

**Unlock before notifying.** If you notify while still holding the mutex, the
woken thread immediately blocks trying to acquire it, costing a second context
switch for nothing. This is sometimes called the "hurry up and wait"
pessimisation. It is a performance bug, not a correctness one, which is exactly
why it survives code review.

**`notify_all` on close, never `notify_one`.** Every blocked thread must wake to
observe the closure. Waking one leaves the others parked forever, and any
`join` on them deadlocks. `test_close_wakes_blocked_consumers` blocks four
consumers and joins them, so a `notify_one` implementation hangs the test rather
than failing it, which is its own kind of clear signal.

**Closed does not mean empty.** A producer whose `push` returned `true` is
entitled to have that element consumed. So `pop` must drain the backlog before
it starts reporting `nullopt`, which means the check is "empty" first and
"closed" second, not the other way round.

**`mutable std::mutex` is not a hack.** `size()` and `closed()` are logically
const but must lock, and `mutable` is the standard way to say "this member is
not part of the object's observable state". A queue whose observers are not
`const` is painful to use.

**The queue is neither copyable nor movable, and it should not be.** A mutex is
neither, and a queue that could be moved out from under a waiting thread would
be a data race by construction. Deleting the operations says so in the type.

**Nothing asserts on cross-thread ordering.** With four producers and four
consumers, the interleaving is genuinely nondeterministic and asserting on it
would make a flaky exercise. What *is* deterministic is that every item is
consumed exactly once, which the test checks with a per-item atomic counter.

## Verifying it yourself

```bash
c++ -std=c++20 -Wall -Wextra -pthread -fsanitize=thread -O1 -o /tmp/q main.cpp && /tmp/q
```

Passing once means little for concurrent code; the suite runs three times for
the same reason.

## Extending it

Add `pop_for(timeout)` using `wait_for`, and notice that the predicate overload
handles the spurious-wakeup-plus-deadline interaction that hand-rolled timeout
loops almost always get wrong. Then try replacing the `std::queue` with a fixed
ring buffer to remove the per-element allocation, which is the change that makes
this competitive with a lock-free queue for small payloads.
