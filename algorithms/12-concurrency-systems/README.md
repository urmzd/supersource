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

---

# Concurrency & Systems Patterns

## Core Insight

Concurrency bugs are **non-deterministic** — they happen under specific interleavings and are hard to reproduce. The key is to reason about **invariants** (what must always be true) and **critical sections** (what must execute atomically). Systems questions test whether you can build reliable software on unreliable foundations.

## Pattern 1: Mutex + Condition Variable (Producer-Consumer)

**When to use**: One or more threads produce data, others consume it, with a bounded buffer.

**C Implementation**:
```c
#include <pthread.h>

typedef struct {
    int buffer[CAPACITY];
    int count, in, out;
    pthread_mutex_t lock;
    pthread_cond_t not_full, not_empty;
} BoundedQueue;

void produce(BoundedQueue *q, int item) {
    pthread_mutex_lock(&q->lock);
    while (q->count == CAPACITY)
        pthread_cond_wait(&q->not_full, &q->lock);
    q->buffer[q->in] = item;
    q->in = (q->in + 1) % CAPACITY;
    q->count++;
    pthread_cond_signal(&q->not_empty);
    pthread_mutex_unlock(&q->lock);
}

int consume(BoundedQueue *q) {
    pthread_mutex_lock(&q->lock);
    while (q->count == 0)
        pthread_cond_wait(&q->not_empty, &q->lock);
    int item = q->buffer[q->out];
    q->out = (q->out + 1) % CAPACITY;
    q->count--;
    pthread_cond_signal(&q->not_full);
    pthread_mutex_unlock(&q->lock);
    return item;
}
```

**Critical rule**: Always use `while`, never `if`, before `pthread_cond_wait` — spurious wakeups are real.

**Interview problems**: Producer-Consumer, Bounded Buffer, Print in Order, Print FooBar Alternately

## Pattern 2: Reader-Writer Lock

**When to use**: Many readers, few writers. Readers can be concurrent; writers need exclusive access.

**Policy choices**:
- **Reader-preference**: Readers never block on other readers (writers may starve)
- **Writer-preference**: If a writer is waiting, no new readers start (readers may starve)
- **Fair**: FIFO order, no starvation

**C Implementation (writer-preference)**:
```c
typedef struct {
    int readers, writers, write_waiters;
    pthread_mutex_t lock;
    pthread_cond_t can_read, can_write;
} RWLock;

void read_lock(RWLock *rw) {
    pthread_mutex_lock(&rw->lock);
    while (rw->writers > 0 || rw->write_waiters > 0)
        pthread_cond_wait(&rw->can_read, &rw->lock);
    rw->readers++;
    pthread_mutex_unlock(&rw->lock);
}

void write_lock(RWLock *rw) {
    pthread_mutex_lock(&rw->lock);
    rw->write_waiters++;
    while (rw->readers > 0 || rw->writers > 0)
        pthread_cond_wait(&rw->can_write, &rw->lock);
    rw->write_waiters--;
    rw->writers++;
    pthread_mutex_unlock(&rw->lock);
}
```

## Pattern 3: Thread Pool

**When to use**: Amortize thread creation cost, limit concurrency.

**Architecture**:
```
[Task Queue] → [Worker Thread 1]
             → [Worker Thread 2]
             → [Worker Thread N]
```

Each worker loops: `lock → dequeue task → unlock → execute task → repeat`.

**Interview problems**: Design Thread Pool, Web Server Architecture, Rate Limiter

## Pattern 4: Lock-Free Data Structures (CAS)

**When to use**: High-contention scenarios where mutex overhead is too high.

**Compare-And-Swap (CAS)**:
```c
// Atomic: if *ptr == expected, set *ptr = desired, return true
bool cas(int *ptr, int expected, int desired) {
    return __sync_bool_compare_and_swap(ptr, expected, desired);
}

// Lock-free stack push
void push(Stack *s, Node *node) {
    do {
        node->next = s->top;
    } while (!cas(&s->top, node->next, node));
}
```

**ABA problem**: Value changes A→B→A, CAS thinks nothing changed. Fix with version counters or hazard pointers.

## Pattern 5: Deadlock Prevention

**Four necessary conditions** (Coffman): Mutual exclusion, Hold-and-wait, No preemption, Circular wait.

**Prevention strategies**:
1. **Lock ordering**: Always acquire locks in the same global order
2. **Try-lock with backoff**: `pthread_mutex_trylock`, release all and retry on failure
3. **Lock hierarchy**: Assign numerical levels, only acquire higher-level locks while holding lower

**Classic problems**: Dining Philosophers, Bank Transfer Deadlock

## Pattern 6: Memory Models and Ordering

**Sequential consistency**: All threads see operations in the same order.

**x86-TSO (Total Store Order)**: Stores can be delayed (store buffer), but loads see latest stores from same thread.

**Barriers/Fences**:
```c
__sync_synchronize();          // full memory barrier
__atomic_thread_fence(__ATOMIC_SEQ_CST);  // C11 sequentially consistent fence
```

**Relevance**: Critical for lock-free code and understanding why `volatile` isn't enough.

## Systems Design Patterns

### Rate Limiter
- **Token bucket**: Tokens added at fixed rate, each request consumes one
- **Sliding window**: Count requests in last N seconds using sorted set

### Distributed Locking
- **Redlock**: Acquire lock on majority of N Redis instances
- **Fencing tokens**: Monotonically increasing token prevents stale lock holders from writing

### Circuit Breaker
- States: CLOSED (normal) → OPEN (failing, reject all) → HALF-OPEN (test one request)
- Prevents cascading failures in microservices

## Company Targeting

| Company | Focus | Difficulty |
|---------|-------|------------|
| Anthropic | Producer-consumer, concurrent data structures, GPU coordination | Hard |
| Google | Lock-free structures, memory ordering | Hard |
| NVIDIA | GPU thread synchronization, warp-level primitives | Very Hard |
| Netflix | Thread pools, circuit breakers, rate limiters | Medium-Hard |
| HRT | Lock-free queues, nanosecond latency | Extreme |
| Citadel | Memory models, cache-line optimization | Very Hard |
| SpaceX | Real-time constraints, deterministic scheduling | Hard |

## Common Concurrency Bugs

| Bug | Cause | Fix |
|-----|-------|-----|
| Data race | Unprotected shared access | Mutex or atomic |
| Deadlock | Circular lock dependency | Lock ordering |
| Livelock | Threads keep retrying, no progress | Randomized backoff |
| Priority inversion | Low-priority thread holds lock needed by high-priority | Priority inheritance |
| Lost wakeup | Signal before wait | Always check predicate in while loop |
