# C++ 07: Coroutine generator

**Concepts:** C++20 coroutines, `co_yield`, lazy sequences
**Difficulty:** ⭐⭐⭐⭐

A `Generator<T>` that turns any function containing `co_yield` into a lazy,
composable sequence.

## The contract

C++20 provides no coroutine *type*. It provides a protocol: write a function
containing `co_yield`, and the compiler looks at the declared return type, finds
its nested `promise_type`, and calls specific members at fixed points. Supplying
those members is the exercise.

| Member | Called |
|--------|--------|
| `get_return_object()` | Once, to build what the caller receives |
| `initial_suspend()` | Before the body's first statement |
| `yield_value(v)` | By each `co_yield` |
| `final_suspend()` | After the body finishes |
| `unhandled_exception()` | If the body throws |
| `~Generator()` | Yours: must `destroy()` the frame |

## What to notice

**`initial_suspend` returning `suspend_always` is what makes infinite
generators possible.** With `suspend_never` the body runs to its first
`co_yield` during construction, so `naturals()` hangs before the constructor
returns. Suspending immediately means nothing executes until the consumer asks,
which is the difference between a lazy sequence and an eager one that happens to
be spelled with `co_yield`.

**`final_suspend` returning `suspend_always` is what keeps the handle usable.**
Return `suspend_never` and the frame is destroyed the instant the body finishes,
so every later `handle_.done()` is a use-after-free. Suspending at the end means
the finished coroutine still owns its frame and your destructor is what releases
it.

**The frame is heap-allocated and uniquely owned.** That is why `Generator` is
move-only with a `destroy()` in the destructor: it is a `unique_ptr` wearing a
different name. `test_abandonment_frees_the_frame` drops a thousand suspended
generators, which under AddressSanitizer is a leak check.

**Stash the exception, do not let it escape.** `unhandled_exception` is called
*inside* the coroutine, where there is no consumer stack to unwind into.
Capturing it with `std::current_exception()` and rethrowing on the consumer's
next `next()` makes a throwing generator behave like a throwing loop, which is
what a caller expects.

**Interleaving is real, not batching.** `test_interleaving_is_real` records a
log from both sides and asserts the exact alternation. A generator that ran the
body to completion and buffered the results would produce every generator line
before every consumer line, and pass a naive values-only test.

**Only an input iterator is honest here.** A single-pass lazy sequence cannot
provide a forward iterator: you cannot revisit an element, and copying the
position does not copy the state. The iterator compares against `nullptr` for
end rather than tracking an index, because there is no length to compare to.

**Composition is where laziness pays.** `take(squares_of(naturals()), 5)`
evaluates exactly five elements and builds no intermediate container, and each
stage is an ordinary function with a loop in it.

## Extending it

Try `co_await` on a generator to flatten nested sequences, which needs an
awaiter type and is how recursive generators avoid O(depth) resumption cost.
Then compare with `std::generator` in C++23, which standardises this exercise
and adds exactly that recursive-yield support.
