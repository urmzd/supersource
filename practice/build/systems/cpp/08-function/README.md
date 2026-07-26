# C++ 08: `std::function` clone

**Concepts:** type erasure, small buffer optimisation, manual vtables
**Difficulty:** ⭐⭐⭐⭐

A `Function<R(Args...)>` that stores any compatible callable behind one type,
keeping small ones inline and spilling large ones to the heap.

## The contract

| Member | Does |
|--------|------|
| `Function(F&& f)` | Erase any compatible callable |
| `operator()(args...)` | Invoke, or throw `bad_function_call` if empty |
| `operator bool()` | Whether it holds anything |
| `uses_inline_storage()` | Exposed so the tests can prove the SBO works |
| Move ctor / assign | Transfer, handling both storage strategies |

Copy operations are deliberately deleted; see below.

## What to notice

**Two independent ideas share one class.** Type erasure is about storing
unrelated types behind one interface. The small buffer is purely a performance
trick. You could have either without the other, and conflating them is what
makes this class confusing to read.

**The vtable is written by hand, and it has to be.** Inheritance would work and
would force every callable onto the heap: a base subobject cannot live in a
buffer whose type the base does not know. A struct of function pointers, built
per callable type by `make_vtable<Callable, Inline>()`, gives the same dispatch
with the storage decision left free.

**The `enable_if` on the converting constructor is not optional.** Without it,
`Function(F&&)` is a *better* match for a non-const `Function&` than the copy
constructor is, so copying a Function recursively wraps it in itself until the
compiler gives up. The guard excludes `std::decay_t<F> == Function`, and
everyone who writes this class hits the bug once.

**A callable goes inline only if it fits *and* its move cannot throw.** Size
alone is not enough. Moving the `Function` has to relocate an inline callable,
and if that move can throw there is no valid state to leave behind. The heap
path has no such problem, because moving is then just copying a pointer, which
is also why the move constructor can be `noexcept` for every callable.

**Relocating an inline callable means calling its move constructor, not
copying bytes.** A callable may hold a pointer into itself, and even when it
does not, `memcpy` on a non-trivially-copyable type is undefined. That is what
`move_to` in the vtable is for.

**Destruction must mirror construction exactly.** An inline callable was
placement-new'd, so only its destructor runs. A heap one came from a
new-expression and needs the matching delete-expression, which also selects the
correctly aligned `operator delete` for over-aligned types. Splitting that
decision between the vtable and the caller is how you get a mismatched
deallocation that works until someone stores an aligned lambda.

**Copying is deleted here, and `std::function` pays a real price for
supporting it.** A copyable erasure needs a fourth vtable entry and forces
every stored callable to be copyable, which is why
`std::function<void()> f = [p = std::make_unique<int>(1)]{};` does not compile.
C++23 added `std::move_only_function` for precisely this reason.
`test_move_only_capture` shows what that buys.

## Extending it

Add copy support and watch the constraint propagate: the converting constructor
now needs `is_copy_constructible`, and every test involving a move-only capture
stops compiling. Then measure the buffer size against real code, because 32
bytes is a guess and the standard library implementations disagree with each
other about the right answer.
