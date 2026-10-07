# C++ 02: `UniquePtr<T>`

**Concepts:** RAII, move-only types, custom deleters
**Difficulty:** ⭐⭐

A `std::unique_ptr` work-alike: exactly one owner, no copying, deletion only in
the destructor.

## The contract

| Member | Does |
|--------|------|
| `release()` | Give up ownership *without* deleting |
| `reset(p)` | Destroy what is owned, adopt `p` |
| `swap(other)` | Exchange pointers and deleters |
| Move ctor / move assign | Transfer ownership, empty the source |
| Copy ctor / copy assign | Deleted, and that is the point |
| `~UniquePtr()` | The only place `delete` appears |

An array specialisation and `make_unique_ptr` are also provided.

## What to notice

**Deleting the copy operations is the design.** A copyable owning pointer is a
double free waiting for a second scope exit. Writing `= delete` moves the
invariant out of the documentation and into the compiler, and the tests assert
it with `static_assert` rather than at runtime, because a violation must fail
to *build*.

**`release` and `reset` differ in exactly one thing and it is the important
one.** `release` hands the raw pointer back and deletes nothing, transferring
responsibility to the caller. `reset` destroys what is held. Confusing them
gives you either a leak or a double free, and both compile perfectly.

**Order matters in `reset`: overwrite the member, then delete the old value.**
Deleting first means that if `~T()` reaches back into this `UniquePtr`, it
observes a pointer to memory that is already gone. Doing it in the safe order
also makes self-reset fall out for free, which `test_self_reset` pins.

**Self-move-assignment is the bug most implementations have.** The obvious
`reset(other.release())` is correct, but `reset(nullptr); ptr_ = other.ptr_;`
is not: on `p = std::move(p)` it destroys the object and then adopts the
dangling pointer. The standard only promises "valid but unspecified" for
moved-from objects, but destroying the thing you are about to keep is never
acceptable.

**Move operations must be `noexcept`, and there is a static_assert for it.**
Containers check that trait: a `std::vector<UniquePtr<T>>` that cannot move
without risk of throwing will *copy* on reallocation instead, and copying is
deleted, so the code stops compiling. That is the same `move_if_noexcept`
machinery as [exercise 01](../01-vector/), seen from the other side.

**`operator bool` is explicit for a reason.** Without `explicit`, `if (p)`
still works, and so do `int x = p;`, `p + 1`, and comparing two unrelated
`UniquePtr`s for equality by accident. The explicit form permits exactly the
contextual conversions you want and rejects the rest.

**The array specialisation exists because `delete` and `delete[]` are not
interchangeable.** Calling the wrong one is undefined behaviour that usually
appears to work: `test_array_specialisation` catches it by counting
destructors, where plain `delete` on an array of five runs one destructor
instead of five.

**`make_unique_ptr` is not just convenience.** In `f(UniquePtr<T>(new T), g())`,
compilers were once permitted to evaluate `new T`, then `g()`, then the
constructor. If `g()` threw, the `T` leaked. C++17 tightened the rules, but the
factory function remains the habit worth having.

## Extending it

Implement `SharedPtr` with a control block, and notice how much you gave up:
an atomic refcount on every copy, a second allocation unless you use the
`make_shared` trick, and cycles that never free. Then implement `WeakPtr` and
see why the control block needs two counts rather than one.
