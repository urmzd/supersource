# C++ 01: `Vector<T>`

**Concepts:** templates, move semantics, manual object lifetime
**Difficulty:** ⭐⭐⭐

A `std::vector` work-alike. The interface is given in `vector.hpp`; you write
the marked regions. `main.cpp` tests it.

## The contract

| Member | Does |
|--------|------|
| `reserve(n)` | Raise capacity, never lower it |
| `emplace_back(args...)` | Construct in place from the arguments |
| `push_back(v)` | Copy or move one element in |
| `pop_back()` | Destroy the last element, keep the capacity |
| `clear()` | Destroy all elements, keep the capacity |
| The rule of five | Copy/move construct, copy/move assign, destructor |

## What to notice

**Allocation and construction are two different things, and this is the whole
exercise.** A vector holds `capacity` worth of *raw storage* and constructs
objects only in the first `size` slots. Reach for `new T[n]` and you have
default-constructed every spare slot: `Vector<T>` then requires `T` to be
default-constructible, wastes work, and destroys objects nobody created.
`test_spare_capacity_holds_no_objects` is what catches it, and a `Vector<int>`
never could.

**Increment `size_` *after* the constructor returns, not before.** If `T`'s
constructor throws, that element never existed, and a size already bumped means
`clear()` and the destructor will run a destructor on raw memory.

**`std::move_if_noexcept` in `reserve` is not a micro-optimisation.** If `T`'s
move constructor can throw, moving elements into the new buffer can fail
halfway, with the originals already destroyed and no way back. Copying instead
keeps the source intact until the new buffer is complete. This is precisely why
marking a move constructor `noexcept` makes vectors of that type measurably
faster: without it, every reallocation silently degrades to copying.

**Copy-and-swap gets self-assignment and exception safety in one move.** The
naive copy assignment destroys the contents and then copies from the source,
which on `v = v` reads objects it has already destroyed. Building the copy
first means the operation either fully succeeds or leaves `*this` untouched.

**A moved-from object must be valid, not merely destructible.** The standard
requires "valid but unspecified", and the test pushes onto a moved-from vector
to prove it. Leaving stale pointers in the source and relying on nobody
touching it works right up until someone reuses the variable.

**The tests count objects, not values.** `Tracked` records every construction,
destruction, copy, and move, which is how a test can assert that a move copied
*nothing* and that a copy really copied ten elements. Most vector bugs are
invisible to value-based assertions and obvious to lifetime-based ones.

## Extending it

Add `insert` and `erase` in the middle, which force you to think about
iterator invalidation and about shifting a range of objects using
move-assignment rather than construction. Then add an `Allocator` template
parameter and discover why `std::vector`'s real signature is so much more
complicated than this one.
