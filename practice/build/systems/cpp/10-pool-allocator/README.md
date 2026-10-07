# C++ 10: Pool allocator

**Concepts:** custom allocators, intrusive free lists, alignment
**Difficulty:** ⭐⭐⭐⭐

A fixed-size-block allocator, plus the `std::allocator`-shaped adaptor that
lets `std::list` and `std::map` use it.

## The contract

| Member | Does |
|--------|------|
| `Pool::allocate()` | Pop a block off the free list |
| `Pool::deallocate(p)` | Push it back |
| `Pool::release()` | Return every chunk to the system |
| `PoolAllocator<T>` | The standard-library adaptor |

## What to notice

**Handling one size collapses the entire problem.** A general allocator must
search for a fit, split blocks, coalesce neighbours on free, and store a header
per block so `free` knows the size. A pool does none of that: every block is
interchangeable, so allocation is "pop the head" and deallocation is "push the
head". Two pointer writes, no search, no metadata, and fragmentation is
impossible by construction.

**The free list is intrusive, and that is why the overhead is zero.** The
"next" pointer lives *inside* the free block's own storage, because a free
block's contents are by definition not being used for anything else. A block is
either holding your object or holding a pointer, never both. This is the same
idea as the [C intrusive list](../../c/03-intrusive-list/), applied to memory
that is not currently an object.

**Which is why blocks are at least pointer-sized.** Ask for a pool of 1-byte
blocks and the free list has nowhere to store its link. The constructor rounds
up, and the test asserts it.

**An allocator is a handle, not an owner.** This is the mistake almost everyone
makes first: putting the `Pool` *inside* `PoolAllocator`. Containers copy their
allocator freely, rebind it to their internal node type, and compare copies for
equality, so an allocator must be cheap to copy and two copies must be
interchangeable. Holding a `Pool*` satisfies all of that;
`test_allocator_is_a_handle` frees through a different copy than it allocated
from, which only works if that is true.

**Rebinding is why a `std::list<int>` allocator is never asked for `int`s.**
The container needs nodes, which hold your value *plus* two pointers, so it
rebinds `PoolAllocator<int>` to `PoolAllocator<ListNode<int>>` via the
templated converting constructor. You cannot know the node size portably, which
is why the tests size the pool generously and rely on the fallback.

**Falling back is not an admission of defeat.** `std::vector` asks for one
allocation of *n* elements, which a fixed-size pool fundamentally cannot serve.
Returning a too-small block would be memory corruption; forwarding to
`::operator new` is correct and keeps the allocator usable with any container.

**Node-based containers are the ones that benefit.** `std::list` and
`std::map` do exactly one fixed-size allocation per element, which is the shape
a pool serves perfectly. `std::vector` does one growing allocation and gains
nothing. Knowing which of your containers is which is the practical takeaway.

**Building the free list backwards is a deliberate small thing.** Threading a
new chunk from the last block to the first leaves the list in ascending address
order, so consuming blocks in sequence walks memory forwards and the
prefetcher can help. Threading it forwards works identically and traverses
backwards.

## Extending it

Add a debug mode that poisons freed blocks with a recognisable byte pattern and
checks it on allocation, which turns use-after-free from silently plausible
data into an immediate assertion. Then try making the pool thread-safe with a
lock-free free list using `compare_exchange` and meet the ABA problem, which the
[Rust lock-free stack exercise](../../rust/) covers from the other direction.
