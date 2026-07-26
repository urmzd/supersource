# C 03: Intrusive linked list

**Concepts:** container-of macro, pointer manipulation
**Difficulty:** ⭐⭐

A circular doubly-linked list in the style the Linux kernel uses, where the node
lives *inside* your struct instead of pointing at it. `container_of` and the
iteration macros are given to you; you write the list operations.

## The contract

`list.h` declares it, `main.c` tests it, you write `list.c`.

| Function | Does |
|----------|------|
| `list_init(head)` | Point the sentinel at itself |
| `list_empty(head)` | True when only the sentinel remains |
| `list_insert_after(n, fresh)` | Splice `fresh` in after `n` |
| `list_insert_before(n, fresh)` | Splice `fresh` in before `n` |
| `list_push_back(head, fresh)` | Append. One line over `insert_before` |
| `list_push_front(head, fresh)` | Prepend. One line over `insert_after` |
| `list_remove(n)` | Unlink `n`, leaving it a valid empty node |
| `list_is_linked(n)` | Whether `n` is currently on a list |
| `list_length(head)` | Count, O(n) |
| `list_splice(dst, src)` | Move all of `src` onto `dst`, O(1) |

No allocation anywhere in `list.c`. The list never owns a node and never knows
what type encloses it.

## What to notice

**The sentinel head deletes every edge case.** Because the list is circular and
the head is a real node, `head->next` always exists, so insert and remove are
four unconditional pointer writes with no branch for empty, first, or last. Write
the same list with `NULL` terminators and you will find yourself checking for
`NULL` in six places and getting one of them wrong. This is a general technique:
a fake element that can never be the answer often removes all the special cases.

**`container_of` is doing pointer arithmetic that must be done in bytes.** The
cast to `char *` is not cosmetic. Subtracting `offsetof` from a `Task *` would
scale by `sizeof(Task)` and land far outside the object. The `1 ? (ptr) : ...`
in the macro is a compile-time type check with no runtime cost: it makes the
compiler reject a pointer to the wrong member, which converts a silent
wrong-offset memory bug into a build error.

**The node is deliberately not the first member.** If it were, `offsetof` would
be zero and a broken `container_of` that ignored the offset entirely would pass
every test. `Task` puts `id` first for exactly that reason.

**Deleting during iteration is a use-after-free waiting to happen.** The plain
`list_for_each` reads `pos->next` at the top of each step, so if the body
unlinks `pos`, the next read comes from a node that is no longer in the list.
`list_for_each_safe` reads the successor *before* running the body, which is why
it needs the extra variable.

**What `list_remove` leaves behind is a design decision with two payoffs.** The
list stays correct whatever you do to the removed node's own pointers, so it is
tempting to leave them dangling. Resetting the node to point at *itself*, the
same state `list_init` produces, buys two things at the cost of two writes:
removing twice becomes harmless instead of writing through stale pointers into
whatever now occupies that memory, and `list_is_linked` becomes expressible at
all, since "not on a list" is exactly "points at itself". Setting them to `NULL`
instead would give you the second property but not the first.

**`push_back` needs no tail pointer, and that surprises people.** The sentinel
sits between the tail and the head, so "append" is just "insert before the
sentinel". A singly-linked list without a sentinel needs an explicit tail
pointer and has to maintain it on every operation; here it falls out of the
shape for free.

## Why intrusive at all

The list allocates nothing, so an insert cannot fail, which matters enormously
in a kernel or an allocator where "handle malloc failing here" is not a
comfortable sentence. One object can also belong to several lists at once by
carrying several nodes, which `test_one_struct_in_two_lists` demonstrates: a
task in the "all tasks" list and the "runnable" list simultaneously, with no
extra allocation and no second lookup.

The cost is that the list is untyped and every access needs `container_of`, and
that a node cannot be in the same list twice. That trade is why you find this
pattern in kernels and allocators and almost nowhere else.
