# C 04: Binary heap

**Concepts:** array-backed trees, `sift_up`/`sift_down`
**Difficulty:** ⭐⭐

A min-heap where the entire data structure is one flat array and the tree shape
exists only in the index arithmetic. Then heapsort, which is the same machinery
run in place with no extra memory at all.

## The contract

`heap.h` declares it, `main.c` tests it, you write `heap.c`.

| Function | Does | Cost |
|----------|------|------|
| `heap_push(h, p, v)` | Insert | O(log n) |
| `heap_pop(h, &out)` | Remove the smallest | O(log n) |
| `heap_peek(h, &out)` | Look at the smallest | O(1) |
| `heap_build(h, items, n)` | Heapify an unordered array | O(n), not O(n log n) |
| `heap_sort(items, n)` | Sort ascending in place | O(n log n), no allocation |

`sift_up` and `sift_down` are given as empty helpers. Everything else is built
from those two.

## What to notice

**There are no pointers, and that is the point.** With a 0-based array the
children of `i` are at `2i+1` and `2i+2` and the parent is at `(i-1)/2`. A
pointer-based tree of the same shape costs two pointers per node, an allocation
per insert, and a cache miss per level. The array version costs nothing per
element and walks contiguous memory, which is why heaps back priority queues
everywhere from schedulers to Dijkstra.

**`sift_down` must descend toward the smaller child.** Taking the left child
unconditionally is the classic bug: it can swap a large value past a smaller
sibling and leave the invariant broken one level down. Every ordering test in
the file passes for a while with this bug, because it only shows once the heap
has been reshaped from both ends. `test_random_interleaving` is what catches it.

**Building is O(n), and the intuition that says O(n log n) is wrong.** Pushing n
items one at a time is O(n log n). Floyd's method, sifting down from the last
internal node backwards, is O(n): most nodes are near the bottom of the tree and
sift down by almost nothing, and the sum of heights over all nodes converges to
2n rather than n log n. The tests do not time this, but the implementation
should still be the linear one.

**A min-heap sorts descending, which is not what `heap_sort` promises.** In-place
heapsort parks each extracted element at the shrinking end, so a min-heap
produces a descending array. Getting ascending order needs a max-heap. Rather
than write `sift_down` twice with the comparison flipped, the reference flips
the *data*: an order-reversing involution applied before and after. One sift
implementation, no second copy to drift out of sync.

**And that flip cannot be plain negation.** `-INT_MIN` overflows, which is
undefined behaviour and in practice leaves `INT_MIN` unchanged, silently
misordering. The map `x → -x-1` covers every `int` with no hole, and it is its
own inverse. `test_heap_sort` includes `INT_MIN` and `INT_MAX` specifically to
catch the naive version.

**Heapsort is not stable, and the array shows you why.** `heap_pop` moves the
*last* element to the root, which throws away any relationship between an
element's position and its arrival order. Equal priorities come out in whatever
order the sifting happened to leave them, which is why the tests carry a payload
but never assert on tie order.

## Extending it

Add `heap_decrease_key`, which is what Dijkstra actually needs and what makes a
heap awkward: you must find the element first, and the array gives you no way to
do that in less than O(n). The usual fix is a side table from element identity
to current index, updated on every swap, which is a good illustration of how a
clean data structure grows a wart the moment a real algorithm uses it.
