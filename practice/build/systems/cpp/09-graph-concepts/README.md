# C++ 09: Graph algorithms with concepts

**Concepts:** C++20 concepts, ranges, generic algorithms
**Difficulty:** ⭐⭐⭐

BFS, DFS, shortest path, and Dijkstra written against a *concept* rather than a
class. The exercise is not the graph algorithms; it is that the same code runs
on three unrelated types that share no base class and do not know about each
other.

## The contract

| Concept | Requires |
|---------|----------|
| `Graph<G>` | `G::vertex_type`, hashable and equality-comparable, plus `g.neighbours(v)` returning an input range |
| `WeightedGraph<G>` | `Graph<G>` and `g.weight(a, b)` |

| Algorithm | Constrained on |
|-----------|----------------|
| `bfs`, `dfs_order`, `shortest_path` | `Graph` |
| `shortest_distances` (Dijkstra) | `WeightedGraph` |

## What to notice

**The concept moves the error to the call site.** An unconstrained template
that fails deep inside its body produces a page of instantiation backtrace
pointing at code the caller never wrote. `template <Graph G>` rejects the call
itself and says which requirement was not met. The tests assert this with
`static_assert(!Graph<NotAGraph>)`, which is a compile-time claim that a
mistake is *impossible*, not merely untested.

**The grid is the case that justifies the whole approach.** `Grid` stores no
adjacency structure at all: `neighbours` computes the four cardinal directions
on demand. A 300×300 grid, tested here, has 90,000 vertices and around 360,000
edges that never exist in memory. With an interface based on inheritance you
would have to materialise them, or bend the base class into something that can
express computed edges. With a concept, `Grid` simply satisfies it.

**`WeightedGraph` refining `Graph` is subsumption, and it does real work.**
Dijkstra is callable only on graphs that report weights, so passing an
unweighted one is a clean overload-resolution failure rather than an error
about a missing `weight` member several frames down. Concepts also order
overloads by specificity, which is what makes constrained templates compose.

**BFS and Dijkstra answer different questions, and the test makes them
disagree on purpose.** In the weighted graph, `start → b` is one hop but costs
5, while `start → a → b` is two hops and costs 2. BFS reports 1, Dijkstra
reports 2.0, and both are right. That disagreement is the entire reason
Dijkstra exists, and a test suite where they happen to agree is not testing
either.

**Lazy deletion beats decrease-key in practice.** `std::priority_queue` has no
way to update an entry's priority, so this pushes a new entry and skips stale
ones on pop. It costs a larger queue and saves implementing an indexed heap.
The [C binary heap exercise](../../c/04-binary-heap/) discusses the other side
of that trade.

**Mark on enqueue, not on dequeue.** `try_emplace` returning whether it
inserted does the check and the mark in one lookup. Marking at dequeue lets the
same vertex be queued once per predecessor, which is quadratic on a dense
graph. This is the same bug as in the C version, and it is worth noticing that
the idiomatic C++ spelling makes it harder to get wrong.

**Watch for macro commas again.** `CHECK(dist[Cell{0, 0}] == 0, "...")` does not
compile: the preprocessor reads the comma inside the braced initialiser as an
argument separator. Extra parentheses around the condition fix it, and clang's
diagnostic even shows you where to put them.

## Extending it

Add A\* by constraining on a `Heuristic` concept, and notice you now need to
express a relationship *between* two template parameters rather than a property
of one. Then try making `neighbours` return a lazy `std::ranges` view instead of
a `std::vector`, which removes the per-vertex allocation the grid currently
pays and is the reason the concept asks only for `input_range`.
