# C 06: Graph (adjacency list)

**Concepts:** growable per-vertex arrays, BFS/DFS, topological order
**Difficulty:** ⭐⭐⭐

A directed graph where each vertex owns a growable array of outgoing edges, plus
the three traversals that most graph problems reduce to.

## The contract

`graph.h` declares it, `main.c` tests it, you write `graph.c`.

| Function | Does |
|----------|------|
| `graph_init(g, n)` | `n` vertices, no edges |
| `graph_add_edge(g, from, to)` | One directed edge. Duplicates allowed |
| `graph_degree(g, v)` | Out-degree |
| `graph_bfs(g, src, dist)` | Hop counts, `GRAPH_UNREACHABLE` where no path exists |
| `graph_dfs(g, src, order, &count)` | First-visit order, iteratively |
| `graph_toposort(g, order)` | Topological order, or -1 if cyclic |

Undirected graphs are built by adding both directions. The tests do that
explicitly rather than the API hiding it, because forgetting the reverse edge is
one of the most common graph bugs and hiding it teaches nothing.

## What to notice

**Mark vertices when you enqueue, not when you dequeue.** This is the BFS bug
that survives every small test. If you only mark on dequeue, a vertex with three
unprocessed predecessors gets pushed three times before it is first examined. The
traversal still terminates and still reports correct distances, but the queue can
hold far more than `n` entries, which overflows the fixed-size queue this
implementation allocates and turns a dense graph quadratic.

**BFS gives shortest paths only because every edge costs the same.** The moment
edges carry weights, the first time you reach a vertex is no longer the cheapest
time, and you need Dijkstra's priority queue instead. `test_bfs_takes_the_short_route`
adds the shortcut edge *last* specifically so that an implementation which
overwrites an already-known distance reports 3 instead of 1.

**DFS must be iterative here, and the test enforces it with 100,000 vertices in
a line.** Recursive DFS is much prettier and its stack depth is the length of the
longest path, so a path graph of any real size overflows. The iterative version
needs an explicit stack whose frames carry *how many neighbours have been
consumed*, because that is the state the call stack was holding for you.

**Kahn's algorithm gets cycle detection for free.** A vertex inside a cycle never
reaches in-degree zero, so it never enters the queue and never comes out. If
fewer than `n` vertices are emitted, the remainder are in cycles. No colouring,
no separate pass, no recursion.

**In-degree counts edges, not neighbours.** Three parallel edges from 0 to 1 give
vertex 1 an in-degree of 3, so processing vertex 0 must decrement three times.
Decrementing once per distinct neighbour leaves the count stuck above zero and
reports a cycle in a graph that plainly has none. `test_toposort_with_parallel_edges`
exists solely to catch this.

**`calloc` for the adjacency array is load-bearing.** Every `AdjList` must start
zeroed, or the first `graph_add_edge` calls `realloc` on an uninitialised
pointer. That is a crash if you are lucky and silent heap corruption if you are
not.

## Extending it

Add Dijkstra using the heap from [exercise 04](../04-binary-heap/), which is the
natural next step and immediately runs into the awkward part: you need
`decrease_key`, and the array-backed heap has no way to find an element without
a side index. After that, Tarjan's strongly connected components, which needs a
recursive formulation or a considerably more intricate iterative one.
