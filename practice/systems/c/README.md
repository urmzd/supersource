# C Practice

Master manual memory management, pointer arithmetic, and undefined behavior avoidance.

## Language Essentials

| Concept | Details |
|---------|---------|
| **Memory Model** | Manual allocation (`malloc`/`free`), stack vs heap, no GC |
| **Type System** | Weak static typing, implicit conversions, `void*` generics |
| **Error Handling** | Return codes, `errno`, `setjmp`/`longjmp` (avoid) |
| **Concurrency** | pthreads, atomics (`stdatomic.h`), no language-level async |
| **Build** | `gcc`/`clang`, `make`, `cmake`, header/source separation |
| **Standard** | C11 (prefer) or C17 |

## Key Idioms

- **Opaque pointers** for encapsulation (`struct Foo;` in header, definition in .c)
- **Goto cleanup** pattern for resource management
- **Flexible array members** for variable-length structs
- **Static inline** functions in headers for zero-cost abstractions
- **Compound literals** for temporary struct values
- **Designated initializers** for readable struct initialization

## Reference Documentation

- [C11 Standard Draft (N1570)](https://www.open-std.org/jtc1/sc22/wg14/www/docs/n1570.pdf) -- free
- [cppreference.com/w/c](https://en.cppreference.com/w/c) -- best online reference
- *The C Programming Language* (K&R, 2nd ed.) -- the classic
- *Modern C* by Jens Gustedt -- [free PDF](https://gustedt.gitlabpages.inria.fr/modern-c/)
- *Expert C Programming* by Peter van der Linden -- deep dives

## Exercises

Implement each using only the C standard library. No external dependencies.

| # | Exercise | Concepts | Difficulty |
|---|----------|----------|------------|
| 01 | Dynamic array (`vec.h`) | `realloc`, growth factor, generics via macros | ⭐⭐ |
| 02 | Hash map (open addressing) | Hash functions, probing, load factor | ⭐⭐⭐ |
| 03 | Linked list (intrusive) | Container-of macro, pointer manipulation | ⭐⭐ |
| 04 | Binary heap | Array-based, `sift_up`/`sift_down` | ⭐⭐ |
| 05 | Merge sort (in-place) | Pointer arithmetic, recursion | ⭐⭐ |
| 06 | Graph (adjacency list) | Flexible arrays, BFS/DFS | ⭐⭐⭐ |
| 07 | Thread pool | pthreads, condition variables, work queue | ⭐⭐⭐⭐ |
| 08 | Arena allocator | Memory management, alignment, bump allocation | ⭐⭐⭐ |
| 09 | String interning | Hash table, `strdup`, lifetime management | ⭐⭐⭐ |
| 10 | Simple HTTP parser | State machine, buffer management, `recv` | ⭐⭐⭐⭐ |
