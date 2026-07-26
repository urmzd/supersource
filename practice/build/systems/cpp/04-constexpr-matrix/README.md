# C++ 04: Compile-time matrix

**Concepts:** `constexpr`, dimensions as template parameters, `requires`
**Difficulty:** ⭐⭐⭐

A fixed-size matrix whose dimensions live in the type and whose arithmetic the
compiler can evaluate. Most of the test suite is `static_assert`, so a wrong
implementation **fails to build** rather than failing at runtime.

## The contract

| Member | Notes |
|--------|-------|
| `operator()(r, c)` | Element access. Not `operator[]`, see below |
| `operator+` `-` | Same-shape only, enforced by the type |
| `operator*(scalar)` | Scale |
| `operator*(Matrix<T, Cols, N>)` | Product. Shape checked at compile time |
| `transpose()` | Returns a *different type* with the dimensions swapped |
| `identity()`, `trace()` | Square matrices only, via `requires` |
| `matrix_pow(m, n)` | Repeated squaring, O(log n) |

## What to notice

**Two separate ideas are doing the work, and they are worth keeping apart.**
Dimensions as template parameters means a shape mismatch is a *type* error:
multiplying a 2×3 by a 4×5 is not a bug you can have at runtime, because the
program will not build. `constexpr` is the orthogonal claim that the same code
can run during compilation. You can have either without the other.

**`transpose()` returning a different type is the payoff.** `Matrix<T,2,3>` and
`Matrix<T,3,2>` are unrelated classes, so the compiler tracks shapes through
every operation for free. That is why the tests can assert
`decltype(a * c) == Matrix<int,2,2>` and why `(AB)^T == B^T A^T` is checkable at
compile time.

**`requires(Rows == Cols)` beats a `static_assert` in the body.** Constraining
the member means `Matrix<int,2,3>::identity()` is *not a member at all* rather
than a member that fails when instantiated. The tests use SFINAE detection
idioms to assert exactly that: `has_identity<M23>::value` must be `false`.
A `static_assert` inside the function would make that test impossible to write,
because the mere attempt would be a hard error.

**Testing that something does *not* compile takes a detection idiom.** You
cannot write "assert this is a compile error" directly. `std::void_t` plus a
partial specialisation gives you a trait that is `true` only when the
expression is well formed, which turns "must not compile" into an ordinary
`static_assert`. This is a technique worth stealing for any constrained API.

**`operator()` rather than `operator[]`, and there is a historical reason.**
Before C++23, the subscript operator took exactly one argument, so `m[1,2]` was
impossible. Every C++ matrix library, Eigen included, settled on `m(i,j)`. In
C++23 multi-argument subscript exists and the convention is slowly changing.

**Watch for macro commas.** `CHECK(Matrix<long, 8, 8>::identity()..., "...")`
does not compile, because the preprocessor knows nothing about angle brackets
and reads those commas as argument separators. An extra pair of parentheses
fixes it. This bites everyone once.

**The Fibonacci matrix makes the tests self-checking.** Powers of `[[1,1],[1,0]]`
have Fibonacci numbers as entries, so `matrix_pow(fib, 10)(0,1) == 55` is an
assertion whose expected value comes from somewhere other than the
implementation being tested.

## Extending it

Add a `constexpr` determinant and inverse for small fixed sizes, which forces
you to decide what an integer matrix's inverse even means and pushes you toward
a rational element type. Then try making the element type a `concept` rather
than an unconstrained `typename`, so misuse produces a readable error instead of
a page of template instantiation backtrace.
