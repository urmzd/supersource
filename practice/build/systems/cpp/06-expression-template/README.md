# C++ 06: Expression templates

**Concepts:** CRTP, operator overloading, lazy evaluation
**Difficulty:** ⭐⭐⭐⭐

Make `d = a + b + c` compute in a single pass with no temporary arrays, by
having `+` return a description of the work rather than the work's result.

## The contract

| Type | Role |
|------|------|
| `Expr<Derived>` | CRTP base. Anything satisfying it joins the algebra |
| `Vec` | The only type that owns memory |
| `Scalar` | A broadcast constant, reporting `size() == 0` |
| `BinaryExpr` / `UnaryExpr` | Nodes holding operands and an operation |
| `operator+ - * /`, unary `-` | Return nodes, never a `Vec` |

You write `Vec`'s expression constructor and assignment, and `BinaryExpr`'s
`operator[]` and `size()`.

## What to notice

**The naive version's cost is memory bandwidth, not arithmetic.** `a + b + c`
with ordinary overloading allocates a temporary for `a + b`, fills it, then
allocates another for the outer sum. Three arrays' worth of traffic to produce
one. On anything large, that traffic *is* the runtime, which is why Eigen,
Blaze, and xtensor are all built on this technique.

**CRTP is static polymorphism: the base downcasts to the derived type.**
`static_cast<const Derived&>(*this)` costs nothing and inlines, where a virtual
call would defeat the whole purpose by preventing the compiler from collapsing
the tree. You get an interface without a vtable.

**The recursion happens at compile time, and the loop happens once.** By the
time `Vec::operator=` runs its single loop, `expr[i]` has expanded into
`a[i] + b[i] + c[i]` as one inlined expression. There is no tree walk at
runtime and no intermediate storage. `test_no_intermediate_materialisation`
proves it by counting reads: four inputs of four elements must be exactly
sixteen reads.

**Storing operands by reference is the obvious choice and it is wrong.** This is
the most famous bug in expression templates, and the reference implementation
was written with it before AddressSanitizer caught it:

```cpp
auto e = a + b + c;   // (a + b) is a temporary...
Vec  v = e;           // ...that died at the semicolon above
```

Assigning directly is safe because every temporary lives to the end of that
full expression. Reach for `auto` and the outer node holds a reference to an
intermediate that no longer exists. The fix real libraries use, and the one
here: hold intermediate *nodes* by value and only the user's *containers* by
reference. Nodes are a couple of pointers, so copying them is free; containers
are precisely the things that must not be copied. Eigen still documents this
as a caveat, because a temporary container on the left of the expression can
dangle regardless.

**`size()` cannot just take the left operand's.** `Scalar` reports 0 so that a
binary node can take its length from whichever side is a real array. Taking the
left unconditionally works for `v * Scalar(2)` and breaks on `Scalar(2) * v`,
which is why the tests do both.

**The tests define their own leaf type.** `CountingVec` lives in `main.cpp`,
inherits `Expr<CountingVec>`, and immediately works with every operator. That
is not a testing trick: it shows the interface is open, and that any type
providing `operator[]` and `size()` participates without the library knowing it
exists.

## Extending it

Add a `Slice` node so `v[range(2,5)] = expr` works and see how quickly aliasing
becomes a real problem: `v = v + shift(v, 1)` reads elements it has already
overwritten. Eigen's answer is an explicit `.eval()` and a documented aliasing
rule, and understanding why they could not simply detect it is the interesting
part.
