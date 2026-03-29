# 13-functional-programming

## Summary
- Contains Scheme exercises: `Factors.scm`, `Factors2.scm`, `IsPrime.scm`, `bst.scm`,
  `determine-binding.scm`, `io.scm`, `list-visitor.scm`, `max2-visitor.scm`,
  `new-sqrt-iterator.scm`, `product.scm`.
- Focused on recursion, list processing, and functional patterns.

## Key takeaways
- Pure functions and recursion are the primary control mechanisms.
- Visitors and higher-order patterns are used for traversal and aggregation.

## How to run
- Load files in a Scheme interpreter (e.g., `racket` or `guile`) and evaluate the top-level forms.

---

# Functional Programming Patterns

## Core Insight

FP treats computation as **evaluation of mathematical functions** — no mutable state, no side effects. This makes reasoning about correctness easier and enables powerful abstractions like higher-order functions, closures, and algebraic data types.

## Pattern 1: Higher-Order Functions (Map/Filter/Reduce)

**Map**: Transform every element.
**Filter**: Keep elements matching a predicate.
**Reduce (Fold)**: Collapse a collection into a single value.

```scheme
;; Scheme
(define (my-map f lst)
  (if (null? lst) '()
      (cons (f (car lst)) (my-map f (cdr lst)))))

(define (my-filter pred lst)
  (cond ((null? lst) '())
        ((pred (car lst)) (cons (car lst) (my-filter pred (cdr lst))))
        (else (my-filter pred (cdr lst)))))

(define (my-fold f init lst)
  (if (null? lst) init
      (my-fold f (f init (car lst)) (cdr lst))))
```

**Interview relevance**: Implement map/filter/reduce from scratch, lazy evaluation.

## Pattern 2: Recursion as Iteration

FP replaces loops with recursion. **Tail recursion** is the FP equivalent of a loop — the recursive call is the last operation.

```scheme
;; Non-tail (stack grows)
(define (factorial n)
  (if (<= n 1) 1
      (* n (factorial (- n 1)))))

;; Tail-recursive (constant stack with TCO)
(define (factorial-tail n)
  (define (helper n acc)
    (if (<= n 1) acc
        (helper (- n 1) (* acc n))))
  (helper n 1))
```

**Key insight**: Any loop can be converted to tail recursion and vice versa. Scheme guarantees tail call optimization (TCO).

## Pattern 3: Persistent Data Structures

**Immutable by default** — "modifications" return new structures sharing most of their memory with the original.

```
Original list: [1] → [2] → [3] → nil
After cons 0:  [0] → [1] → [2] → [3] → nil
                       ↑ shared structure
```

**BST insertion** creates a new path from root to inserted node, sharing all other subtrees.

**Interview relevance**: Implement persistent BST, understand structural sharing.

## Pattern 4: Pattern Matching and Algebraic Data Types

```
data Tree a = Leaf | Node (Tree a) a (Tree a)

depth :: Tree a -> Int
depth Leaf = 0
depth (Node l _ r) = 1 + max (depth l) (depth r)
```

**Visitor pattern** in FP: Pattern matching replaces the OOP visitor — each case is handled by a match arm.

## Pattern 5: Closures and Currying

```scheme
;; Closure: function captures its environment
(define (make-counter)
  (let ((count 0))
    (lambda ()
      (set! count (+ count 1))
      count)))

;; Currying: function that returns a function
(define (add x)
  (lambda (y) (+ x y)))
;; ((add 3) 4) => 7
```

## Company Targeting

| Company | Focus | Difficulty |
|---------|-------|------------|
| Jane Street | OCaml, pattern matching, immutability | Hard |
| Meta | Haskell-influenced Hack, functional React patterns | Medium |
| Google | Functional concepts in distributed systems | Medium |
| Anthropic | Functional patterns in Python, immutable state | Medium |
