# Polyglot Practice

Retain and sharpen coding skills across 9 languages organized by domain. Each exercise is a from-scratch implementation of a core algorithm or data structure in the target language, emphasizing **language-specific idioms** over direct translation.

## Tiers

### Systems Programming

Languages where you manage memory, understand hardware, and write performance-critical code.

| Language | Focus | Reference |
|----------|-------|-----------|
| [C](systems/c/) | Manual memory, pointers, undefined behavior | K&R, C11 standard |
| [C++](systems/cpp/) | RAII, templates, move semantics, STL | ISO C++, cppreference.com |
| [Rust](systems/rust/) | Ownership, borrowing, lifetimes, zero-cost abstractions | The Rust Book (free) |
| [Zig](systems/zig/) | Comptime, no hidden allocations, C interop | ziglearn.org (free) |

### Cloud & Backend

Languages for distributed systems, microservices, and data pipelines.

| Language | Focus | Reference |
|----------|-------|-----------|
| [Go](cloud/go/) | Goroutines, channels, interfaces, simplicity | Go Tour + Effective Go (free) |
| [Scala](cloud/scala/) | FP + OOP, type system, pattern matching, Akka | Scala Book (free) |
| [Java](cloud/java/) | JVM internals, generics, concurrency (virtual threads) | JLS, Effective Java |

### General Purpose

Languages for rapid prototyping, scripting, ML, and full-stack development.

| Language | Focus | Reference |
|----------|-------|-----------|
| [Python](general/python/) | Generators, decorators, data model, asyncio | CPython docs, Fluent Python |
| [TypeScript](general/typescript/) | Type narrowing, generics, mapped types, runtime safety | TS Handbook (free) |

## How to Practice

1. Pick a topic from the [algorithm curriculum](../algorithms/) or [math track](../math/)
2. Use the `practice-impl` skill to scaffold an empty implementation: invoke with language and topic
3. Implement from scratch using only the language's standard library
4. Compare with reference implementations and note idiomatic differences
5. Repeat in a different language to build cross-language fluency

## Exercise Categories

Each language directory contains exercises in these categories:

| Category | Examples | Tests |
|----------|----------|-------|
| **Data Structures** | Linked list, hash map, BST, heap, graph | Correctness + edge cases |
| **Algorithms** | Sort, search, DP, graph traversal, string matching | Correctness + complexity |
| **Concurrency** | Producer-consumer, thread pool, async pipelines | Race-free + deadlock-free |
| **Systems** | Memory allocator, file I/O, network socket | Platform-specific |
| **Language Idioms** | Iterators, error handling, generics, macros | Idiomatic patterns |
