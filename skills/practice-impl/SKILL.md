---
name: practice-impl
description: "Scaffold an empty practice implementation for any language and exercise in the polyglot practice track. Creates skeleton source files with type signatures, test stubs, and build instructions."
invoke: user
arguments:
  - name: args
    description: "<language> <exercise-number-or-name> — e.g., 'rust 03' or 'go worker-pool'"
---

# Practice Implementation Scaffolder

Generate a ready-to-code practice file for any exercise in the polyglot practice track.

## Instructions

1. Parse the language and exercise identifier from the arguments: `{args}`
2. Map the language to its tier and directory:
   - **Systems**: c → `practice/systems/c/`, cpp → `practice/systems/cpp/`, rust → `practice/systems/rust/`, zig → `practice/systems/zig/`
   - **Cloud**: go → `practice/cloud/go/`, scala → `practice/cloud/scala/`, java → `practice/cloud/java/`
   - **General**: python → `practice/general/python/`, typescript → `practice/general/typescript/`
3. Read the language's README.md to find the exercise by number or name match
4. Extract: exercise name, concepts, difficulty
5. Create the implementation file using the language-appropriate template below
6. Create a companion test file if the language convention supports it

## File Naming

| Language | Source | Test |
|----------|--------|------|
| C | `{nn}-{name}.c` | `{nn}-{name}_test.c` |
| C++ | `{nn}-{name}.cpp` | `{nn}-{name}_test.cpp` |
| Rust | `{nn}_{name}.rs` | (tests inline via `#[cfg(test)]`) |
| Zig | `{nn}_{name}.zig` | (tests inline via `test` blocks) |
| Go | `{nn}_{name}.go` | `{nn}_{name}_test.go` |
| Scala | `{PascalName}.scala` | `{PascalName}Suite.scala` |
| Java | `{PascalName}.java` | `{PascalName}Test.java` |
| Python | `{nn}_{name}.py` | `test_{nn}_{name}.py` |
| TypeScript | `{nn}-{name}.ts` | `{nn}-{name}.test.ts` |

## Templates

### C Template
```c
/**
 * Exercise {nn}: {title}
 * Difficulty: {stars}
 * Concepts: {concepts}
 *
 * Build: gcc -std=c11 -Wall -Wextra -o {name} {nn}-{name}.c
 * Test:  gcc -std=c11 -Wall -Wextra -o {name}_test {nn}-{name}.c {nn}-{name}_test.c && ./{name}_test
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>

// TODO: Implement {title}
```

### C++ Template
```cpp
/**
 * Exercise {nn}: {title}
 * Difficulty: {stars}
 * Concepts: {concepts}
 *
 * Build: g++ -std=c++20 -Wall -Wextra -o {name} {nn}-{name}.cpp
 */

#include <iostream>
#include <vector>
#include <cassert>

// TODO: Implement {title}
```

### Rust Template
```rust
//! Exercise {nn}: {title}
//! Difficulty: {stars}
//! Concepts: {concepts}
//!
//! Run: rustc {nn}_{name}.rs && ./{nn}_{name}
//! Test: rustc --test {nn}_{name}.rs && ./{nn}_{name}

// TODO: Implement {title}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_basic() {
        todo!("Write tests")
    }
}

fn main() {
    println!("{title} — not yet implemented");
}
```

### Zig Template
```zig
//! Exercise {nn}: {title}
//! Difficulty: {stars}
//! Concepts: {concepts}
//!
//! Build: zig build-exe {nn}_{name}.zig
//! Test:  zig test {nn}_{name}.zig

const std = @import("std");

// TODO: Implement {title}

test "{title} basic" {
    // TODO: Write tests
}

pub fn main() !void {
    std.debug.print("{title} — not yet implemented\n", .{});
}
```

### Go Template
```go
// Exercise {nn}: {title}
// Difficulty: {stars}
// Concepts: {concepts}
//
// Run:  go run {nn}_{name}.go
// Test: go test -v -run Test{PascalName}

package main

import "fmt"

// TODO: Implement {title}

func main() {
	fmt.Println("{title} — not yet implemented")
}
```

### Scala Template
```scala
/** Exercise {nn}: {title}
  * Difficulty: {stars}
  * Concepts: {concepts}
  *
  * Run: scala-cli run {PascalName}.scala
  * Test: scala-cli test {PascalName}.scala {PascalName}Suite.scala
  */

// TODO: Implement {title}
```

### Java Template
```java
/**
 * Exercise {nn}: {title}
 * Difficulty: {stars}
 * Concepts: {concepts}
 *
 * Build: javac {PascalName}.java
 * Run:   java {PascalName}
 */

// TODO: Implement {title}

public class {PascalName} {
    public static void main(String[] args) {
        System.out.println("{title} — not yet implemented");
    }
}
```

### Python Template
```python
#!/usr/bin/env python3
"""
Exercise {nn}: {title}
Difficulty: {stars}
Concepts: {concepts}

Run:  python {nn}_{name}.py
Test: python -m pytest test_{nn}_{name}.py
"""

# TODO: Implement {title}

if __name__ == "__main__":
    print("{title} — not yet implemented")
```

### TypeScript Template
```typescript
/**
 * Exercise {nn}: {title}
 * Difficulty: {stars}
 * Concepts: {concepts}
 *
 * Run:  tsx {nn}-{name}.ts
 * Test: vitest run {nn}-{name}.test.ts
 */

// TODO: Implement {title}
```

## After Scaffolding

Tell the user:
1. The file(s) created and their location
2. The exercise concepts and difficulty
3. Key language-specific hints for the exercise (from the README)
4. How to build and run tests
