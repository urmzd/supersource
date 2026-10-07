# C 09: String interning

**Concepts:** hash table, stable addresses, lifetime management
**Difficulty:** ⭐⭐⭐

A table that owns exactly one copy of each distinct string and returns the same
pointer every time you ask for it. Once strings are interned, `a == b` replaces
`strcmp(a, b) == 0` everywhere downstream.

## The contract

`intern.h` declares it, `main.c` tests it, you write `intern.c`.

| Function | Does |
|----------|------|
| `intern(t, s)` | Intern a C string, returning a stable handle |
| `intern_n(t, s, len)` | Intern arbitrary bytes, including embedded NULs |
| `intern_find(t, s)` | Look up without inserting |
| `intern_count(t)` | Distinct strings held |
| `intern_free(t)` | Release the table and every string in it |

The hash and the match helper are given. What you write is the table.

## What to notice

**Every pointer handed out must stay valid for the life of the table, and that
constrains the whole design.** Growth rehashes the table, so if the strings live
in one big buffer that gets reallocated, every handle a caller is holding
dangles. The fix is that entries are individually allocated and the table stores
*pointers*: rehashing moves pointers between slots and never touches the string
data. `test_pointers_survive_growth` interns 5,000 strings, forcing many
rehashes, then checks that the very first handle is still correct.

**One allocation per entry, not two.** The header and the bytes go in a single
`malloc` using a flexible array member, so the string cannot be separated from
its metadata and the allocator sees half the traffic. This is the same technique
as [exercise 06](../06-graph/)'s blocks and worth having in reflex.

**Cache the hash in the entry.** It costs one `size_t` and buys two things:
growth never rehashes the text, and a probe can reject a colliding entry by
comparing integers instead of calling `memcmp`. With long strings that share a
prefix, which is exactly what identifier tables are full of, that is most of the
lookup cost.

**No tombstones, because interning never deletes.** [Exercise 02](../02-hash-map/)
needed tombstones and a load factor that counted them, and all of that
complexity exists to support removal. Drop removal from the interface and the
table gets substantially simpler. It is worth noticing which of your data
structure's complications are paying for a feature you are not using.

**`strlen` cannot be the internal length.** The table stores bytes, and
`intern_n` must copy exactly the length it was given: `"a\0b"` and `"a"` have the
same `strlen` and are different strings. It still writes a terminator after the
bytes, so C consumers can treat `str` normally, but the length is what defines
identity. `test_embedded_nuls` and `test_unterminated_input` pin both halves.

**Interning is a space/time trade that can lose.** You pay a hash and a
comparison once per string to make every subsequent comparison free. That wins
overwhelmingly for compiler identifiers, JSON keys, and log field names, where a
small set of strings is compared constantly. It loses for strings you compare
once, and it leaks by construction for unbounded input, since nothing is ever
removed. A table interning attacker-supplied strings is an unbounded memory
sink.

## Extending it

Add a per-entry reference count and a `intern_release`, and watch how much
complexity re-enters: tombstones come back, pointer stability gets harder, and
you have to decide what happens to a handle whose count hit zero. That is the
usual reason production interners simply never free, or tie their lifetime to an
[arena](../08-arena-allocator/) that is reset wholesale.
