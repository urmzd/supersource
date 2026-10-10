# C++ 03: Hash map (open addressing)

**Concepts:** hashing, probing, iterator invalidation
**Difficulty:** ⭐⭐⭐⭐

A hash map with a `std::unordered_map`-shaped interface but a completely
different implementation underneath, which is the interesting part.

## The contract

| Member | Does |
|--------|------|
| `find(key)` | Pointer to the value, or `nullptr` |
| `insert(k, v)` | `{element, inserted}`. Does **not** overwrite |
| `insert_or_assign(k, v)` | Does overwrite |
| `operator[](key)` | Default-constructs a missing value, like the standard's |
| `erase(key)` | True when something was removed |
| `rehash(n)` | Resize and reclaim tombstones |
| `begin()`/`end()` | Forward iteration over live elements |

## What to notice

**`std::unordered_map` is specified as separate chaining, and that is a
performance decision baked into the standard.** Its guarantees, particularly
that references to elements survive a rehash, can only be met by allocating
each element separately. Every lookup therefore chases a pointer into
unpredictable memory. Open addressing stores elements inline, so a probe walks
contiguous cache lines, and pays for it by invalidating everything on rehash.
This exercise's interface promises less than the standard's on purpose. Noticing
*which* guarantee costs the performance is the lesson.

**Deletion cannot write an empty slot.** Entries that collided with the deleted
one were placed after it in the probe run, and a lookup stops at the first empty
slot. Clearing to empty does not delete one entry, it makes every entry behind
it unreachable while it still occupies space.
`test_erase_from_the_middle_of_a_probe_run` uses a hash that sends everything to
bucket zero, so every key is in one long run and the bug is unmissable.

**Which means the load factor must count tombstones.** A tombstone costs a probe
step forever and holds nothing. Resize on the live count alone and a
delete-heavy workload fills the table with tombstones while reporting a low load
factor, decaying to a linear scan. `test_churn_does_not_degrade` runs 20,000
insert/erase pairs and asserts the table stays small.

**And the resize target must come from the live count, not the table size.**
This is the bug the reference was written with. Asking for `bucket_count() * 2`
whenever the used-count trips means a table that is *empty* but tombstone-heavy
doubles forever, growing without bound while holding nothing. Sizing from
`size()` rebuilds at the same capacity and simply drops the tombstones.

**Test with a deliberately terrible hash.** `AlwaysCollide` returns 0 for
everything. With a good hash, collisions are rare enough that probing bugs
survive thousands of operations undetected; with this one, every insert and
lookup exercises the full probe path. A hash map test suite without an
adversarial hash is not testing the hash map.

**`insert` and `insert_or_assign` differ, and the standard's naming is
unhelpfully subtle.** `insert` leaves an existing value alone and reports
`false`; only `insert_or_assign` overwrites. Getting this backwards produces a
map that silently discards updates, and the tests pin both.

## Extending it

Implement Robin Hood probing: on insert, if the element you are placing has
travelled further from its ideal slot than the one sitting there, steal the slot
and displace the incumbent. It barely changes the average probe length and
dramatically flattens the *variance*, which is the number that actually
determines tail latency. The course module `ds.02` (chapter home:
[Systems Data Structures](../../../algorithms/16-systems-data-structures/))
goes further into what modern implementations do instead.
