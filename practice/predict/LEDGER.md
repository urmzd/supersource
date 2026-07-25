# Gap Ledger

Every miss goes here. A miss you do not write down is a miss you will repeat,
and the point of this whole directory is to stop repeating them.

Keep entries short. The valuable column is the last one: not what the code did,
but which belief of yours was wrong. If you cannot name the belief, you have not
finished learning the thing yet.

## Format

| Date | Snippet | I predicted | It printed | The rule I had wrong |
|------|---------|-------------|------------|----------------------|

## Worked example

This row is here to show the shape of a good entry. Delete it once you have your
own.

| Date | Snippet | I predicted | It printed | The rule I had wrong |
|------|---------|-------------|------------|----------------------|
| 2026-07-24 | `go/01-loop-var-capture` | `2 2 2` for case 1 | `0 1 2` | Go 1.22 made the three-clause loop variable per-iteration. I was still carrying the pre-1.22 rule, and the fix I "knew" (`i := i` shadowing) is now redundant there. Case 2 still prints `3 3 3` because that variable is declared outside the header, which is the part I would still get wrong. |

## My misses

| Date | Snippet | I predicted | It printed | The rule I had wrong |
|------|---------|-------------|------------|----------------------|
|      |         |             |            |                      |

## Review

Once a month, reread the ledger before touching a new snippet. Rows that you now
find obvious can be deleted. Rows you still hesitate over are the ones to
re-drill: `lr reset <lang> <id>` clears your old prediction so the snippet can be
attempted cold.

Patterns across rows matter more than any single row. Three entries about
aliasing in three different languages is not three misses, it is one missing
model of value versus reference semantics.
