"""04-aliasing-and-copies: `*` on a list repeats the reference, not the object,
and `copy` stops at the first level. Both facts are invisible until something
nested gets mutated.
"""

import copy

grid = [[0] * 3] * 3
grid[0][0] = 9
print("1.", grid)

grid2 = [[0] * 3 for _ in range(3)]
grid2[0][0] = 9
print("2.", grid2)

nested = [[1, 2], [3, 4]]
shallow = copy.copy(nested)
deep = copy.deepcopy(nested)
nested[0].append(99)
print("3.", shallow)
print("4.", deep)

# dict(d) and d.copy() are both shallow.
d = {"k": [1]}
e = dict(d)
d["k"].append(2)
print("5.", e)

# Rebinding is not mutation: the other name still sees the old object.
s = "abc"
t = s
s += "d"
print("6.", s, t)

xs = [1, 2]
ys = xs
xs = xs + [3]
print("7.", xs, ys)

# ... but += on a list mutates in place, so the alias does see it.
ps = [1, 2]
qs = ps
ps += [3]
print("8.", ps, qs)

# Slicing copies the outer level only.
rows = [[1], [2]]
sliced = rows[:]
rows[0].append(0)
sliced.append([3])
print("9.", rows, sliced)

# Default sort is stable and in place; sorted() returns a new list.
words = ["bb", "a", "ccc", "dd"]
by_len = sorted(words, key=len)
print("10.", by_len, words)
