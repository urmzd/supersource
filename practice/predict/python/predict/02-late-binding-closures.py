"""02-late-binding-closures: a closure captures the variable, not its value at
capture time. Python resolves the name when the function runs, which is usually
long after the loop that created it finished.
"""

fs = [lambda: i for i in range(3)]
print("1.", [f() for f in fs])

# Binding it as a default argument snapshots the value instead.
gs = [lambda i=i: i for i in range(3)]
print("2.", [g() for g in gs])

# Comprehension variables live in their own scope and do not leak.
i = "outer"
_ = [i for i in range(3)]
print("3.", i)

# A plain for loop does leak, and the leaked value is the last one.
for j in range(3):
    pass
print("4.", j)


def make_counter():
    total = 0

    def bump(n):
        nonlocal total
        total += n
        return total

    return bump


c = make_counter()
print("5.", c(1), c(2), c(3))

# Two calls to the factory get two independent cells.
d = make_counter()
print("6.", c(10), d(10))


# Without nonlocal, the assignment creates a fresh local and the read fails.
def broken():
    count = 0

    def bump():
        # ruff is right: this is a bug. It is the point of the snippet.
        count += 1  # noqa: F823

    try:
        bump()
    except UnboundLocalError as exc:
        return type(exc).__name__
    return count


print("7.", broken())

# Closures are inspectable: this is the cell the lambda reads at call time.
print("8.", [cell.cell_contents for cell in (c.__closure__ or ())])
