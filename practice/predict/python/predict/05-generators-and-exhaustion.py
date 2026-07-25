"""05-generators-and-exhaustion: a generator is a one-shot cursor, not a
sequence. Nothing in the body runs until the first `next`, and once it is
drained it is silently empty rather than an error.
"""


def gen():
    print("  body starts")
    yield 1
    print("  between yields")
    yield 2
    print("  body ends")


g = gen()
print("1. created, nothing has run")
print("2.", next(g))
print("3.", list(g))
print("4.", list(g))

squares = (n * n for n in range(4))
print("5.", sum(squares), sum(squares))

# zip pulls from the left first, so the extra pull is visible afterwards.
it = iter([1, 2, 3, 4])
print("6.", list(zip(it, "ab")))
print("7.", list(it))

# any/all short-circuit, so the side effects stop early too.
seen = []


def note(x):
    seen.append(x)
    return x > 1


print("8.", any(note(x) for x in [1, 2, 3]), seen)


# return inside a generator becomes StopIteration.value, not a yielded item.
def with_return():
    yield "a"
    return "done"


w = with_return()
print("9.", next(w))
try:
    next(w)
except StopIteration as stop:
    print("10.", stop.value)


# Generators are lazy enough to be infinite; the consumer sets the bound.
def naturals():
    n = 0
    while True:
        yield n
        n += 1


print("11.", [x for _, x in zip(range(4), naturals())])
