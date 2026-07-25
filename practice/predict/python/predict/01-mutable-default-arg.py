"""01-mutable-default-arg: a default value is evaluated once, when the `def`
line runs, and then lives on the function object forever. Every call that does
not pass the argument shares that one object.
"""

import itertools


def add(item, bucket=[]):
    bucket.append(item)
    return bucket


print("1.", add(1))
print("2.", add(2))
print("3.", add(3, []))
print("4.", add(4))


def add_safe(item, bucket=None):
    bucket = [] if bucket is None else bucket
    bucket.append(item)
    return bucket


print("5.", add_safe(1), add_safe(2))

# The default is an expression, so it is *called* once at definition time.
counter = itertools.count()


def stamped(n=next(counter)):
    return n


print("6.", stamped(), stamped(), stamped())
print("7.", next(counter))

# Defaults are visible (and mutable) on the function object itself.
print("8.", add.__defaults__)

# The same rule applies to class bodies, which also run exactly once.


class Registry:
    seen = []

    def note(self, item):
        self.seen.append(item)
        return len(self.seen)


a, b = Registry(), Registry()
print("9.", a.note("x"), b.note("y"))
print("10.", Registry.seen)
