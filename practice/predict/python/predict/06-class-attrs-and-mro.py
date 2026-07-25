"""06-class-attrs-and-mro: attribute lookup walks the instance, then the class,
then the MRO. Writes never walk: they always land on the instance, which is how
a shared counter quietly becomes a per-instance one.
"""


class Registry:
    entries = []

    def __init__(self, name):
        self.name = name
        self.entries.append(name)


a = Registry("a")
b = Registry("b")
print("1.", Registry.entries, a.entries is Registry.entries)

# Assigning through the instance shadows the class attribute for that instance.
a.entries = ["shadowed"]
print("2.", a.entries, b.entries)


class Counter:
    count = 0

    def bump(self):
        self.count += 1
        return self.count


c1, c2 = Counter(), Counter()
print("3.", c1.bump(), c1.bump(), c2.bump(), Counter.count)


class A:
    def who(self):
        return "A"


class B(A):
    def who(self):
        return "B" + super().who()


class C(A):
    def who(self):
        return "C" + super().who()


class D(B, C):
    pass


print("4.", D().who())
print("5.", [cls.__name__ for cls in D.__mro__])

# super() follows the MRO of the instance, not the lexical parent, so C.who
# runs even though B inherits from A.
print("6.", B().who(), C().who())


class Slotted:
    __slots__ = ("x",)

    def __init__(self, x):
        self.x = x


s = Slotted(1)
try:
    s.y = 2
    result = "assigned"
except AttributeError as exc:
    result = type(exc).__name__
print("7.", result, hasattr(s, "__dict__"))
