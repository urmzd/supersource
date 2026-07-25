"""03-identity-vs-equality: `==` asks the objects, `is` asks the allocator.
CPython caches and interns enough small values that `is` appears to work, right
up until the day the value comes from input instead of a literal.
"""

small_a = 256
small_b = 256
print("1.", small_a is small_b)

# Same value, but built at runtime, so no cache hit.
big_a = int("257")
big_b = int("257")
print("2.", big_a == big_b, big_a is big_b)

list_a = [1, 2]
list_b = [1, 2]
print("3.", list_a == list_b, list_a is list_b)

# Literals in the same code object are interned.
s1 = "hello"
s2 = "hello"
s3 = "".join(["hel", "lo"])
print("4.", s1 is s2, s1 == s3, s1 is s3)

# Containers compare by identity first, then equality, which NaN exposes.
nan = float("nan")
print("5.", nan == nan, [nan] == [nan], nan in [nan])

# Float equality is about representation, not arithmetic.
print("6.", 0.1 + 0.2 == 0.3, abs((0.1 + 0.2) - 0.3) < 1e-9)

# bool is a subclass of int, so it hashes and compares like one.
# ruff flags the duplicate key, which is exactly the behaviour on display.
print("7.", True == 1, isinstance(True, int), {1: "int", True: "bool"})  # noqa: F601

# Empty immutable containers are shared; empty mutable ones are not.
empty_tuple_a = ()
empty_tuple_b = ()
empty_list_a = []
empty_list_b = []
print("8.", empty_tuple_a is empty_tuple_b, empty_list_a is empty_list_b)

# `is not None` is the only identity check you should be writing.
value = None
print("9.", value is None, value == None)  # noqa: E711
