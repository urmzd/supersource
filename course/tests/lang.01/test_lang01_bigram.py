"""lang.01 course tests: the vectorized bigram count (primers/lang.01/bigram.py).

Annotated exemplars (DESIGN 5.12). The count here is the whole of "training"
the L0.0 byte bigram, so every pitfall below would later show up as a wrong
model rather than a crash.
"""

import ast
import inspect

import numpy as np
from _lib.close import assert_close
from _lib.pcg32 import PCG32

import bigram
from bigram import bigram_counts, row_normalize


def _random_bytes(rng: PCG32, n: int, alphabet: int) -> bytes:
    # A small alphabet makes repeated pairs common, which is where counting bugs hide.
    return bytes(rng.below(alphabet) for _ in range(n))


def _loop_counts(data: bytes) -> np.ndarray:
    # The obviously-correct oracle: one Python loop, one pair at a time.
    c = np.zeros((256, 256), dtype=np.int64)
    for a, b in zip(data, data[1:]):
        c[a, b] += 1
    return c


def test_bigram_hand_example():
    # WHY: section 3 worked by hand: "banana" has the pairs ba, an, na, an, na,
    #      so C[b,a] = 1, C[a,n] = 2, C[n,a] = 2, and nothing else.
    # KIND: unit
    # CHAPTER: lang.01 section 3
    c = bigram_counts(b"banana")
    assert c.shape == (256, 256)
    assert c[ord("b"), ord("a")] == 1
    assert c[ord("a"), ord("n")] == 2
    assert c[ord("n"), ord("a")] == 2
    assert int(c.sum()) == 5


def test_bigram_counts_every_repeated_pair():
    # WHY: `C[a, b] += 1` with index ARRAYS writes each repeated (a, b) once,
    #      not once per occurrence: "aaaa" would count aa as 1, not 3. Use
    #      np.add.at or np.bincount.
    # KIND: boundary
    # CHAPTER: lang.01 section 5, pitfall 7
    c = bigram_counts(b"aaaa")
    assert c[ord("a"), ord("a")] == 3
    assert int(c.sum()) == 3


def test_bigram_counts_do_not_wrap_at_256():
    # WHY: the counts must be int64. A uint8 count array wraps at 256, so
    #      999 repeats read as 231; the bigram would then think "aa" is rare.
    # KIND: boundary
    # CHAPTER: lang.01 section 5, pitfall 8
    c = bigram_counts(b"a" * 1000)
    assert c.dtype == np.int64
    assert c[ord("a"), ord("a")] == 999


def test_bigram_counts_of_short_inputs_are_all_zero():
    # WHY: zero or one byte has no adjacent pair. The function must still
    #      return the full (256, 256) int64 table, not crash or shrink.
    # KIND: boundary
    for data in (b"", b"x"):
        c = bigram_counts(data)
        assert c.shape == (256, 256) and c.dtype == np.int64
        assert int(c.sum()) == 0


def test_bigram_counts_use_all_256_byte_values():
    # WHY: a byte is 0..255 and the tracer vocabulary is all 256 of them
    #      (D32). Bytes >= 128 (UTF-8 continuation bytes) must land in their
    #      own rows, not be folded or dropped as negative int8 values.
    # KIND: boundary
    data = bytes([0, 255, 128, 0, 255])
    c = bigram_counts(data)
    assert c[0, 255] == 2 and c[255, 128] == 1 and c[128, 0] == 1
    assert int(c.sum()) == 4


def test_bigram_row_sums_count_each_leading_byte():
    # WHY: law: row a sums to the number of times a appears in data[:-1]
    #      (every byte but the last starts exactly one pair).
    # KIND: property
    rng = PCG32(seed=2)
    for n in (2, 3, 17, 500):
        data = _random_bytes(rng, n, alphabet=4)
        c = bigram_counts(data)
        assert int(c.sum()) == n - 1
        lead = np.bincount(np.frombuffer(data[:-1], dtype=np.uint8), minlength=256)
        assert c.sum(axis=1).tolist() == lead.tolist()


def test_bigram_counts_match_a_python_loop():
    # WHY: differential test against the slow obvious loop on random bytes,
    #      full alphabet and tiny alphabet.
    # KIND: differential
    rng = PCG32(seed=3)
    for n, alphabet in ((1000, 256), (1000, 3), (64, 2)):
        data = _random_bytes(rng, n, alphabet)
        assert bigram_counts(data).tolist() == _loop_counts(data).tolist()


def test_bigram_counts_has_no_python_loop():
    # WHY: "vectorize a count" is the exercise. A Python loop runs one
    #      interpreted step per byte; L0.0 counts megabytes. No for, while,
    #      or comprehension may appear in bigram_counts.
    # KIND: unit
    # CHAPTER: lang.01 section 4
    #      A helper of your own that bigram_counts calls counts as part of
    #      it, so the scan follows calls to functions defined in bigram.py.
    loops = (
        ast.For,
        ast.While,
        ast.ListComp,
        ast.SetComp,
        ast.DictComp,
        ast.GeneratorExp,
    )
    module_tree = ast.parse(inspect.getsource(bigram))
    defs = {
        n.name: n
        for n in ast.walk(module_tree)
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))
    }
    found, seen, todo = [], set(), ["bigram_counts"]
    while todo:
        name = todo.pop()
        if name in seen or name not in defs:
            continue
        seen.add(name)
        for node in ast.walk(defs[name]):
            if isinstance(node, loops):
                found.append(f"{type(node).__name__} in {name}")
            elif isinstance(node, ast.Name) and node.id in defs:
                todo.append(node.id)
    assert "bigram_counts" in seen, "bigram.py defines no bigram_counts"
    assert not found, f"bigram_counts runs a Python loop: {found}"


def test_row_normalize_hand_example():
    # WHY: section 3: [[1, 3], [1, 1]] has row sums 4 and 2, so
    #      P = [[1/4, 3/4], [1/2, 1/2]].
    # KIND: unit
    # CHAPTER: lang.01 section 3
    p = row_normalize(np.array([[1, 3], [1, 1]]))
    assert p.dtype == np.float64
    assert_close(p, np.array([[0.25, 0.75], [0.5, 0.5]]))


def test_row_normalize_divides_rows_not_columns():
    # WHY: `c / c.sum(axis=1)` (no keepdims) broadcasts the (n,) sums along
    #      the LAST axis and divides column j by row j's total. A square
    #      table gives no shape error, only wrong numbers. The rows of a
    #      correct result sum to 1.
    # KIND: boundary
    # CHAPTER: lang.01 section 5, pitfall 9
    c = np.array([[1, 3, 0], [1, 1, 2], [5, 0, 5]])
    p = row_normalize(c)
    assert_close(p.sum(axis=1), np.ones(3))
    assert_close(p[1], np.array([0.25, 0.25, 0.5]))


def test_row_normalize_leaves_empty_rows_zero():
    # WHY: most of the 256 rows are empty for real text. 0/0 is NaN, and one
    #      NaN in a probability table poisons every sum that touches it.
    # KIND: boundary
    # CHAPTER: lang.01 section 5, pitfall 10
    p = row_normalize(bigram_counts(b"banana"))
    assert not np.isnan(p).any()
    sums = p.sum(axis=1)
    nonzero = sorted({ord(ch) for ch in "ban"})
    assert_close(sums[nonzero], np.ones(3))
    assert np.count_nonzero(sums) == 3
