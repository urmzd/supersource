"""The frozen PCG32 every course test draws from (DESIGN D10, spec/pcg32.md).

PCG32 is PCG-XSH-RR with 64-bit state and 32-bit output (O'Neill 2014):

    old    = state
    state  = old * 6364136223846793005 + inc            (mod 2^64)
    xs     = ((old >> 18) ^ old) >> 27                   (low 32 bits)
    rot    = old >> 59
    output = (xs >> rot) | (xs << ((-rot) & 31))         (32-bit rotate right)

Seeding with (seed, seq) is the reference `pcg32_srandom_r`: inc = (seq << 1) | 1,
state = 0, step, state += seed, step. `uniform()` builds a 53-bit double from
two draws, `((a >> 5) * 2^26 + (b >> 6)) / 2^53`, in [0, 1). `split(i)` derives
an independent child generator with SplitMix64 so tests can give each case
its own stream from one seed.
"""

from __future__ import annotations

import math

MASK64 = (1 << 64) - 1
MASK32 = (1 << 32) - 1
MULT = 6364136223846793005


def splitmix64(x: int) -> tuple[int, int]:
    """One SplitMix64 step: returns (next_state, output)."""
    x = (x + 0x9E3779B97F4A7C15) & MASK64
    z = x
    z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & MASK64
    z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & MASK64
    return x, z ^ (z >> 31)


class PCG32:
    __slots__ = ("state", "inc")

    def __init__(self, seed: int = 0, seq: int = 54):
        self.state = 0
        self.inc = ((seq << 1) | 1) & MASK64
        self.next_u32()
        self.state = (self.state + (seed & MASK64)) & MASK64
        self.next_u32()

    def next_u32(self) -> int:
        old = self.state
        self.state = (old * MULT + self.inc) & MASK64
        xs = (((old >> 18) ^ old) >> 27) & MASK32
        rot = old >> 59
        return ((xs >> rot) | (xs << ((-rot) & 31))) & MASK32

    def uniform(self) -> float:
        a, b = self.next_u32() >> 5, self.next_u32() >> 6
        return (a * 67108864.0 + b) / 9007199254740992.0

    def normal(self) -> float:
        """Box-Muller, cosine branch; consumes two uniforms."""
        u1, u2 = self.uniform(), self.uniform()
        return math.sqrt(-2.0 * math.log(1.0 - u1)) * math.cos(2.0 * math.pi * u2)

    def below(self, n: int) -> int:
        """Uniform integer in [0, n) without modulo bias (rejection)."""
        if not 0 < n <= 1 << 32:
            raise ValueError("n must be in 1..2^32")
        threshold = ((1 << 32) - n) % n
        while True:
            r = self.next_u32()
            if r >= threshold:
                return r % n

    def split(self, i: int) -> "PCG32":
        _, a = splitmix64((self.state ^ (i * 0xD1B54A32D192ED03)) & MASK64)
        _, b = splitmix64(a)
        return PCG32(a, b)

    def uniform_array(self, shape, lo: float = 0.0, hi: float = 1.0):
        """A numpy float64 array of uniforms in [lo, hi), drawn in C order."""
        import numpy as np

        n = int(np.prod(shape)) if shape != () else 1
        out = np.fromiter(
            (lo + (hi - lo) * self.uniform() for _ in range(n)),
            dtype=np.float64,
            count=n,
        )
        return out.reshape(shape)

    def normal_array(self, shape, scale: float = 1.0):
        import numpy as np

        n = int(np.prod(shape)) if shape != () else 1
        out = np.fromiter(
            (scale * self.normal() for _ in range(n)), dtype=np.float64, count=n
        )
        return out.reshape(shape)
