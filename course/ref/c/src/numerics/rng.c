/* c/src/numerics/rng.c (M06.3): PCG32 and FNV-1a 64 in C.
 * Contract: tinyllm/numerics.h; algorithms: spec/pcg32.md. The Python twin is
 * python/tinyllm/num/rng.py, and the two must produce the same stream bit for
 * bit. uint64_t and uint32_t arithmetic wraps modulo 2^64 and 2^32, which is
 * exactly the modular arithmetic the spec asks for.
 */
#include <stddef.h>
#include <stdint.h>

#include "tinyllm/numerics.h"

#define TL_PCG32_MULT 6364136223846793005ull
#define TL_FNV1A64_PRIME 0x100000001B3ull

void tl_pcg32_seed(tl_pcg32 *r, uint64_t seed, uint64_t seq) {
/* SOLUTION-BEGIN M06.3 */
    r->state = 0u;
    r->inc = (seq << 1u) | 1u; /* odd: the LCG has full period 2^64 */
    (void)tl_pcg32_next(r);
    r->state += seed;
    (void)tl_pcg32_next(r);
/* SOLUTION-END */
}

uint32_t tl_pcg32_next(tl_pcg32 *r) {
/* SOLUTION-BEGIN M06.3 */
    uint64_t old = r->state;
    r->state = old * TL_PCG32_MULT + r->inc;
    uint32_t xs = (uint32_t)(((old >> 18u) ^ old) >> 27u);
    uint32_t rot = (uint32_t)(old >> 59u);
    /* -rot & 31 keeps the left shift below 32 when rot == 0: shifting a
     * 32-bit value by 32 is undefined behavior in C. */
    return (xs >> rot) | (xs << ((0u - rot) & 31u));
/* SOLUTION-END */
}

double tl_pcg32_uniform(tl_pcg32 *r) {
/* SOLUTION-BEGIN M06.3 */
    uint64_t a = tl_pcg32_next(r) >> 5u; /* 27 bits */
    uint64_t b = tl_pcg32_next(r) >> 6u; /* 26 bits */
    return (double)((a << 26u) | b) * (1.0 / 9007199254740992.0); /* * 2^-53 */
/* SOLUTION-END */
}

uint64_t tl_fnv1a64(const void *data, size_t n, uint64_t h) {
/* SOLUTION-BEGIN M06.3 */
    const unsigned char *p = (const unsigned char *)data;
    for (size_t i = 0; i < n; i++) {
        h ^= (uint64_t)p[i];
        h *= TL_FNV1A64_PRIME;
    }
    return h;
/* SOLUTION-END */
}
