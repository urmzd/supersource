/* tinyllm/numerics.h: the random generator, hashing, low-precision formats,
 * and the two transcendental kernels the runtime needs. Rules in c/ABI.md.
 *
 *   tl_pcg32_*, tl_fnv1a64   M06.3  c/src/numerics/rng.c     spec/pcg32.md
 *   tl_f32_to_* / tl_*_to_f32 M09.4  c/src/numerics/lowp.c
 *   tl_rsqrtf, tl_rsqrt_f32   M09.5  c/src/numerics/rsqrt.c   (Newton)
 *   tl_expf, tl_exp_f32       M09.6  c/src/numerics/expf.c    (range reduction + polynomial)
 *
 * Every function here is pure or touches only its own generator, so all are
 * reentrant. */
#ifndef TINYLLM_NUMERICS_H
#define TINYLLM_NUMERICS_H

#include <stddef.h>
#include <stdint.h>

#include "tinyllm/abi.h"

#ifdef __cplusplus
extern "C" {
#endif

/* -- M06.3: PCG32 and FNV-1a (spec/pcg32.md) -------------------------------- */

typedef struct {
    uint64_t state, inc;
} tl_pcg32; /* 16 bytes */

/* O'Neill's pcg32_srandom_r(r, seed, seq): inc = (seq << 1) | 1, state = 0,
 * one step, state += seed, one step. The default stream is seq = 54. */
void tl_pcg32_seed(tl_pcg32 *r, uint64_t seed, uint64_t seq);

/* PCG-XSH-RR 64/32: one step, then the permuted output of the old state. */
uint32_t tl_pcg32_next(tl_pcg32 *r);

/* Two draws a, b: ((a >> 5) * 2^26 + (b >> 6)) * 2^-53, in [0, 1). */
double tl_pcg32_uniform(tl_pcg32 *r);

/* FNV-1a 64 over n bytes, continuing from h (pass TL_FNV1A64_OFFSET to
 * start): for each byte, h ^= byte, then h *= 0x100000001B3 (mod 2^64).
 * tl_fnv1a64("a", 1, TL_FNV1A64_OFFSET) == 0xAF63DC4C8601EC8C. */
#define TL_FNV1A64_OFFSET 0xCBF29CE484222325ull
uint64_t tl_fnv1a64(const void *data, size_t n, uint64_t h);

/* -- M09.4: low-precision conversions ------------------------------------------ */

/* f32 to 8- and 16-bit formats round to nearest, ties to even. Edge rules:
 *
 *   e4m3 (OCP E4M3FN: bias 7, no infinity, max 448)
 *        |x| > 448 (and +-inf) saturates to +-448 (0x7E, 0xFE);
 *        NaN gives 0x7F; |x| <= 2^-10 (half the smallest subnormal,
 *        2^-9) rounds to +-0 with the sign kept.
 *   e5m2 (OCP E5M2: bias 15, max finite 57344)
 *        finite |x| that rounds above 57344 saturates to +-57344
 *        (0x7B, 0xFB); +-inf stays +-inf (0x7C, 0xFC); NaN gives 0x7F.
 *   bf16 the upper 16 bits after rounding; NaN gives 0x7FC0 (quiet,
 *        sign dropped); overflow goes to +-inf.
 *   f16  IEEE binary16 with subnormals; overflow goes to +-inf (0x7C00);
 *        NaN gives 0x7E00.
 *
 * The *_to_f32 directions are exact for every code (all 256 or 65536 codes
 * are tested bitwise; any NaN code gives a quiet f32 NaN). */
uint8_t tl_f32_to_e4m3(float x);
float tl_e4m3_to_f32(uint8_t b);
uint8_t tl_f32_to_e5m2(float x);
float tl_e5m2_to_f32(uint8_t b);
uint16_t tl_f32_to_bf16(float x);
float tl_bf16_to_f32(uint16_t b);
uint16_t tl_f32_to_f16(float x);
float tl_f16_to_f32(uint16_t b);

/* -- M09.5: reciprocal square root ----------------------------------------------- */

/* 1 / sqrt(x) by a bit-level initial guess and Newton steps, within 2 ulp of
 * the correctly rounded result for every positive normal and subnormal x.
 * +0 gives +inf, -0 gives -inf, +inf gives +0, x < 0 and NaN give NaN.
 * The array form applies it elementwise; x and y may alias exactly. */
float tl_rsqrtf(float x);
void tl_rsqrt_f32(const float *x, float *y, int64_t n);

/* -- M09.6: exponential ---------------------------------------------------------------- */

/* e^x by reduction x = k ln2 + r and a polynomial in r, within 2 ulp of the
 * correctly rounded result wherever that result is a normal float.
 * x > 88.7228... gives +inf; results below the normal range may be flushed
 * to +0; -inf gives +0; NaN gives NaN. Same aliasing rule as above. */
float tl_expf(float x);
void tl_exp_f32(const float *x, float *y, int64_t n);

#ifdef __cplusplus
}
#endif

#endif /* TINYLLM_NUMERICS_H */
