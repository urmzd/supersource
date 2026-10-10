/* tinyllm/topk.h (ds.04): top-k by a binary min-heap of size k, O(n log k).
 * Rules in c/ABI.md. The Rust sampler calls it through tl-sys for top_k
 * (spec/sampling.md step 4).
 *
 * module: ds.04 (c/src/ds/topk.c)
 * chapter: algorithms/16-systems-data-structures/04-binary-heap-top-k.md
 */
#ifndef TINYLLM_TOPK_H
#define TINYLLM_TOPK_H

#include <stdint.h>

#include "tinyllm/abi.h"

#ifdef __cplusplus
extern "C" {
#endif

/* Writes the k largest values of x[0..n) and their indices, sorted by value
 * descending; among equal values the lower index comes first (and wins the
 * last place). -inf is an ordinary value. Needs 0 <= k <= n; k == 0 writes
 * nothing. TL_EINVAL, with idx and val untouched, for k out of range, a
 * NULL pointer that would be written, or any NaN in x. Allocates nothing:
 * the heap lives in idx and val. */
tl_status tl_topk_f32(const float *x, int64_t n, int64_t k, int32_t *idx, float *val);

#ifdef __cplusplus
}
#endif

#endif /* TINYLLM_TOPK_H */
