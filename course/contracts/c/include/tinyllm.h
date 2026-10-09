/* contracts/c/include/tinyllm.h: the umbrella header of the C ABI (ABI v1).
 *
 * Each unit has its own header under tinyllm/, so one module owns one file.
 * The rules every header follows are in c/ABI.md.
 *
 * Contract v0 holds the units of Pass 1 only. Later units add their header
 * and one #include line here when their batch lands (arena.h rt.02, pool.h
 * rt.03, kv_pool.h rt.04, ds.h ds.01 to ds.03, topk.h ds.04, numerics.h,
 * softmax.h L9.2, attention.h L9.3 and L9.4, qmatmul.h L9.5,
 * elementwise.h L9.6). Adding a header is a minor contract bump. */
#ifndef TINYLLM_H
#define TINYLLM_H

#include "tinyllm/abi.h"    /* rt.01 */
#include "tinyllm/matmul.h" /* M03.1 (v0), upgraded by L9.1 */

#endif /* TINYLLM_H */
