/* contracts/c/include/tinyllm.h: the umbrella header of the C ABI (ABI v1).
 *
 * Each unit has its own header under tinyllm/, so one module owns one file.
 * The rules every header follows are in c/ABI.md.
 *
 * Contract 0.2 declares every unit of ABI v1. A unit that is not built yet
 * is linked as a stub (each function sets the error slot and returns
 * TL_EUNSUPPORTED or its type's zero value), so the whole header is always
 * safe to include. tinyllm/kv_pool_v2.h (KV format v2) is not included
 * here: the craft.13 migration adds it (D13). */
#ifndef TINYLLM_H
#define TINYLLM_H

#include "tinyllm/abi.h"         /* rt.01 */
#include "tinyllm/arena.h"       /* rt.02 */
#include "tinyllm/pool.h"        /* rt.03 */
#include "tinyllm/kv_pool.h"     /* rt.04 */
#include "tinyllm/ds.h"          /* ds.01 to ds.03 */
#include "tinyllm/topk.h"        /* ds.04 */
#include "tinyllm/numerics.h"    /* M06.3, M09.4, M09.5, M09.6 */
#include "tinyllm/matmul.h"      /* M03.1 (v0), upgraded by L9.1 */
#include "tinyllm/softmax.h"     /* L9.2 */
#include "tinyllm/attention.h"   /* L9.3, L9.4 */
#include "tinyllm/qmatmul.h"     /* L9.5 */
#include "tinyllm/elementwise.h" /* L9.6 */

#endif /* TINYLLM_H */
