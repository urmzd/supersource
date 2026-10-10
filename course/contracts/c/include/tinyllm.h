/* contracts/c/include/tinyllm.h: optional C-only convenience umbrella.
 *
 * Include an individual header under tinyllm/ when a smaller interface is
 * sufficient. These declarations are for standalone C callers; they do not
 * define bindings for Python or Rust, or require a shared library.
 *
 * The rules every header follows are in c/ABI.md. The v2 KV format is
 * intentionally separate and is declared in tinyllm/kv_pool_v2.h. */
#ifndef TINYLLM_H
#define TINYLLM_H

#include "tinyllm/abi.h"         /* shared C-only support, owned by rt.02 */
#include "tinyllm/arena.h"       /* rt.02 */
#include "tinyllm/pool.h"        /* rt.03 */
#include "tinyllm/kv_pool.h"     /* rt.04 */
#include "tinyllm/ds.h"          /* ds.01 to ds.03 */
#include "tinyllm/topk.h"        /* ds.04 */
#include "tinyllm/numerics.h"    /* optional C numerics: M09.5, M09.6, M09.7 */
#include "tinyllm/matmul.h"      /* optional C matmul: L9.1 */
#include "tinyllm/softmax.h"     /* L9.2 */
#include "tinyllm/attention.h"   /* L9.3, L9.4 */
#include "tinyllm/qmatmul.h"     /* L9.5 */
#include "tinyllm/elementwise.h" /* L9.6 */

#endif /* TINYLLM_H */
