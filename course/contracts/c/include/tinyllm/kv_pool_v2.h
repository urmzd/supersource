/* tinyllm/kv_pool_v2.h (craft.13): KV format v2, fp8 e4m3 with scales.
 * Rules in c/ABI.md; the v2 payload in formats/kv-block.md.
 *
 * Contract 0.x ships this header but tinyllm.h does not include it: D13
 * keeps v2 out of the system until the craft.13 migration, whose contract
 * sync adds the #include line and lets tl_kv_pool_create accept
 * format = TL_KV_FORMAT_V2. Before that, nothing implements these symbols.
 *
 * A v2 pool (cfg.format == 2, cfg.dtype == TL_F8_E4M3) stores K and V as
 * e4m3 bytes (tl_f32_to_e4m3 of x / scale) with one f32 scale per
 * (layer, K or V, KV head) per block. tl_kv_block_ptr returns the e4m3
 * slab; the scales sit beside it. A v1 reader refuses a v2 envelope with
 * TL_EFORMAT (version mismatch), which is what "v1 readers refuse v2
 * cleanly" in drill ops.04 checks.
 *
 * module: craft.13 (upgrades c/src/runtime/kv_pool.c and paged_attn.c) */
#ifndef TINYLLM_KV_POOL_V2_H
#define TINYLLM_KV_POOL_V2_H

#include <stdint.h>

#include "tinyllm/abi.h"
#include "tinyllm/kv_pool.h"

#ifdef __cplusplus
extern "C" {
#endif

#define TL_KV_FORMAT_V2 2u

/* The n_kv_heads scales of the K (is_v == 0) or V slab of one layer of one
 * block: element (h, t, d) of the slab decodes to
 * tl_e4m3_to_f32(slab[(h * block_tokens + t) * head_dim + d]) * scales[h].
 * A writer chooses scales[h] = amax / 448 over the head's slab (1.0 when
 * amax is 0), so no value saturates. NULL, with the error slot set, for a
 * v1 pool or an id or layer out of range. */
float *tl_kv_block_scales(const tl_kv_pool *p, uint32_t id, uint32_t layer, int is_v);

/* The envelope versions a pool of this build can import: writes up to cap
 * values into out (ascending) and returns how many exist. A v2 build
 * returns 2 (versions 1 and 2): a v2 pool imports a v1 envelope by
 * converting each f16 slab to e4m3 with the scales chosen as above, which
 * is how a v2 decode worker serves v1 prefill workers during a rolling
 * upgrade. The engine reports the list in InfoResponse.kv_formats_read. */
uint32_t tl_kv_formats_supported(uint32_t *out, uint32_t cap);

#ifdef __cplusplus
}
#endif

#endif /* TINYLLM_KV_POOL_V2_H */
