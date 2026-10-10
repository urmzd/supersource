/* My tests for ds.02 (rung R2: the chapter named them, I wrote the bodies).
 * They include only the contract headers and the test kit. */
#include <stdint.h>
#include <string.h>

#include "tinyllm.h"
#include "ss_prop.h"
#include "ss_test.h"

static uint64_t mix(uint64_t x) {
    x ^= x >> 30;
    x *= 0xBF58476D1CE4E5B9ull;
    x ^= x >> 27;
    x *= 0x94D049BB133111EBull;
    return x ^ (x >> 31);
}

SS_TEST(fourteen_fit_then_grow) {
    tl_map *m = NULL;
    SS_EQ(tl_map_create(14, &m), TL_OK);
    ss_alloc_fail_after(0);
    for (uint64_t k = 0; k < 14; k++) SS_EQ(tl_map_put(m, k, k), TL_OK);
    SS_EQ(tl_map_put(m, 99, 1), TL_ENOMEM);
    ss_alloc_fail_after(-1);
    SS_EQ(tl_map_len(m), 14);
    SS_EQ(tl_map_put(m, 99, 1), TL_OK);
    tl_map_destroy(m);
}

SS_TEST(overwrite_keeps_len) {
    tl_map *m = NULL;
    SS_EQ(tl_map_create(0, &m), TL_OK);
    SS_EQ(tl_map_put(m, 5, 1), TL_OK);
    SS_EQ(tl_map_put(m, 5, 2), TL_OK);
    uint64_t v = 0;
    SS_EQ(tl_map_get(m, 5, &v), 1);
    SS_EQ(v, 2);
    SS_EQ(tl_map_len(m), 1);
    tl_map_destroy(m);
}

SS_TEST(shared_fingerprint_keys) {
    tl_map *m = NULL;
    SS_EQ(tl_map_create(14, &m), TL_OK);
    uint64_t keys[14];
    int n = 0;
    for (uint64_t k = 1; n < 14; k++)
        if ((mix(k) & 0x7F) == 0x11) keys[n++] = k;
    for (int i = 0; i < 14; i++) SS_EQ(tl_map_put(m, keys[i], (uint64_t)i), TL_OK);
    for (int i = 0; i < 14; i++) {
        uint64_t v = 99;
        SS_EQ(tl_map_get(m, keys[i], &v), 1);
        SS_EQ(v, (uint64_t)i);
    }
    tl_map_destroy(m);
}

SS_TEST(spilled_keys_survive_deletes) {
    tl_map *m = NULL;
    SS_EQ(tl_map_create(28, &m), TL_OK);
    uint64_t keys[20];
    int n = 0;
    for (uint64_t k = 1; n < 20; k++)
        if (((mix(k) >> 7) & 1) == 0) keys[n++] = k;
    for (int i = 0; i < 20; i++) SS_EQ(tl_map_put(m, keys[i], 1), TL_OK);
    for (int i = 0; i < 16; i += 2) SS_EQ(tl_map_del(m, keys[i]), 1);
    for (int i = 0; i < 20; i++) SS_EQ(tl_map_get(m, keys[i], NULL), (i < 16 && i % 2 == 0) ? 0 : 1);
    SS_EQ(tl_map_put(m, keys[19], 7), TL_OK); /* overwrite, not a second copy */
    SS_EQ(tl_map_len(m), 12);
    SS_EQ(tl_map_del(m, keys[19]), 1);
    SS_EQ(tl_map_get(m, keys[19], NULL), 0);
    tl_map_destroy(m);
}

SS_TEST(failed_growth_keeps_keys) {
    tl_map *m = NULL;
    SS_EQ(tl_map_create(14, &m), TL_OK);
    for (uint64_t k = 0; k < 14; k++) SS_EQ(tl_map_put(m, k * 3, k), TL_OK);
    for (long f = 0; f < 4; f++) {
        ss_alloc_fail_after(f);
        tl_status st = tl_map_put(m, 1000, 1);
        ss_alloc_fail_after(-1);
        if (st == TL_OK) break;
        for (uint64_t k = 0; k < 14; k++) SS_EQ(tl_map_get(m, k * 3, NULL), 1);
    }
    SS_EQ(tl_map_len(m), 15);
    tl_map_destroy(m);
}

static int vs_model(ss_gen *g, int size) {
    enum { K = 512 };
    static uint64_t val[K];
    static int has[K];
    memset(has, 0, sizeof has);
    tl_map *m = NULL;
    if (tl_map_create(0, &m) != TL_OK) return 0;
    int ok = 1;
    size_t len = 0;
    for (int i = 0; i < 2000 + 200 * size && ok; i++) {
        int k = (int)ss_gen_int(g, 0, K - 1), op = (int)ss_gen_int(g, 0, 2);
        if (op == 0) {
            uint64_t v = ss_gen_u64(g);
            ok &= tl_map_put(m, (uint64_t)k, v) == TL_OK;
            len += !has[k];
            has[k] = 1;
            val[k] = v;
        } else if (op == 1) {
            uint64_t v = 0;
            int f = tl_map_get(m, (uint64_t)k, &v);
            ok &= f == has[k] && (!f || v == val[k]);
        } else {
            ok &= tl_map_del(m, (uint64_t)k) == has[k];
            len -= has[k];
            has[k] = 0;
        }
        ok &= tl_map_len(m) == len;
    }
    tl_map_destroy(m);
    return ok;
}

SS_TEST(matches_a_simple_model) { SS_CHECK_PROP(vs_model, 20, 20); }

int main(void) { return SS_RUN_ALL(); }
