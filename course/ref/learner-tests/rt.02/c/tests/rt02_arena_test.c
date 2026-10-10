/* My tests for rt.02 (rung R4): the arena against a model of live allocations, with ss_prop.h, plus the
 * boundary and fault cases. Only the contract headers are included. */
#include <stdint.h>
#include <string.h>

#include "tinyllm.h"
#include "ss_prop.h"
#include "ss_test.h"

static uintptr_t addr(const void *p) { return (uintptr_t)p; }

static tl_arena_stats stats(const tl_arena *a) {
    tl_arena_stats s;
    memset(&s, 0xAB, sizeof s);
    tl_arena_stats_get(a, &s);
    return s;
}

SS_TEST(hand_example) {
    tl_arena *ar = NULL;
    SS_EQ(tl_arena_create(256, &ar), TL_OK);
    char *a = tl_arena_alloc(ar, 10, 0);
    SS_TRUE(a != NULL && addr(a) % 64 == 0);
    size_t u0 = stats(ar).bytes_used;
    SS_TRUE(u0 >= 10 && u0 <= 10 + 63);
    char *b = tl_arena_alloc(ar, 4, 64);
    SS_EQ(b - a, 64);
    SS_EQ(stats(ar).bytes_used, u0 + 58);
    tl_arena_mark m = tl_arena_mark_get(ar);
    char *c = tl_arena_alloc(ar, 100, 8);
    SS_EQ(c - a, 72);
    SS_EQ(stats(ar).bytes_used, u0 + 58 + 104);
    tl_arena_reset_to(ar, m);
    SS_EQ(stats(ar).bytes_used, u0 + 58);
    SS_EQ(stats(ar).high_water, u0 + 58 + 104);
    char *c2 = tl_arena_alloc(ar, 100, 8);
    SS_TRUE(c2 == c);
    tl_arena_stats s = stats(ar);
    SS_EQ(s.n_blocks, 1);
    SS_EQ(s.bytes_reserved, 256);
    tl_arena_destroy(ar);
}

SS_TEST(every_power_of_two_alignment) {
    tl_arena *ar = NULL;
    SS_EQ(tl_arena_create(1 << 16, &ar), TL_OK);
    for (size_t al = 1; al <= 4096; al *= 2) {
        SS_TRUE(tl_arena_alloc(ar, 3, 1) != NULL); /* knock the offset off any boundary */
        void *p = tl_arena_alloc(ar, 7, al);
        SS_TRUE(p != NULL);
        SS_EQ(addr(p) % al, 0);
    }
    SS_TRUE(tl_arena_alloc(ar, 3, 1) != NULL);
    SS_EQ(addr(tl_arena_alloc(ar, 1, 0)) % 64, 0);
    tl_arena_destroy(ar);
}

SS_TEST(bad_requests_change_nothing) {
    tl_arena *ar = NULL;
    SS_EQ(tl_arena_create(1024, &ar), TL_OK);
    char *a = tl_arena_alloc(ar, 16, 16);
    tl_arena_stats before = stats(ar);
    tl_set_last_error("");
    SS_TRUE(tl_arena_alloc(ar, 0, 16) == NULL);
    SS_TRUE(strstr(tl_last_error(), "tl_arena_alloc") != NULL);
    tl_set_last_error("");
    SS_TRUE(tl_arena_alloc(ar, 8, 3) == NULL);
    SS_TRUE(strstr(tl_last_error(), "tl_arena_alloc") != NULL);
    SS_TRUE(tl_arena_alloc(ar, 8, 8192) == NULL);
    tl_arena_stats after = stats(ar);
    SS_EQ(after.bytes_used, before.bytes_used);
    SS_EQ(after.n_blocks, before.n_blocks);
    SS_TRUE(tl_arena_alloc(ar, 16, 16) == a + 16);
    SS_EQ(tl_arena_create(64, NULL), TL_EINVAL);
    tl_arena_destroy(ar);
    tl_arena_destroy(NULL);
}

SS_TEST(nested_marks) {
    tl_arena *ar = NULL;
    SS_EQ(tl_arena_create(256, &ar), TL_OK); /* each step below needs a new block */
    (void)tl_arena_alloc(ar, 100, 0);
    tl_arena_mark outer = tl_arena_mark_get(ar);
    char *x = tl_arena_alloc(ar, 200, 0);
    tl_arena_mark inner = tl_arena_mark_get(ar);
    char *y = tl_arena_alloc(ar, 300, 0);
    tl_arena_reset_to(ar, inner);
    SS_TRUE(tl_arena_alloc(ar, 300, 0) == y);
    tl_arena_reset_to(ar, inner);
    tl_arena_reset_to(ar, outer);
    SS_TRUE(tl_arena_alloc(ar, 200, 0) == x);
    tl_arena_destroy(ar);
}

SS_TEST(a_later_mark_is_ignored) {
    tl_arena *ar = NULL;
    SS_EQ(tl_arena_create(128, &ar), TL_OK);
    tl_arena_mark start = tl_arena_mark_get(ar);
    (void)tl_arena_alloc(ar, 100, 0);
    (void)tl_arena_alloc(ar, 100, 0); /* a second block */
    tl_arena_mark late = tl_arena_mark_get(ar);
    tl_arena_reset_to(ar, start);
    char *p = tl_arena_alloc(ar, 50, 0);
    size_t used = stats(ar).bytes_used;
    tl_set_last_error("");
    tl_arena_reset_to(ar, late);
    SS_TRUE(strstr(tl_last_error(), "tl_arena_reset_to") != NULL);
    SS_EQ(stats(ar).bytes_used, used);
    char *q = tl_arena_alloc(ar, 10, 1);
    SS_TRUE(q == p + 50);
    tl_arena_destroy(ar);
}

SS_TEST(blocks_are_kept_and_reused) {
    tl_arena *ar = NULL;
    SS_EQ(tl_arena_create(128, &ar), TL_OK);
    tl_arena_mark start = tl_arena_mark_get(ar);
    char *a = tl_arena_alloc(ar, 100, 0), *b = tl_arena_alloc(ar, 100, 0);
    SS_EQ(stats(ar).n_blocks, 2);
    SS_EQ(stats(ar).bytes_reserved, 256);
    long live = ss_live_allocs();
    tl_arena_reset_to(ar, start);
    SS_EQ(ss_live_allocs(), live);
    SS_TRUE(tl_arena_alloc(ar, 100, 0) == a);
    SS_TRUE(tl_arena_alloc(ar, 100, 0) == b);
    SS_EQ(stats(ar).n_blocks, 2);
    SS_EQ(ss_live_allocs(), live);
    tl_arena_destroy(ar);
}

SS_TEST(exact_fit_uses_the_last_byte) {
    tl_arena *ar = NULL;
    SS_EQ(tl_arena_create(128, &ar), TL_OK);
    char *a = tl_arena_alloc(ar, 64, 0);
    char *b = tl_arena_alloc(ar, 64, 0);
    SS_TRUE(a != NULL && b == a + 64);
    SS_EQ(stats(ar).n_blocks, 1);
    tl_arena_destroy(ar);
}

SS_TEST(oversized_request_gets_its_own_block) {
    tl_arena *ar = NULL;
    SS_EQ(tl_arena_create(128, &ar), TL_OK);
    unsigned char *p = tl_arena_alloc(ar, 1000, 256);
    SS_TRUE(p != NULL && addr(p) % 256 == 0);
    memset(p, 0x5A, 1000);
    SS_TRUE(stats(ar).bytes_reserved >= 1000);
    unsigned char *q = tl_arena_alloc(ar, 20, 0);
    SS_TRUE(q != NULL);
    memset(q, 0x33, 20);
    for (int i = 0; i < 1000; i++) SS_EQ(p[i], 0x5A);
    tl_arena_destroy(ar);
}

/* Model-based random operations: a stack of marks, live allocations filled
 * with a byte pattern, and the stats the contract promises. */
enum { MAXLIVE = 256, MAXMARKS = 32 };
typedef struct {
    unsigned char *p;
    size_t n;
    unsigned char tag;
} live_t;

static int random_ops(ss_gen *g, int size) {
    tl_arena *ar = NULL;
    if (tl_arena_create((size_t)ss_gen_int(g, 64, 512), &ar) != TL_OK) return 0;
    live_t live[MAXLIVE];
    int nlive = 0;
    struct {
        tl_arena_mark m;
        int nlive;
        size_t used;
    } marks[MAXMARKS];
    int nmarks = 0, ok = 1;
    size_t peak = 0;
    for (int step = 0; step < 20 + size && ok; step++) {
        int op = (int)ss_gen_int(g, 0, 9);
        if (op < 6 && nlive < MAXLIVE) {
            size_t n = (size_t)ss_gen_int(g, 1, 700), al = (size_t)1 << ss_gen_int(g, 0, 8);
            unsigned char *p = tl_arena_alloc(ar, n, al);
            if (p == NULL || addr(p) % al != 0) ok = 0;
            else {
                for (int i = 0; i < nlive; i++) /* no overlap with a live allocation */
                    if (p < live[i].p + live[i].n && live[i].p < p + n) ok = 0;
                unsigned char tag = (unsigned char)ss_gen_int(g, 1, 255);
                memset(p, tag, n);
                live[nlive++] = (live_t){p, n, tag};
            }
        } else if (op < 8 && nmarks < MAXMARKS) {
            marks[nmarks].m = tl_arena_mark_get(ar);
            marks[nmarks].nlive = nlive;
            marks[nmarks].used = 0;
            tl_arena_stats s;
            tl_arena_stats_get(ar, &s);
            marks[nmarks++].used = s.bytes_used;
        } else if (nmarks > 0) {
            int k = (int)ss_gen_int(g, 0, nmarks - 1); /* any earlier mark */
            tl_arena_reset_to(ar, marks[k].m);
            nlive = marks[k].nlive;
            tl_arena_stats s;
            tl_arena_stats_get(ar, &s);
            if (s.bytes_used != marks[k].used) ok = 0;
            nmarks = k;
        }
        tl_arena_stats s;
        tl_arena_stats_get(ar, &s);
        if (s.bytes_used > peak) peak = s.bytes_used;
        size_t sum = 0;
        for (int i = 0; i < nlive; i++) sum += live[i].n;
        if (s.bytes_used < sum || s.high_water != peak || s.bytes_used > s.bytes_reserved) ok = 0;
    }
    for (int i = 0; i < nlive && ok; i++) /* nothing live was overwritten */
        for (size_t j = 0; j < live[i].n; j++)
            if (live[i].p[j] != live[i].tag) {
                ok = 0;
                break;
            }
    tl_arena_destroy(ar);
    return ok;
}

SS_TEST(random_operations_against_a_model) {
    SS_CHECK_PROP(random_ops, 300, 200);
}

SS_TEST(zero_hook_calls_after_warmup) {
    tl_arena *ar = NULL;
    SS_EQ(tl_arena_create(4096, &ar), TL_OK);
    static const size_t sizes[] = {1000, 3000, 64, 5000, 200, 4000};
    for (int step = 0; step < 1003; step++) {
        if (step == 3) ss_alloc_fail_after(0);
        tl_arena_mark m = tl_arena_mark_get(ar);
        for (int i = 0; i < 6; i++) {
            if (tl_arena_alloc(ar, sizes[i], 64) == NULL) {
                ss_alloc_fail_after(-1);
                tl_arena_destroy(ar);
                SS_FAIL("an allocation after warmup reached the hook");
            }
        }
        tl_arena_reset_to(ar, m);
    }
    ss_alloc_fail_after(-1);
    tl_arena_stats s = stats(ar);
    SS_EQ(s.bytes_used, 0);
    SS_TRUE(s.high_water >= 13264);
    tl_arena_destroy(ar);
}

SS_TEST(hook_failure_leaves_the_arena_usable) {
    for (long n = 0; n < 40; n++) {
        ss_alloc_fail_after(n);
        tl_arena *ar = NULL;
        tl_status st = tl_arena_create(64, &ar);
        if (st != TL_OK) {
            SS_EQ(st, TL_ENOMEM);
            SS_TRUE(ar == NULL);
            continue;
        }
        int failed = 0;
        for (int i = 0; i < 30; i++) {
            tl_arena_stats before = stats(ar);
            void *p = tl_arena_alloc(ar, 48, 0); /* one block each time */
            if (p == NULL) {
                failed = 1;
                tl_arena_stats after = stats(ar);
                SS_EQ(after.bytes_used, before.bytes_used);
                SS_EQ(after.n_blocks, before.n_blocks);
            } else {
                memset(p, 1, 48);
            }
        }
        ss_alloc_fail_after(-1);
        /* Half the time destroy at once, half after a successful request,
         * so both "never got a block" and "recovered" end states are freed. */
        if (n % 2 == 1) SS_TRUE(tl_arena_alloc(ar, 48, 0) != NULL || !failed);
        tl_arena_destroy(ar);
    }
    ss_alloc_fail_after(-1);
}

int main(void) { return SS_RUN_ALL(); }
