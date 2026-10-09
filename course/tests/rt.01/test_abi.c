/* Course tests for rt.01 (C side): c/src/runtime/abi.c against tinyllm/abi.h.
 *
 * Built by `ss check rt.01` with ASan and UBSan and -DSS_COUNTING_ALLOC=1:
 * SS_RUN_ALL installs a counting allocator through YOUR tl_set_allocator and
 * fails any test that ends with more blocks live than it started with.
 * Python-side tests (the ctypes loader) are in test_loader.py.
 */
#include <stdint.h>
#include <string.h>

#include "tinyllm.h"
#include "ss_test.h"

/* A hook that counts its own calls through the `user` pointer, so a test can
 * see which allocator served a request. */
typedef struct {
    long allocs, frees;
} hook_calls;

static void *counting_alloc(void *user, size_t n, size_t align) {
    hook_calls *h = user;
    h->allocs++;
    if (align < sizeof(void *)) align = sizeof(void *);
    size_t size = (n + align - 1) / align * align;
    return aligned_alloc(align, size);
}

static void counting_free(void *user, void *p) {
    hook_calls *h = user;
    h->frees++;
    free(p);
}

/* Put the harness's counting allocator back after a test replaced the hook,
 * so the leak check keeps working for the tests that follow. */
static void restore_harness_hook(void) {
#if SS__COUNTING
    tl_allocator a = {ss__alloc, ss__free, NULL};
    tl_set_allocator(&a);
#else
    tl_set_allocator(NULL);
#endif
}

SS_TEST(abi_version_matches_header) {
    /* WHY: bindings compare tl_abi_version() with the TL_ABI_VERSION they were
     *      written against and refuse a mismatch; a library that reports
     *      anything else cannot be loaded by Python or Rust at all.
     * KIND: unit, smoke
     * CATCHES: s01
     * CHAPTER: rt.01 section 4 */
    SS_EQ(tl_abi_version(), TL_ABI_VERSION);
    SS_EQ(tl_abi_version(), 1);
}

SS_TEST(status_str_names_every_code) {
    /* WHY: every status has its constant's name, so a log line or a Python
     *      exception says TL_EINVAL instead of 1. The table is the one the
     *      chapter's worked example walks.
     * KIND: unit
     * CATCHES: s02
     * CHAPTER: rt.01 section 3 */
    static const char *names[] = {"TL_OK", "TL_EINVAL", "TL_ENOMEM", "TL_ESHAPE", "TL_EDTYPE", "TL_EFULL",
                                  "TL_ENOTFOUND", "TL_EFORMAT", "TL_EBUSY", "TL_EUNSUPPORTED", "TL_EIO"};
    for (int32_t s = 0; s <= TL_EIO; s++) {
        const char *got = tl_status_str(s);
        SS_TRUE(got != NULL);
        if (strcmp(got, names[s]) != 0) SS_FAIL(names[s]);
    }
}

SS_TEST(status_str_unknown_codes) {
    /* WHY: tl_status crosses the boundary as a plain int32_t, so a caller can
     *      hand in any value: a negative one, or a code from a newer ABI. An
     *      array indexed by s would read out of bounds; the answer must be
     *      TL_UNKNOWN, never NULL and never a crash.
     * KIND: boundary
     * CATCHES: m001
     * CHAPTER: rt.01 section 5, Pitfalls */
    SS_TRUE(strcmp(tl_status_str(11), "TL_UNKNOWN") == 0);
    SS_TRUE(strcmp(tl_status_str(-1), "TL_UNKNOWN") == 0);
    SS_TRUE(strcmp(tl_status_str(INT32_MAX), "TL_UNKNOWN") == 0);
    SS_TRUE(strcmp(tl_status_str(INT32_MIN), "TL_UNKNOWN") == 0);
}

SS_TEST(error_slot_copies_and_truncates) {
    /* WHY: the error slot is given code, but every unit and every binding
     *      relies on it: it copies the message (the caller's buffer may be a
     *      stack array that dies on return), treats NULL as "", and truncates
     *      to TL_LAST_ERROR_CAP - 1 bytes instead of overflowing.
     * KIND: boundary, regression
     * CHAPTER: rt.01 section 2 */
    char buf[16] = "from a buffer";
    tl_set_last_error(buf);
    memset(buf, 'X', sizeof buf - 1);
    SS_TRUE(strcmp(tl_last_error(), "from a buffer") == 0);
    tl_set_last_error(NULL);
    SS_TRUE(tl_last_error() != NULL && tl_last_error()[0] == '\0');
    char big[TL_LAST_ERROR_CAP + 50];
    memset(big, 'a', sizeof big - 1);
    big[sizeof big - 1] = '\0';
    tl_set_last_error(big);
    SS_EQ((long long)strlen(tl_last_error()), TL_LAST_ERROR_CAP - 1);
}

SS_TEST(alloc_goes_through_the_hook) {
    /* WHY: the harness counts live blocks through the hook it installed with
     *      your tl_set_allocator. A tl_alloc or tl_free that bypasses the hook
     *      makes every leak check in the course blind, and freeing hook memory
     *      with free() corrupts any allocator that is not malloc.
     * KIND: fault
     * CATCHES: s03, s04
     * CHAPTER: rt.01 section 2 */
#if SS__COUNTING
    long before = ss_live_allocs();
    void *p = tl_alloc(10, 64);
    SS_TRUE(p != NULL);
    SS_EQ(ss_live_allocs(), before + 1);
    tl_free(p);
    SS_EQ(ss_live_allocs(), before);
#endif
    hook_calls calls = {0, 0};
    tl_allocator mine = {counting_alloc, counting_free, &calls};
    SS_EQ(tl_set_allocator(&mine), TL_OK);
    void *q = tl_alloc(24, 16);
    tl_free(q);
    long allocs = calls.allocs, frees = calls.frees;
    restore_harness_hook();
    SS_TRUE(q != NULL);
    SS_EQ(allocs, 1);
    SS_EQ(frees, 1);
}

SS_TEST(alloc_alignment_by_hand) {
    /* WHY: the chapter's worked example: a request for 10 bytes at alignment
     *      64 returns an address divisible by 64. Kernels rely on it for
     *      vector loads; the default hook must round the size up to a
     *      multiple of the alignment, which aligned_alloc requires.
     * KIND: unit
     * CATCHES: s05
     * CHAPTER: rt.01 section 3 */
    tl_set_allocator(NULL); /* the default hook, the one your code supplies */
    int ok = 1;
    for (size_t align = 1; align <= 4096; align *= 2) {
        void *p = tl_alloc(10, align);
        if (p == NULL || (uintptr_t)p % align != 0) ok = 0;
        if (p) memset(p, 0xAB, 10); /* ASan: the 10 bytes are really ours */
        tl_free(p);
    }
    restore_harness_hook();
    SS_TRUE(ok);
}

SS_TEST(set_allocator_null_restores_the_default) {
    /* WHY: tl_set_allocator(NULL) puts the default hook back. A library that
     *      keeps the last hook after NULL keeps calling a hook whose owner may
     *      be gone (a test's counters on its stack, a freed arena).
     * KIND: unit
     * CHAPTER: rt.01 section 2 */
    hook_calls calls = {0, 0};
    tl_allocator mine = {counting_alloc, counting_free, &calls};
    SS_EQ(tl_set_allocator(&mine), TL_OK);
    SS_EQ(tl_set_allocator(NULL), TL_OK);
    void *p = tl_alloc(32, 16);
    tl_free(p);
    long allocs = calls.allocs, frees = calls.frees;
    restore_harness_hook();
    SS_TRUE(p != NULL);
    SS_EQ(allocs, 0);
    SS_EQ(frees, 0);
}

SS_TEST(alloc_rejects_zero_and_bad_alignment) {
    /* WHY: tl_alloc(0, ...) and a non power of two alignment are caller bugs
     *      the contract defines: NULL with the error slot set, and the hook
     *      never called (a hook may not accept them either).
     * KIND: boundary
     * CATCHES: m002
     * CHAPTER: rt.01 section 4 */
    hook_calls calls = {0, 0};
    tl_allocator mine = {counting_alloc, counting_free, &calls};
    SS_EQ(tl_set_allocator(&mine), TL_OK);
    tl_set_last_error(NULL);
    void *a = tl_alloc(0, 8);
    int err_a = tl_last_error()[0] != '\0';
    tl_set_last_error(NULL);
    void *b = tl_alloc(8, 3);
    int err_b = tl_last_error()[0] != '\0';
    void *c = tl_alloc(8, 0);
    long hook_used = calls.allocs;
    restore_harness_hook();
    SS_TRUE(a == NULL && b == NULL && c == NULL);
    SS_TRUE(err_a && err_b);
    SS_EQ(hook_used, 0);
}

SS_TEST(alloc_failure_sets_the_error) {
    /* WHY: when the hook returns NULL, tl_alloc returns NULL and says why;
     *      every constructor turns that into TL_ENOMEM. This is the fault
     *      path the fail-after-n tests of later modules walk.
     * KIND: fault
     * CATCHES: s09
     * CHAPTER: rt.01 section 4 */
#if SS__COUNTING
    tl_set_last_error(NULL);
    ss_alloc_fail_after(0);
    void *p = tl_alloc(16, 16);
    SS_TRUE(p == NULL);
    SS_TRUE(strstr(tl_last_error(), "tl_alloc") != NULL);
#endif
}

SS_TEST(set_allocator_rejects_half_a_hook) {
    /* WHY: a hook with alloc but no free would leak every block it hands
     *      out; the contract says TL_EINVAL, and the old hook stays installed.
     * KIND: boundary
     * CATCHES: s06
     * CHAPTER: rt.01 section 4 */
    tl_allocator half = {counting_alloc, NULL, NULL};
    SS_EQ(tl_set_allocator(&half), TL_EINVAL);
    SS_TRUE(strstr(tl_last_error(), "tl_set_allocator") != NULL);
    tl_allocator other_half = {NULL, counting_free, NULL};
    SS_EQ(tl_set_allocator(&other_half), TL_EINVAL);
#if SS__COUNTING
    long before = ss_live_allocs();
    void *p = tl_alloc(8, 8); /* still the harness hook */
    SS_EQ(ss_live_allocs(), before + 1);
    tl_free(p);
#endif
}

SS_TEST(set_allocator_copies_the_struct) {
    /* WHY: callers build the tl_allocator on the stack and return. Keeping a
     *      pointer to it reads a dead stack frame on the next tl_alloc; the
     *      hook must be copied when it is installed.
     * KIND: fault
     * CATCHES: s07
     * CHAPTER: rt.01 section 5, Pitfalls */
    hook_calls calls = {0, 0};
    tl_allocator mine = {counting_alloc, counting_free, &calls};
    SS_EQ(tl_set_allocator(&mine), TL_OK);
    mine.alloc = NULL; /* the caller reuses its struct */
    mine.free = NULL;
    mine.user = NULL;
    void *p = tl_alloc(8, 8);
    tl_free(p);
    long allocs = calls.allocs;
    restore_harness_hook();
    SS_TRUE(p != NULL);
    SS_EQ(allocs, 1);
}

SS_TEST(free_null_is_a_no_op) {
    /* WHY: cleanup paths call tl_free on pointers that may never have been
     *      allocated; like free(NULL), it must do nothing (and must not hand
     *      NULL to a hook that does not expect it).
     * KIND: boundary
     * CATCHES: s08
     * CHAPTER: rt.01 section 4 */
    hook_calls calls = {0, 0};
    tl_allocator mine = {counting_alloc, counting_free, &calls};
    SS_EQ(tl_set_allocator(&mine), TL_OK);
    tl_free(NULL);
    long frees = calls.frees;
    restore_harness_hook();
    SS_EQ(frees, 0);
}

int main(void) { return SS_RUN_ALL(); }
