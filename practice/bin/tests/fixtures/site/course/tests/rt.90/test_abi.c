#include <string.h>

#include "tinyllm.h"
#include "ss_test.h"

SS_TEST(abi_version_matches_header) {
    /* WHY: bindings refuse a library whose major ABI version differs from the
     *      header they were written against.
     * KIND: unit */
    SS_EQ(tl_abi_version(), TL_ABI_VERSION);
}

SS_TEST(status_strings) {
    /* WHY: every status has a readable name and an unknown code still gets one.
     * KIND: unit */
    SS_TRUE(tl_status_str(TL_OK) != NULL);
    SS_TRUE(strcmp(tl_status_str(TL_OK), "ok") == 0);
    SS_TRUE(strcmp(tl_status_str(12345), "unknown status") == 0);
}

SS_TEST(alloc_goes_through_the_hook) {
    /* WHY: the counting allocator only sees allocations that use tl_alloc, and
     *      64-byte alignment is what the kernels rely on.
     * KIND: fault */
    long before = ss_live_allocs();
    void *p = tl_alloc(10, 64);
    SS_TRUE(p != NULL);
    SS_EQ((long long)((uintptr_t)p % 64), 0);
    SS_EQ(ss_live_allocs(), before + 1);
    tl_free(p);
    SS_EQ(ss_live_allocs(), before);
}

SS_TEST(set_allocator_rejects_half_a_hook) {
    /* WHY: a hook with alloc but no free would leak every block it hands out.
     * KIND: boundary */
    tl_allocator half = {0};
    SS_EQ(tl_set_allocator(&half), TL_EINVAL);
    SS_TRUE(tl_last_error() != NULL);
}

int main(void) { return SS_RUN_ALL(); }
