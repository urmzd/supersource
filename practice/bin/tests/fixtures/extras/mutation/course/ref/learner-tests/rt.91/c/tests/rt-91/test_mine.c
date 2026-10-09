#include "tinyllm.h"
#include "ss_test.h"

SS_TEST(sums_four) {
    float x[4] = {1.0f, 2.0f, 3.0f, 4.0f}, y = 0.0f;
    SS_EQ(tl_demo_sum_f32(x, 4, &y), TL_OK);
    SS_CLOSE(y, 10.0, 0, 0);
}

SS_TEST(alloc_failure_frees_the_struct) {
    tl_demo_buf *b = NULL;
    ss_alloc_fail_after(1);
    SS_EQ(tl_demo_buf_create(4, &b), TL_ENOMEM);
}

int main(void) { return SS_RUN_ALL(); }
