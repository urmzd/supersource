#include "tinyllm.h"
#include "tinyllm/race.h"
#include "ss_test.h"

SS_TEST(hand_example) {
    /* WHY: four threads adding 1000 each give 4000, with no data race (TSan build).
     * KIND: unit */
    int64_t n = 0;
    SS_EQ(tl_demo_count_parallel(4, 1000, &n), TL_OK);
    SS_EQ(n, 4000);
}

int main(void) { return SS_RUN_ALL(); }
