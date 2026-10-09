/* primers/lang.03/roundtrip.c: the C half of the ctypes round trip. */
#include "roundtrip.h"

#include <stddef.h> /* offsetof */

int64_t rt_sum_i32(const int32_t *xs, size_t n) {
    /* SOLUTION-BEGIN lang.03 */
    int64_t s = 0; /* widen before adding: int32 + int32 overflows */
    for (size_t i = 0; i < n; i++) s += xs[i];
    return s;
    /* SOLUTION-END */
}

void rt_scale_f32(float *xs, size_t n, float a) {
    /* SOLUTION-BEGIN lang.03 */
    for (size_t i = 0; i < n; i++) xs[i] *= a;
    /* SOLUTION-END */
}

int rt_best_pair(const rt_pair *ps, size_t n, rt_pair *out) {
    /* SOLUTION-BEGIN lang.03 */
    if (n == 0) return -1;
    size_t best = 0;
    for (size_t i = 1; i < n; i++) {
        if (ps[i].score > ps[best].score) best = i; /* strict: ties keep the lower index */
    }
    *out = ps[best];
    return 0;
    /* SOLUTION-END */
}

size_t rt_pair_size(void) {
    /* SOLUTION-BEGIN lang.03 */
    return sizeof(rt_pair);
    /* SOLUTION-END */
}

size_t rt_pair_score_offset(void) {
    /* SOLUTION-BEGIN lang.03 */
    return offsetof(rt_pair, score);
    /* SOLUTION-END */
}
