/* c/src/runtime/race.c (fixture rt.92) */
#include <pthread.h>
#include <stdatomic.h>

#include "tinyllm/race.h"

typedef struct {
    _Atomic int64_t *total;
    int64_t per;
} race_arg;

static void *race_work(void *p) {
/* SOLUTION-BEGIN rt.92 */
    race_arg *a = p;
    for (int64_t i = 0; i < a->per; i++) atomic_fetch_add_explicit(a->total, 1, memory_order_relaxed);
    return NULL;
/* SOLUTION-END */
}

tl_status tl_demo_count_parallel(int nthreads, int64_t per, int64_t *out) {
/* SOLUTION-BEGIN rt.92 */
    if (out == NULL || nthreads < 1 || nthreads > 16 || per < 0) {
        tl_set_last_error("tl_demo_count_parallel: out is NULL or nthreads/per out of range");
        return TL_EINVAL;
    }
    _Atomic int64_t total = 0;
    race_arg a = {&total, per};
    pthread_t th[16];
    for (int i = 0; i < nthreads; i++) pthread_create(&th[i], NULL, race_work, &a);
    for (int i = 0; i < nthreads; i++) pthread_join(th[i], NULL);
    *out = atomic_load(&total);
    return TL_OK;
/* SOLUTION-END */
}
