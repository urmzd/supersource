/* tinyllm/race.h (fixture rt.92): a counter shared by threads.
 * chapter: chapters/rt-92-race.md */
#ifndef TINYLLM_RACE_H
#define TINYLLM_RACE_H
#include "tinyllm/abi.h"

/* Starts nthreads (1 to 16) threads that each add 1 to one shared counter
 * `per` times; *out is nthreads * per. */
tl_status tl_demo_count_parallel(int nthreads, int64_t per, int64_t *out);
#endif
