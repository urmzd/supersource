/* tinyllm/demo.h (fixture rt.91): a float sum and an owned buffer.
 * chapter: chapters/rt-91-demo-sum.md */
#ifndef TINYLLM_DEMO_H
#define TINYLLM_DEMO_H
#include "tinyllm/abi.h"

typedef struct tl_demo_buf tl_demo_buf;
tl_status tl_demo_sum_f32(const float *x, int64_t n, float *out); /* accumulates in double */
tl_status tl_demo_buf_create(int64_t n, tl_demo_buf **out);       /* zeroed, through tl_alloc */
float *tl_demo_buf_data(tl_demo_buf *b);
void tl_demo_buf_destroy(tl_demo_buf *b);
#endif
