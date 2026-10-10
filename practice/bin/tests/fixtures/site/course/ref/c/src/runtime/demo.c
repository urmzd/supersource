/* c/src/runtime/demo.c (fixture rt.91) */
#include "tinyllm/demo.h"

struct tl_demo_buf {
    int64_t n;
    float *data;
};

tl_status tl_demo_sum_f32(const float *x, int64_t n, float *out) {
/* SOLUTION-BEGIN rt.91 */
    if (out == NULL || n < 0 || (n > 0 && x == NULL)) {
        tl_set_last_error("tl_demo_sum_f32: out is NULL, n is negative, or x is NULL");
        return TL_EINVAL;
    }
    double acc = 0.0; /* a float accumulator loses about sqrt(n) ulps */
    for (int64_t i = 0; i < n; i++) acc += x[i];
    *out = (float)acc;
    return TL_OK;
/* SOLUTION-END */
}

tl_status tl_demo_buf_create(int64_t n, tl_demo_buf **out) {
/* SOLUTION-BEGIN rt.91 */
    if (out == NULL || n < 0) return TL_EINVAL;
    tl_demo_buf *b = tl_alloc(sizeof *b, _Alignof(struct tl_demo_buf));
    if (b == NULL) return TL_ENOMEM;
    b->data = tl_alloc((size_t)(n ? n : 1) * sizeof(float), 64);
    if (b->data == NULL) {
        tl_free(b); /* the half-built object must not leak */
        return TL_ENOMEM;
    }
    for (int64_t i = 0; i < n; i++) b->data[i] = 0.0f;
    b->n = n;
    *out = b;
    return TL_OK;
/* SOLUTION-END */
}

float *tl_demo_buf_data(tl_demo_buf *b) {
/* SOLUTION-BEGIN rt.91 */
    return b ? b->data : NULL;
/* SOLUTION-END */
}

void tl_demo_buf_destroy(tl_demo_buf *b) {
/* SOLUTION-BEGIN rt.91 */
    if (b == NULL) return;
    tl_free(b->data);
    tl_free(b);
/* SOLUTION-END */
}
