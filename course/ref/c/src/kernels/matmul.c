/* c/src/kernels/matmul.c (M03.1, naive v0): C = alpha * A @ op(B) + beta * C.
 * Contract: tinyllm/matmul.h. L9.1 later takes this file over with a tiled,
 * packed, batch-invariant version behind the same symbol and signature.
 */
#include <stdint.h>

#include "tinyllm/abi.h"
#include "tinyllm/matmul.h"

tl_status tl_matmul_f32(const float *A, const float *B, float *C,
                        int64_t M, int64_t N, int64_t K,
                        int64_t lda, int64_t ldb, int64_t ldc,
                        float alpha, float beta, int trans_b, tl_pool *tp) {
/* SOLUTION-BEGIN M03.1 */
    (void)tp; /* v0 is serial; rt.03 and L9.1 bring the pool */

    /* 1. Validate everything before touching C, so a rejected call leaves C
     *    exactly as it was. */
    if (M < 0 || N < 0 || K < 0) {
        tl_set_last_error("tl_matmul_f32: negative dimension");
        return TL_EINVAL;
    }
    if (M == 0 || N == 0) return TL_OK; /* no element of C exists */
    if (ldc < N) {
        tl_set_last_error("tl_matmul_f32: ldc < N");
        return TL_EINVAL;
    }
    if (C == NULL) {
        tl_set_last_error("tl_matmul_f32: C is NULL");
        return TL_EINVAL;
    }
    if (K > 0) {
        if (lda < K) {
            tl_set_last_error("tl_matmul_f32: lda < K");
            return TL_EINVAL;
        }
        if (trans_b ? ldb < K : ldb < N) {
            tl_set_last_error(trans_b ? "tl_matmul_f32: ldb < K (trans_b)" : "tl_matmul_f32: ldb < N");
            return TL_EINVAL;
        }
        if (A == NULL || B == NULL) {
            tl_set_last_error("tl_matmul_f32: A or B is NULL");
            return TL_EINVAL;
        }
    }

    /* 2. One dot product per output element. op(B)[k][j] is B[k*ldb + j]
     *    when B is stored K x N, and B[j*ldb + k] when it is stored N x K. */
    for (int64_t i = 0; i < M; i++) {
        float *c = C + i * ldc; /* row i of C */
        for (int64_t j = 0; j < N; j++) {
            float acc = 0.0f;
            for (int64_t k = 0; k < K; k++) {
                float b = trans_b ? B[j * ldb + k] : B[k * ldb + j];
                acc += A[i * lda + k] * b;
            }
            /* beta == 0 means "overwrite": never read c[j], which may hold
             * NaN or garbage (0 * NaN is NaN, not 0). */
            c[j] = beta == 0.0f ? alpha * acc : alpha * acc + beta * c[j];
        }
    }
    return TL_OK;
/* SOLUTION-END */
}
