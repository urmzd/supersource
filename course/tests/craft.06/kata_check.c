/* course/tests/craft.06/kata_check.c: is a kata's kernels.c correct?
 *
 * Built by test_craft06_gate.py with the kernels.c under test (yours, the
 * course's, or the course's with one planted slowdown) and run. Each kernel
 * is compared with its plain version on fixed inputs (no RNG), including
 * sizes that are not multiples of the kata's tiles or of 8. Exit 0 when all
 * agree, 1 otherwise; one line per failure on stdout. */
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#include "kernels.h"

static int fails;

static void check(int ok, const char *what, int n) {
    if (!ok) {
        printf("FAIL %s (n = %d)\n", what, n);
        fails++;
    }
}

static float val(int i, int m) { return (float)((i * 37 + 11) % m) / (float)m - 0.5f; }

int main(void) {
    const int sizes[] = {1, 7, 64, 100, 257};
    for (int si = 0; si < 5; si++) {
        int n = sizes[si];
        float *A = malloc(sizeof(float) * n * n), *B = malloc(sizeof(float) * n * n);
        float *C = malloc(sizeof(float) * n * n);
        for (int i = 0; i < n * n; i++) {
            A[i] = val(i, 17);
            B[i] = val(i + 3, 23);
            C[i] = NAN; /* the kata must overwrite C, never read it */
        }
        kata_matmul(A, B, C, n);
        int ok = 1;
        for (int i = 0; i < n && ok; i++)
            for (int j = 0; j < n && ok; j++) {
                double ref = 0.0, mag = 0.0;
                for (int k = 0; k < n; k++) {
                    ref += (double)A[i * n + k] * B[k * n + j];
                    mag += fabs((double)A[i * n + k] * B[k * n + j]);
                }
                ok = fabs(C[i * n + j] - ref) <= 2.0 * n * ldexp(1.0, -24) * mag + 1e-30;
            }
        check(ok, "kata_matmul", n);
        free(A);
        free(B);
        free(C);
    }
    const int lens[] = {0, 1, 7, 8, 9, 1000, 4099};
    for (int li = 0; li < 7; li++) {
        int n = lens[li];
        float *x = malloc(sizeof(float) * (n ? n : 1));
        double ref = 0.0, mag = 0.0;
        for (int i = 0; i < n; i++) {
            x[i] = val(i, 29) * 4.0f;
            ref += x[i];
            mag += fabs(x[i]);
        }
        check(fabs(kata_sum(x, n) - ref) <= (n + 1) * ldexp(1.0, -24) * mag, "kata_sum", n);
        free(x);
    }
    const int dims[] = {1, 5, 8, 33, 1024};
    for (int di = 0; di < 5; di++) {
        int d = dims[di], rows = 3;
        float *x = malloc(sizeof(float) * rows * d), *w = malloc(sizeof(float) * d), *y = malloc(sizeof(float) * rows * d);
        for (int i = 0; i < rows * d; i++) x[i] = val(i, 31) * 3.0f;
        for (int i = 0; i < d; i++) w[i] = 1.0f + val(i, 7);
        kata_rmsnorm(x, w, y, rows, d, 1e-5f);
        int ok = 1;
        for (int r = 0; r < rows; r++) {
            double ss = 0.0;
            for (int i = 0; i < d; i++) ss += (double)x[r * d + i] * x[r * d + i];
            double inv = 1.0 / sqrt(ss / d + 1e-5);
            for (int i = 0; i < d; i++) {
                double ref = x[r * d + i] * inv * w[i];
                ok &= fabs(y[r * d + i] - ref) <= (d + 8) * ldexp(1.0, -24) * fabs(ref) + 1e-30;
            }
        }
        check(ok, "kata_rmsnorm", d);
        free(x);
        free(w);
        free(y);
    }
    const int cn[] = {0, 1, 2, 1000};
    for (int ci = 0; ci < 4; ci++) {
        int n = cn[ci];
        float *s = malloc(sizeof(float) * (n ? n : 1));
        for (int i = 0; i < n; i++) s[i] = (float)(i / 3); /* runs of equal keys */
        int ok = 1;
        for (int q = -4; q <= 2 * (n / 3 + 2) && ok; q++) {
            float t = 0.5f * (float)q; /* on keys and between them */
            int ref = 0;
            while (ref < n && s[ref] < t) ref++;
            ok = kata_count_below(s, n, t) == ref;
        }
        check(ok, "kata_count_below", n);
        free(s);
    }
    if (!fails) printf("kata ok\n");
    return fails ? 1 : 0;
}
