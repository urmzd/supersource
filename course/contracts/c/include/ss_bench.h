/* ss_bench.h: the single-header C benchmark kit (course/DESIGN.md 5.9, 5.11).
 *
 * Frozen helper. A course benchmark is a program under
 * course/tests/<ID>/bench/<name>.c that `ss bench <ID>` builds with -O2
 * against the overlay's objects and runs. It times the code under test with
 * ss_bench_best and reports named metrics; the last stdout line is one JSON
 * object, which `ss bench` compares with the module's budget relative to the
 * machine calibration (`ss bench --calibrate`).
 *
 *   #include "tinyllm.h"
 *   #include "ss_bench.h"
 *
 *   static float a[1 << 16], b[1 << 16], c[1 << 16];
 *   static void step(void *ctx) { (void)ctx; tl_matmul_f32(a, b, c, ...); }
 *
 *   int main(void) {
 *       double s = ss_bench_best(step, NULL, 3, 0.2);   // best seconds per call
 *       ss_bench_metric("gflops", 2.0 * M * N * K / s / 1e9);
 *       return ss_bench_done();                          // prints {"gflops": ...}
 *   }
 */
#ifndef SS_BENCH_H
#define SS_BENCH_H

#ifndef _POSIX_C_SOURCE
#define _POSIX_C_SOURCE 199309L
#endif
#include <stdio.h>
#include <string.h>
#include <time.h>

#ifndef SS_BENCH_MAX_METRICS
#define SS_BENCH_MAX_METRICS 16
#endif

static inline double ss_bench_now(void) {
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return (double)t.tv_sec + 1e-9 * (double)t.tv_nsec;
}

/* Best time per call, in seconds: `reps` repetitions, each calling fn enough
 * times to take at least min_s seconds (one warm-up call first). */
static inline double ss_bench_best(void (*fn)(void *), void *ctx, int reps, double min_s) {
    fn(ctx);
    double best = 1e300;
    for (int r = 0; r < reps; r++) {
        long n = 0;
        double t0 = ss_bench_now(), dt;
        do {
            fn(ctx);
            n++;
            dt = ss_bench_now() - t0;
        } while (dt < min_s);
        if (dt / (double)n < best) best = dt / (double)n;
    }
    return best;
}

static const char *ss_bench__names[SS_BENCH_MAX_METRICS];
static double ss_bench__values[SS_BENCH_MAX_METRICS];
static int ss_bench__n;

static inline void ss_bench_metric(const char *name, double value) {
    for (int i = 0; i < ss_bench__n; i++)
        if (strcmp(ss_bench__names[i], name) == 0) {
            ss_bench__values[i] = value;
            return;
        }
    if (ss_bench__n < SS_BENCH_MAX_METRICS) {
        ss_bench__names[ss_bench__n] = name;
        ss_bench__values[ss_bench__n++] = value;
    }
}

/* Prints the metrics as the last stdout line; returns 0 (an exit code). */
static inline int ss_bench_done(void) {
    putchar('{');
    for (int i = 0; i < ss_bench__n; i++)
        printf("%s\"%s\": %.6g", i ? ", " : "", ss_bench__names[i], ss_bench__values[i]);
    puts("}");
    fflush(stdout);
    return 0;
}

#endif /* SS_BENCH_H */
