/* primers/lang.03/roundtrip.h: three C functions Python calls through
 * ctypes (given; do not edit). You write roundtrip.c and roundtrip.py. */
#ifndef ROUNDTRIP_H
#define ROUNDTRIP_H

#include <stddef.h>
#include <stdint.h>

/* 4 bytes of id, 4 bytes of padding, 8 bytes of score: 16 in all. */
typedef struct {
    int32_t id;
    double score;
} rt_pair;

/* Sum of n int32 values as an int64 (no overflow for n < 2^32).
 * xs may be NULL when n is 0. */
int64_t rt_sum_i32(const int32_t *xs, size_t n);

/* xs[i] *= a for every i < n, in place: the caller sees the new values. */
void rt_scale_f32(float *xs, size_t n, float a);

/* The index of the pair with the highest score (ties: the lowest index),
 * copied to *out. -1 when n is 0, else 0. */
int rt_best_pair(const rt_pair *ps, size_t n, rt_pair *out);

/* sizeof(rt_pair) and offsetof(rt_pair, score), so Python can check that
 * its ctypes.Structure has the same layout. */
size_t rt_pair_size(void);
size_t rt_pair_score_offset(void);

#endif /* ROUNDTRIP_H */
