/* Parity driver for `rng` (M06.3, C): one JSON case per stdin line
 * ({seed, seq, n_u32, n_uniform, n_normal}), one JSON object per stdout line
 * ({"u32": [...], "uniform": [...], "normal": [...]}), each list from a fresh
 * tl_pcg32_seed(seed, seq). Normals are spec/pcg32.md's Box-Muller with a
 * spare, computed here from tl_pcg32_uniform (M06.3 owns only the uniforms).
 * Doubles print with 17 significant digits, which round-trips exactly.
 * spec/pcg32.md compares normals within 4 ulp (they go through libm); on one
 * machine the C and Python libm calls agree exactly once sin and cos are
 * not fused. */
#include <math.h>
#include <stdio.h>

#include "tinyllm.h"
#include "ss_parity.h"

static void print_normals(uint64_t seed, uint64_t seq, int64_t n) {
    tl_pcg32 r;
    tl_pcg32_seed(&r, seed, seq);
    const double two_pi = 6.283185307179586; /* 2 * M_PI, the double Python's math.pi * 2.0 gives */
    double spare = 0.0;
    int have_spare = 0;
    putchar('[');
    for (int64_t i = 0; i < n; i++) {
        double z;
        if (have_spare) {
            z = spare;
            have_spare = 0;
        } else {
            double u1 = tl_pcg32_uniform(&r), u2 = tl_pcg32_uniform(&r);
            double rad = sqrt(-2.0 * log(1.0 - u1));
            /* volatile: separate cos and sin calls, as Python makes them.
             * Without it clang -O2 on macOS fuses the pair into __sincos,
             * whose results can differ from cos and sin by 1 ulp. */
            volatile double angle = two_pi * u2;
            z = rad * cos(angle);
            spare = rad * sin(angle);
            have_spare = 1;
        }
        printf(i ? ",%.17g" : "%.17g", z);
    }
    putchar(']');
}

int main(void) {
    static char line[1 << 16];
    while (ssp_line(line, sizeof line)) {
        uint64_t seed = ssp_u64(line, "seed", 0), seq = ssp_u64(line, "seq", 54);
        int64_t n_u32 = ssp_int(line, "n_u32", 0), n_uni = ssp_int(line, "n_uniform", 0);
        int64_t n_nor = ssp_int(line, "n_normal", 0);
        tl_pcg32 r;
        tl_pcg32_seed(&r, seed, seq);
        printf("{\"u32\":[");
        for (int64_t i = 0; i < n_u32; i++) printf(i ? ",%u" : "%u", (unsigned)tl_pcg32_next(&r));
        printf("],\"uniform\":[");
        tl_pcg32_seed(&r, seed, seq);
        for (int64_t i = 0; i < n_uni; i++) printf(i ? ",%.17g" : "%.17g", tl_pcg32_uniform(&r));
        printf("],\"normal\":");
        print_normals(seed, seq, n_nor);
        printf("}\n");
        fflush(stdout);
    }
    return 0;
}
