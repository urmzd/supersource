/* Course tests for M06.3 (C half): c/src/numerics/rng.c against
 * tinyllm/numerics.h and spec/pcg32.md, built with ASan and UBSan, so a
 * 32-bit shift by 32 (undefined behavior in C) stops the run with a report.
 * The Python half is test_rng.py; test_rng_c_vs_python.py compares the two
 * streams through ctypes.
 */
#include <stdint.h>
#include <string.h>

#include "tinyllm.h"
#include "ss_test.h"

/* spec/pcg32.vectors.json: the first 16 outputs of pcg32(s) for s = 0, 1,
 * 2^63, and output number 1024 of each (index 1023). */
static const uint32_t V0[16] = {0x47C28B93u, 0xB98F6A27u, 0x7D3DCB1Eu, 0xF0761116u,
                                0x9CC33F5Bu, 0xBE0E744Du, 0x5752C556u, 0x43369132u,
                                0x173D6C87u, 0x5AF69B49u, 0xD1E5D614u, 0x2B566AF8u,
                                0xD6B77452u, 0x63A2F961u, 0x7192495Cu, 0xB3F92FDFu};
static const uint32_t V1[16] = {0x9B6BDDA9u, 0x0C31FC48u, 0xCE97F8EFu, 0x822C03D4u,
                                0x943E8CF2u, 0x1B75DDE3u, 0xAFDE6F30u, 0xC9748B79u,
                                0x23A9A7B4u, 0x7703C579u, 0x5B3186B8u, 0x55EF7CFCu,
                                0x840CD0ECu, 0x73494C1Cu, 0xC6B031DAu, 0xE6BB375Au};
static const uint32_t V63[16] = {0x8B93C7C2u, 0x6A27B98Du, 0xCB1E753Du, 0x1116F876u,
                                 0x3F5B1CC3u, 0x744DBE0Cu, 0xC5565552u, 0xD1324336u,
                                 0x6C87173Fu, 0x9B495AFEu, 0xDE14D1E5u, 0x6AF82A56u,
                                 0x7452D6A7u, 0xF9616382u, 0x495C7182u, 0x6FDFB3F9u};
static const uint32_t LAST[3] = {1247616890u, 1730178602u, 528107613u};
static const double U0[8] = {0x1.1f0a2e5cc7b50p-2, 0x1.f4f72c783b088p-2, 0x1.39867eaf839d1p-1,
                             0x1.5d4b15219b488p-2, 0x1.73d6c8b5ed368p-4, 0x1.a3cbac0ad59abp-1,
                             0x1.ad6ee898e8be5p-1, 0x1.c6492559fc97ep-2};

SS_TEST(worked_example_seed_0) {
    /* WHY: the chapter's worked example: pcg32_srandom_r(0, 54) leaves
     *      state 0x9AE4F7499BA72696 and inc 109, and the first output is
     *      rotr32(0x5C9A3E14, 19) = 0x47C28B93.
     * KIND: unit, smoke
     * CATCHES: s13, s17, m04
     * CHAPTER: M06.3 section 3 */
    tl_pcg32 r;
    tl_pcg32_seed(&r, 0, 54);
    SS_EQ(r.inc, (uint64_t)109);
    SS_EQ(r.state, (uint64_t)0x9AE4F7499BA72696ull);
    SS_EQ(tl_pcg32_next(&r), 0x47C28B93u);
}

SS_TEST(oneill_demo_line) {
    /* WHY: the first line of O'Neill's pcg32-demo (seed 42, sequence 54):
     *      an oracle from outside the course.
     * KIND: golden
     * CATCHES: s13, s17, m04
     * CHAPTER: M06.3 section 2.3 */
    static const uint32_t want[6] = {0xa15c02b7u, 0x7b47f409u, 0xba1d3330u,
                                     0x83d2f293u, 0xbfa4784bu, 0xcbed606eu};
    tl_pcg32 r;
    tl_pcg32_seed(&r, 42, 54);
    for (int i = 0; i < 6; i++) SS_EQ(tl_pcg32_next(&r), want[i]);
}

SS_TEST(spec_vectors_three_seeds) {
    /* WHY: 1024 outputs per seed, 16 checked at the start and the 1024th at
     *      the end, for seeds 0, 1, and 2^63 (the top bit of the seed). Every
     *      step also exercises the rotation by rot = old >> 59, which is 0
     *      for one state in 32: xs << 32 there is undefined behavior, and
     *      UBSan stops the run.
     * KIND: golden
     * CATCHES: s13, s14, s17, m04
     * CHAPTER: M06.3 section 2.3 */
    const uint64_t seeds[3] = {0, 1, 1ull << 63};
    const uint32_t *heads[3] = {V0, V1, V63};
    for (int s = 0; s < 3; s++) {
        tl_pcg32 r;
        tl_pcg32_seed(&r, seeds[s], 54);
        uint32_t x = 0;
        for (int i = 0; i < 1024; i++) {
            x = tl_pcg32_next(&r);
            if (i < 16) SS_EQ(x, heads[s][i]);
        }
        SS_EQ(x, LAST[s]);
    }
}

SS_TEST(uniform_bit_exact) {
    /* WHY: uniform = ((a >> 5) * 2^26 + (b >> 6)) * 2^-53 from two draws:
     *      integers only until the final exact scaling, so C and Python agree
     *      to the last bit (the hex-float literals below are exact).
     * KIND: golden
     * CATCHES: s15
     * CHAPTER: M06.3 section 2.4 */
    tl_pcg32 r;
    tl_pcg32_seed(&r, 0, 54);
    for (int i = 0; i < 8; i++) SS_EQ(tl_pcg32_uniform(&r), U0[i]);
}

SS_TEST(uniform_in_unit_interval_with_53_bits) {
    /* WHY: [0, 1) exactly, and u * 2^53 is an integer: no draw is ever 1.0,
     *      which inverse-CDF sampling (L8.1) relies on.
     * KIND: property
     * CATCHES: s14
     * CHAPTER: M06.3 section 2.4 */
    tl_pcg32 r;
    tl_pcg32_seed(&r, 99, 7);
    for (int i = 0; i < 100000; i++) {
        double u = tl_pcg32_uniform(&r);
        SS_TRUE(u >= 0.0 && u < 1.0);
        double s = u * 9007199254740992.0;
        SS_TRUE(s == (double)(uint64_t)s);
    }
}

SS_TEST(streams_differ_by_sequence) {
    /* WHY: seq selects one of 2^63 streams: same seed, different seq must
     *      give different outputs (this is how sub-streams stay independent).
     * KIND: property
     * CATCHES: m04
     * CHAPTER: M06.3 section 2.3 */
    tl_pcg32 a, b;
    tl_pcg32_seed(&a, 5, 1);
    tl_pcg32_seed(&b, 5, 2);
    int same = 0;
    for (int i = 0; i < 64; i++) same += tl_pcg32_next(&a) == tl_pcg32_next(&b);
    SS_TRUE(same < 4);
    SS_TRUE(a.inc % 2 == 1 && b.inc % 2 == 1);
}

SS_TEST(fnv1a64_published_vectors) {
    /* WHY: FNV-1a 64 reference test vectors (Noll's FNV test suite), and the
     *      contract's own example: rt.04 builds every KV block hash from this
     *      function, and the Rust engine must reproduce it.
     * KIND: golden, smoke
     * CATCHES: s16
     * CHAPTER: M06.3 section 2.7 */
    SS_EQ(tl_fnv1a64("", 0, TL_FNV1A64_OFFSET), (uint64_t)0xCBF29CE484222325ull);
    SS_EQ(tl_fnv1a64("a", 1, TL_FNV1A64_OFFSET), (uint64_t)0xAF63DC4C8601EC8Cull);
    SS_EQ(tl_fnv1a64("b", 1, TL_FNV1A64_OFFSET), (uint64_t)0xAF63DF4C8601F1A5ull);
    SS_EQ(tl_fnv1a64("foobar", 6, TL_FNV1A64_OFFSET), (uint64_t)0x85944171F73967E8ull);
    SS_EQ(tl_fnv1a64(NULL, 0, 12345u), (uint64_t)12345u);
}

SS_TEST(fnv1a64_chains_over_buffers) {
    /* WHY: the KV block hash continues from the parent block's hash, so
     *      hashing "foo" then "bar" with the first result as h must equal
     *      hashing "foobar" at once.
     * KIND: property
     * CATCHES: m05
     * CHAPTER: M06.3 section 2.7 */
    uint64_t h = tl_fnv1a64("foo", 3, TL_FNV1A64_OFFSET);
    SS_EQ(tl_fnv1a64("bar", 3, h), tl_fnv1a64("foobar", 6, TL_FNV1A64_OFFSET));
    unsigned char bytes[256];
    for (int i = 0; i < 256; i++) bytes[i] = (unsigned char)(255 - i); /* high bytes too */
    uint64_t whole = tl_fnv1a64(bytes, 256, TL_FNV1A64_OFFSET);
    SS_EQ(tl_fnv1a64(bytes + 100, 156, tl_fnv1a64(bytes, 100, TL_FNV1A64_OFFSET)), whole);
}

int main(void) { return SS_RUN_ALL(); }
