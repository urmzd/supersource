/* ss_test.h: the single-header C test kit for course and learner tests.
 *
 * Frozen helper (course/DESIGN.md D35, 5.9). Usage:
 *
 *   #include "tinyllm.h"      // optional; enables the counting allocator
 *   #include "ss_test.h"
 *
 *   SS_TEST(hand_example) {
 *       // WHY: ...   KIND: unit
 *       float y = 0;
 *       SS_EQ(tl_demo_sum_f32(x, 3, &y), TL_OK);
 *       SS_CLOSE(y, 6.0, 1e-6, 1e-7);
 *   }
 *
 *   int main(void) { return SS_RUN_ALL(); }
 *
 * Tests in several .c files link into one binary: the case table is a weak
 * global shared by every translation unit. SS_RUN_ALL() runs every case (or
 * the comma-separated names in $SS_ONLY), prints one line per case, and
 * returns the exit code: 0 when every case passed and at least one ran.
 *
 * Counting allocator: built with -DSS_COUNTING_ALLOC=1 (ss check does this)
 * and with tinyllm/abi.h included first, SS_RUN_ALL installs an allocator
 * through tl_set_allocator that counts live blocks, and fails any case that
 * returns with more blocks live than it started with. This is the leak check
 * on every platform (Apple clang has no LeakSanitizer). ss_alloc_fail_after(n)
 * makes the n-th allocation from now return NULL, for fault tests.
 */
#ifndef SS_TEST_H
#define SS_TEST_H

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifndef SS_MAX_CASES
#define SS_MAX_CASES 1024
#endif

typedef struct {
    const char *name;
    void (*fn)(void);
    const char *file;
} ss_case;

__attribute__((weak)) ss_case ss__cases[SS_MAX_CASES];
__attribute__((weak)) int ss__ncases;
__attribute__((weak)) int ss__failed;
__attribute__((weak)) long ss__live_allocs;
__attribute__((weak)) long ss__fail_after = -1;

static inline void ss__register(const char *name, void (*fn)(void), const char *file) {
    if (ss__ncases >= SS_MAX_CASES) {
        fprintf(stderr, "ss_test.h: more than %d cases; raise SS_MAX_CASES\n", SS_MAX_CASES);
        abort();
    }
    ss__cases[ss__ncases].name = name;
    ss__cases[ss__ncases].fn = fn;
    ss__cases[ss__ncases].file = file;
    ss__ncases++;
}

#define SS_TEST(name)                                                              \
    static void name(void);                                                        \
    __attribute__((constructor)) static void ss__reg_##name(void) {                \
        ss__register(#name, name, __FILE__);                                       \
    }                                                                              \
    static void name(void)

static inline int ss__fail(const char *file, int line, const char *what) {
    fprintf(stdout, "    %s:%d: %s\n", file, line, what);
    ss__failed = 1;
    return 0;
}

static inline int ss__eq_i(long long a, long long b, const char *ea, const char *eb, const char *file, int line) {
    if (a == b) return 1;
    char buf[512];
    snprintf(buf, sizeof buf, "SS_EQ(%s, %s): %lld != %lld", ea, eb, a, b);
    return ss__fail(file, line, buf);
}

static inline int ss__eq_d(double a, double b, const char *ea, const char *eb, const char *file, int line) {
    if (a == b) return 1;
    char buf[512];
    snprintf(buf, sizeof buf, "SS_EQ(%s, %s): %.17g != %.17g", ea, eb, a, b);
    return ss__fail(file, line, buf);
}

static inline double ss__abs(double x) { return x < 0 ? -x : x; }

static inline int ss__close(double a, double b, double rtol, double atol, const char *ea, const char *eb,
                     const char *file, int line) {
    double diff = ss__abs(a - b), allowed = atol + rtol * ss__abs(b);
    if (diff <= allowed) return 1;
    char buf[512];
    snprintf(buf, sizeof buf, "SS_CLOSE(%s, %s): %.17g vs %.17g, |diff| %.3e > allowed %.3e", ea, eb, a, b,
             diff, allowed);
    return ss__fail(file, line, buf);
}

/* Each assertion returns from the test on failure. */
#define SS_EQ(a, b)                                                                                 \
    do {                                                                                            \
        if (!_Generic((a) + (b), float: ss__eq_d, double: ss__eq_d, long double: ss__eq_d,       \
                      default: ss__eq_i)((a), (b), #a, #b, __FILE__, __LINE__))                     \
            return;                                                                                 \
    } while (0)
#define SS_TRUE(c)                                                                                  \
    do {                                                                                            \
        if (!(c)) {                                                                                 \
            ss__fail(__FILE__, __LINE__, "SS_TRUE(" #c ")");                                        \
            return;                                                                                 \
        }                                                                                           \
    } while (0)
#define SS_CLOSE(a, b, rtol, atol)                                                                  \
    do {                                                                                            \
        if (!ss__close((double)(a), (double)(b), (rtol), (atol), #a, #b, __FILE__, __LINE__))     \
            return;                                                                                 \
    } while (0)
#define SS_FAIL(msg)                                                                                \
    do {                                                                                            \
        ss__fail(__FILE__, __LINE__, (msg));                                                        \
        return;                                                                                     \
    } while (0)

static inline void ss_alloc_fail_after(long n) { ss__fail_after = n; }
static inline long ss_live_allocs(void) { return ss__live_allocs; }

#if defined(SS_COUNTING_ALLOC) && defined(TL_ABI_VERSION)
#define SS__COUNTING 1
static inline void *ss__alloc(void *user, size_t n, size_t align) {
    (void)user;
    if (ss__fail_after == 0) return NULL;
    if (ss__fail_after > 0) ss__fail_after--;
    if (align < sizeof(void *)) align = sizeof(void *);
    size_t size = n ? n : 1;
    size = (size + align - 1) / align * align;
    void *p = aligned_alloc(align, size);
    if (p) ss__live_allocs++;
    return p;
}
static inline void ss__free(void *user, void *p) {
    (void)user;
    if (p) {
        ss__live_allocs--;
        free(p);
    }
}
#else
#define SS__COUNTING 0
#endif

static inline int ss__selected(const char *name) {
    const char *only = getenv("SS_ONLY");
    if (!only || !*only) return 1;
    size_t n = strlen(name);
    for (const char *p = only; *p;) {
        const char *q = strchr(p, ',');
        size_t len = q ? (size_t)(q - p) : strlen(p);
        if (len == n && strncmp(p, name, n) == 0) return 1;
        if (!q) break;
        p = q + 1;
    }
    return 0;
}

static inline int ss_run_all(void) {
    int counting = 0;
#if SS__COUNTING
    tl_allocator a = {ss__alloc, ss__free, NULL};
    tl_status st = tl_set_allocator(&a);
    counting = st == TL_OK;
    if (!counting) fprintf(stdout, "note: counting allocator unavailable (tl_set_allocator returned %d)\n", (int)st);
#endif
    int ran = 0, failed = 0;
    for (int i = 0; i < ss__ncases; i++) {
        if (!ss__selected(ss__cases[i].name)) continue;
        long before = ss__live_allocs;
        ss__failed = 0;
        ss__fail_after = -1;
        ss__cases[i].fn();
        ran++;
        if (counting && !ss__failed && ss__live_allocs != before) {
            fprintf(stdout, "    leak: %ld allocation(s) still live after the test\n", ss__live_allocs - before);
            ss__failed = 1;
        }
        fprintf(stdout, "%s %s\n", ss__failed ? "FAIL" : "ok  ", ss__cases[i].name);
        failed += ss__failed;
    }
#if SS__COUNTING
    if (counting) tl_set_allocator(NULL);
#endif
    if (ran == 0) {
        fprintf(stdout, "no test ran%s\n", getenv("SS_ONLY") ? " (SS_ONLY matched nothing)" : "");
        return 1;
    }
    fprintf(stdout, "%d passed, %d failed\n", ran - failed, failed);
    fflush(stdout);
    return failed ? 1 : 0;
}

#define SS_RUN_ALL() ss_run_all()

#endif /* SS_TEST_H */
