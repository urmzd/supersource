/* course/tests/lang.03/count_alloc.h: force-included (cc -include) into YOUR
 * vec.c by the lang.03 check, never into the test file.
 *
 * It routes vec.c's malloc, calloc, realloc, and free through counting
 * wrappers that test_vec.c defines, so the tests can see leaks (macOS ASan
 * has no leak checker) and make the next allocation fail on purpose.
 * Including <stdlib.h> first means the real declarations come before the
 * macros, and your own #include <stdlib.h> later is a no-op.
 */
#ifndef LANG03_COUNT_ALLOC_H
#define LANG03_COUNT_ALLOC_H
#include <stdlib.h>
void *lang03_malloc(size_t n);
void *lang03_calloc(size_t n, size_t size);
void *lang03_realloc(void *p, size_t n);
void lang03_free(void *p);
#define malloc(n) lang03_malloc(n)
#define calloc(n, size) lang03_calloc(n, size)
#define realloc(p, n) lang03_realloc(p, n)
#define free(p) lang03_free(p)
#endif
