/* c/src/runtime/abi.c (fixture rt.90) */
#include <stdlib.h>

#include "tinyllm/abi.h"

/* Given, outside the markers: the error slot and its setter, so a stub of
 * any unit can still report "unimplemented: <id>". */
static _Thread_local const char *tl__err = NULL;
void tl_set_last_error(const char *msg) { tl__err = msg; }
const char *tl_last_error(void) { return tl__err; }

static tl_allocator tl__hook;
static int tl__hooked = 0;

uint32_t tl_abi_version(void) {
/* SOLUTION-BEGIN rt.90 */
    return TL_ABI_VERSION;
/* SOLUTION-END */
}

const char *tl_status_str(tl_status s) {
/* SOLUTION-BEGIN rt.90 */
    switch (s) {
    case TL_OK: return "ok";
    case TL_EINVAL: return "invalid argument";
    case TL_ENOMEM: return "out of memory";
    case TL_EUNSUPPORTED: return "unsupported";
    default: return "unknown status";
    }
/* SOLUTION-END */
}

tl_status tl_set_allocator(const tl_allocator *a) {
/* SOLUTION-BEGIN rt.90 */
    if (a == NULL) {
        tl__hooked = 0;
        return TL_OK;
    }
    if (a->alloc == NULL || a->free == NULL) {
        tl_set_last_error("tl_set_allocator: alloc and free are both required");
        return TL_EINVAL;
    }
    tl__hook = *a;
    tl__hooked = 1;
    return TL_OK;
/* SOLUTION-END */
}

void *tl_alloc(size_t n, size_t align) {
/* SOLUTION-BEGIN rt.90 */
    if (tl__hooked) return tl__hook.alloc(tl__hook.user, n, align);
    if (align < sizeof(void *)) align = sizeof(void *);
    size_t size = n ? n : 1;
    size = (size + align - 1) / align * align;
    return aligned_alloc(align, size);
/* SOLUTION-END */
}

void tl_free(void *p) {
/* SOLUTION-BEGIN rt.90 */
    if (p == NULL) return;
    if (tl__hooked) tl__hook.free(tl__hook.user, p);
    else free(p);
/* SOLUTION-END */
}
