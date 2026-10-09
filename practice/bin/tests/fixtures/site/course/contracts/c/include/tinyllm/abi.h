/* tinyllm/abi.h (fixture rt.90): status codes, the error slot, the allocator hook.
 * chapter: chapters/rt-90-abi.md
 *
 * tl_set_last_error, tl_alloc, and tl_free are the seams other units use:
 * a unit never reaches into abi.c except through these names. */
#ifndef TINYLLM_ABI_H
#define TINYLLM_ABI_H
#include <stddef.h>
#include <stdint.h>

#define TL_ABI_VERSION 1
typedef int32_t tl_status; /* never a C enum type across the ABI */
enum { TL_OK = 0, TL_EINVAL = 1, TL_ENOMEM = 2, TL_EUNSUPPORTED = 9 };

uint32_t tl_abi_version(void);
const char *tl_status_str(tl_status s);
const char *tl_last_error(void);
void tl_set_last_error(const char *msg);

typedef struct {
    void *(*alloc)(void *user, size_t n, size_t align);
    void (*free)(void *user, void *p);
    void *user;
} tl_allocator;
tl_status tl_set_allocator(const tl_allocator *a); /* NULL restores the default */
void *tl_alloc(size_t n, size_t align);            /* through the hook */
void tl_free(void *p);
#endif
