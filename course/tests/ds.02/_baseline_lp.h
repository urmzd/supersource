/* _baseline_lp.h: the ds.02 baseline, a linear-probing u64 -> u64 map.
 *
 * This is practice c/02 (string keys, FNV-1a, load 3/4, tombstones) carried
 * over to the key type of tl_map: u64 keys hashed with the same SplitMix64
 * finalizer the chapter gives tl_map, so the differential test and the
 * benchmark compare probing strategies, not hash functions. It is the
 * oracle of test_swiss.c's differential test and the bench's reference.
 * Course-side code: plain malloc, nothing through the tl_ allocator hook.
 */
#ifndef DS02_BASELINE_LP_H
#define DS02_BASELINE_LP_H

#include <stdint.h>
#include <stdlib.h>

enum { LP_EMPTY = 0, LP_FULL = 1, LP_TOMB = 2 };

typedef struct {
    uint64_t key, value;
    int state;
} lp_slot;

typedef struct {
    lp_slot *slots;
    size_t cap, len, used; /* used = live + tombstones */
} lp_map;

static inline uint64_t lp_mix64(uint64_t x) {
    x ^= x >> 30;
    x *= 0xBF58476D1CE4E5B9ull;
    x ^= x >> 27;
    x *= 0x94D049BB133111EBull;
    return x ^ (x >> 31);
}

static inline void lp_init(lp_map *m) {
    m->slots = NULL;
    m->cap = m->len = m->used = 0;
}

static inline void lp_free(lp_map *m) {
    free(m->slots);
    lp_init(m);
}

/* 1 and *idx = slot when present; 0 and *idx = the slot to insert into. */
static inline int lp_probe(const lp_map *m, uint64_t k, size_t *idx) {
    size_t mask = m->cap - 1, i = (size_t)lp_mix64(k) & mask, tomb = SIZE_MAX;
    for (size_t step = 0; step < m->cap; step++, i = (i + 1) & mask) {
        const lp_slot *s = &m->slots[i];
        if (s->state == LP_EMPTY) {
            *idx = tomb == SIZE_MAX ? i : tomb;
            return 0;
        }
        if (s->state == LP_TOMB) {
            if (tomb == SIZE_MAX) tomb = i;
        } else if (s->key == k) {
            *idx = i;
            return 1;
        }
    }
    *idx = tomb;
    return 0;
}

static inline int lp_resize(lp_map *m, size_t want) {
    size_t cap = 8;
    while (cap < want) cap *= 2;
    lp_slot *slots = calloc(cap, sizeof *slots);
    if (!slots) return -1;
    for (size_t i = 0; i < m->cap; i++) {
        if (m->slots[i].state != LP_FULL) continue;
        size_t j = (size_t)lp_mix64(m->slots[i].key) & (cap - 1);
        while (slots[j].state == LP_FULL) j = (j + 1) & (cap - 1);
        slots[j] = m->slots[i];
    }
    free(m->slots);
    m->slots = slots;
    m->cap = cap;
    m->used = m->len;
    return 0;
}

static inline int lp_put(lp_map *m, uint64_t k, uint64_t v) {
    if (m->cap == 0 || (m->used + 1) * 4 >= m->cap * 3)
        if (lp_resize(m, (m->len + 1) * 2) != 0) return -1;
    size_t i;
    if (lp_probe(m, k, &i)) {
        m->slots[i].value = v;
        return 0;
    }
    if (m->slots[i].state == LP_EMPTY) m->used++;
    m->slots[i] = (lp_slot){k, v, LP_FULL};
    m->len++;
    return 0;
}

static inline int lp_get(const lp_map *m, uint64_t k, uint64_t *v) {
    size_t i;
    if (m->cap == 0 || !lp_probe(m, k, &i)) return 0;
    *v = m->slots[i].value;
    return 1;
}

static inline int lp_del(lp_map *m, uint64_t k) {
    size_t i;
    if (m->cap == 0 || !lp_probe(m, k, &i)) return 0;
    m->slots[i].state = LP_TOMB;
    m->len--;
    return 1;
}

#endif /* DS02_BASELINE_LP_H */
