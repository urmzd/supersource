/* c/src/ds/list.c (ds.03): the intrusive doubly linked list under tl_lru.
 * Contract: tinyllm/ds.h (ds.03 section).
 *
 * A list is circular with a sentinel: l->head is a node that holds no
 * element, head.next is the first (least recently used) node and head.prev
 * the last (most recent). With the sentinel, unlinking or linking any node
 * is the same four pointer writes; there is no "first node" or "last node"
 * special case. A node that is in no list has prev == next == NULL, which
 * is how remove recognizes it.
 *
 * lru.c builds the policy (touch, pop the oldest) on these primitives.
 */
#include <stddef.h>

#include "tinyllm/ds.h"

void tl_lru_init(tl_lru *l) {
/* SOLUTION-BEGIN ds.03 */
    l->head.prev = &l->head; /* empty: the sentinel points at itself */
    l->head.next = &l->head;
    l->len = 0;
/* SOLUTION-END */
}

void tl_lru_remove(tl_lru *l, tl_list_node *n) {
/* SOLUTION-BEGIN ds.03 */
    if (n == NULL || n->prev == NULL || n->next == NULL) return; /* in no list */
    n->prev->next = n->next; /* the neighbours now skip n */
    n->next->prev = n->prev;
    n->prev = NULL; /* mark n as unlinked, so a second remove is harmless */
    n->next = NULL;
    l->len--;
/* SOLUTION-END */
}

tl_list_node *tl_lru_pop_oldest(tl_lru *l) {
/* SOLUTION-BEGIN ds.03 */
    if (l->head.next == &l->head) return NULL; /* only the sentinel: empty */
    tl_list_node *n = l->head.next;
    tl_lru_remove(l, n);
    return n;
/* SOLUTION-END */
}
