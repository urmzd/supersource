/* My tests for ds.03 (rung R2). */
#include <stddef.h>

#include "tinyllm.h"
#include "ss_test.h"

typedef struct {
    int id;
    tl_list_node node;
} item;

static int id(tl_list_node *n) { return TL_CONTAINER_OF(n, item, node)->id; }

SS_TEST(touch_order_by_hand) {
    item a = {1, {0}}, b = {2, {0}}, c = {3, {0}};
    tl_lru l;
    tl_lru_init(&l);
    tl_lru_touch(&l, &a.node);
    tl_lru_touch(&l, &b.node);
    tl_lru_touch(&l, &c.node);
    tl_lru_touch(&l, &a.node);
    SS_EQ(l.len, 3);
    SS_EQ(id(tl_lru_pop_oldest(&l)), 2);
    SS_EQ(id(tl_lru_pop_oldest(&l)), 3);
    SS_EQ(id(tl_lru_pop_oldest(&l)), 1);
}

SS_TEST(pop_empty_is_null) {
    tl_lru l;
    tl_lru_init(&l);
    SS_TRUE(tl_lru_pop_oldest(&l) == NULL);
    SS_EQ(l.len, 0);
}

SS_TEST(retouch_moves) {
    item a = {1, {0}}, b = {2, {0}};
    tl_lru l;
    tl_lru_init(&l);
    tl_lru_touch(&l, &a.node);
    tl_lru_touch(&l, &b.node);
    tl_lru_touch(&l, &a.node);
    tl_lru_touch(&l, &a.node);
    SS_EQ(l.len, 2);
    SS_EQ(id(l.head.next), 2);
    SS_EQ(id(l.head.prev), 1);
    SS_TRUE(l.head.next->next == l.head.prev);
}

SS_TEST(double_remove_is_harmless) {
    item a = {1, {0}}, b = {2, {0}};
    tl_lru l;
    tl_lru_init(&l);
    tl_lru_touch(&l, &a.node);
    tl_lru_touch(&l, &b.node);
    tl_lru_remove(&l, &a.node);
    SS_TRUE(a.node.prev == NULL && a.node.next == NULL);
    tl_lru_remove(&l, &a.node);
    SS_EQ(l.len, 1);
    SS_EQ(id(tl_lru_pop_oldest(&l)), 2);
    SS_TRUE(tl_lru_pop_oldest(&l) == NULL);
}

SS_TEST(walk_and_remove) {
    item it[6];
    tl_lru l;
    tl_lru_init(&l);
    for (int i = 0; i < 6; i++) {
        it[i] = (item){i, {NULL, NULL}};
        tl_lru_touch(&l, &it[i].node);
    }
    for (tl_list_node *n = l.head.next, *nx = n->next; n != &l.head; n = nx, nx = n->next)
        if (id(n) < 3) tl_lru_remove(&l, n);
    SS_EQ(l.len, 3);
    SS_EQ(id(l.head.next), 3);
    SS_EQ(id(l.head.prev), 5);
}

int main(void) { return SS_RUN_ALL(); }
