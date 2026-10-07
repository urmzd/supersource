/* list.c - the part you write. */
#include "list.h"

void list_init(ListNode *head) {
  /* SOLUTION-BEGIN */
  head->prev = head;
  head->next = head;
  /* SOLUTION-END */
}

void list_insert_after(ListNode *node, ListNode *fresh) {
  /* SOLUTION-BEGIN */
  /* No NULL checks and no special case for an empty list: the sentinel means
   * node->next always exists, even when it is node itself. */
  fresh->prev = node;
  fresh->next = node->next;
  node->next->prev = fresh;
  node->next = fresh;
  /* SOLUTION-END */
}

void list_insert_before(ListNode *node, ListNode *fresh) {
  /* SOLUTION-BEGIN */
  list_insert_after(node->prev, fresh);
  /* SOLUTION-END */
}

void list_push_back(ListNode *head, ListNode *fresh) {
  /* SOLUTION-BEGIN */
  /* The sentinel is the element after the tail, so inserting before it appends.
   * That is the whole reason push_back needs no tail pointer. */
  list_insert_before(head, fresh);
  /* SOLUTION-END */
}

void list_push_front(ListNode *head, ListNode *fresh) {
  /* SOLUTION-BEGIN */
  list_insert_after(head, fresh);
  /* SOLUTION-END */
}

void list_remove(ListNode *node) {
  /* SOLUTION-BEGIN */
  node->prev->next = node->next;
  node->next->prev = node->prev;

  /* Reset to the empty-node state rather than leaving stale pointers into a
   * list this node is no longer part of. Two things fall out of that: removing
   * twice is harmless instead of corrupting whatever the old neighbours have
   * become, and list_is_linked can tell the difference. */
  list_init(node);
  /* SOLUTION-END */
}

int list_is_linked(const ListNode *node) {
  /* SOLUTION-BEGIN */
  return node->next != node;
  /* SOLUTION-END */
}

int list_empty(const ListNode *head) {
  /* SOLUTION-BEGIN */
  return head->next == head;
  /* SOLUTION-END */
}

size_t list_length(const ListNode *head) {
  /* SOLUTION-BEGIN */
  size_t n = 0;
  for (const ListNode *p = head->next; p != head; p = p->next) n++;
  return n;
  /* SOLUTION-END */
}

void list_splice(ListNode *dst, ListNode *src) {
  /* SOLUTION-BEGIN */
  if (list_empty(src)) return;

  ListNode *first = src->next;
  ListNode *last = src->prev;
  ListNode *tail = dst->prev;

  /* Four pointer writes, whatever the length. The elements themselves are
   * never touched, which is what "O(1) splice" actually means and why an
   * intrusive list beats an array when you move items between collections. */
  tail->next = first;
  first->prev = tail;
  last->next = dst;
  dst->prev = last;

  list_init(src);
  /* SOLUTION-END */
}
