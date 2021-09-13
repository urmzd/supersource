#ifndef LINKED_LIST_H
#define LINKED_LIST_H

typedef struct OneWayNode {
  int data;
  struct OneWayNode *next;
} one_way_node;

typedef struct LinkedList {
  one_way_node *head;
} linked_list;

extern linked_list *new_linked_list();
extern one_way_node *new_way_node(int data);

#endif
