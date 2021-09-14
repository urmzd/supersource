#ifndef LINKED_LIST_H
#define LINKED_LIST_H

typedef struct Node {
  void *data;
  struct Node *next;
} node;

typedef node *(*node_new_fn)(void *data);

extern node *node_new(void *data);

extern node *node_insert(node *head, void *data, node_new_fn *node_new_fn);

extern int node_free(node *node);

#endif
