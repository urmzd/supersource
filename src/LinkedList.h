#ifndef LINKED_LIST_H
#define LINKED_LIST_H

typedef struct Node {
  void *data;
  struct Node *next;
} node;

extern node *node_new(void *data);
extern node *node_insert(node *head, void *data);
extern node *node_free(node *node);

#endif
