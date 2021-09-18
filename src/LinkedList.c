#include <stdio.h>
#include <stdlib.h>

#include "LinkedList.h"

extern int Node_free(Node *node) { return 0; }

extern Node *Node_append(Node *head, Node *node_to_append) {
  if (head == NULL) {
    return node_to_append;
  }

  return Node_append(head->next, node_to_append);
}

extern Node *Node_new(void *data) {
  Node *new_node = malloc(sizeof(Node));
  new_node->data = data;
  return new_node;
}
