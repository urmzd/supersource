#include <stdio.h>
#include <stdlib.h>

#include "LinkedList.h"

extern int node_free(node *node) { return 0; }

extern node *node_append(node *head, void *data, node_new_fn *node_new_fn) {

  return NULL;
}

extern node *node_new(void *data) {
  node *new_node = malloc(sizeof(node));
  new_node->data = data;
  return new_node;
}
