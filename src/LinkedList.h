#ifndef LINKED_LIST_H
#define LINKED_LIST_H

/**
 * struct Node - An instance of a LinkedList holding data and a pointer to the
 * next LinkedList.
 */
typedef struct Node {
  void *data;
  struct Node *next;
} Node;

/**
 * @brief - Creates a `Node` instance given some generic data.
 *
 * @param data - A pointer to some structure.
 * @return -  A `Node` instance holding the data provided.
 */
extern Node *Node_new(void *data);

/**
 * @brief - Puts the `node_to_append` at the end of the `head` LinkedList.
 *
 * @param head - The head of the LinkedList.
 * @param node_to_append - The node containing the data to append.
 * @return - The tail of the linked list.
 */
extern Node *Node_append(Node *head, Node *node_to_append);

/**
 * @brief - Deletes the allocated memory of the `Node` structure.
 *
 * @param node - The node to free.
 * @return - 1 if successful, 0 otherwise.
 */
extern int Node_free(Node *node);

#endif
