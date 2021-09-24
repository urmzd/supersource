/**
 * @author: Urmzd Mukhammadnaim
 * @course: CSCI 3110
 * @purpose: Implements transaction list and its corresponding methods.
 * @date 2021-06-13
 */

#include <math.h>
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "transactions.h"

const transaction transaction_default = {0, {'\0'}, {'\0'}, 0, 0};

extern transaction_list *list_new() {
  transaction_list *list = malloc(sizeof(transaction_list));
  list->head = NULL;
  list->tail = NULL;
  return list;
};

extern unsigned int list_append(transaction_list *list,
                                transaction transaction) {

  if (!list || !transaction.id) {
    return 0;
  }

  transaction_node *node = malloc(sizeof(transaction_node));
  node->data = transaction;
  node->next = NULL;

  if (!list->head) {
    list->head = node;
    return 1;
  }

  if (!list->tail) {
    list->tail = node;
    list->head->next = node;
    return 1;
  }

  list->tail->next = node;
  list->tail = node;

  return 1;
};

extern transaction_node *list_remove_first(transaction_list *list) {
  if (!list || !list->head) {
    return NULL;
  }

  transaction_node *first = list->head;

  list->head = list->head->next;

  if (!list->head || (list->head && !list->head->next)) {
    list->tail = NULL;
  }

  return first;
}

extern transaction_node *list_remove_next(transaction_list *list,
                                          transaction_node *node) {
  if (!list || !node) {
    return NULL;
  }

  transaction_node *next = node->next;
  node->next = node->next->next;

  return next;
}

extern transaction_node *list_remove(transaction_list *list, unsigned int id) {
  if (!list || !list->head || !id) {
    return NULL;
  }

  if (list->head->data.id == id) {
    return list_remove_first(list);
  }

  transaction_node *previous_node = NULL;
  transaction_node *current_node = list->head;
  unsigned int current_id;

  while (current_node->next) {
    previous_node = current_node;
    current_node = current_node->next;
    current_id = current_node->data.id;

    if (current_id == id) {
      return list_remove_next(list, previous_node);
    }
  }

  return NULL;
};

extern transaction_node *list_remove_size(transaction_list *list,
                                          unsigned int available_size) {
  transaction_node *curr = list->head;
  transaction_node *prev;

  while (curr) {
    if (transaction_sizeof(curr->data) <= available_size) {
      if (curr->data.id != list->head->data.id) {
        return list_remove_next(list, prev);
      }

      return list_remove_first(list);
    }

    prev = curr;
    curr = curr->next;
  }

  return NULL;
}

extern unsigned int transaction_priority(transaction transaction) {
  int priority = transaction.fee / transaction_sizeof(transaction);
  return priority > 9 ? 9 : (int)priority;
}

extern void transaction_printf(transaction transaction,
                               transaction_print_states print_state) {
  char *tag = print_state == NONE     ? ""
              : print_state == REMOVE ? "Removing transaction: "
                                      : "Adding transaction: ";

  printf("%s%u %s %s %u %u", tag, transaction.id, transaction.payer,
         transaction.payee, transaction.amount, transaction.fee);
  printf("\n");
}

extern unsigned int transaction_sizeof(transaction transaction) {
  return sizeof(transaction.id) + sizeof(transaction.amount) +
         (strlen(transaction.payee) + 1) + (strlen(transaction.payer) + 1) +
         sizeof(transaction.fee);
}

extern void transaction_node_free(transaction_node *node) {
  if (node) {
    free(node);
    node = NULL;
  }
}

extern void list_free(transaction_list *list) {
  transaction_node *prev = NULL;
  transaction_node *curr = list->head;

  while (curr) {
    transaction_node_free(prev);
    curr = curr->next;
  }

  transaction_node_free(prev);
}

extern transaction *transaction_scanf() {
  transaction *payload = calloc(1, sizeof(transaction));

  scanf("%u %s %s %u %u", &(payload->id), payload->payer, payload->payee,
        &(payload->amount), &(payload->fee));

  return payload;
}
