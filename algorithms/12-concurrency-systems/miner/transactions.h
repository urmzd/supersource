/**
 * @author: Urmzd Mukhammadnaim
 * @course: CSCI 3110
 * @purpose: Defines TransactionList object and its corresponding methods.
 * @date 2021-06-13
 */

#ifndef TRANSACTIONS_H
#define TRANSACTIONS_H

typedef struct Transaction {
  unsigned int id;
  char payer[32];
  char payee[32];
  unsigned int amount;
  unsigned int fee;
} transaction;

typedef struct TransactionNode {
  transaction data;
  struct TransactionNode *next;
} transaction_node;

typedef struct TransactionList {
  transaction_node *head;
  transaction_node *tail;
} transaction_list;

typedef enum TransactionPrintStates {
  NONE,
  REMOVE,
  ADD,
} transaction_print_states;

extern const transaction transaction_default;

extern transaction_list *list_new();

extern transaction_node *list_remove(transaction_list *list, unsigned int id);
extern transaction_node *list_remove_next(transaction_list *list,
                                          transaction_node *node);
extern transaction_node *list_remove_first(transaction_list *list);
extern transaction_node *list_remove_size(transaction_list *list,
                                          unsigned int available_size);

extern unsigned int list_append(transaction_list *list,
                                transaction transaction);

extern unsigned int transaction_sizeof(transaction transaction);
extern unsigned int transaction_priority(transaction transaction);
extern void transaction_printf(transaction transaction,
                               transaction_print_states removed);

extern void transaction_node_free(transaction_node *node);
extern void list_free(transaction_list *list);

extern transaction *transaction_scanf();
#endif
