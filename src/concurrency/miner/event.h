/**
 * @author: Urmzd Mukhammadnaim
 * @course: CSCI 3110
 * @purpose: Contains definitions for a thread-safe queue.
 * @revision 2021-07-25
 */

#include "memory_pool.h"
#include "transactions.h"
#include <stdlib.h>
#ifndef EVENT_H
#define EVENT_H

typedef struct EventData {
  char *event_name;
  void *event_body;
} event_data;

typedef struct EventNode {
  event_data *data;
  struct EventNode *next;
} event_node;

typedef struct EventQueue {
  event_node *head;
} event_q;

typedef struct EventBodyBlock {
  unsigned int prev_id;
  unsigned int prev_sig;
  unsigned int length;
  transaction_list *transactions;
} event_body_block;

typedef struct EventBodyMine {
  unsigned int n_of_threads;
} event_body_mine;

extern event_data *event_data_init(char *event, void *body);

extern event_q *event_q_init();

extern void event_q_free(event_q *q);

extern void event_q_append(event_q *q, event_data *input);

extern event_data *event_q_pop(event_q *q);

extern void event_mine_handler(memory_pool *memory_pool, unsigned int *prev_sig,
                               unsigned int *prev_id, event_body_mine input);

extern void event_transaction_handler(memory_pool *memory_pool,
                                      transaction payload);

extern void event_block_handler(memory_pool *memory_pool,
                                unsigned int *prev_sig, unsigned int *prev_id,
                                event_body_block input);

extern void event_end_handler(memory_pool *memory_pool);

extern void event_epoch_handler(memory_pool *memory_pool);

extern char *event_mine;
extern char *event_transaction;
extern char *event_block;
extern char *event_epoch;
extern char *event_end;

#endif
