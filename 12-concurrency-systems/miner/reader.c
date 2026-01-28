/**
 * @author: Urmzd Mukhammadnaim
 * @course: CSCI 3110
 * @purpose: Contains definitions for concurrent system input handlers and
 * helper functions.
 * @revision 2021-07-25
 */

#include "reader.h"
#include "event.h"

#include <assert.h>
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/*
 * Mutex to prevent race conditions.
 * Condition to prevent consuming from empty queue.
 * Thread to allow for concurrent input and system operations.
 */
pthread_mutex_t reader_mutex;
pthread_cond_t reader_condition;
pthread_t reader_thread;

/**
 * Initializes a separate thread to handle user input and
 * provides the event queue containing inputs from stdin.
 * @return: An initialized event queue where input from stdin will be stored.
 */
extern event_q *reader_init() {
  event_q *q = event_q_init();
  pthread_mutex_init(&reader_mutex, NULL);
  pthread_cond_init(&reader_condition, NULL);

  int reader_thread_failed =
      pthread_create(&reader_thread, NULL, &reader_runner, (void *)q);

  assert(!reader_thread_failed);

  return q;
}

/**
 * Waits until reader thread is finished before destroying
 * attributes associated with reader thread operations.
 */
extern void reader_join() {
  pthread_join(reader_thread, NULL);
  pthread_mutex_destroy(&reader_mutex);
  pthread_cond_destroy(&reader_condition);
}

/**
 * Handles input from users.
 * @param args: The event queue where stdin should be stored.
 */
extern void *reader_runner(void *args) {
  event_q *q = (event_q *)args;
  event_data *input;

  for (;;) {
    input = reader_handler();
    event_q_append(q, input);
    if (!strcmp(input->event_name, event_end)) {
      break;
    }
  }

  return NULL;
}

/**
 * Handles the mine event.
 * @param event: The tag associated with the MINE operation.
 * @return: User input collected from stdin.
 */
extern event_data *reader_mine_handler(char *event) {
  reader_printf(event, NULL);
  event_body_mine *data = calloc(1, sizeof(event_body_mine));

  scanf("%u", &data->n_of_threads);

  return event_data_init(event, data);
};

/**
 * Handles the block event.
 * @param event: The tag associated with the BLK operation.
 * @return: User input collected from stdin.
 */
extern event_data *reader_block_handler(char *event) {
  event_body_block *data = calloc(1, sizeof(event_body_block));

  scanf("%u %*u %*i %u", &data->prev_id, &data->length);
  reader_printf(event, &data->prev_id);
  data->transactions = list_new();

  for (unsigned int i = 0; i < data->length; i++) {
    transaction *trx = transaction_scanf();
    list_append(data->transactions, *trx);
  }

  scanf("%*i %i", &data->prev_sig);

  return event_data_init(event, data);
};

/**
 * Handles the end event.
 * @param event: The tag associated with the END operation.
 * @return: User input collected from stdin.
 */
extern event_data *reader_end_handler(char *event) {
  reader_printf(event, NULL);
  return event_data_init(event, NULL);
};


/**
 * Handles the transaction event.
 * @param event: The tag associated with the TRX operation.
 * @return: User input collected from stdin.
 */
extern event_data *reader_transaction_handler(char *event) {
  transaction *trx = transaction_scanf();
  reader_printf(event, &trx->id);
  return event_data_init(event, trx);
};

/**
 * Handles the epoch event.
 * @param event: The tag associated with the EPOCH operation.
 * @return: User input collected from stdin.
 */
extern event_data *reader_epoch_handler(char *event) {
  reader_printf(event, NULL);
  return event_data_init(event, NULL);
};

/**
 * Invokes the appropriate handler for each associated tag.
 * @return: The input received from the invoked handler.
 */
extern event_data *reader_handler() {
  char *event = calloc(6, sizeof(char));
  scanf("%s", event);

  if (!(strcmp(event, event_transaction))) {
    return reader_transaction_handler(event);
  }

  if (!(strcmp(event, event_mine))) {
    return reader_mine_handler(event);
  }

  if (!(strcmp(event, event_block))) {
    return reader_block_handler(event);
  }

  if (!(strcmp(event, event_epoch))) {
    return reader_epoch_handler(event);
  }

  return reader_end_handler(event);
}

/**
 * Helper function to print output once event has been received.
 * @param event: The tag associated with the current event.
 * @param id: Optional parameter for BLK and TRX events.
 */
extern void reader_printf(char *event, unsigned int *id) {
  if (!strcmp(event, event_block) || !(strcmp(event, event_transaction))) {
    printf("Received event %s with ID %u", event, *id);
  } else {
    printf("Received event %s", event);
  }

  printf("\n");
}
