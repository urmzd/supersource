/**
 * @author: Urmzd Mukhammadnaim
 * @course: CSCI 3110
 * @purpose: Contains implementations for a thread-safe queue.
 * @revision 2021-07-25
 */
#include "event.h"
#include "memory.h"
#include "nonce.h"
#include "reader.h"

#include <pthread.h>
#include <stdlib.h>

// Tags for event operations.
char *event_mine = "MINE";
char *event_transaction = "TRX";
char *event_block = "BLK";
char *event_epoch = "EPOCH";
char *event_end = "END";

/**
 * A helper function to construct an event data object given the necessary attributes.
 * @param event: The name of the event.
 * @param body: The input associated with the event.
 * @return: An event data object with a name of `event` and the body `body`.
 */
extern event_data *event_data_init(char *event, void *body) {
  event_data *data = calloc(1, sizeof(event_data));
  data->event_name = event;
  data->event_body = body;

  return data;
}

/**
 * Helper function to free queue memory.
 * @param q: The queue where memory should be freed.
 */
extern void event_q_free(event_q *q) {
  free(q->head);
  free(q);
}

/**
 * Helper function to construct an event queue object.
 * @return: The constructed event queue.
 */
extern event_q *event_q_init() {
  event_q *q = calloc(1, sizeof(event_q));
  return q;
}

/**
 * THREAD-SAFE
 * Removes and provides the first element in the queue.
 * @param q: The queue to traverse.
 * @return: The first element of `q`.
 */
extern event_data *event_q_pop(event_q *q) {
  pthread_mutex_lock(&reader_mutex);
  if (q->head == NULL) {
    pthread_cond_wait(&reader_condition, &reader_mutex);
  }

  event_node *temp_head = q->head;
  q->head = temp_head->next;

  pthread_mutex_unlock(&reader_mutex);
  return temp_head->data;
}

/**
 * THREAD-SAFE
 * Adds an element to the end of the queue.
 * @param q: The queue to append object to.
 * @param input: The data to provide the constructed node to.
 */
extern void event_q_append(event_q *q, event_data *input) {
  pthread_mutex_lock(&reader_mutex);
  event_node *e_n = calloc(1, sizeof(event_node));
  e_n->data = input;

  if (q->head == NULL) {
    q->head = e_n;
  } else {
    event_node *curr = q->head;

    while (curr->next != NULL) {
      curr = curr->next;
    }

    curr->next = e_n;
  }

    pthread_mutex_unlock(&reader_mutex);
    pthread_cond_signal(&reader_condition);
}

/**
 * Consumes input for a BLK event and handles the operation.
 * @param memory_pool: The memory pool holding all transactions.
 * @param prev_sig: The previous signature computed.
 * @param prev_id: The previous id computed.
 * @param input: The data for the event.
 */
extern void event_block_handler(memory_pool *memory_pool,
                                unsigned int *prev_sig, unsigned int *prev_id,
                                event_body_block input) {
  (*prev_id) = input.prev_id;
  (*prev_sig) = input.prev_sig;

  transaction_node *current_node = input.transactions->head;

  while (current_node) {
    if (memory_pool_remove(memory_pool, current_node->data)) {
      transaction data = current_node->data;
      transaction_printf(data, REMOVE);
    };
    current_node = current_node->next;
  }

  free(input.transactions);
}

/**
 * Consumes input for a TRX event and adds transaction to memory pool
 * @param memory_pool: The memory pool containing the transactions.
 * @param payload: Details for the transaction event.
 */
extern void event_transaction_handler(memory_pool *memory_pool,
                                      transaction payload) {
  memory_pool_insert(memory_pool, payload);
  transaction_printf(payload, ADD);
}

/**
 * Consumes input for a EPOCH event and ages all transactions.
 * @param memory_pool: The memory pool containing the transactions.
 */
extern void event_epoch_handler(memory_pool *memory_pool) {
  memory_pool_age(memory_pool);
}

/**
 * Consumes input for a END event and frees the memory pools space.
 * @param memory_pool: The memory pool containing the transactions.
 */
extern void event_end_handler(memory_pool *memory_pool) { free(memory_pool); }

/**
 * Consumes input for a MINE event and begins to compute nonce.
 * @param memory_pool: The memory pool contain the transactions.
 * @param prev_sig: The previous computed signature.
 * @param prev_id: The previous computed id.
 * @param input: Input containing the number of threads to be used to compute nonce.
 */
extern void event_mine_handler(memory_pool *memory_pool, unsigned int *prev_sig,
                               unsigned int *prev_id, event_body_mine input) {

  int N = input.n_of_threads;

  // Initialize computation block.
  block *computed_block = block_new();
  computed_block->id = (*prev_id) + 1;
  computed_block->previous_id = (*prev_id);
  computed_block->previous_sig = (*prev_sig);
  computed_block->transactions = block_transactions_new();
  computed_block->nonce = 0;

  // Declare available size.
  unsigned int available_space = 256;
  int block_size;
  transaction selected_transaction;

  do {
    selected_transaction = memory_pool_pop(memory_pool, available_space);
    if (!selected_transaction.id) {
      break;
    }
    block_transactions_append(computed_block, selected_transaction);
    block_size = block_sizeof(computed_block);
    available_space = block_size >= 256 ? 0 : 256 - block_size;
  } while (selected_transaction.id);

  unsigned int sig_no_nonce = block_signature_without_nonce(computed_block);

  // Initialize shared thread args.
  nonce_shared_compute_args *nonce_shared_args =
      malloc(sizeof(nonce_shared_compute_args));
  nonce_shared_args->found = 0;
  nonce_shared_args->nonce = 0;

  // Create N threads;
  pthread_t nonce_threads[N];
  pthread_mutex_init(&nonce_mutex, NULL);

  unsigned int tid;

  for (tid = 0; tid < N; tid++) {
    // Initialize thread specific args.
    nonce_compute_args *nonce_args = malloc(sizeof(nonce_compute_args));

    nonce_args->tid = tid;
    nonce_args->sig_no_nonce = sig_no_nonce;
    nonce_args->no_of_threads = N;
    nonce_args->shared_args = nonce_shared_args;

    // Initialize our threads.
    pthread_create(&nonce_threads[tid], NULL, &nonce_compute,
                   (void *)nonce_args);
  }

  // Wait until N threads finish processing.
  for (tid = 0; tid < N; tid++) {
    pthread_join(nonce_threads[tid], NULL);
  }

  // Destroy the nonce_mutex.
  pthread_mutex_destroy(&nonce_mutex);

  // Update nonce base on multi-thread computation.
  computed_block->nonce = nonce_shared_args->nonce;
  computed_block->signature =
      block_signature_with_nonce(nonce_shared_args->nonce, &sig_no_nonce);

  // Print out block.
  block_printf(computed_block);

  (*prev_sig) = computed_block->signature;
  (*prev_id) = computed_block->id;

  // Free memory.
  block_free(computed_block);
}
