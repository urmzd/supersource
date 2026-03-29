/**
 * @author: Urmzd Mukhammadnaim
 * @course: CSCI 3110
 * @purpose: Executes miner using user input actions.
 * @date 2021-06-13
 */

#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "block.h"
#include "event.h"
#include "memory_pool.h"
#include "nonce.h"
#include "reader.h"

int main() {
  // Memory pool init.
  memory_pool *memory_pool = memory_pool_new(10);

  // Event queue init.
  event_q *q = reader_init();

  // Store most recent sig and id for next mine.
  unsigned int prev_sig = 0;
  unsigned int prev_id = 0;

  // Hold produced element.
  event_data *data;

  while (1) {
    // Get data.
    data = event_q_pop(q);

    char *event = data->event_name;
    void *body = data->event_body;

    // Call relevant handlers.
    if (!strcmp(event, event_transaction)) {
      event_transaction_handler(memory_pool, *(transaction *)body);
    } else if (!strcmp(event, event_mine)) {
      event_mine_handler(memory_pool, &prev_sig, &prev_id,
                         *(event_body_mine *)body);
    } else if (!strcmp(event, event_block)) {
      event_block_handler(memory_pool, &prev_sig, &prev_id,
                          *(event_body_block *)body);
    } else if (!strcmp(event, event_epoch)) {
      event_epoch_handler(memory_pool);
    } else {
      event_end_handler(memory_pool);
      break;
    }
  }

  // Begin cleaning process.
  reader_join();
  event_q_free(q);

  return 0;
}
