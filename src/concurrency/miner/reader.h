/**
 * @author: Urmzd Mukhammadnaim
 * @course: CSCI 3110
 * @purpose: Contains implementations to handle concurrent system input.
 * @revision 2021-07-25
 */
#include "event.h"

#ifndef READER_H
#define READER_H

extern pthread_mutex_t reader_mutex;
extern pthread_cond_t reader_condition;

extern event_q *reader_init();

extern void reader_join();

extern void *reader_runner(void *args);

extern event_data *reader_mine_handler(char *event);
extern event_data *reader_block_handler(char *event);
extern event_data *reader_end_handler(char *event);
extern event_data *reader_transaction_handler(char *event);
extern event_data *reader_epoch_handler(char *event);
extern void reader_printf(char *event, unsigned int *id);

extern event_data *reader_handler();

#endif
