#include "block.h"
#include "sys/types.h"
#include <pthread.h>

#ifndef NONCE_H
#define NONCE_H

typedef struct NonceSharedComputeArgs {
  unsigned int nonce;
  unsigned int found;
} nonce_shared_compute_args;

typedef struct NonceComputeArgs {
  unsigned int sig_no_nonce;
  unsigned int no_of_threads;
  unsigned int tid;
  nonce_shared_compute_args *shared_args;
} nonce_compute_args;

extern void *nonce_compute(void *args);

extern void nonce_printf(unsigned int tid, unsigned int sig_nonce);

extern pthread_mutex_t nonce_mutex;

#endif
