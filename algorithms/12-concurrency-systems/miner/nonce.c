/**
 * @author: Urmzd Mukhammadnaim
 * @course: CSCI 3110
 * @purpose: Handle multithreading nonce computation.
 * @date 2021-06-13
 */
#include "nonce.h"
#include "block.h"

#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>

// Declare global nonce_mutex.
pthread_mutex_t nonce_mutex;

/**
 *
 * @param tid: Integer indicating the the thread id.
 * @param nonce: Integer indicating the nonce thread tid is calculating.
 */
extern void nonce_printf(unsigned int tid, unsigned int nonce) {
  printf("Thread %u checking nonce 0x%8.8x", tid, nonce);
  printf("\n");
}

/**
 * @param args: Pointer containing unique and shared arguments.
 * @return: NULL to indicate thread is being closed.
 */
extern void *nonce_compute(void *args) {
  nonce_compute_args *nonce_args = (nonce_compute_args *)args;
  nonce_shared_compute_args *nonce_shared_args = nonce_args->shared_args;

  unsigned int tid = nonce_args->tid;
  unsigned int sig_no_nonce = nonce_args->sig_no_nonce;
  unsigned int no_of_threads = nonce_args->no_of_threads;

  unsigned int msb = 1;
  unsigned int sig_nonce = 0;
  unsigned int nonce = tid;

  while (1) {
    pthread_mutex_lock(&nonce_mutex);
    nonce_printf(tid, nonce);

    // Only stop when nonce has been found.
    if (nonce_shared_args->found) {
      if (nonce > nonce_shared_args->nonce) {
        break;
      }
    }

    // Compute the nonce.
    sig_nonce = block_signature_with_nonce(nonce, &sig_no_nonce);
    msb = block_signature_msb(&sig_nonce);

    // Check if nonce has been computed.
    if (!msb) {
      nonce_shared_args->found = 1;
      nonce_shared_args->nonce = nonce;
      break;
    }

    nonce += no_of_threads;
    pthread_mutex_unlock(&nonce_mutex);
  }

  pthread_mutex_unlock(&nonce_mutex);
  return NULL;
}
