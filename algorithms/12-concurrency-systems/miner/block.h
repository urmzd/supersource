/**
 * @author: Urmzd Mukhammadnaim
 * @course: CSCI 3110
 * @purpose: Defines Block object and its corresponding methods.
 * @date 2021-06-13
 */

#include "transactions.h"
#ifndef BLOCK_H
#define BLOCK_H

typedef struct BlockTransactions {
  unsigned int length;
  transaction_list *list;
} block_transactions;

typedef struct Block {
  unsigned int id;
  unsigned int previous_id;
  unsigned int previous_sig;
  block_transactions *transactions;
  unsigned int nonce;
  unsigned int signature;
} block;

extern unsigned int block_sizeof(block *block);
extern unsigned int block_signature_without_nonce(block *block);
extern unsigned int
block_signature_with_nonce(unsigned int nonce,
                           unsigned int *block_signature_without_nonce);
extern unsigned int block_signature_msb(unsigned int *signature);

extern void block_transactions_append(block *block, transaction transactions);
extern block_transactions *block_transactions_new();

extern void block_printf(block *block);

extern block *block_new();
extern void block_free(block *blk);

#endif
