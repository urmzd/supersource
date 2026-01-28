/**
 * @author: Urmzd Mukhammadnaim
 * @course: CSCI 3110
 * @purpose: Implements Block object and its corresponding methods.
 * @date 2021-06-13
 */

#include <stdio.h>
#include <stdlib.h>

#include "block.h"
#include "siggen.h"

/**
 * @return: A new instance of a linked list to contain transactions.
 */
extern block_transactions *block_transactions_new() {
    block_transactions *transactions = malloc(sizeof(block_transactions));
    transactions->list = list_new();
    transactions->length = 0;
    return transactions;
}

/**
 * @param block: The block to evaluate.
 * @return: The size of the block.
 */
extern unsigned int block_sizeof(block *block) {
    unsigned int size = sizeof(block->id) + sizeof(block->previous_id) +
                        sizeof(block->previous_sig) + sizeof(block->nonce) +
                        sizeof(block->signature) +
                        sizeof(block->transactions->length);

    transaction_node *current = block->transactions->list->head;

    while (current) {
        size += transaction_sizeof(current->data);
        current = current->next;
    }

    return size;
}

/**
 * @param block: The block that should hold the transaction.
 * @param transaction: The transaction to insert into the block.
 */
extern void block_transactions_append(block *block, transaction transaction) {
    list_append(block->transactions->list, transaction);
    block->transactions->length += 1;
}

/**
 * @param nonce: The nonce to add to the signature.
 * @param block_signature_without_nonce: The current signature not containing a
 * nonce.
 * @return: The signature containing the nonce.
 */
extern unsigned int
block_signature_with_nonce(unsigned int nonce,
                           unsigned int *block_signature_without_nonce) {
    return siggen_int(*block_signature_without_nonce, nonce);
}


/**
 * @param signature: The signature to evaluate.
 * @return: The most significant bit of the signature.
 */
extern unsigned int block_signature_msb(unsigned int *signature) {
    if (*signature == 0) {
        return *signature;
    }

    unsigned int msb = *signature >> 24;

    return msb;
}


/**
 * @param block: The block to compute a signature from.
 * @return: The computed signature.
 */
extern unsigned int block_signature_without_nonce(block *block) {
    unsigned int sig = siggen_new();
    sig = siggen_int(sig, block->id);
    sig = siggen_int(sig, block->previous_id);
    sig = siggen_int(sig, block->previous_sig);
    sig = siggen_int(sig, block->transactions->length);

    transaction_node *current_node = block->transactions->list->head;

    while (current_node) {
        transaction d = current_node->data;
        sig = siggen_int(sig, d.id);
        sig = siggen_string(sig, d.payer);
        sig = siggen_string(sig, d.payee);
        sig = siggen_int(sig, d.amount);
        sig = siggen_int(sig, d.fee);

        current_node = current_node->next;
    }

    return sig;
}


/**
 * @param block: The block to print.
 */
extern void block_printf(block *block) {
    printf("Block mined: %u %u 0x%8.8x %u", block->id, block->previous_id,
           block->previous_sig, block->transactions->length);
    printf("\n");

    transaction_node *current_node = block->transactions->list->head;

    while (current_node) {
        transaction_printf(current_node->data, NONE);

        current_node = current_node->next;
    }

    printf("0x%8.8x 0x%8.8x\n", block->nonce, block->signature);
}

/**
 * @return: A new instance of a block.
 */
extern block *block_new() {
    block *b = malloc(sizeof(block));
    return b;
}

/**
 * @param blk: The block pointer to free.
 */
extern void block_free(block *blk) { list_free(blk->transactions->list); }
