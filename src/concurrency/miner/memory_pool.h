/**
 * @author: Urmzd Mukhammadnaim
 * @course: CSCI 3110
 * @purpose: Defines Memory Pool object and its corresponding methods.
 * @date 2021-06-13
 */

#include "transactions.h"
#ifndef MEMORY_POOL_H
#define MEMORY_POOL_H

typedef struct MemoryPool {
    transaction_list **pool;
    unsigned int length;
} memory_pool;

extern transaction_node *memory_pool_remove(memory_pool *memory_pool,
                                            transaction transaction);

extern void memory_pool_age(memory_pool *memory_pool);

extern unsigned int memory_pool_insert(memory_pool *memory_pool,
                                       transaction transaction);

extern memory_pool *memory_pool_new(unsigned int size);

extern transaction memory_pool_pop(memory_pool *memory_pool,
                                   unsigned int available_space);

#endif
