/**
 * @author: Urmzd Mukhammadnaim
 * @course: CSCI 3110
 * @purpose: Implements memory pool and its corresponding methods.
 * @date 2021-06-13
 */
#include <stdio.h>
#include <stdlib.h>

#include "memory_pool.h"

extern transaction_node *memory_pool_remove(memory_pool *memory_pool,
                                            transaction transaction) {
    int priority_level;
    transaction_node *removed;

    for (priority_level = memory_pool->length - 1; priority_level >= 0;
         priority_level--) {

        removed = list_remove(memory_pool->pool[priority_level], transaction.id);
        if (removed) {
            break;
        }
    }

    return removed;
}

extern memory_pool *memory_pool_new(unsigned int size) {
    memory_pool *memory_pool = malloc(sizeof(*memory_pool));

    unsigned int priority_level;

    memory_pool->length = size;
    memory_pool->pool = malloc(sizeof(transaction_list) * size);

    for (priority_level = 0; priority_level < size; priority_level++) {
        memory_pool->pool[priority_level] = list_new();
    }

    return memory_pool;
}

extern unsigned int memory_pool_insert(memory_pool *memory_pool,
                                       transaction transaction) {
    unsigned int priority = transaction_priority(transaction);

    return list_append(memory_pool->pool[priority], transaction);
}

extern transaction memory_pool_pop(memory_pool *memory_pool,
                                   unsigned int available_size) {
    int priority_level;
    transaction_node *found_transaction;

    for (priority_level = memory_pool->length - 1; priority_level >= 0;
         priority_level--) {

        found_transaction =
                list_remove_size(memory_pool->pool[priority_level], available_size);

        if (found_transaction) {
            return found_transaction->data;
        }
    }

    return transaction_default;
}

extern void memory_pool_age(memory_pool *memory_pool) {

    unsigned int priority_level;

    for (priority_level = memory_pool->length - 1; priority_level > 0;
         priority_level--) {
        transaction_node *found =
                list_remove_first(memory_pool->pool[priority_level - 1]);
        if (found) {
            list_append(memory_pool->pool[priority_level], found->data);
            printf("Aging transaction (%i):", priority_level);
            transaction_printf(found->data, NONE);
        }
    }
}
