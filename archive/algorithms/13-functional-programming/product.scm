#!/usr/bin/chezscheme

#|
  Calculate product of a list.
  --------------------------------
  @param L: The list to aggregate.
  @return: The result of multiplying all numbers in list non-empty list L 
  together, otherwise 1.
|#
(define (product-tr L) (cond ((null? L) 1) (else (* (car L) (product-tr (cdr L))))))

;Test
(display (product-tr (list 1 2 3 4)))
