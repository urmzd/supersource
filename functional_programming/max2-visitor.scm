#!/usr/bin/chezscheme

(load "list-visitor.scm")

#|
  Provides a reducer object which determines the second greatest 
  number as each number in a list L with length > 1 is provided.
  --------------------------------
  Reducer object:
  @param current-value: An instance of an element in list L.
  @param accumulator: The second greatest number found in list L up until
  the element `current-value`.
  @return: -inf.0 if not enough elements have been provided, otherwise 
  the second greatest number.
|#

(define (new-max2) 
  (let ((current-max -inf.0) (iterations 0))
    (lambda (current-value accumulator) 
      (cond ((= iterations 0)
             (set! iterations 1) 
             (set! current-max current-value) 
             -inf.0) 
           (else (if (> current-value current-max) 
                    (let ((max2 current-max)) 
                      (set! current-max current-value) 
                      max2) 
                    accumulator)))
      )
    )
  )


; -- TEST --
(define x (list-visitor (list 1 2 3 4 5 6) (new-max2) -inf.0))
(display x)
(newline)
