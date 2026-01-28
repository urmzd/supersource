#!/usr/bin/chezscheme

#|
  Provides an iterator which returns the square root of each 
  element in list L.
  --------------------------------
  Iterator:
  @return: The square root of the next element in the list L, and false if
  iterator has reached end of list.
|#
(define (new-sqrt-iterator L) 
  (let ((UL L)) 
    (lambda () 
      (cond ((pair? UL) 
             (let ((answer (sqrt (car UL)))) 
               (set! UL (cdr UL))
               answer) 
             ) 
           (else #f))))
  )

#|
  Finds a sublist where the first element satifies the 
  test method.
  --------------------------------
  @param L: A list to evaluate.
  @param test: A function which tests the first element of L.
  @return: An empty list if no element satifies `test`, otherwise a sublist L'.
|#
(define (find-with-condition L test) 
  (if (null? L) (list)
     (let ((elem (car L))) 
       (cond ((test elem) L) 
            (else (find-with-condition 
                   (cdr L) 
                   test)))
       )))


#|
  Tests if element is a perfect square.
  --------------------------------
  @param elem: The element to test.
  @return: True if perfect square, otherwise false.
|#
(define (test-square elem) (integer? (sqrt elem)))

#|
  Provides an iterator which outputs the next perfect square in a list L.
  --------------------------------
  @param L: The list to iterate over.
  @return: The next perfect square, otherwise false.
|#
(define (new-square-iterator L) 
  (let ((UL L)) 
    (lambda () 
      (let ((find-result (find-with-condition UL test-square)))
        (cond 
          ((null? find-result) #f) 
          (else 
           (set! UL (cdr find-result))
           (car find-result)))
        ))  
    ) )


; Test
(define sq (new-sqrt-iterator (list 1 2 3 4)))
( display (sq) )
(newline)
( display (sq) )
(newline)
( display (sq) )
(newline)
( display (sq) )
(newline)
( display (sq) )
(newline)

; Test 2
(define sq2 (new-square-iterator (list 1 2 3 4)))
( display (sq2) )
(newline)
( display (sq2) )
(newline)
( display (sq2) )
(newline)
