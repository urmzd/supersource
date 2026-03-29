#!/usr/bin/chezscheme

#| list visitor 
 | visitor(L, f, p):
 |   if L is empty:
 |     return p
 |   else:
 |     head = removeFirst(L)  // L contains one element less
 |     return visitor(L, f, f(head, p))
 | Example:
 |   L = [1, 2, 3, 4, 5]
 |   f = add(a,b): return a + b
 |   visitor(L, f, 0) => 1 + 2 + 3 + 4 + 5
 |     1 + 0 => 1
 |     2 + 1 => 3
 |     3 + 3 => 6
 |     4 + 6 => 10
 |     5 + 10 => 15
 |#

(define list-visitor (lambda (L func partial-result) 
    (if (null? L)
      partial-result
      (let* ((head (car L)) 
             (rest (cdr L)) 
             (new-result (func head partial-result))
            )
         (list-visitor rest func new-result)
      )
    )
  )
)

(display (list-visitor '(1 2 3 4 5) + 0))
(newline)
(display (list-visitor '(1 2 3 4) + 0))
(newline)
(display (list-visitor '(1 2 3 4 5) cons '()))
(newline)
(display (list-visitor '(11 8 31 7) max 0))
(newline)
