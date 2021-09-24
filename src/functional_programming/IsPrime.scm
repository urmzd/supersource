#!/usr/bin/racket
#lang racket

; Export function
(provide read-number)

; Read in line and evaluate number
(define read-number (lambda () 
                      (string->number (read-line (current-input-port)))
                      ))

; For [i, n), if n % i == 0, we have found a composite number
; Else we have found a prime number 
(define is-prime (lambda (n [i 2]) 
                   (if (< i n) (if (= (modulo n i) 0) #f (is-prime n (+ i 1))) #t
                      )))

; Execute function
(is-prime (read-number))
