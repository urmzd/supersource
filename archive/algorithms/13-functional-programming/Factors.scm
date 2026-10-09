#!/usr/bin/racket
#lang racket

; Load read-line function
(require "IsPrime.scm")

(define factor 
  (lambda (num [start 2] [factors (list)]) 
    (if (= num 1) factors 
       (if (= (modulo num start) 0) 
          (factor (/ num start) start (append factors (list start))) 
          (factor num (+ start 1) factors)
          )
       )
    ))

; Read in number and print out factors.
(factor (read-number))
