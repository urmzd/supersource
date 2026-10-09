#!/usr/bin/racket
#lang racket

(define factors (lambda (num) 
                  (letrec ([factors-helper 
                            (lambda (s n) 
                              (if (equal? 1 n) 
                                 '()                 
                                 (if (equal? 0 (remainder n s))  
                                    (cons s (factors-helper s(/ n s)))
                                    (factors-helper (+ s 1) n)       
                                    )
                                 )
                              )]) 
                    (factors-helper 2 num)) 
                  ))

; read from console to test and display result
(require "io.scm")
(display (factors (string->number (getline))))
(newline)
(exit)
