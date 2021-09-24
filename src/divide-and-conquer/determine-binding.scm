#!/usr/bin/racket
#lang racket


; Code modified from Dr. Alex Brodsky (Lecture 19 + 20)
; Determine type of binding by checking what value of x is evaluated.
; If last read x is evaluated, we have deep binding
; Else we have shallow binding.
(define (increase-x) (set! x (+ x 1)))
(define (execute f) 
  (let ((x 20)) 
    (define before x)
    (f) 
    (define after x)
    (list before after)
    ))

; Determine scope by outputting the value attached to current scope.
; If lexical scope, "lexical" will be printed.
; Else a dynamic variable will be found and outputted, "dynamic".
(define check-binding 
  ; Define a nested scope.
  (let (( scope "lexical" ))
    (lambda () scope)
    ))

(define x 1)
(define out-b x)
(define in (execute increase-x))
(define out-a x)

(if (and (= out-b 1) (= out-a 2) (= (car in) 20) (= (car (cdr in)) 20)) "deep" "shallow")

(let ((scope "dynamic")) (check-binding))
