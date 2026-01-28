#!/usr/bin/chezscheme

; Binary Search Tree
; Idea: represent a BST node as (value left right)
; if left == () there is no left child
; if right == () there is no right child

(load "io.scm")

; main binary search tree loader
; params: list of strings to be inserted into the BST
; return: a BST
(define load-bst (lambda (values)
    (let ((dummy '("@" () ())))           ; use a dummy node to avoid boundary
      (load-bst-helper dummy values)      ; cases
    )
  )
)

; recursive helper used to iterate over the list and insert items into
; a BST
; params: root: root of BST
;         values: remaining list of strings to be inserted
; return: a BST
(define load-bst-helper (lambda (root values)
    (if (null? values)                      ; empty list is the base case
      root
      (let ((elem (car values)) (remaining (cdr values)))   ; use a couple
        (let ((new-root (add-to-bst root elem)))            ; let to make call
          (load-bst-helper new-root remaining)              ; simpler
        )
      )
    )
  )
)

; insert a new string into the BST, using standard algorithm
; params: root: root of BST
;         values: remaining list of strings to be inserted
; return: a BST
(define add-to-bst (lambda (root value)
    ; we use root-val, left, and right, a couple times
    ; so compute once, use many
    (let ((root-val (car root)) (left (car (cdr root))) 
                                (right (car (cdr (cdr root)))))
       (if (string<? value root-val)                        ; do we go left?
          (if (null? left)                                  ;   at bottom?
             (list root-val (list value '() '()) right)     ;     yes, add node
             (list root-val (add-to-bst left value) right)  ;     no, recurse
          )                                                 ; else going right 
          (if (null? right)                                 ;   at bottom?
             (list root-val left (list value '() '()))      ;     yes, add node
             (list root-val left (add-to-bst right value))  ;     no, recurse
          )
       )
       ; Note we are rebuilding each node as we move back up the tree
       ; This is because we cannot modify a node, only create a new one to
       ; replace it
    )
  )
)

; Determine if a key is in the BST.  Use standard search algorithm
; params: root: root of BST
;         key: key to find
; return: "Key found!" if key is in BST and "Key not found" if not.
(define find-in-bst (lambda (root key)
    (if (null? root)                  ; standard base case (empty root)
      "Key not found"
      (let ((root-val (car root)) (left (car (cdr root))) 
                                  (right (car (cdr (cdr root)))))
        ; computed root-val, left, and right to make code nicer
        (if (equal? key root-val)      ; have we found the key?
          "Key found!"                 ;   yes
          (if (string<? key root-val)  ;   no, do we go left right?
             (find-in-bst left key)    ;     yes, recurse left
             (find-in-bst right key)   ;     no, recurse right
          )
        )
      )
    )
  )
)

; Run test from console
; Input consists of a list and a line
;   - List of words to be inserted into a BST, terminated by "end"
;   - Line consists of key to search for in the BST
; params: none
; return: none
(define load-from-console (lambda ()
    (define bst-root (load-bst (getlist)))
    (display (find-in-bst bst-root (getline)))
    (newline)
  )
)


;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;;;;;;; Assignment 8 Solution below 
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;


; Determine height of BST
; params: root: root of BST
; return: height of BST
(define get-bst-height (lambda (root)
    ; assume that we will never start with an empty tree
    (if (null? root)                  ; standard base case (empty root)
      -1                              ;   empty trees have -1 height
      (let ((left (car (cdr root))) (right (car (cdr (cdr root)))))
        (+ 1 (max (get-bst-height left) (get-bst-height right)))
      )
    )
  )
)

; Determine size of BST
; params: root: root of BST
; return: size of BST
(define get-bst-size (lambda (root)
    (if (null? root)                  ; standard base case (empty root)
      0                               ;   empty trees have 0 nodes
      (let ((left (car (cdr root))) (right (car (cdr (cdr root)))))
        (+ 1 (get-bst-size left) (get-bst-size right))  ; not empty, so count
      )
    )
  )
)

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;;;;;;; Example tests for Assignment 8 Solution below 
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

; Run tests to get size of BST and display results
; params: none
; return: none
(define test-size (lambda ()
    (let ((T (load-bst '())))
      (display (get-bst-size T))        ; should output 1
      (newline)
    )
    (let ((T (load-bst '("Me"))))
      (display (get-bst-size T))        ; should output 2
      (newline)
    )
    (let ((T (load-bst '("Me" "How" "You" "Tau" "Jade" "Awesome")))) 
      (display (get-bst-size T))        ; should output 7
      (newline)
    )
  )
)

; Run tests to get height of BST and display results
; params: none
; return: none
(define test-height (lambda ()
    (let ((T (load-bst '())))
      (display (get-bst-height T))        ; should output 0
      (newline)
    )
    (let ((T (load-bst '("Me"))))
      (display (get-bst-height T))        ; should output 1
      (newline)
    )
    (let ((T (load-bst '("Me" "How" "You" "Tau" "Jade" "Awesome")))) 
      (display (get-bst-height T))        ; should output 3
      (newline)
    )
  )
)

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;;;;;;; Assignment 9 Solution below for Q2
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
; Perform an preorder visitor pattern on bst
; params: root: root of BST
;         visitor: function to on each item
;         partial-result: partial result of traversal so far
; return: partial-result after traversal completes
(define bst-preorder-visitor (lambda (root visitor partial-result)
    (if (null? root)                  ; standard base case (empty root)
      partial-result
      (let* ((root-val (car root)) 
             (left (car (cdr root))) 
             (right (car (cdr (cdr root))))
             (node-partial (visitor root-val partial-result))
             (left-partial (bst-preorder-visitor left visitor node-partial))
            )
        (bst-preorder-visitor right visitor left-partial)
      )
    )
  )
)

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;;;;;;; Example tests for Assignment 9 Solution below 
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

; Run tests to perform visitor pattern on BST
; params: none
; return: none
(define test-visitor (lambda ()
    (let ((T (load-bst '())))
      (display (bst-preorder-visitor T cons '()))        ; should output (@)
      (newline)
    )
    (let ((T (load-bst '("Me"))))
      (display (bst-preorder-visitor T cons '()))        ; should output (Me @)
      (newline)
    )
    (let ((T (load-bst '("Me" "How" "You" "Tau" "Jade" "Awesome")))) 
      (display (bst-preorder-visitor T cons '()))        
                                 ; should output (Tau You Jade Awesome How Me @)
      (newline)
    )
  )
)

(test-visitor)


#| ---------------- Solution for Assignment 10: Urmzd B00800045  --- |#

#|
  Applies some reducer function `visitor` to a BST while using a 
  post-order traversal.
  --------------------------------
  @param T: The bst to evaluate.
  @param visitor: The reducer function to apply to T's elements.
  @param partial-result: The default value.
  @return: The accumulation of applying visitor to each node in T, default
  value otherwise.
|#
(define (bst-postorder-visitor T visitor partial-result) 
  (if (null? T) partial-result 
     (let*
         ((N (car T))
          (L (car (cdr T)))
          (R (car (cdr (cdr T))))
          (LP (bst-postorder-visitor L visitor partial-result))
          (RP (bst-postorder-visitor R visitor LP))) 
       (visitor N RP)
       )
     )
  )

#|
  Creates a stack by accumulating a sequence of values.
  --------------------------------
  @param current: Some value.
  @param accumulated: The stack not including current.
  @return: The stack including current.
|#
(define (stack-visitor current accumulated) (append accumulated (list current)))


#|
  Provides an post-order iterator for a tree `T`.
  --------------------------------
  @param T: The tree to traverse.
  @return: The next value of the binary tree, otherwise false.
|#
(define (new-bst-iterator T) 
  (let ((stack (bst-postorder-visitor T stack-visitor (list)))) 
    (lambda () 
      (cond ((null? stack) #f) 
           (else 
            (let ((node (car stack))) 
              (set! stack (cdr stack)) node) ) )
      ) ))



; Test
(define T (load-bst (list "4" "2" "6")))
(define iter (new-bst-iterator T))
(display (iter))
(newline)
(display (iter))
(newline)
(display (iter))
(newline)
(display (iter))
(newline)
(display (iter))

(exit)
