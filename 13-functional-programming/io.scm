#!/usr/bin/chezscheme

;#| These functions are for reading a line of text from the console and
; | for reading a list of lines from the console, ending with "end"
; |#

(define getline (lambda () 
                  (read-line (current-input-port))
                  )
  )

(define getlist (lambda () 
                  (let ((input (getline)))
                    (if (not (equal? input "end")) 
                       (cons input (getlist)) 
                       ' ()
                       )
                    )
                  )
  )

; (display (getlist))
