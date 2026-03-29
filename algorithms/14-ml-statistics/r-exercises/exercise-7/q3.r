#!/usr/bin/env Rscript

n1 = 28
n2 = 30

s1 = 14.697
s2 = 15.425

x1 = 30.493
x2 = 30.813

top = (n1-1)*s1^2 + (n2-1)*s2^2
bottom = n1 + n2 - 2

sx = sqrt(top/bottom)

s = sx * sqrt(1/n1 + 1/n2)
s

t_test = (x1 - x2) / s
t_test

pt(t_test, n1 + n2 - 2, lower.tail=TRUE)
