#!/usr/bin/env Rscript

n1 = 17
n2 = 5.3
s1 = 15
s2 = 4.6

pooled_variance <- ((n1-1)*s1^2 + (n2-1)*s2^2)/(n1+n2-2)
pooled_variance

