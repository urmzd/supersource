#!/usr/bin/env Rscript

f_obs <- 9.01
dfa <- 5
dfe <- 3

sst <- 648

sse <- sst / ((f_obs / (dfe / dfa)) + 1)
ssa <- sst - sse

sst
sse
ssa

mse <- sse / dfe
msa <- ssa / dfa

mse
msa

