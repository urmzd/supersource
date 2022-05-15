#!/usr/bin/env Rscript

ssf = 85.78 
ssp = 23.39 

mse = 13.67 
dfe = 5 * 3 * (2 - 1) 

sse = mse * dfe

sd = 8.1197
dft = 5 * 3 * 2 - 1
sst = (sd^2) * dft 

ssi = sst - (ssf + ssp + sse)
ssi

sse

dfi = (5 - 1) * (3 - 1)
msi = ssi/dfi

msi


f = msi/mse
f

p = pf(f, dfi, dfe, lower.tail=FALSE)
p
