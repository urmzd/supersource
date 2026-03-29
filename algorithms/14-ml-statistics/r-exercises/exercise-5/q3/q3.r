#!/usr/bin/env Rscript

x_mean <- 1.250
y_mean <- 14.167
ssxx <- 4.375
ssxy <- 45.750
ssyy <- 484.833

b1 <- ssxy / ssxx
b0 <- y_mean - b1 * x_mean

b1
b0

residual <- 5 - ((b1 * 0.5) + b0)
residual

ssr <- b1 * ssxy
msr <- ssr / 1

sse <- (ssxx * ssyy - ssxy^2) / ssxx
mse <- sse / 4

sst <- ssyy

ssr
sse

msr
mse


r2 <- 1 - sse / sst
r2

r <- sqrt(r2)
r

