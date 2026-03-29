#!/usr/bin/env Rscript

xmean <- 62.05
ymean <- 117.28
ssxx <- 23.21
ssyy <- 93.58
ssxy <- 32.62
n <- 11

b1 <- ssxy / ssxx
b1

b0 <- ymean - b1 * xmean
b0

r <- ssxy / sqrt(ssxx * ssyy)
r

sse <- ssyy - b1 * ssxy
mse <- sse / (n - 2)
mse

dfr <- 1
dfe <- n - 2

ssr <- (ssxy)^2/(ssxx)
msr <- ssr/dfr
msr

f <- msr/mse
f

seb1 <- sqrt(mse / ssxx)
seb0 <- sqrt(mse * (1/n + (xmean)^2 /ssxx))
seb1
seb0

t <- 3.250
t
cib1 <- c(b1 - t * seb1, b1 + t * seb1)
cib1
cib0 <- c(b0 - t * seb0, b0 + t * seb0)
cib0

x <- 120
y <- b0 + b1*x
y

ciyv <- t * sqrt(mse) * sqrt (1 + 1/n + (x - xmean)^2/(ssxx))
ciy <- c(y + ciyv, y - ciyv)
ciy
