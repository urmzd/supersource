#!/usr/bin/env Rscript

dfr <- 1
dfe <- 14
dft <- dfr + dfe

pr <- 0.25
f_obs <- qf(pr, dfr, dfe, lower.tail = FALSE)

mse <- 3.67
ssr <- dfr * (f_obs * mse)

ssr

sse <- mse * dfe
sse

sst <- ssr + sse
sst

msr <- ssr * dfr
msr

mst <- ssr + sse

f_obs

r2 <- ssr / mst
r2

ssxx <- 80.08
ssxy <- sqrt(ssr * ssxx)

b1 <- ssxy / ssxx
y_mean <- 14.99
x_mean <- 80.16
b0 <- y_mean - b1 * x_mean

paste("B1", b1)
paste("B0", b0)

n <- dft + 2
seb1 <- sqrt(mse) / sqrt(ssxx)
seb0 <- sqrt(mse * (1 / n + x_mean^2 / ssxx))

seb1
seb0

sig_level_b1 <- 0.01
t_b1 <- qt(1 - (sig_level_b1 / 2), dft)
err_b1 <- t_b1 * seb1
ci_b1 <- list(b1 - err_b1, b1 + err_b1)
ci_b1

sig_level_b0 <- 0.01
t_b0 <- qt(1 - (sig_level_b0 / 2), dft)
err_b0 <- t_b0 * seb0
ci_b0 <- list(b0 - err_b0, b0 + err_b0)
ci_b0

x_pred <- 127
y_pred <- b0 + b1 * x_pred
y_pred
