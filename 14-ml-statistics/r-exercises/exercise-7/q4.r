#!/usr/bin/env Rscript
mean_x1 <- 121.01
mean_x2 <- 124.83
mean_x3 <- 122.28
mean_x4 <- 123.63

# mean_x1 <- 105.68
# mean_x2 <- 110.02
# mean_x3 <- 108.00
# mean_x4 <- 107.43

means <- c(mean_x1, mean_x2, mean_x3, mean_x4)

grandmean <- mean(means)
grandmean


dff <- 3
ssf <- sum((dff * (means - grandmean))^2)
ssf

pf(23.5960, 3, 30)

mse <- 1.0437
# mse <- 0.7265
dm <- sqrt(mse * ((1 / 9 + 1 / 9)))

(mean_x1 - mean_x4) / dm
(mean_x2 - mean_x3) / dm
