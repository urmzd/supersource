#!/usr/bin/env Rscript

data <- read.delim("data.txt")
x <- data$x
y <- data$y

ss_xx <- sum((x - mean(x))^2)
ss_yy <- sum((y - mean(y))^2)
ss_xy <- sum((x - mean(x)) * (y - mean(y)))

ss_xx
ss_yy
ss_xy

#b1
slope <- ss_xy/ss_xx
#b0
intercept <- mean(y) - slope*mean(x) 

slope
intercept


