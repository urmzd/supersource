#!/usr/bin/env Rscript
library(readr)

data <- read_csv("./q1.csv")

allobs <- c(data$A, data$B)
ranked <- rank(allobs, na.last = NA)

n1 <- length(data$A)
n <- length(ranked)
n2 <- n - n1
w <- sum(ranked[1:n1])

mu <- n1 * (n + 1) / 2
sd <- sqrt(n1 * n2 * (n + 1) / 12)
z <- (w - mu) / sd

print(w)

pvalue <- pnorm(z, 0, 1, lower.tail = F)
pvalue <- 2 * pnorm(abs(z), 0, 1, lower.tail = F)
wilcox.test(data$A, data$B, alternative = "greater", exact = F, correct = F)
