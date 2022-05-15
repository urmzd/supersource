#!/usr/bin/env Rscript
data <- read.csv("./q5.csv")

a <- data$Toads[!is.na(data$Toads)]
b <- data$Frogs[!is.na(data$Frogs)]
c <- data$Salamanders[!is.na(data$Salamanders)]

ranks <- rank(c(a, b, c), na.last = FALSE, ties.method = "average")
a_ranks <- ranks[0:length(a)]
b_ranks <- ranks[(length(a) + 1):(length(b) + length(a))]
c_ranks <- ranks[(length(b) + length(a) + 1):length(ranks)]

r_sum <- c(mean(a_ranks), mean(b_ranks), mean(c_ranks))
r_len <- c(length(a_ranks), length(b_ranks), length(c_ranks))

K <- 12 / (sum(r_len) * (sum(r_len) - 1))
c17 <- r_sum / r_len
print(c17)
print(K)
K <- (K * sum(r_sum^2 / r_len)) - 3 * (sum(r_len) + 1)

print(K)

kruskal.test(data)
