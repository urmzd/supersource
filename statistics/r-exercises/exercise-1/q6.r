#!/usr/bin/env Rscript

data <- read.csv("./inputs/athletes.csv")
x_0 <- data$B
x_1 <- data$C

x_0
x_1

diff <- x_1 - x_0

t.test(diff, alternative = "less")
