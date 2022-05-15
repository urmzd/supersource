#!/usr/bin/env Rscript
library(coin)

data <- read.csv("./q5.csv")
video <- data$Video
book <- data$Book

independence_test(book~video, alternative = "greater")
