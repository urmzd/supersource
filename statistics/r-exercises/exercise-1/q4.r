#!/usr/bin/env Rscript

data <- read.csv("./inputs/cholestrol.csv")
test_1 <- data$Test1
test_2 <- data$Test2
diff <- test_1 - test_2

result <- t.test(diff)
result
