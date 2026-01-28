#!/usr/bin/env Rscript

data <- read.csv("./inputs/q5.csv")

x_1 <- data$Morning.Class
x_2 <- data$Afternoon.Class

x_2[x_2 == ""] <- NA
x_1 <- na.omit(x_1)
x_2 <- na.omit(x_2)


pooled_variance <- function(a, b) {
  n1 <- length(a)
  n2 <- length(b)
  s1_2 <- var(a, na.rm = TRUE)
  s2_2 <- var(b, na.rm = TRUE)

  ((n1 - 1) * s1_2 + (n2 - 1) * s2_2) / pooled_df(a, b)
}

pooled_df <- function(a, b) {
  length(a) + length(b) - 2
}

pooled_t_test <- function(a, b) {
  (mean(a) - mean(b)) / sqrt(pooled_variance(a, b) * ((1 / 5) + (1 / 6)))
}

significant <- function(a, b, alpha) {
  t_a <- abs(qt(alpha / 2, pooled_df(a, b)))

  if (pooled_t_test(a, b) >= t_a || pooled_t_test(a, b) <= -t_a) {
    print("REJECT")
    return(FALSE)
  }

  print("ACCEPT")
  return(TRUE)
}

print("Pooled Variance")
pooled_variance(x_1, x_2)
print("Pooled DF")
pooled_t_test
print("Test Statistic")
pooled_t_test(x_1, x_2)
print("Critical Value")
significant(x_1, x_2, 0.01)
