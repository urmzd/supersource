#!/usr/bin/env Rscript

# Pearson correlation

pearson_correlation <- function() {
  data <- read.delim("pc.txt")
  x <- data$x
  y <- data$y
  x_mean <- mean(x)
  y_mean <- mean(y)

  corr <- cor.test(x=x, y=y, method="pearson")

  corr
}

spearman_correlation <- function() {
  data <- read.delim("sc.txt")
  x <- data$x
  y <- data$y

  corr <- cor.test(x=x, y=y, method="spearman")

  corr
}

pearson_correlation()
spearman_correlation()
