#!/usr/bin/env Rscript

data <- read.csv("./q1.csv")

x_bar <- function(data) {
  totals <- sum(data$Mean * data$Size)
  total_size <- sum(data$Size)

  totals / total_size
}

ssa <- function(data) {
  sum(data$Size * (data$Mean - x_bar(data))^2)
}

sse <- function(data) {
  sum((data$Size - 1) * (data$SD)^2)
}

sst <- function(data) {
  sum(ssa(data), sse(data))
}

ssa_df <- function(data) {
  length(data$Mean) - 1
}

sse_df <- function(data) {
  sum(data$Size - 1)
}

sst_df <- function(data) {
  ssa_df(data) + sse_df(data)
}

test <- function(data, q = 0.99) {
  (ssa(data) / ssa_df(data)) / (sse(data) / sse_df(data))
}

p <- function(data) {
  pf(test(data), df1 = ssa_df(data), df2 = sse_df(data), lower.tail = FALSE)
}


x_bar(data)
ssa(data)
sse(data)
sst(data)

ssa_df(data)
sse_df(data)
sst_df(data)

test(data)
p(data)
