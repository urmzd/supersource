#!/usr/bin/env Rscript

# Toques	Panama Hats	Fedoras
# Winter	114	117	119
# Spring	123	135	115
# Fall	136	134	116
# Summer	56	195	104

total <- c(114, 117, 119, 123, 135, 115, 136, 134, 116, 56, 195, 104)

sum(117, 135, 134, 195) * sum(123, 135, 115) / sum(total)
sum(114, 117, 119) * sum(114, 123, 136, 56) / sum(total)

data <- read.delim("./q4.txt", row.names = 1)
data

chisq.test(data)
