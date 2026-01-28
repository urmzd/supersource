#!/usr/bin/env Rscript

# grand mean
tcm = sum(64.74, 88.65, 96.53) / 3

# df
dfs <- 3 - 1
dft <- 2 - 1
dfi <- dfs * dft
dfe <- 3 * 2 * (3 - 1)

dfs
dft
dfi
dfe

ssab <- 0
input <- read.csv("q1.csv")
input

rowmean <- input$RowMean
s1 <- input$X25C
s2 <- input$X30C

  #Temerature  X25C   X30C RowMean
#1        10% 53.94  75.54   64.74
#2        25% 75.84 101.46   88.65
#3        40% 82.42 110.63   96.53
#4 ColumnMean 70.73 95.88      NA

tcm
y00 = (53.94 - 64.74 - 70.73 + tcm)^2
y01 = (75.84 - 88.65 - 70.73 + tcm)^2
y02 = (82.84 - 96.53 - 70.73 + tcm)^2

y10 = (75.54 - 64.74 - 95.88 + tcm)^2
y11 = (101.45 - 88.65 - 95.88 + tcm)^2
y12 = (110.64 - 96.53 - 95.88 + tcm)^2

ssi = 3 * (y00 + y01 + y02 + y10 + y11 + y12)
msi = ssi / dfi
ssi 
dfi
msi
