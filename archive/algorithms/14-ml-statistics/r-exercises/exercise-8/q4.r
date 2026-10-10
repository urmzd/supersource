#!/usr/bin/Rscript

ssab <- 12
sse <- 36
dfab <- (3 - 1) * (3 - 1)
dfe <- (3 * 3 * 3) - (2 + 2 + dfab) - 1
f <- (ssab / dfab) / (sse / dfe)
f

fstat <- ssab^2 / sse^2
prob <- pf(fstat, dfab, dfe)
prob
round(1 - prob, 7)
pvalue <- 2 * min(prob, 1 - prob)
round(pvalue, 7)

print("E")

y <- 50
a1 <- 5.4
a2 <- 7.2
b1 <- -2.7
ab <- -0.33

50 + a1 + b1 + ab

dfe

t <- (5.4 - 7.2) / sqrt(2 * ((1 / 9) + (1 / 9)))
