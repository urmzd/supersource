#!/usr/bin/env Rscript
## import the data set

options(digits = 10)
keyboard <- read.csv("keyboard.csv")

## Two way ANOVA (without interaction)

res <- aov(keyboard$KeysPressed ~ keyboard$Switch + keyboard$Backplate)
summary(res)


## Two way ANOVA (with interaction)

res1 <- aov(keyboard$KeysPressed ~ keyboard$Switch + keyboard$Backplate + keyboard$Switch:keyboard$Backplate)
summary(res1)

## Find grand mean

mean(keyboard$KeysPressed)

## Find the sample mean for each level in backplate

tapply(keyboard$KeysPressed, keyboard$Backplate, mean)

## Find the sample mean for each level in switch

tapply(keyboard$KeysPressed, keyboard$Switch, mean)

## Find the cell means
tapply(keyboard$KeysPressed, list(keyboard$Switch, keyboard$Backplate), mean)

## One way ANOVA (with only switch factor)

res2 <- aov(keyboard$KeysPressed ~ keyboard$Switch)
summary(res2)
