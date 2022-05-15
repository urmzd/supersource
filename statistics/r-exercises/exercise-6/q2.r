#!/usr/bin/env Rscript

data <- read.csv("./housedat.csv")
data

y <- data$SalePrice
x1 <- data$HouseSize
x2 <- data$BackyardSize
model <- lm(y ~ x1 + x2)
model

# summary(model)

x1_pred <- 1280
x2_pred <- 4285

y <- 152231.2000 + x1_pred * 1.1932 + x2_pred * 1.6652

anova <- anova(model)
anova

