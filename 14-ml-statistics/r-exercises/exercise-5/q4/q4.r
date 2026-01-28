#!/usr/bin/env Rscript
mydata <- read.csv("heartrate.csv")

## attach mydata in R so I can call the variables directly

attach(mydata)

## scatterplot, Weight is the independent variable and HeartRate is dependent variable

plot(Weight, HeartRate)

## calculate the SSxx, SSyy and SSxy

SSxx <- sum((Weight - mean(Weight))^2)

SSyy <- sum((HeartRate - mean(HeartRate))^2)

SSxy <- sum((Weight - mean(Weight))*(HeartRate - mean(HeartRate)))

## beta1 hat and beta0 hat

beta1hat <- SSxy/SSxx
beta0hat <- mean(HeartRate) - beta1hat*mean(Weight)

print("Hat")
beta1hat
beta0hat

## use lm function to find the estimated regression equation

model <- lm(HeartRate ~ Weight)

summary(model)

## ANOVA table for the regression
anova(model)

## remove the outlier, outlier is the observation in the last row
newdata <- mydata[1:(nrow(mydata)-1),]

## estimated regression equation without outlier

modelnew <- lm(newdata$HeartRate ~ newdata$Weight)
summary(modelnew)
