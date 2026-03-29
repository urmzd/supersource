#!/usr/bin/env Rscript

data <- read.csv("./salinity.csv")

sites <- data$Site
salinity <- data$Salinity

sites_type <- factor(sites)
sites_type

res1 <- aov(salinity ~ sites_type)
summary(res1)


plot(res1)
