#!/usr/bin/env Rscript
mydata <- read.csv("./meatpacking.csv")

## Meat packaged in ambient air only has a shelf life of about two days before the ## quality of the meat begins to degrade due to microbial growth. A group of scientists ## performed an experiment to test alternative methods of meat packing to assess whether ## any method reduced microbial growth. The scientists applied four different packing ## methods to equally sized batches of meat, standard air, 100% CO2, vacuum packaging and ## a mix of different gases. They then counted the amount of baterial growth that occured ## per cm squared and took the natural log of that. They are interested in knowing if ## there is a difference in microbial growth between the four methods. They have decided ## to conduct their experiment at significance level α = 0.001.

## import the dataset
## call the package "tidyr"

library(tidyr)

## convert the data from the csv file to have one column for the factor levels and one ## column for the observations

mydatanew <- gather(mydata, key = "method", value = "bacterial")

## ANOVA

res <- aov(mydatanew$bacterial ~ mydatanew$method)

## show the anova table 

summary(res)

## QQ plot and residual plots

par(mfrow=c(2,2))
plot(res)


################# Calculate by Hand

## calculate the sample mean and sample variance for each groups

means <- tapply(mydatanew$bacterial,mydatanew$method,mean)
vars  <- tapply(mydatanew$bacterial,mydatanew$method,var)

## find the grand mean

grandmean = mean(mydatanew$bacterial)
grandmean

## calculate MSTR

nj=dim(mydata)[1]
k =dim(mydata)[2]
MSTR <- sum(nj *(means-grandmean)^2) / (k-1)

## calculate MSE
nt = nj*k
MSE <- sum((nj-1)*vars)/(nt-k)

## F test statistic
Ftest <- MSTR/MSE

## P-value
pvalue <- pf(Ftest,k-1,nt-k,lower.tail=FALSE)
pvalue
