#!/usr/bin/env Rscript

mydata = read.csv("troutlengths.csv")

allobs = c(mydata$Lake1, mydata$Lake2)
ranked <- rank(allobs,ties.method="average",na.last=NA)

n1=length(mydata$Lake1)
n=length(ranked)
n2=n-n1
w=sum(ranked[1:n1])

w=sum(ranked[1:n1])
mu=n1*(n+1)/2
sd=sqrt(n1*n2*(n+1)/12)
z=(w-mu)/sd

print(w)


pvalue <- pnorm(z,0,1,lower.tail=T)
print(pvalue)

pvalue <- 2*pnorm(abs(z),0,1,lower.tail=F)

wilcox.test(mydata$Lake1,mydata$Lake2,alternative="less",exact=F,correct=F)
