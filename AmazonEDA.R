##Libraries
library(tidyverse)
library(tidymodels)
library(mosaic)
library(embed)

trainCsv <- read_csv("train.csv")

trainCsv
##create a table 

library(ggmosaic)
ggplot(data=trainCsv) + geom_mosaic(aes(x=product(RESOURCE), fill=ACTION))


ggplot(trainCsv, mapping = aes(x=RESOURCE,y = ACTION)) + geom_point()

