#data preprocessing
#import the data set
dataset = read.csv('Data.csv')
# dataset = dataset[,2:3]
#install.packages('caTools') #installing caTools for split
library(caTools)
set.seed(123)
split = sample.split(dataset$Purchased, SplitRatio = 0.8)
training_set = subset(dataset,split == TRUE)
testing_set = subset(dataset,split == FALSE)

#featurescaling
# training_set[,2:3] = scale(training_set[,2:3])
# testing_set[,2:3] = scale(testing_set[,2:3])

