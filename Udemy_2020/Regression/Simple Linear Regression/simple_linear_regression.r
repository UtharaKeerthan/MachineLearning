#import dataset
Dataset = read.csv('Salary_Data.csv')

#splitting the testing and training data set
library(caTools)
set.seed(123)
split = sample.split(Dataset$Salary,SplitRatio = 2/3)
trainingset = subset(Dataset, split == TRUE)
testingset = subset(Dataset, split == FALSE)

#fitting the regressor for training set
regressor = lm(formula = Salary ~ YearsExperience,
               data =  trainingset)

#predcition

Y_pred = predict(regressor,newdata = testingset)

#Visualisations
#install.packages("ggplot2")
library(ggplot2)
ggplot() +
  geom_point(aes(x = trainingset$YearsExperience, y = trainingset$Salary),
             colour = 'red') +
  geom_line(aes(x = trainingset$YearsExperience, y = predict(regressor,newdata = trainingset)),
                colour = 'blue') +
  xlab("Years of Experience") +
  ylab("Salary")+
  ggtitle("Plot Training set")

#Visualisaton testing set
library(ggplot2)
ggplot() +
  geom_point(aes(x = testingset$YearsExperience, y = testingset$Salary),
             colour = 'red') +
  geom_line(aes(x = trainingset$YearsExperience,y = predict(regressor,newdata = trainingset)),
            colour = 'blue') +
  xlab("Years of Experience") +
  ylab("Salary")+
  ggtitle("Plot Testing set")