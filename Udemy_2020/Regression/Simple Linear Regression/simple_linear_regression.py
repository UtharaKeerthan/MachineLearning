# -*- coding: utf-8 -*-
"""
Created on Wed Mar 18 00:11:29 2020

@author: keert
"""

#importing the libraries

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#importing the data sets
dataset = pd.read_csv('Salary_Data.csv')
#separating the dependent and the independent variables
#independent variables in the data set.
X = dataset.iloc[:,:-1].values
#dependentvariable vector
Y = dataset.iloc[:,1].values


#splitting test and training set
from sklearn.cross_validation import train_test_split
X_train,X_test,Y_train,Y_test = train_test_split(X,Y, test_size = 1/3, random_state = 0)

#simple linear regression model creation
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train,Y_train)

#prediction of values
y_pred = regressor.predict(X_test)

#visualisation training set
plt.scatter(X_train,Y_train, color = "red")
plt.plot(X_train, regressor.predict(X_train), color = "blue")
plt.xlabel("Years of Experience")
plt.ylabel("Salary")
plt.title("Training set")

#visualisation testing set
plt.scatter(X_test,Y_test, color = "red")
plt.plot(X_train,  regressor.predict(X_train), color = "blue")
plt.xlabel("Years of Experience")
plt.ylabel("Salary")
plt.title("Testing set")