# -*- coding: utf-8 -*-
"""
Created on Thu Aug 20 16:56:00 2020

@author: gchak
"""

#importing libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#importing dataset
dataset = pd.read_csv('Salary_Data.csv')
x = dataset.iloc[:, 0]
x = pd.DataFrame(x)
y = dataset.iloc[:, 1]
y = pd.DataFrame(y)


#spliting data in to training and testing data
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 1/3, random_state = 0)

#fitting the model to the training data
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(x_train, y_train)

#predictions using the model
y_pred = regressor.predict(x_test)


#visualizing the model on training data
plt.scatter(x_train, y_train, color = 'red')
plt.plot(x_train, regressor.predict(x_train), color = 'blue')
plt.title('years of experiance vs salary (training data)')
plt.xlabel('years of experiance')
plt.ylabel('salary')
plt.show()

#visualizing the model on test data
plt.scatter(x_test, y_test, color = 'red')
plt.plot(x_train, regressor.predict(x_train), color = 'blue')
plt.title('years of experiance vs salary (test data)')
plt.xlabel('years of experiance')
plt.ylabel('salary')
plt.show()













