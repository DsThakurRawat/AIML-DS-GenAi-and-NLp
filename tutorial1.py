"""
predict a students final exam score  based on their  number hour they study
"""
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split #split data into train and test
#why we dont import sklearn directly 
# if we do so then all the library will be imported
#step-2
#we will take linear model
from sklearn.linear_model import LinearRegression
dev = {"Hours_sturdy" : [1,2,3,4,5,6,7,8,9,10],"exam_score" : [15,20,25,30,35,40,45,50,55,60]}
#step _ -3
df = pd.DataFrame(dev)
#print(df)
#step-4 assigning the variable
x = df[["Hours_sturdy"]]
y = df[["exam_score"]]
# training the model
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
# 0.2 means -- test data will be 20% of the total data
# random state is used to shuffle the data before splitting it into train and test sets
