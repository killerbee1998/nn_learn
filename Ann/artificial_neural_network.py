# import libs
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer 
from keras.models import Sequential
from keras.layers import Dense

#import dataset and select x,y
dataset = pd.read_csv("Churn_Modelling.csv")
X = dataset.iloc[:, 3:13].values
y = dataset.iloc[:, 13].values

# labelEncode country and gender
labelEncoder_X1 = LabelEncoder()
X[:, 1] = labelEncoder_X1.fit_transform(X[:, 1])
labelEncoder_X2 = LabelEncoder()
X[:, 2] = labelEncoder_X2.fit_transform(X[:, 2])

# onehotencode X
ct = ColumnTransformer([("Country", OneHotEncoder(),[1])], remainder="passthrough") 
ct.fit_transform(X)

# train and test split
x_train, x_test, y_train, y_test = train_test_split(X, y, random_state = 0)

# start work on nn
classifier = Sequential()
classifier.add(Dense(6, 'relu', input_dim = 11))
classifier.add(Dense(6, 'relu'))
classifier.add(Dense(1, 'relu'))