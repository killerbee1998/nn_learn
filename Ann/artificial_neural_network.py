# import libs
import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder, LabelEncoder

#import dataset and select x,y
dataset = pd.read_csv("Churn_Modelling.csv")
X = dataset.iloc[:, 3:13].values
y = dataset.iloc[:, 13].values

# labelEncode country and gender
labelEncoder_X1 = LabelEncoder()
X[:, 1] = labelEncoder_X1.fit_transform(X[:, 1])
labelEncoder_X2 = LabelEncoder()
X[:, 2] = labelEncoder_X2.fit_transform(X[:, 2])