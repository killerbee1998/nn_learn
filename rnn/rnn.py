# import libs
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

# import datasets
train_set = pd.read_csv("Google_Stock_Price_Train.csv")
train_ara = train_set.iloc[:, 1:2].values

test_set = pd.read_csv("Google_Stock_Price_Test.csv")
test_ara = test_set.iloc[:, 1:2].values

# feature scaling
sc = MinMaxScaler(feature_range=(0,1))
train_ara_scaled = sc.fit_transform(train_ara)

# create data structure with timesteps
# create data structure with timesteps
x_train = []
y_train = []
for i in range(0, 1198):
    x_train.append(train_ara_scaled[i:i+60, 0])
    y_train.append(train_ara_scaled[i+60, 0])

x_train = np.array(x_train)
y_train = np.array(y_train)

# reshape
x_train = np.reshape(x_train, (1198,x_train.shape[0], x_train.shape[1], 1 ) )