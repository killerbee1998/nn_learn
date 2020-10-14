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