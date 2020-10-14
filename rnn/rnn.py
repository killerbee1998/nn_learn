# import libs
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# import datasets
train_set = pd.read_csv("Google_Stock_Price_Train.csv")
train_ara = train_set.iloc[:, 1:2].values

test_set = pd.read_csv("Google_Stock_Price_Test.csv")
test_ara = test_set.iloc[:, 1:2].values