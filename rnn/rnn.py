# import libs
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout

# import datasets
train_set = pd.read_csv("Google_Stock_Price_Train.csv")
train_ara = train_set.iloc[:, 1:2].values

test_set = pd.read_csv("Google_Stock_Price_Test.csv")
test_ara = test_set.iloc[:, 1:2].values

# feature scaling
sc = MinMaxScaler(feature_range=(0,1))
train_ara_scaled = sc.fit_transform(train_ara)

# create data structure with timesteps
x_train = []
y_train = []
for i in range(0, 1198):
    x_train.append(train_ara_scaled[i:i+60, 0])
    y_train.append(train_ara_scaled[i+60, 0])

x_train = np.array(x_train)
y_train = np.array(y_train)

# reshape
x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1 ) )

# build rnn
rnn = Sequential()
rnn.add(LSTM(units = 50, return_sequences=True, input_shape=(x_train.shape[1], 1)))
rnn.add(Dropout(0.2))
rnn.add(LSTM(units=50, return_sequences=True))
rnn.add(Dropout(0.2))
rnn.add(LSTM(units=50, return_sequences=True))
rnn.add(Dropout(0.2))
rnn.add(LSTM(units=50, return_sequences=False))
rnn.add(Dropout(0.2))
rnn.add(Dense(units=1))

# compile the rnn
rnn.compile(optimizer='adam', loss='mean_square_error')

# fit the rnn
rnn.fit(x_train, y_train, epochs=100, batch_size=32)