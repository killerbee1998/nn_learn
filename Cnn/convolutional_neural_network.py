# import libs
from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPool2D, Flatten

# cnn layers
cnn = Sequential()
cnn.add(Conv2D(filters=64, kernel_size=3, activation='relu', input_shape=[64,64,3]))