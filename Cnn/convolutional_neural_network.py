# import libs
from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPool2D, Flatten
from keras.preprocessing import image_dataset_from_directory

# load images
train_datagen = image_dataset_from_directory('dataset/training_set',
batch_size=32, image_size=(64,64))
test_datagen = image_dataset_from_directory('dataset/test_set',
batch_size=32, image_size=(64,64))


# cnn layers
cnn = Sequential()
cnn.add(Conv2D(filters=64, kernel_size=3, activation='relu', input_shape=[64,64,3]))
cnn.add(MaxPool2D(pool_size=(2,2), strides=2))
cnn.add(Flatten())

# cnn hidden layers
cnn.add(Dense(units=128, activation='relu'))
cnn.add(Dense(units=1, activation='sigmoid'))

# compile cnn
cnn.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
cnn.fit()