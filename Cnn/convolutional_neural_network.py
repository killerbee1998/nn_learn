# import libs
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPool2D, Flatten
from keras.preprocessing import image_dataset_from_directory
from keras.preprocessing.image import img_to_array, load_img

# load images
train_set = image_dataset_from_directory('dataset/training_set',
batch_size=32, image_size=(64,64))
test_set = image_dataset_from_directory('dataset/test_set',
batch_size=32, image_size=(64,64))


# cnn layers
cnn = Sequential()
cnn.add(Conv2D(filters=64, kernel_size=3, activation='relu', input_shape=[64,64,3]))
cnn.add(MaxPool2D(pool_size=(2,2), strides=2))
cnn.add(Flatten())

# cnn hidden layers
cnn.add(Dense(units=128, activation='relu'))
cnn.add(Dense(units=1, activation='sigmoid'))

# compile cnnload_img('dataset/single_prediction/cat_or_dog_1.jpg', target_size=(64,64))
cnn.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
cnn.fit(x=train_set, validation_data=test_set, epochs=25)

# test cnn
test_img1 = load_img('dataset/single_prediction/cat_or_dog_1.jpg', target_size=(64,64))
test_img1 = img_to_array(test_img1)
test_img1 = np.expand_dims(test_img1, axis = 0)
res1 = cnn.predict(test_img1)
print(res1)

test_img2 = load_img('dataset/single_prediction/cat_or_dog_2.jpg', target_size=(64,64))
test_img2 = img_to_array(test_img2)
test_img2 = np.expand_dims(test_img2, axis = 0)
res2 = cnn.predict(test_img2)
print(res2)
