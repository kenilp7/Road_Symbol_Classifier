import keras
from keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from keras.layers import Dense
from keras.layers import Flatten, Dropout
from keras.utils.np_utils import to_categorical
from keras.layers.convolutional import Conv2D, MaxPooling2D

def LeNet_model_modified():
   model = Sequential()
   model.add(Conv2D(60, (5, 5), input_shape=(32, 32, 1), activation='relu'))
   model.add(Conv2D(60, (5, 5), activation='elu'))
   model.add(MaxPooling2D(pool_size=(2, 2)))
  
   model.add(Conv2D(30, (3, 3), activation='elu'))
   model.add(Conv2D(30, (3, 3), activation='elu'))
   model.add(MaxPooling2D(pool_size=(2, 2)))
  
   model.add(Flatten())
   model.add(Dense(500, activation='elu'))
   model.add(Dense(43, activation='softmax'))
  
   model.compile(Adam(learning_rate = 0.0001), loss='categorical_crossentropy', metrics=['accuracy'])
   return model