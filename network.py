# -*- coding: utf-8 -*-

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.utils import to_categorical

from config import CNN_INPUT_SIZE, CNN_CHANNEL, CNN_OUTPUT_NUM


class Network:
    @classmethod
    def build(cls):
        model = Sequential()
        model.add(
            Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(CNN_INPUT_SIZE, CNN_INPUT_SIZE, CNN_CHANNEL))
        )
        model.add(Conv2D(64, (3, 3), activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.25))
        model.add(Flatten())
        model.add(Dense(128, activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(CNN_OUTPUT_NUM, activation='softmax'))

        model.compile(loss='categorical_crossentropy',
                      optimizer='adam',
                      metrics=['accuracy'])
        return model
