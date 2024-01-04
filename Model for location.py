

import os
import cv2
import random
import numpy as np
import pickle
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Activation, Flatten, Dense

def create_training_data():
    for Class in class_names:
        path = os.path.join(tumor_location, Class)
        class_num = class_names.index(Class)
        for img in os.listdir(path):
            try:
                img_array = cv2.imread(os.path.join(path, img), cv2.IMREAD_GRAYSCALE)
                new_array = cv2.resize(img_array, (IMG_Size, IMG_Size))
                training_data.append([new_array, class_num])
            except Exception as e:
                print(f"Error loading/resizing image: {os.path.join(path, img)}")
                continue

create_training_data()

random.shuffle(training_data)

print("Number of samples in training_data:", len(training_data))

features_x = []
target_y = []

for features, label in training_data:
    features_x.append(features)
    target_y.append(label)

features_x = np.array(features_x).reshape(-1, IMG_Size, IMG_Size, 1)
target_y = np.array(target_y)

features_x = features_x / 255

model = Sequential()
model.add(Conv2D(64, (3, 3), input_shape=features_x.shape[1:]))
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(66, (3, 3)))
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())
model.add(Dense(64, activation='relu'))

model.add(Dense(len(class_names)))
model.add(Activation('softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

from tensorflow.keras.utils import to_categorical
target_y = to_categorical(target_y)

model.fit(features_x, target_y, batch_size=100, validation_split=0.1, epochs=50)

model.save('tumor_location.h5')













