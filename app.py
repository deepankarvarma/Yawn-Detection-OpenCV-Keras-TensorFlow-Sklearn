import os
import cv2

import numpy as np
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split

# Set path to the data folder
data_dir = 'data'

# Set the size of the input images
img_size = (224, 224)

# Set the batch size for training
batch_size = 32

# Create data generators for training and validation
train_data_gen = ImageDataGenerator(rescale=1./255, shear_range=0.2, zoom_range=0.2, horizontal_flip=True)
val_data_gen = ImageDataGenerator(rescale=1./255)

# Load the images and labels from the data folder
x = []
y = []
for label, folder_name in enumerate(['no yawn', 'yawn']):
    folder_path = os.path.join(data_dir, folder_name)
    for img_name in os.listdir(folder_path):
        img_path = os.path.join(folder_path, img_name)
        img = cv2.imread(img_path)
        img = cv2.resize(img, img_size)
        x.append(img)
        y.append(label)

# Convert the images and labels to numpy arrays
x = np.array(x)
y = np.array(y)

# Split the data into training and validation sets
x_train, x_val, y_train, y_val = train_test_split(x, y, test_size=0.2, stratify=y)

# Create data generators for training and validation
train_generator = train_data_gen.flow(x_train, y_train, batch_size=batch_size)
val_generator = val_data_gen.flow(x_val, y_val, batch_size=batch_size)
# Define the model architecture
model = Sequential()
model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(img_size[0], img_size[1], 3)))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(1, activation='sigmoid'))

# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train the model
model.fit_generator(train_generator, steps_per_epoch=train_generator.n // batch_size, epochs=3, validation_data=val_generator, validation_steps=val_generator.n // batch_size)

# Save the model
model.save('yawn_detection_model.h5')