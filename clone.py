import csv
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
import random

def generator(samples, batch_size=32):
    num_samples = len(samples)
    correction = 0.2
    image_x = 320
    image_y = 160
    while True: # Forever
        shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]
            images = []
            measurements = []
            for batch_sample in batch_samples:
                # center,left,right,steering,throttle,brake,speed
                for i in range(3):
                    source_path = batch_sample[i]
                    filename = source_path.split('/')[-1]
                    #if (len(source_path.split('/') > 2)): # TODO: Change hack to better parse
                    #    image_path = 'data/run1/IMG/'
                    #else:
                    #    image_path = 'data/data/IMG/'
                    image_path = 'data/data/IMG/'
                    current_path = image_path + filename
                    image = cv2.imread(current_path)
                    # Crop
                    # images.append(image[60:(image_y - 20)]) # Don't use - Crop in model instead
                    measurement = float(batch_sample[3])
                    if i == 1: # left
                        measurement += correction # Steer right
                    if i == 2: # Right
                        measurement -= correction # Steer left
                    if random.random() < 0.5: # Randomly flip
                        image = cv2.flip(image, 1) # Around y-axis
                        measurement = -measurement
                    images.append(image)
                    measurements.append(measurement)

            X_train = np.array(images)
            y_train = np.array(measurements)
            yield shuffle(X_train, y_train)


from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda
from keras.layers import Conv2D, MaxPooling2D, Cropping2D, BatchNormalization
from keras.callbacks import ModelCheckpoint

import matplotlib.pyplot as plt

print('Reading data')
lines = []
with open('data/data/driving_log.csv', 'r') as csvfile:
    reader = csv.reader(csvfile)
    next(reader) # Skip header
    for line in reader:
        lines.append(line)


train_samples, validation_samples = train_test_split(lines, test_size = 0.2)
train_generator = generator(train_samples, batch_size=32)
validation_generator = generator(validation_samples, batch_size=32)

ch, row, col = 3, 80, 320 # Trimmed image format

model = Sequential()
#model.add(Lambda(lambda x: x / 127.5 - 1., input_shape = (ch, row, col), output_shape = (ch, row, col)))
#model.add(Lambda(lambda x: x / 127.5 - 1., input_shape = (160, 320, 3)))
model.add(Lambda(lambda x: x / 255.0 - 0.5, input_shape = (160, 320, 3)))
model.add(Cropping2D(cropping=((70,25), (0,0))))

model.add(Conv2D(24, (5, 5), strides=(2, 2), activation="relu"))
model.add(BatchNormalization())# Added Batchnorm

model.add(Conv2D(64, (1, 1), activation="relu")) # Added 1x1 convolutions

model.add(Conv2D(36, (5, 5), strides=(2, 2), activation="relu"))
model.add(BatchNormalization())# Added Batchnorm

model.add(Conv2D(48, (5, 5), strides=(2, 2), activation="relu"))
model.add(BatchNormalization())# Added Batchnorm

model.add(Conv2D(64, (1, 1), activation="relu")) # Added 1x1 convolutions

model.add(Conv1D(64, (3, 3), activation="relu"))
model.add(BatchNormalization())# Added Batchnorm

model.add(Conv2D(64, (3, 3), activation="relu"))
model.add(BatchNormalization())# Added Batchnorm

model.add(Conv2D(64, (1, 1), activation="relu")) # Added 1x1 convolutions

model.add(Flatten())

model.add(Dense(100, activation = 'relu'))
model.add(BatchNormalization()) # Added Batchnorm

model.add(Dense(50, activation = 'relu'))
model.add(Dense(10, activation = 'relu'))
model.add(Dense(1))

print('Training model')
model.compile(loss='mse', optimizer='adam')
checkpointer = ModelCheckpoint(filepath="model_checkpoint.hdf5", verbose=1, save_best_only=True)


history_object = model.fit_generator(train_generator, steps_per_epoch=len(train_samples), \
        validation_data = validation_generator, validation_steps = len(validation_samples), epochs=10, callbacks=[checkpointer]) 

print('saving model')
model.save('model.h5')

print('Plotting curves')

# Plot loss curve
plt.plot(history_object.history['loss'])
plt.plot(history_object.history['val_loss'])
plt.title('model mean squared error loss')
plot.ylabel('mean squared error loss')
plot.xlabel('epoch')
plt.legend(['training set', 'validation set'], loc='upper right')
plt.save('loss.png')

