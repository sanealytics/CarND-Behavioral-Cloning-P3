import csv
import cv2
import numpy as np

lines = []
with open('data/data/driving_log.csv', 'r') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        lines.append(line)

print('Reading file')
images = []
measurements = []
correction = 0.2
for line in lines[1:]:
    # center,left,right,steering,throttle,brake,speed
    for i in range(3):
        source_path = line[i]
        filename = source_path.split('/')[-1]
        current_path = 'data/data/IMG/' + filename
        image = cv2.imread(current_path)
        images.append(image)
        measurement = float(line[3])
        if i == 1: # left
            measurement += correction # Steer right
        if i == 2: # Right
            measurement -= correction # Steer left
        measurements.append(measurement)

print('Training model')
X_train = np.array(images)
y_train = np.array(measurements)

from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda
from keras.layers import Conv2D, MaxPooling2D, Cropping2D
#from keras.backend import tf

model = Sequential()
model.add(Lambda(lambda x: x / 255.0 - 0.5, input_shape = (160, 320, 3)))
model.add(Cropping2D(cropping=((70,25), (0,0))))
#model.add(Convolution2D(6, 5, 5, subsample = (2,2), activation = 'relu'))
model.add(Conv2D(6, (5, 5), strides=(2, 2), activation="relu"))
model.add(MaxPooling2D(pool_size=(2, 2)))
#model.add(Convolution2D(6, 5, 5, subsample = (2,2), activation = 'relu'))
model.add(Conv2D(6, (5, 5), strides=(2, 2), activation="relu"))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dense(128, activation = 'relu'))
model.add(Dense(84, activation = 'relu'))
model.add(Dense(1))

model.compile(loss='mse', optimizer='adam')
model.fit(X_train, y_train, validation_split=0.2, shuffle=True, epochs=2)

print('saving model')
model.save('model.h5')

