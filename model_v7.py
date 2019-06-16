#!/usr/bin/env python
# coding: utf-8

# In[1]:


from __future__ import division, print_function, absolute_import

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
from sklearn.utils import shuffle

import cv2
import csv
import pickle

from scipy import ndimage
from sklearn.model_selection import train_test_split
import tensorflow as tf
from keras.models import *
from keras.layers import *
from keras.optimizers import Adam


# ---
# ## Step 0: Generate Data

# In[2]:



# data was generated using downlaoded simulator
# data was collected from 8 laps each in forward and reverse directions
# augument data set by performing vertical flip
# save image and steering angle as pickle files

# data from forward laps
lines = []
with open('./data2/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        lines.append(line)


images = []
measurements = []

imgs_aug = []
meas_aug = []

for line in lines:
    source_path = line[0]
    filename = source_path.split('/')[-1]
    current_path = './data2/IMG/' + filename
    # image = cv2.imread(current_path)
    image = ndimage.imread(current_path)
    images.append(image)
    imgs_aug.append(image)
    measurement = float(line[3])
    measurements.append(measurement)
    meas_aug.append(measurement)


# data from reverse laps
lines = []
with open('./data0/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        lines.append(line)


for line in lines:
    source_path = line[0]
    filename = source_path.split('/')[-1]
    current_path = './data0/IMG/' + filename
    # image = cv2.imread(current_path)
    image = ndimage.imread(current_path)
    images.append(image)
    imgs_aug.append(image)    
    measurement = float(line[3])
    measurements.append(measurement)
    meas_aug.append(measurement)    



# data augmentation for viol1
lines = []
with open('./data4/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        lines.append(line)


for line in lines:
    source_path = line[0]
    filename = source_path.split('/')[-1]
    current_path = './data4/IMG/' + filename
    # image = cv2.imread(current_path)
    image = ndimage.imread(current_path)
    images.append(image)
    imgs_aug.append(image)    
    measurement = float(line[3])
    measurements.append(measurement)
    meas_aug.append(measurement)    
    

    
ic = 0
while(ic< len(images)):
    
    cur_img = images[ic]
    cur_meas = measurements[ic]
    image_flipped = cv2.flip(cur_img, 1)
    measurement_flipped = -cur_meas
    imgs_aug.append(image_flipped)
    meas_aug.append(measurement_flipped)
        
    ic = ic+1
    


# (?, 160, 320, 3)

X_train = np.array(imgs_aug)
y_train = np.array(meas_aug)
    
print('done')


# ----
# 
# ## Step 2: Design and Test a Model Architecture

# In[3]:


# define model and architecture

model = Sequential()
model.add(Lambda(lambda x: x/255-0.5, input_shape=(160,320,3)))
model.add(Cropping2D(cropping=((70,25),(0,0))))
model.add(Conv2D(24, 5, 5, subsample=(2,2), activation='relu'))
model.add(Conv2D(36, 5, 5, subsample=(2,2), activation='relu'))
model.add(Conv2D(48, 5, 5, subsample=(2,2), activation='relu'))
model.add(Conv2D(64, 3, 3, activation='relu'))
model.add(Conv2D(64, 3, 3, activation='relu'))
model.add(Flatten())
model.add(Dense(100))
model.add(Dense(50))
model.add(Dense(10))
model.add(Dense(1))

model.summary()

# compile and run the model
model.compile(loss='mse', optimizer='adam')
history = model.fit(X_train, y_train, validation_split=0.2, shuffle=True, batch_size=128, nb_epoch=5)

print(history.history.keys())
# # "Loss"
# plt.plot(history.history['loss'])
# plt.plot(history.history['val_loss'])
# plt.title('Model Loss')
# plt.ylabel('Loss')
# plt.xlabel('N Epochs')
# plt.legend(['train', 'validation'], loc='upper left')
# plt.show()

model.save('model.h5')
print('done')


