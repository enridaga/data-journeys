
# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.
# common imports

import numpy as np
import json
import matplotlib.pyplot as plt

file = open('../input/ships-in-satellite-imagery/shipsnet.json')
dataset = json.load(file)
file.close()
dataset.keys()
# write data to numpy arrays
data = np.array(dataset['data']).astype('uint8')
data.shape
# extract label data 

label_data = np.array(dataset['labels']).astype('uint8')
label_data.shape
# reshape data for visualization
channels = 3
width = 80
height = 80

X = data.reshape(-1, 3, width, height).transpose([0,2,3,1])
X.shape
# check sample shape and plot
print(X[800].shape)
sample_pic = X[800]
plt.imshow(X[800])
type(sample_pic)
from skimage import color
sample_pic_gr = color.rgb2gray(sample_pic)
sample_pic_gr.shape
plt.imshow(sample_pic_gr)
plt.set_cmap('Greys')
# converting all images to greyscale. Output is a list

X_grey = [ color.rgb2gray(i) for i in X]
X_grey = np.array(X_grey)
X_grey.shape
plt.imshow(X_grey[800])
label_data[800]

X_grey.shape
X_grey[:2]
import tensorflow as tf
from tensorflow import keras
print(tf.__version__,"|", keras.__version__)
print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))
from tensorflow.python.client import device_lib
print(device_lib.list_local_devices())
from keras import backend as K
K.tensorflow_backend._get_available_gpus()
tf.random.set_seed(42)
np.random.seed(42)
# Split dataset into train, valid and test sets.
from sklearn.model_selection import train_test_split
# split training, validation and test sets

X_train_full, X_test, y_train_full, y_test = train_test_split(X_grey, label_data, random_state=42)

X_train, X_valid, y_train, y_valid = train_test_split(X_train_full, y_train_full, random_state=42)
# Standardize features by subtracting the mean and scaling to unit variance

pixel_means = X_train.mean(axis=0, keepdims=True)
pixel_stds = X_train.std(axis=0, keepdims=True)

X_train_scaled = (X_train - pixel_means) / pixel_stds
X_valid_scaled = (X_valid - pixel_means) / pixel_stds
X_test_scaled = (X_test - pixel_means) / pixel_stds
keras.backend.clear_session()
# Model

model = keras.models.Sequential()
model.add(keras.layers.Flatten(input_shape=[80,80]))
model.add(keras.layers.Dense(300, activation='selu',
                             kernel_initializer='lecun_normal'))

for layer in range(9):
#     model.add(keras.layers.Dropout(0.3)) 
    model.add(keras.layers.Dense(100, activation='selu', kernel_initializer='lecun_normal'))

# output layer
model.add(keras.layers.Dense(1, activation='sigmoid'))
model.summary()
# compile

model.compile(loss='binary_crossentropy',
              optimizer=keras.optimizers.Nadam(decay=1e-4),
              metrics=["accuracy"])
# train
early_stop = keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

history = model.fit(X_train_scaled, y_train, epochs=40,
                    validation_data=(X_valid_scaled, y_valid),
                    callbacks=[early_stop])
model.evaluate(X_test_scaled, y_test, verbose=0)
