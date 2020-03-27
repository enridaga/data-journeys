
# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
# for dirname, _, filenames in os.walk('/kaggle/input'):
#     for filename in filenames:
#         print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.
TRAIN_DATA_PATH = '/kaggle/input/dogs-vs-cats/train/train/'
TEST_DATA_PATH = '/kaggle/input/dogs-vs-cats/test/test/'
import keras.backend as K
from keras.models import Sequential
from keras.layers import Activation
from keras.layers.normalization import BatchNormalization
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.utils import to_categorical
from keras.preprocessing import image
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical
imgs_per_cat = 3000
img_size = 200
labels = []
train_image = []
cat = 0
dog = 0
for item in os.listdir(TRAIN_DATA_PATH):
    if item.split('.')[0] == 'cat':
        if cat >= imgs_per_cat:
            continue
        cat += 1
    elif item.split('.')[0] == 'dog':
        if dog >= imgs_per_cat:
            continue
        dog += 1
    img = image.load_img(TRAIN_DATA_PATH + str(item), target_size=(img_size, img_size))
    img = image.img_to_array(img)
    img = img/255
    train_image.append(img)
    labels.append(item.split('.')[0])
train_image[0].shape
X = np.array(train_image)
plt.imshow(X[1])
y = []
for item in labels:
    if item == 'cat':
        y.append(0)
    else:
        y.append(1)
new_labels = pd.get_dummies(labels)
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, new_labels, random_state=42, test_size=0.2)
def build(width, height, depth, classes):
      #initialize the model along with the input shape to be
      #'channels last' and the channels dimension itself
      model = Sequential()
      inputShape = (height, width, depth)
      chanDim = -1

      #if we are using 'channels first', update the input shape
      #and channels dimension
      if K.image_data_format() == "channels_first":
          inputShape = (depth, height, width)
          chanDim = 1

      # CONV => RELU => POOL
      model.add(Conv2D(32, (3,3), padding='same', input_shape=inputShape))
      model.add(Activation('relu'))
      model.add(BatchNormalization(axis=chanDim))
      model.add(MaxPooling2D(pool_size=(3,3)))
      model.add(Dropout(0.25))

      # (CONV => RELU) * 2 => POOL
      model.add(Conv2D(64, (3,3), padding='same'))
      model.add(Activation('relu'))
      model.add(BatchNormalization(axis=chanDim))
      model.add(Conv2D(64, (3,3), padding='same'))
      model.add(Activation('relu'))
      model.add(BatchNormalization(axis=chanDim))
      model.add(MaxPooling2D(pool_size=(2,2)))
      model.add(Dropout(0.25))

      # (CONV => RELU) * 2 => POOL
      model.add(Conv2D(128, (3,3), padding='same'))
      model.add(Activation('relu'))
      model.add(BatchNormalization(axis=chanDim))
      model.add(Conv2D(128, (3,3), padding='same'))
      model.add(Activation('relu'))
      model.add(BatchNormalization(axis=chanDim))
      model.add(MaxPooling2D(pool_size=(2,2)))
      model.add(Dropout(0.25))

      #first (and only) set of FC => RELU layers
      model.add(Flatten())
      model.add(Dense(1024))
      model.add(Activation('relu'))
      model.add(BatchNormalization())
      model.add(Dropout(0.5))

      #softmax Classifier
      model.add(Dense(classes))
      model.add(Activation('softmax'))

      #return the constructed network architecture
      return model
model = build(200, 200, 3, 2)
model.compile(loss='binary_crossentropy',optimizer='rmsprop',metrics=['accuracy'])
model.summary()
model.fit(X_train, y_train, batch_size=32, epochs=200, validation_split=0.3)
model.evaluate(X_test, y_test)
print(model.metrics_names)
predictions = model.predict_classes(X_test)
predictions
from sklearn.metrics import classification_report, confusion_matrix
y_testing = []
for item in y_test.index:
    if y_test['cat'][item] == 1:
        y_testing.append(0)
    else:
        y_testing.append(1)
print(classification_report(y_testing, predictions))
print(confusion_matrix(y_testing, predictions))
