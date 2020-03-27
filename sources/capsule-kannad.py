
# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

from __future__ import division, print_function, unicode_literals
import numpy as np # linear algebra
import pandas as pd# data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()


# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.
tf.reset_default_graph()
train_data=pd.read_csv("/kaggle/input/Kannada-MNIST/train.csv")
train_data.shape
train_data.head(1)
array
samples=5
plt.figure(figsize=(samples*6,700))
for index in range(samples):
    plt.subplot(1, samples, index+1)
    image=np.array(train_data.iloc[index, 1:]).reshape(28,28)
    plt.imshow(image, cmap="binary")
    plt.axis("off")
plt.show()
train_data.label[:samples]
X=tf.placeholder(shape=[None, 28, 28, 1], dtype=tf.float32, name="X")
conv_layer1_params={
    "filters" : 256,
    "kernel_size" : 9,
    "strides" : 1,
    "padding" : "valid",
    "activation" : tf.nn.relu,
}
caps1_maps=32
caps1_caps=caps=caps1_maps*6*6
caps1_dims=8
conv_layer2_params={
    "filters" : caps1_maps*caps1_dims,
    "kernel_size" : 9,
    "strides" : 2,
    "padding" : "valid",
    "activation" : tf.nn.relu
}
conv1=tf.layers.conv2d(X, name="conv1", **conv_layer1_params)
conv2=tf.layers.conv2d(conv1, name="conv2", **conv_layer2_params)
