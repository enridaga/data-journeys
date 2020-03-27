
import pandas as pd
import tensorflow as tf
import numpy as np
import datetime
from tensorflow.keras.layers import Conv2D, Dropout, Flatten, Dense, MaxPool2D, BatchNormalization
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
train = pd.read_csv("/kaggle/input/digit-recognizer/train.csv")
test = pd.read_csv("/kaggle/input/digit-recognizer/test.csv")

y_train = train["label"]

x_train = train.drop(labels = ["label"],axis = 1) 

# free some space
del train
label_counts = y_train.value_counts()
print(label_counts)

plt.bar(label_counts.index, label_counts.values)
plt.xticks(np.arange(0, 10, 1)) 
plt.xlabel('digit') 
plt.ylabel('count') 
#plt.show()
# grayscale normalizataion
x_train = x_train / 255.0
test = test / 255.0
x_train = x_train.values.reshape(-1,28,28,1)
test = test.values.reshape(-1,28,28,1)
# One hot encode labels
y_train = tf.keras.utils.to_categorical(y_train, num_classes = 10)
# Split the train and the validation set for the fitting
x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size = 0.1, random_state=16)
# Define the model
model = tf.keras.models.Sequential()

# model.add(Conv2D(filters = 64, kernel_size = (5,5), activation ='relu', input_shape = (28,28,1)))
# model.add(MaxPool2D(pool_size=(2,2)))
# model.add(Conv2D(filters = 128, kernel_size = (3,3), activation ='relu'))
# model.add(MaxPool2D(pool_size=(2,2)))
# model.add(Flatten())
# model.add(Dense(128, activation = "relu"))
# model.add(Dense(64, activation = "relu"))
# model.add(Dense(10, activation = "softmax"))

model.add(Conv2D(filters = 64, kernel_size = (3,3), activation ="relu", input_shape = (28,28,1)))
model.add(BatchNormalization())
model.add(Conv2D(filters = 64, kernel_size = (3,3), activation ="relu"))
model.add(MaxPool2D(pool_size=(2,2)))
model.add(Dropout(0.3))
model.add(Conv2D(filters = 128, kernel_size = (3,3), activation ="relu"))
model.add(BatchNormalization())
model.add(Conv2D(filters = 128, kernel_size = (3,3), activation ="relu"))
model.add(MaxPool2D(pool_size=(2,2)))
model.add(Dropout(0.3))
model.add(Conv2D(filters = 256, kernel_size = (3,3), activation ="relu"))
model.add(MaxPool2D(pool_size=(2,2)))
model.add(Dropout(0.3))
model.add(Flatten())
model.add(Dense(128, activation = "relu"))
model.add(Dense(64, activation = "relu"))
model.add(Dense(10, activation = "softmax"))

model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])

model.summary()
# Callbacks

# log_dir="logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
# tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, profile_batch=0)

# early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5)
epochs = 32
batch_size = 1048
history = model.fit(x_train, y_train, batch_size = batch_size, epochs = epochs, validation_data = (x_val, y_val), verbose = 2)
loss, accuracy = model.evaluate(x_val, y_val)
print("Accuracy", accuracy)
# predict results
predictions = model.predict(test)

# select the indix with the maximum probability
predictions = np.argmax(predictions, axis = 1)

predictions = pd.Series(predictions, name="Label")

submission = pd.concat([pd.Series(range(1, 28001), name="ImageId"), predictions], axis=1)

submission.to_csv("/kaggle/working/submission.csv", index=False)