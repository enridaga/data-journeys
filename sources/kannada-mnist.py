
import pandas as pd
import numpy as np # linear algebra


dataset = pd.read_csv("/kaggle/input/Kannada-MNIST/train.csv", sep=",")
dataset.head()
y = dataset.label
dataset.drop("label", inplace=True, axis=1)
X = dataset
print(X.shape, y.shape)
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)
X_train = X_train.to_numpy().reshape(-1, 28,28, 1) # 20*20, 3 channels (R-G-B)
X_test = X_test.to_numpy().reshape(-1, 28,28, 1) # 20*20, 3 channels (R-G-B)
print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train = X_train / 255.
X_test = X_test / 255.
X_train,X_valid,y_train,y_valid = train_test_split(X_train, y_train, test_size=0.2, random_state=13)
print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)
import keras
import tensorflow as tf
from keras.models import Sequential,Input,Model
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.layers.normalization import BatchNormalization
from keras.layers.advanced_activations import LeakyReLU

from tensorflow.python.client import device_lib 
device_lib.list_local_devices() # let's list all available computing devices
batch_size = 64
epochs = 20
num_classes = 10
model = Sequential()

model.add(Conv2D(32, kernel_size=(3, 3),activation='linear',input_shape=(28,28,1),padding='same'))
model.add(LeakyReLU(alpha=0.1))
model.add(MaxPooling2D((2, 2),padding='same'))
model.add(Conv2D(64, (3, 3), activation='linear',padding='same'))
model.add(LeakyReLU(alpha=0.1))
model.add(MaxPooling2D(pool_size=(2, 2),padding='same'))
model.add(Conv2D(128, (3, 3), activation='linear',padding='same'))
model.add(LeakyReLU(alpha=0.1))                  
model.add(MaxPooling2D(pool_size=(2, 2),padding='same'))
model.add(Flatten())
model.add(Dense(128, activation='linear'))
model.add(LeakyReLU(alpha=0.1))                  
model.add(Dense(num_classes, activation='softmax'))
model.compile(loss=keras.losses.sparse_categorical_crossentropy, optimizer=keras.optimizers.Adam(),metrics=['accuracy'])
model.summary()
with tf.device('/GPU:0'):
    model_train = model.fit(X_train, y_train, batch_size=batch_size,epochs=epochs,verbose=1,validation_data=(X_valid, y_valid))
plt.figure(figsize=(7, 7), dpi=80)
plt.subplot(2,1,1)
plt.title("Training History - Accuracy")
plt.plot(range(epochs), model_train.history["accuracy"], label="accuracy", color="red")
plt.scatter(range(epochs), model_train.history["val_accuracy"], label="val_accuracy")
plt.xticks(range(0,epochs,1))
min_y = min(np.min(model_train.history["val_accuracy"]), np.min(model_train.history["accuracy"]))
plt.yticks(np.linspace(min_y-0.1,1,11))
plt.legend()


plt.subplot(2,1,2)
plt.title("Training History - Loss")
plt.plot(range(epochs), model_train.history["val_loss"], label="val_loss", color="red")
plt.scatter(range(epochs), model_train.history["loss"], label="loss")
plt.xticks(range(0,epochs,1))
max_y = max(np.max(model_train.history["val_loss"]), np.max(model_train.history["loss"]))
plt.yticks(np.linspace(0,max_y+0.1,11))
plt.legend()
test_eval = model.evaluate(X_test, y_test, verbose=0)
print('Test loss:', test_eval[0])
print('Test accuracy:', test_eval[1]*100, "%")
from sklearn.metrics import classification_report

y_pred = np.argmax(np.round(model.predict(X_test)), axis=1)

target_names = ["Class {}".format(i) for i in range(num_classes)]
print(classification_report(y_test, y_pred, target_names=target_names))
from sklearn.metrics import confusion_matrix
import seaborn as sn

cf = confusion_matrix(y_test, y_pred)
sn.heatmap(cf, annot=True)
plt.title("Confusion Matrix")
test_data = pd.read_csv("/kaggle/input/Kannada-MNIST/test.csv", sep=",")
test_data.drop("id", inplace=True, axis=1)
test_data.head()
test_data = test_data.to_numpy().reshape(-1, 28,28, 1) 
print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)
test_data = test_data.astype('float32')
test_data = test_data / 255.
y_pred = np.argmax(np.round(model.predict(X_test)), axis=1)
with open("submission.csv", "w") as out:
    out.write("ImageId,Label\n")
    for i in range(len(y_pred)):
        out.write(str(i)+","+str(y_pred[i])+"\n")
result = pd.read_csv("/kaggle/working/submission.csv", sep=",")
result.head()
