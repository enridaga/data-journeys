
import pandas as pd
import numpy as np

train_data = pd.read_csv("/kaggle/input/sign-language-mnist/sign_mnist_train/sign_mnist_train.csv", sep=",")
train_data.head()
test_data = pd.read_csv("/kaggle/input/sign-language-mnist/sign_mnist_test/sign_mnist_test.csv", sep=",")

train_data.append(test_data)
dataset = train_data
dataset.head()
import matplotlib.pyplot as plt
#%matplotlib inline

mapping_values = {
    0:"A", 1:"B", 2:"C", 3:"D", 4:"E", 5:"F",
    6:"G", 7:"H", 8:"I", 10:"K", 11:"L", 12:"M", 13:"N",
    14:"O", 15:"P", 16:"Q", 17:"R", 18:"S", 19:"T", 20:"U",
    21:"V", 22:"W", 23:"X", 24:"Y"}

plt.figure(figsize=(10, 18))
for i in range(26):
    if i in (9, 25): 
        continue
    plt.subplot(7, 4, i+1)
    plt.imshow(np.array(dataset[dataset.label == i].iloc[0,1:].values.tolist()).reshape(28,28), cmap='Greys')
    plt.xticks([])
    plt.yticks([])
    plt.title(mapping_values[i])
y = dataset.label
dataset.drop("label", axis=1, inplace=True)
X = dataset
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)
X_train = X_train.to_numpy().reshape(-1, 28,28, 1) # 28*28
X_test = X_test.to_numpy().reshape(-1, 28,28, 1) # 28*28
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
batch_size = 128
epochs = 20
num_classes = 26
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
model.compile(loss=keras.losses.sparse_categorical_crossentropy, 
              optimizer=keras.optimizers.Adam(),
              metrics=['accuracy'])
model.summary()
with tf.device('/GPU:0'):
    model_train = model.fit(X_train, y_train, 
                            batch_size=batch_size,
                            epochs=epochs,
                            verbose=1,
                            validation_data=(X_valid, y_valid))
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
print(classification_report(y_test, y_pred))
from sklearn.metrics import confusion_matrix
import seaborn as sn

cf = confusion_matrix(y_test, y_pred)
sn.heatmap(cf, annot=True)
plt.title("Confusion Matrix")
