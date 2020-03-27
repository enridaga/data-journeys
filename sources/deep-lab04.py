
from tensorflow.keras.preprocessing.image import ImageDataGenerator

from tensorflow.keras.models import Sequential, Model
from tensorflow.keras import layers
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.optimizers import RMSprop
from sklearn.utils import shuffle
from tensorflow.keras import optimizers

import numpy as np

from datetime import datetime
BATCH = 128
datagen = ImageDataGenerator(rescale=1./255)
    
train_data = datagen.flow_from_directory('/kaggle/input/intel-image-classification/seg_train/seg_train/',
                                        target_size=(150, 150),
                                        batch_size=14034,
                                        class_mode='categorical',
                                        shuffle=True)

test_data = datagen.flow_from_directory('/kaggle/input/intel-image-classification/seg_test/seg_test/',
                                        target_size=(150, 150),
                                        batch_size=3000,
                                        class_mode='categorical',
                                        shuffle=True)


#train_images, train_labels = train_data.next()
#test_images, test_labels = test_data.next()
train_images, train_labels = train_data.next()
test_images, test_labels = test_data.next()
train_images.shape
#train_images = np.reshape( train_images,  (len(train_images),  150 ,  150 ,  3 ))   # adapt this if using `channels_first` image data format
#test_images =  np.reshape( test_images,  (len(test_images),  150 ,  150 ,  3 ))
from keras.layers import Input, Dense, Flatten, Reshape
from keras.models import Model
def create_dense_ae():
    
    encoding_dim = 1000

    input_img = Input(shape=(150, 150, 3))
    flat_img = Flatten()(input_img)

    encoded = Dense(encoding_dim, activation='relu')(flat_img)
    
    input_encoded = Input(shape=(encoding_dim,))
    flat_decoded = Dense(150*150*3, activation='relu')(input_encoded)
    decoded = Reshape((150, 150, 3))(flat_decoded)

    encoder = Model(input_img, encoded, name="encoder")
    decoder = Model(input_encoded, decoded, name="decoder")
    autoencoder = Model(input_img, decoder(encoder(input_img)), name="autoencoder")
    return encoder, decoder, autoencoder
encoder, decoder, autoencoder = create_dense_ae()
autoencoder.summary()

autoencoder.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])

autoencoder.fit( train_images, train_images, epochs= 2, batch_size= 128, shuffle= True, 
                validation_data= (test_images,  test_images))
autoencoder.layers[1].summary()
def dense(autoencoder):
    input_img = Input(shape=(150, 150, 3)) # 28, 28, 1 - размерности строк, столбцов, фильтров одной картинки, без батч-размерности
    
    encoder_dim = Input(shape=(1000,))
    #flat_img = Flatten()(encoder_dim)
    
    dens_1 = Dense(500, activation='relu')(encoder_dim)
    
    input_dens_1 = Input(shape=(500,))
    dens_2 = Dense(6, activation='softmax')(input_dens_1)

    
    auto = Model(input_img, autoencoder.layers[1](input_img), name="encoder")    
    dens_1 = Model(encoder_dim, dens_1, name="dens_1")
    dens_2 = Model(input_dens_1, dens_2, name="dens_2")
    fc = Model(input_img, dens_2(dens_1(auto(input_img))), name="fc")
    
    fc.summary()
    fc.compile(optimizer='rmsprop', 
              loss='categorical_crossentropy',
              metrics=['accuracy'])


    fc.fit(train_images, train_labels, epochs=2, batch_size=128, shuffle=True, validation_data=(test_images,  test_labels))

    return fc

dense(autoencoder)