
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
import sys
from tqdm import tqdm
from skimage.transform import resize
#print(os.listdir("../input/best-artworks-of-all-time"))
print("Reading all CSV files...")
root_dir = "../input/best-artworks-of-all-time/"
images_dir = root_dir + 'images/images/'

#Get all authors
df = pd.read_csv(root_dir + "artists.csv")
df.replace(' ', '_', regex=True, inplace=True)
all_authors = list(df.name.values)
all_paintings = list(df.paintings)

from random import choice as rc
from random import randint as ri

def read_input(n):
    for _ in range(n):
        path = 'nonexistent'
        while not os.path.exists(path):
            auth = rc(all_authors)
            n = ri(1, all_paintings[all_authors.index(auth)])
            path = images_dir + auth + '/' + auth+'_'+str(n)+'.jpg'
        print('Current Author: %s\tNumber: %i' %(auth, n), file=sys.stderr) #For debugging
        image = plt.imread(path)
        new_image = resize(image, (512, 512), anti_aliasing=True)
        new_image = new_image/256
        yield new_image.flatten()
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Dense, Dropout, Input, LeakyReLU
from tensorflow.keras.optimizers import Adam

def adam_optimizer():
    return Adam(lr=0.0004, beta_1=0.5)

def create_generator():
    generator = Sequential()
    generator.add(Dense(units=256, input_dim=100))
    generator.add(LeakyReLU(0.2))
    generator.add(Dense(units=512))
    generator.add(LeakyReLU(0.2))
    generator.add(Dense(units=512*512*3, activation='sigmoid'))
    generator.compile(loss='binary_crossentropy', optimizer=adam_optimizer())
    return generator

g = create_generator()
g.summary()

def create_discriminator():
    discriminator=Sequential()
    discriminator.add(Dense(units=512,input_dim=512*512*3))
    discriminator.add(LeakyReLU(0.2))
    discriminator.add(Dropout(0.3))
    discriminator.add(Dense(units=256))
    discriminator.add(LeakyReLU(0.2))
    discriminator.add(Dropout(0.3))
    discriminator.add(Dense(units=128))
    discriminator.add(LeakyReLU(0.2))
    discriminator.add(Dense(units=1, activation='sigmoid'))
    
    discriminator.compile(loss='mse', optimizer=adam_optimizer())
    return discriminator

d = create_discriminator()
d.summary()
def create_gan(discriminator, generator):
    discriminator.trainable=False
    gan_input = Input(shape=(100,))
    x = generator(gan_input)
    gan_output= discriminator(x)
    gan= Model(inputs=gan_input, outputs=gan_output)
    gan.compile(loss='binary_crossentropy', optimizer='adam')
    return gan
test = create_gan(d,g)
test.summary()
del test, d, g
def plot_generated_images(epoch, generator, examples=20, dim=(5,4), figsize=(5,4)):
    noise= np.random.normal(loc=0, scale=1, size=[examples, 100])
    generated_images = generator.predict(noise)
    generated_images = generated_images.reshape(examples,512,512,3)
    plt.figure(figsize=figsize)
    for i in range(generated_images.shape[0]):
        plt.subplot(dim[0], dim[1], i+1)
        plt.imshow(generated_images[i], interpolation='nearest')
        plt.axis('off')
    plt.tight_layout()
    plt.savefig('gan_generated_image %d.png' %epoch)

def load_data(n=8):
    x_train = np.array([x for x in read_input(n)])
    y_train = np.array([1 for x in range(n)])
    return x_train, y_train
def training(epochs=1, batch_size=16):
    # Creating GAN
    generator= create_generator()
    discriminator= create_discriminator()
    gan = create_gan(discriminator, generator)
    
    for e in range(1,epochs+1 ):
        print("Epoch %d" %e)
    
        #Loading the data
        #To increase the number of samples, we load new data for every epoch
        X_train, y_train = load_data(batch_size)
        batch_count = X_train.shape[0] / batch_size
    
        for _ in tqdm(range(batch_size)):
        #generate  random noise as an input  to  initialize the  generator
            noise= np.random.normal(0,1, [batch_size, 100])
            
            # Generate fake MNIST images from noised input
            generated_images = generator.predict(noise)
            
            # Get a random set of  real images
            image_batch = X_train
            
            #Construct different batches of  real and fake data 
            #print(image_batch, file=sys.stderr)
            #print(image_batch.shape, generated_images.shape, file=sys.stderr)
            if image_batch.shape != generated_images.shape:
                print('Error in shape!')
                image_batch = generated_images
            X= np.concatenate([image_batch, generated_images])
            
            # Labels for generated and real data
            y_dis=np.zeros(2*batch_size)
            y_dis[:batch_size]=1.0
            
            #Pre train discriminator on  fake and real data  before starting the gan. 
            discriminator.trainable=True
            discriminator.train_on_batch(X, y_dis)
            
            #Tricking the noised input of the Generator as real data
            noise= np.random.normal(0,1, [batch_size, 100])
            y_gen = np.ones(batch_size)
            
            # During the training of gan, 
            # the weights of discriminator should be fixed. 
            #We can enforce that by setting the trainable flag
            discriminator.trainable=False
            
            #training  the GAN by alternating the training of the Discriminator 
            #and training the chained GAN model with Discriminator’s weights freezed.
            gan.train_on_batch(noise, y_gen)
            
        if e%50==0:# e == 1 or e % 5 == 0 or e:
            plot_generated_images(e, generator)
    
    #Now, we'll save our models to preserve them
    #generator.save('generator.h5')
    #discriminator.save('discriminator.h5')
    #gan.save('GAN.h5')

training(500,3)