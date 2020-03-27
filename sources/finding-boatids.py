
import pandas as pd

import numpy as np

import glob

from sklearn import cluster

from scipy.misc import imread

import cv2

import skimage.measure as sm

# import progressbar

import multiprocessing

import random

import matplotlib.pyplot as plt

import seaborn as sns

#%matplotlib inline

new_style = {'grid': False}

plt.rc('axes', **new_style)



# Function to show 4 images

def show_four(imgs, title):

    #select_imgs = [np.random.choice(imgs) for _ in range(4)]

    select_imgs = [imgs[np.random.choice(len(imgs))] for _ in range(4)]

    _, ax = plt.subplots(1, 4, sharex='col', sharey='row', figsize=(20, 3))

    plt.suptitle(title, size=20)

    for i, img in enumerate(select_imgs):

        ax[i].imshow(img)



# Function to show 8 images

def show_eight(imgs, title):

    select_imgs = [imgs[np.random.choice(len(imgs))] for _ in range(8)]

    _, ax = plt.subplots(2, 4, sharex='col', sharey='row', figsize=(20, 6))

    plt.suptitle(title, size=20)

    for i, img in enumerate(select_imgs):

        ax[i // 4, i % 4].imshow(img)
select = 500 # Only load 500 images for speed

# Data loading

train_files = sorted(glob.glob('../input/train/*/*.jpg'), key=lambda x: random.random())[:select]

train = np.array([imread(img) for img in train_files])

print('Length of train {}'.format(len(train)))
print('Sizes in train:')

shapes = np.array([str(img.shape) for img in train])

pd.Series(shapes).value_counts()
for uniq in pd.Series(shapes).unique():

    show_four(train[shapes == uniq], 'Images with shape: {}'.format(uniq))

    plt.show()
# Function for computing distance between images

def compare(args):

    img, img2 = args

    img = (img - img.mean()) / img.std()

    img2 = (img2 - img2.mean()) / img2.std()

    return np.mean(np.abs(img - img2))



# Resize the images to speed it up.

train = [cv2.resize(img, (224, 224), cv2.INTER_LINEAR) for img in train]



# Create the distance matrix in a multithreaded fashion

pool = multiprocessing.Pool(8)

#bar = progressbar.ProgressBar(max=len(train))

distances = np.zeros((len(train), len(train)))

for i, img in enumerate(train): #enumerate(bar(train)):

    all_imgs = [(img, f) for f in train]

    dists = pool.map(compare, all_imgs)

    distances[i, :] = dists
print(distances)

plt.hist(distances.flatten(), bins=50)

plt.title('Histogram of distance matrix')

print('')
cls = cluster.DBSCAN(metric='precomputed', min_samples=5, eps=0.6)

y = cls.fit_predict(distances)

print(y)

print('Cluster sizes:')

print(pd.Series(y).value_counts())
for uniq in pd.Series(y).value_counts().index:

    if uniq != -1:

        size = len(np.array(train)[y == uniq])

        if size > 10:

            show_eight(np.array(train)[y == uniq], 'BoatID: {} - Image count {}'.format(uniq, size))

            plt.show()

        else:

            show_four(np.array(train)[y == uniq], 'BoatID: {} - Image count {}'.format(uniq, size))

            plt.show()            
size = len(np.array(train)[y == -1])

show_eight(np.array(train)[y == -1], 'BoatID: {} (Unclassified images) - Image count {}'.format(-1, size))