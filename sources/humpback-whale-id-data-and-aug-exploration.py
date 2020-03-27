
import math

from collections import Counter



import matplotlib.pyplot as plt

import pandas as pd

import numpy as np

from PIL import Image



from tqdm import tqdm



#%matplotlib inline
INPUT_DIR = '../input'
def plot_images_for_filenames(filenames, labels, rows=4):

    imgs = [plt.imread(f'{INPUT_DIR}/train/{filename}') for filename in filenames]

    

    return plot_images(imgs, labels, rows)

    

        

def plot_images(imgs, labels, rows=4):

    # Set figure to 13 inches x 8 inches

    figure = plt.figure(figsize=(13, 8))



    cols = len(imgs) // rows + 1



    for i in range(len(imgs)):

        subplot = figure.add_subplot(rows, cols, i + 1)

        subplot.axis('Off')

        if labels:

            subplot.set_title(labels[i], fontsize=16)

        plt.imshow(imgs[i], cmap='gray')
np.random.seed(42)
train_df = pd.read_csv('../input/train.csv')

train_df.head()
rand_rows = train_df.sample(frac=1.)[:20]

imgs = list(rand_rows['Image'])

labels = list(rand_rows['Id'])



plot_images_for_filenames(imgs, labels)
num_categories = len(train_df['Id'].unique())

     

print(f'Number of categories: {num_categories}')
size_buckets = Counter(train_df['Id'].value_counts().values)
plt.figure(figsize=(10, 6))



plt.bar(range(len(size_buckets)), list(size_buckets.values())[::-1], align='center')

plt.xticks(range(len(size_buckets)), list(size_buckets.keys())[::-1])

plt.title("Num of categories by images in the training set")



plt.show()
train_df['Id'].value_counts().head(3)
total = len(train_df['Id'])

print(f'Total images in training set {total}')
w_1287fbc = train_df[train_df['Id'] == 'w_1287fbc']

plot_images_for_filenames(list(w_1287fbc['Image']), None, rows=9)
w_98baff9 = train_df[train_df['Id'] == 'w_98baff9']

plot_images_for_filenames(list(w_98baff9['Image']), None, rows=9)
one_image_ids = train_df['Id'].value_counts().tail(8).keys()

one_image_filenames = []

labels = []

for i in one_image_ids:

    one_image_filenames.extend(list(train_df[train_df['Id'] == i]['Image']))

    labels.append(i)

    

plot_images_for_filenames(one_image_filenames, labels, rows=3)
def is_grey_scale(img_path):

    """Thanks to https://stackoverflow.com/questions/23660929/how-to-check-whether-a-jpeg-image-is-color-or-gray-scale-using-only-python-stdli"""

    im = Image.open(img_path).convert('RGB')

    w,h = im.size

    for i in range(w):

        for j in range(h):

            r,g,b = im.getpixel((i,j))

            if r != g != b: return False

    return True
is_grey = [is_grey_scale(f'{INPUT_DIR}/train/{i}') for i in train_df['Image'].sample(frac=0.1)]

grey_perc = round(sum([i for i in is_grey]) / len([i for i in is_grey]) * 100, 2)

print(f"% of grey images: {grey_perc}")
img_sizes = Counter([Image.open(f'{INPUT_DIR}/train/{i}').size for i in train_df['Image']])



size, freq = zip(*Counter({i: v for i, v in img_sizes.items() if v > 1}).most_common(20))



plt.figure(figsize=(10, 6))



plt.bar(range(len(freq)), list(freq), align='center')

plt.xticks(range(len(size)), list(size), rotation=70)

plt.title("Image size frequencies (where freq > 1)")



plt.show()
from keras.preprocessing.image import (

    random_rotation, random_shift, random_shear, random_zoom,

    random_channel_shift, transform_matrix_offset_center, img_to_array)
img = Image.open(f'{INPUT_DIR}/train/ff38054f.jpg')
img_arr = img_to_array(img)
plt.imshow(img)
imgs = [

    random_rotation(img_arr, 30, row_axis=0, col_axis=1, channel_axis=2, fill_mode='nearest') * 255

    for _ in range(5)]

plot_images(imgs, None, rows=1)
imgs = [

    random_shift(img_arr, wrg=0.1, hrg=0.3, row_axis=0, col_axis=1, channel_axis=2, fill_mode='nearest') * 255

    for _ in range(5)]

plot_images(imgs, None, rows=1)
imgs = [

    random_shear(img_arr, intensity=0.4, row_axis=0, col_axis=1, channel_axis=2, fill_mode='nearest') * 255

    for _ in range(5)]

plot_images(imgs, None, rows=1)
imgs = [

    random_zoom(img_arr, zoom_range=(1.5, 0.7), row_axis=0, col_axis=1, channel_axis=2, fill_mode='nearest') * 255

    for _ in range(5)]

plot_images(imgs, None, rows=1)
import random



def random_greyscale(img, p):

    if random.random() < p:

        return np.dot(img[...,:3], [0.299, 0.587, 0.114])

    

    return img



imgs = [

    random_greyscale(img_arr, 0.5) * 255

    for _ in range(5)]



plot_images(imgs, None, rows=1)
def augmentation_pipeline(img_arr):

    img_arr = random_rotation(img_arr, 18, row_axis=0, col_axis=1, channel_axis=2, fill_mode='nearest')

    img_arr = random_shear(img_arr, intensity=0.4, row_axis=0, col_axis=1, channel_axis=2, fill_mode='nearest')

    img_arr = random_zoom(img_arr, zoom_range=(0.9, 2.0), row_axis=0, col_axis=1, channel_axis=2, fill_mode='nearest')

    img_arr = random_greyscale(img_arr, 0.4)



    return img_arr
imgs = [augmentation_pipeline(img_arr) * 255 for _ in range(5)]

plot_images(imgs, None, rows=1)