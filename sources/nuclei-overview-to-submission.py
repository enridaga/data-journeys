
import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from glob import glob

import os

from skimage.io import imread

import matplotlib.pyplot as plt

import seaborn as sns

### matplotlib inline

dsb_data_dir = os.path.join('..', 'input')

stage_label = 'stage1'
train_labels = pd.read_csv(os.path.join(dsb_data_dir,'{}_train_labels.csv'.format(stage_label)))

train_labels['EncodedPixels'] = train_labels['EncodedPixels'].map(lambda ep: [int(x) for x in ep.split(' ')])

train_labels.sample(3)
all_images = glob(os.path.join(dsb_data_dir, 'stage1_*', '*', '*', '*'))

img_df = pd.DataFrame({'path': all_images})

img_id = lambda in_path: in_path.split('/')[-3]

img_type = lambda in_path: in_path.split('/')[-2]

img_group = lambda in_path: in_path.split('/')[-4].split('_')[1]

img_stage = lambda in_path: in_path.split('/')[-4].split('_')[0]

img_df['ImageId'] = img_df['path'].map(img_id)

img_df['ImageType'] = img_df['path'].map(img_type)

img_df['TrainingSplit'] = img_df['path'].map(img_group)

img_df['Stage'] = img_df['path'].map(img_stage)

img_df.sample(2)
### %time

train_df = img_df.query('TrainingSplit=="train"')

train_rows = []

group_cols = ['Stage', 'ImageId']

for n_group, n_rows in train_df.groupby(group_cols):

    c_row = {col_name: col_value for col_name, col_value in zip(group_cols, n_group)}

    c_row['masks'] = n_rows.query('ImageType == "masks"')['path'].values.tolist()

    c_row['images'] = n_rows.query('ImageType == "images"')['path'].values.tolist()

    train_rows += [c_row]

train_img_df = pd.DataFrame(train_rows)    

IMG_CHANNELS = 3

def read_and_stack(in_img_list):

    return np.sum(np.stack([imread(c_img) for c_img in in_img_list], 0), 0)/255.0

train_img_df['images'] = train_img_df['images'].map(read_and_stack).map(lambda x: x[:,:,:IMG_CHANNELS])

train_img_df['masks'] = train_img_df['masks'].map(read_and_stack).map(lambda x: x.astype(int))

train_img_df.sample(1)
n_img = 6

fig, m_axs = plt.subplots(2, n_img, figsize = (12, 4))

for (_, c_row), (c_im, c_lab) in zip(train_img_df.sample(n_img).iterrows(), 

                                     m_axs.T):

    c_im.imshow(c_row['images'])

    c_im.axis('off')

    c_im.set_title('Microscope')

    

    c_lab.imshow(c_row['masks'])

    c_lab.axis('off')

    c_lab.set_title('Labeled')
train_img_df['Red'] = train_img_df['images'].map(lambda x: np.mean(x[:,:,0]))

train_img_df['Green'] = train_img_df['images'].map(lambda x: np.mean(x[:,:,1]))

train_img_df['Blue'] = train_img_df['images'].map(lambda x: np.mean(x[:,:,2]))

train_img_df['Gray'] = train_img_df['images'].map(lambda x: np.mean(x))

train_img_df['Red-Blue'] = train_img_df['images'].map(lambda x: np.mean(x[:,:,0]-x[:,:,2]))

sns.pairplot(train_img_df[['Gray', 'Red', 'Green', 'Blue', 'Red-Blue']])
train_img_df['images'].map(lambda x: x.shape).value_counts()
from keras.models import Sequential

from keras.layers import BatchNormalization, Conv2D, UpSampling2D, Lambda

simple_cnn = Sequential()

simple_cnn.add(BatchNormalization(input_shape = (None, None, IMG_CHANNELS), 

                                  name = 'NormalizeInput'))

simple_cnn.add(Conv2D(8, kernel_size = (3,3), padding = 'same'))

simple_cnn.add(Conv2D(8, kernel_size = (3,3), padding = 'same'))

# use dilations to get a slightly larger field of view

simple_cnn.add(Conv2D(16, kernel_size = (3,3), dilation_rate = 2, padding = 'same'))

simple_cnn.add(Conv2D(16, kernel_size = (3,3), dilation_rate = 2, padding = 'same'))

simple_cnn.add(Conv2D(32, kernel_size = (3,3), dilation_rate = 3, padding = 'same'))



# the final processing

simple_cnn.add(Conv2D(16, kernel_size = (1,1), padding = 'same'))

simple_cnn.add(Conv2D(1, kernel_size = (1,1), padding = 'same', activation = 'sigmoid'))

simple_cnn.summary()
from keras import backend as K

smooth = 1.

def dice_coef(y_true, y_pred):

    y_true_f = K.flatten(y_true)

    y_pred_f = K.flatten(y_pred)

    intersection = K.sum(y_true_f * y_pred_f)

    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)

def dice_coef_loss(y_true, y_pred):

    return -dice_coef(y_true, y_pred)

simple_cnn.compile(optimizer = 'adam', 

                   loss = dice_coef_loss, 

                   metrics = [dice_coef, 'acc', 'mse'])
def simple_gen():

    while True:

        for _, c_row in train_img_df.iterrows():

            yield np.expand_dims(c_row['images'],0), np.expand_dims(np.expand_dims(c_row['masks'],-1),0)



simple_cnn.fit_generator(simple_gen(), 

                         steps_per_epoch=train_img_df.shape[0],

                        epochs = 3)
### %time

test_df = img_df.query('TrainingSplit=="test"')

test_rows = []

group_cols = ['Stage', 'ImageId']

for n_group, n_rows in test_df.groupby(group_cols):

    c_row = {col_name: col_value for col_name, col_value in zip(group_cols, n_group)}

    c_row['images'] = n_rows.query('ImageType == "images"')['path'].values.tolist()

    test_rows += [c_row]

test_img_df = pd.DataFrame(test_rows)    



test_img_df['images'] = test_img_df['images'].map(read_and_stack).map(lambda x: x[:,:,:IMG_CHANNELS])

print(test_img_df.shape[0], 'images to process')

test_img_df.sample(1)
### %time

test_img_df['masks'] = test_img_df['images'].map(lambda x: simple_cnn.predict(np.expand_dims(x, 0))[0, :, :, 0])
n_img = 3

from skimage.morphology import closing, opening, disk

def clean_img(x):

    return opening(closing(x, disk(1)), disk(3))

fig, m_axs = plt.subplots(3, n_img, figsize = (12, 6))

for (_, d_row), (c_im, c_lab, c_clean) in zip(test_img_df.sample(n_img).iterrows(), 

                                     m_axs):

    c_im.imshow(d_row['images'])

    c_im.axis('off')

    c_im.set_title('Microscope')

    

    c_lab.imshow(d_row['masks'])

    c_lab.axis('off')

    c_lab.set_title('Predicted')

    

    c_clean.imshow(clean_img(d_row['masks']))

    c_clean.axis('off')

    c_clean.set_title('Clean')
from skimage.morphology import label # label regions

def rle_encoding(x):

    '''

    x: numpy array of shape (height, width), 1 - mask, 0 - background

    Returns run length as list

    '''

    dots = np.where(x.T.flatten()==1)[0] # .T sets Fortran order down-then-right

    run_lengths = []

    prev = -2

    for b in dots:

        if (b>prev+1): run_lengths.extend((b+1, 0))

        run_lengths[-1] += 1

        prev = b

    return run_lengths



def prob_to_rles(x, cut_off = 0.5):

    lab_img = label(x>cut_off)

    if lab_img.max()<1:

        lab_img[0,0] = 1 # ensure at least one prediction per image

    for i in range(1, lab_img.max()+1):

        yield rle_encoding(lab_img==i)
_, train_rle_row = next(train_img_df.tail(5).iterrows()) 

train_row_rles = list(prob_to_rles(train_rle_row['masks']))
tl_rles = train_labels.query('ImageId=="{ImageId}"'.format(**train_rle_row))['EncodedPixels']
match, mismatch = 0, 0

for img_rle, train_rle in zip(sorted(train_row_rles, key = lambda x: x[0]), 

                             sorted(tl_rles, key = lambda x: x[0])):

    for i_x, i_y in zip(img_rle, train_rle):

        if i_x == i_y:

            match += 1

        else:

            mismatch += 1

print('Matches: %d, Mismatches: %d, Accuracy: %2.1f%%' % (match, mismatch, 100.0*match/(match+mismatch)))
test_img_df['rles'] = test_img_df['masks'].map(clean_img).map(lambda x: list(prob_to_rles(x)))
out_pred_list = []

for _, c_row in test_img_df.iterrows():

    for c_rle in c_row['rles']:

        out_pred_list+=[dict(ImageId=c_row['ImageId'], 

                             EncodedPixels = ' '.join(np.array(c_rle).astype(str)))]

out_pred_df = pd.DataFrame(out_pred_list)

print(out_pred_df.shape[0], 'regions found for', test_img_df.shape[0], 'images')

out_pred_df.sample(3)
out_pred_df[['ImageId', 'EncodedPixels']].to_csv('predictions.csv', index = False)