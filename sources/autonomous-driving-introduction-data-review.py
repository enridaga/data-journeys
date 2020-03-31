
import pandas as pd
import numpy as np
import matplotlib.pylab as plt
import seaborn as sns
import pickle
import cv2
import json
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

sns.set_style("whitegrid")
my_pal = sns.color_palette(n_colors=10)
# Look at the data folder
### ls -GFlash --color ../input/pku-autonomous-driving/
train = pd.read_csv('../input/pku-autonomous-driving/train.csv')
train.head()
print('Example Prediction String....')
print(train['PredictionString'].values[0])
train_expanded = pd.concat([train, train['PredictionString'].str.split(' ', expand=True)], axis=1)
train_expanded = train_expanded.rename(columns={0 : '1_model_type', 1 : '1_yaw', 2 : '1_pitch',
                                                3 : '1_roll', 4 : '1_x', 5 : '1_y', 6 : '1_z'})
train_expanded.drop('PredictionString', axis=1).head()
train_expanded.groupby('1_model_type')['ImageId'] \
    .count() \
    .sort_values() \
    .plot(kind='barh',
          figsize=(15, 8),
          title='First Car, Count by Model Type',
          color=my_pal[0])
plt.show()
train_expanded['1_yaw'] = pd.to_numeric(train_expanded['1_yaw'])
train_expanded['1_yaw'] \
    .dropna() \
    .plot(kind='hist',
          figsize=(15, 3),
          bins=100,
          title='Distribution of First car YAW',
          color=my_pal[1])
plt.show()
train_expanded['1_pitch'] = pd.to_numeric(train_expanded['1_pitch'])
train_expanded['1_pitch'] \
    .dropna() \
    .plot(kind='hist',
          figsize=(15, 3),
          bins=100,
          title='Distribution of First car pitch',
          color=my_pal[2])
plt.show()
train_expanded['1_roll'] = pd.to_numeric(train_expanded['1_roll'])
train_expanded['1_roll'] \
    .dropna() \
    .plot(kind='hist',
          figsize=(15, 3),
          bins=100,
          title='Distribution of First car roll',
          color=my_pal[3])
plt.show()
train_expanded['1_x'] = pd.to_numeric(train_expanded['1_x'])
train_expanded['1_y'] = pd.to_numeric(train_expanded['1_y'])
train_expanded['1_z'] = pd.to_numeric(train_expanded['1_z'])
train_expanded['1_x'] \
    .dropna() \
    .plot(kind='hist',
          figsize=(15, 3),
          bins=100,
          title='Distribution of x',
          color=my_pal[0])
plt.show()
train_expanded['1_y'] \
    .dropna() \
    .plot(kind='hist',
          figsize=(15, 3),
          bins=100,
          title='Distribution of y',
          color=my_pal[1])
plt.show()
train_expanded['1_z'] \
    .dropna() \
    .plot(kind='hist',
          figsize=(15, 3),
          bins=100,
          title='Distribution of z',
          color=my_pal[2])
plt.show()
ss = pd.read_csv('../input/pku-autonomous-driving/sample_submission.csv')
ss.head()
# Lets look at the first few images on disk
### ls -GFlash ../input/pku-autonomous-driving/train_images | head
plt.rcParams["axes.grid"] = False

train_ids = train['ImageId'].values
img_name = train.loc[2742]['ImageId']
fig, ax = plt.subplots(figsize=(15, 15))
img = load_img('../input/pku-autonomous-driving/train_images/' + img_name + '.jpg')
plt.imshow(img)
plt.show()
fig, ax = plt.subplots(figsize=(15, 15))
mask = load_img('../input/pku-autonomous-driving/train_masks/' + img_name + '.jpg')
plt.imshow(mask)
plt.show()
fig, ax = plt.subplots(figsize=(15, 15))
plt.imshow(img)
plt.imshow(mask, cmap=plt.cm.viridis, interpolation='none', alpha=0.5)
plt.show()
ids = train['ImageId'].values
fig, axes = plt.subplots(4, 3, figsize=(18, 20))
for i in range(4):
    img = load_img('../input/pku-autonomous-driving/train_images/' + ids[i] + '.jpg')
    img_mask = load_img('../input/pku-autonomous-driving/train_masks/' + ids[i] + '.jpg')
    #plt.subplot(1,2*(1+len(ids)),q*2-1)
    ax=axes[i][0].imshow(img)
    #plt.subplot(1,2*(1+len(ids)),q*2)
    ax=axes[i][1].imshow(img_mask)
    ax=axes[i][2].imshow(img)
    ax=axes[i][2].imshow(img_mask, cmap=plt.cm.viridis, interpolation='none', alpha=0.4)
plt.show()
### cat ../input/pku-autonomous-driving/camera/camera_intrinsic.txt
### ls -GFlash ../input/pku-autonomous-driving/car_models/ | head
# model = '../input/pku-autonomous-driving/car_models/aodi-Q7-SUV.pkl'
# with open(model, "rb") as file:
#     pickle.load(file, encoding="latin1")
### ls -GFlash ../input/pku-autonomous-driving/car_models_json/ | head
with open('../input/pku-autonomous-driving/car_models_json/mazida-6-2015.json') as json_file:
    car_model_data = json.load(json_file)
for keys in enumerate(car_model_data):
    print(keys)
def plot_3d_car(model_json_file):
    with open(f'../input/pku-autonomous-driving/car_models_json/{model_json_file}') as json_file:
        car_model_data = json.load(json_file)

    vertices = np.array(car_model_data['vertices'])
    faces = np.array(car_model_data['faces']) - 1
    car_type = car_model_data['car_type']
    x, y, z = vertices[:,0], vertices[:,2], -vertices[:,1]
    fig = plt.figure(figsize=(30, 10))
    ax = plt.axes(projection='3d')
    ax.plot_trisurf(x, y, faces, z,
                    cmap='viridis', edgecolor='none')
    ax.set_title(car_type)
    ax.view_init(30, 0)
    plt.show()
    fig = plt.figure(figsize=(30, 10))
    ax = plt.axes(projection='3d')
    ax.plot_trisurf(x, y, faces, z,
                    cmap='viridis', edgecolor='none')
    ax.set_title(car_type)
    ax.view_init(60, 0)
    plt.show()
    fig = plt.figure(figsize=(30, 10))
    ax = plt.axes(projection='3d')
    ax.plot_trisurf(x, y, faces, z,
                    cmap='viridis', edgecolor='none')
    ax.set_title(car_type)
    ax.view_init(-20, 180)
    plt.show()
    return
plot_3d_car('MG-GT-2015.json')
plot_3d_car('aodi-Q7-SUV.json')
plot_3d_car('mazida-6-2015.json')