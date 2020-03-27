
import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from subprocess import check_output

import matplotlib.pyplot as plt

from scipy.stats import bernoulli

import seaborn as sns

print(check_output(["ls", "../input"]).decode("utf8"))

#print(check_output(["ls", "../input/train-jpg"]).decode("utf8"))
sample = pd.read_csv('../input/sample_submission.csv')

sample.head()
df = pd.read_csv('../input/train.csv')

df.head()
df.shape
all_tags = [item for sublist in list(df['tags'].apply(lambda row: row.split(" ")).values) for item in sublist]

print('total of {} non-unique tags in all training images'.format(len(all_tags)))

print('average number of labels per image {}'.format(1.0*len(all_tags)/df.shape[0]))
tags_counted_and_sorted = pd.DataFrame({'tag': all_tags}).groupby('tag').size().reset_index().sort_values(0, ascending=False)

tags_counted_and_sorted.head()
tags_counted_and_sorted.plot.barh(x='tag', y=0, figsize=(12,8))
tag_probas = tags_counted_and_sorted[0].values/tags_counted_and_sorted[0].values.sum()

indicators = np.hstack([bernoulli.rvs(p, 0, sample.shape[0]).reshape(sample.shape[0], 1) for p in tag_probas])

indicators = np.array(indicators)

indicators.shape
indicators[:10,:]
sorted_tags = tags_counted_and_sorted['tag'].values

all_test_tags = []

for index in range(indicators.shape[0]):

    all_test_tags.append(' '.join(list(sorted_tags[np.where(indicators[index, :] == 1)[0]])))

len(all_test_tags)
sample['tags'] = all_test_tags

sample.head()

sample.to_csv('bernoulli_submission.csv', index=False)
from glob import glob

image_paths = glob('../input/train-jpg/*.jpg')[0:1000]

image_names = list(map(lambda row: row.split("/")[-1][:-4], image_paths))

image_names[0:10]
plt.figure(figsize=(12,8))

for i in range(6):

    plt.subplot(2,3,i+1)

    plt.imshow(plt.imread(image_paths[i]))

    plt.title(str(df[df.image_name == image_names[i]].tags.values))
import cv2



n_imgs = 600



all_imgs = []



for i in range(n_imgs):

    img = plt.imread(image_paths[i])

    img = cv2.resize(img, (50, 50), cv2.INTER_LINEAR)

    img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY).astype('float') / 255.0

    img = img.reshape(1, -1)

    all_imgs.append(img)



img_mat = np.vstack(all_imgs)

img_mat.shape
from scipy.spatial.distance import pdist, squareform



sq_dists = squareform(pdist(img_mat))

print(sq_dists.shape)

sns.clustermap(

    sq_dists,

    figsize=(12,12),

    cmap=plt.get_cmap('viridis')

)
