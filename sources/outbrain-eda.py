
import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os

import gc # We're gonna be clearing memory a lot

import matplotlib.pyplot as plt

import seaborn as sns

#%matplotlib inline



p = sns.color_palette()



print('# File sizes')

for f in os.listdir('../input'):

    if 'zip' not in f:

        print(f.ljust(30) + str(round(os.path.getsize('../input/' + f) / 1000000, 2)) + 'MB')
df_train = pd.read_csv('../input/clicks_train.csv')

df_test = pd.read_csv('../input/clicks_test.csv')
sizes_train = df_train.groupby('display_id')['ad_id'].count().value_counts()

sizes_test = df_test.groupby('display_id')['ad_id'].count().value_counts()

sizes_train = sizes_train / np.sum(sizes_train)

sizes_test = sizes_test / np.sum(sizes_test)



plt.figure(figsize=(12,4))

sns.barplot(sizes_train.index, sizes_train.values, alpha=0.8, color=p[0], label='train')

sns.barplot(sizes_test.index, sizes_test.values, alpha=0.6, color=p[1], label='test')

plt.legend()

plt.xlabel('Number of Ads in display', fontsize=12)

plt.ylabel('Proportion of set', fontsize=12)
ad_usage_train = df_train.groupby('ad_id')['ad_id'].count()



for i in [2, 10, 50, 100, 1000]:

    print('Ads that appear less than {} times: {}%'.format(i, round((ad_usage_train < i).mean() * 100, 2)))



plt.figure(figsize=(12, 6))

plt.hist(ad_usage_train.values, bins=50, log=True)

plt.xlabel('Number of times ad appeared', fontsize=12)

plt.ylabel('log(Count of displays with ad)', fontsize=12)

plt.show()
ad_prop = len(set(df_test.ad_id.unique()).intersection(df_train.ad_id.unique())) / len(df_test.ad_id.unique())

print('Proportion of test ads in test that are in training: {}%'.format(round(ad_prop * 100, 2)))
try:del df_train,df_test # Being nice to Azure

except:pass;gc.collect()



events = pd.read_csv('../input/events.csv')

print('Shape:', events.shape)

print('Columns', events.columns.tolist())

events.head()
plat = events.platform.value_counts()



print(plat)

print('\nUnique values of platform:', events.platform.unique())
events.platform = events.platform.astype(str)

plat = events.platform.value_counts()



plt.figure(figsize=(12,4))

sns.barplot(plat.index, plat.values, alpha=0.8, color=p[2])

plt.xlabel('Platform', fontsize=12)

plt.ylabel('Occurence count', fontsize=12)
uuid_counts = events.groupby('uuid')['uuid'].count().sort_values()



print(uuid_counts.tail())



for i in [2, 5, 10]:

    print('Users that appear less than {} times: {}%'.format(i, round((uuid_counts < i).mean() * 100, 2)))

    

plt.figure(figsize=(12, 4))

plt.hist(uuid_counts.values, bins=50, log=True)

plt.xlabel('Number of times user appeared in set', fontsize=12)

plt.ylabel('log(Count of users)', fontsize=12)

plt.show()
try:del events

except:pass;gc.collect()



topics = pd.read_csv('../input/documents_topics.csv')

print('Columns:',topics.columns.tolist())

print('Number of unique topics:', len(topics.topic_id.unique()))



topics.head()
topic_ids = topics.groupby('topic_id')['confidence_level'].count().sort_values()



for i in [10000, 50000, 100000, 200000]:

    print('Number of topics that appear more than {} times: {}'

          .format(i, (topic_ids > i).sum()))



plt.figure(figsize=(12, 4))

sns.barplot(topic_ids.index, topic_ids.values, order=topic_ids.index, alpha=1, color=p[5])

plt.xlabel('Document Topics', fontsize=12)

plt.ylabel('Total occurences', fontsize=12)

plt.show()
cat = pd.read_csv('../input/documents_categories.csv')

print('Columns:', cat.columns.tolist())

print('Number of unique categories:', len(cat.category_id.unique()))



cat_ids = cat.groupby('category_id')['confidence_level'].count().sort_values()



for i in [1000, 10000, 50000, 100000]:

    print('Number of categories that appear more than {} times: {}'

          .format(i, (cat_ids > i).sum()))



plt.figure(figsize=(12, 4))

sns.barplot(cat_ids.index, cat_ids.values, order=cat_ids.index, alpha=1, color=p[3])

plt.xlabel('Document Categories', fontsize=12)

plt.ylabel('Total occurences', fontsize=12)

plt.show()