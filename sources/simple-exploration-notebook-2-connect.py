
import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns

color = sns.color_palette()



#%matplotlib inline



pd.options.mode.chained_assignment = None  # default='warn'
train_df = pd.read_json("../input/train.json")

train_df.head()
test_df = pd.read_json("../input/test.json")

print("Train Rows : ", train_df.shape[0])

print("Test Rows : ", test_df.shape[0])
int_level = train_df['interest_level'].value_counts()



plt.figure(figsize=(8,4))

sns.barplot(int_level.index, int_level.values, alpha=0.8, color=color[1])

plt.ylabel('Number of Occurrences', fontsize=12)

plt.xlabel('Interest level', fontsize=12)

plt.show()
cnt_srs = train_df['bathrooms'].value_counts()



plt.figure(figsize=(8,4))

sns.barplot(cnt_srs.index, cnt_srs.values, alpha=0.8, color=color[0])

plt.ylabel('Number of Occurrences', fontsize=12)

plt.xlabel('bathrooms', fontsize=12)

plt.show()
train_df['bathrooms'].ix[train_df['bathrooms']>3] = 3

plt.figure(figsize=(8,4))

sns.violinplot(x='interest_level', y='bathrooms', data=train_df)

plt.xlabel('Interest level', fontsize=12)

plt.ylabel('bathrooms', fontsize=12)

plt.show()
cnt_srs = train_df['bedrooms'].value_counts()



plt.figure(figsize=(8,4))

sns.barplot(cnt_srs.index, cnt_srs.values, alpha=0.8, color=color[2])

plt.ylabel('Number of Occurrences', fontsize=12)

plt.xlabel('bedrooms', fontsize=12)

plt.show()
plt.figure(figsize=(8,6))

sns.countplot(x='bedrooms', hue='interest_level', data=train_df)

plt.ylabel('Number of Occurrences', fontsize=12)

plt.xlabel('bedrooms', fontsize=12)

plt.show()
plt.figure(figsize=(8,6))

plt.scatter(range(train_df.shape[0]), np.sort(train_df.price.values))

plt.xlabel('index', fontsize=12)

plt.ylabel('price', fontsize=12)

plt.show()
ulimit = np.percentile(train_df.price.values, 99)

train_df['price'].ix[train_df['price']>ulimit] = ulimit



plt.figure(figsize=(8,6))

sns.distplot(train_df.price.values, bins=50, kde=True)

plt.xlabel('price', fontsize=12)

plt.show()
llimit = np.percentile(train_df.latitude.values, 1)

ulimit = np.percentile(train_df.latitude.values, 99)

train_df['latitude'].ix[train_df['latitude']<llimit] = llimit

train_df['latitude'].ix[train_df['latitude']>ulimit] = ulimit



plt.figure(figsize=(8,6))

sns.distplot(train_df.latitude.values, bins=50, kde=False)

plt.xlabel('latitude', fontsize=12)

plt.show()
llimit = np.percentile(train_df.longitude.values, 1)

ulimit = np.percentile(train_df.longitude.values, 99)

train_df['longitude'].ix[train_df['longitude']<llimit] = llimit

train_df['longitude'].ix[train_df['longitude']>ulimit] = ulimit



plt.figure(figsize=(8,6))

sns.distplot(train_df.longitude.values, bins=50, kde=False)

plt.xlabel('longitude', fontsize=12)

plt.show()
from mpl_toolkits.basemap import Basemap

from matplotlib import cm



west, south, east, north = -74.02, 40.64, -73.85, 40.86



fig = plt.figure(figsize=(14,10))

ax = fig.add_subplot(111)

m = Basemap(projection='merc', llcrnrlat=south, urcrnrlat=north,

            llcrnrlon=west, urcrnrlon=east, lat_ts=south, resolution='i')

x, y = m(train_df['longitude'].values, train_df['latitude'].values)

m.hexbin(x, y, gridsize=200,

         bins='log', cmap=cm.YlOrRd_r);
train_df["created"] = pd.to_datetime(train_df["created"])

train_df["date_created"] = train_df["created"].dt.date

cnt_srs = train_df['date_created'].value_counts()





plt.figure(figsize=(12,4))

ax = plt.subplot(111)

ax.bar(cnt_srs.index, cnt_srs.values, alpha=0.8)

ax.xaxis_date()

plt.xticks(rotation='vertical')

plt.show()
test_df["created"] = pd.to_datetime(test_df["created"])

test_df["date_created"] = test_df["created"].dt.date

cnt_srs = test_df['date_created'].value_counts()



plt.figure(figsize=(12,4))

ax = plt.subplot(111)

ax.bar(cnt_srs.index, cnt_srs.values, alpha=0.8)

ax.xaxis_date()

plt.xticks(rotation='vertical')

plt.show()
train_df["hour_created"] = train_df["created"].dt.hour

cnt_srs = train_df['hour_created'].value_counts()



plt.figure(figsize=(12,6))

sns.barplot(cnt_srs.index, cnt_srs.values, alpha=0.8, color=color[3])

plt.xticks(rotation='vertical')

plt.show()
cnt_srs = train_df.groupby('display_address')['display_address'].count()



for i in [2, 10, 50, 100, 500]:

    print('Display_address that appear less than {} times: {}%'.format(i, round((cnt_srs < i).mean() * 100, 2)))



plt.figure(figsize=(12, 6))

plt.hist(cnt_srs.values, bins=100, log=True, alpha=0.9)

plt.xlabel('Number of times display_address appeared', fontsize=12)

plt.ylabel('log(Count)', fontsize=12)

plt.show()
train_df["num_photos"] = train_df["photos"].apply(len)

cnt_srs = train_df['num_photos'].value_counts()



plt.figure(figsize=(12,6))

sns.barplot(cnt_srs.index, cnt_srs.values, alpha=0.8)

plt.xlabel('Number of Photos', fontsize=12)

plt.ylabel('Number of Occurrences', fontsize=12)

plt.show()
train_df['num_photos'].ix[train_df['num_photos']>12] = 12

plt.figure(figsize=(12,6))

sns.violinplot(x="num_photos", y="interest_level", data=train_df, order =['low','medium','high'])

plt.xlabel('Number of Photos', fontsize=12)

plt.ylabel('Interest Level', fontsize=12)

plt.show()
train_df["num_features"] = train_df["features"].apply(len)

cnt_srs = train_df['num_features'].value_counts()



plt.figure(figsize=(12,6))

sns.barplot(cnt_srs.index, cnt_srs.values, alpha=0.8)

plt.ylabel('Number of Occurrences', fontsize=12)

plt.xlabel('Number of features', fontsize=12)

plt.show()
train_df['num_features'].ix[train_df['num_features']>17] = 17

plt.figure(figsize=(12,10))

sns.violinplot(y="num_features", x="interest_level", data=train_df, order =['low','medium','high'])

plt.xlabel('Interest Level', fontsize=12)

plt.ylabel('Number of features', fontsize=12)

plt.show()
from wordcloud import WordCloud



text = ''

text_da = ''

text_desc = ''

for ind, row in train_df.iterrows():

    for feature in row['features']:

        text = " ".join([text, "_".join(feature.strip().split(" "))])

    text_da = " ".join([text_da,"_".join(row['display_address'].strip().split(" "))])

    #text_desc = " ".join([text_desc, row['description']])

text = text.strip()

text_da = text_da.strip()

text_desc = text_desc.strip()



plt.figure(figsize=(12,6))

wordcloud = WordCloud(background_color='white', width=600, height=300, max_font_size=50, max_words=40).generate(text)

wordcloud.recolor(random_state=0)

plt.imshow(wordcloud)

plt.title("Wordcloud for features", fontsize=30)

plt.axis("off")

plt.show()



# wordcloud for display address

plt.figure(figsize=(12,6))

wordcloud = WordCloud(background_color='white', width=600, height=300, max_font_size=50, max_words=40).generate(text_da)

wordcloud.recolor(random_state=0)

plt.imshow(wordcloud)

plt.title("Wordcloud for Display Address", fontsize=30)

plt.axis("off")

plt.show()
