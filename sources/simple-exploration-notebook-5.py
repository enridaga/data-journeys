
# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns



### matplotlib inline



pd.set_option('display.max_columns', 120)
with pd.HDFStore("../input/train.h5", "r") as train:

    # Note that the "train" dataframe is the only dataframe in the file

    df = train.get("train")
df.head()
df.describe()
labels = []

values = []

for col in df.columns:

    labels.append(col)

    values.append(df[col].isnull().sum())

    print(col, values[-1])
ind = np.arange(len(labels))

width = 0.9

fig, ax = plt.subplots(figsize=(12,50))

rects = ax.barh(ind, np.array(values), color='y')

ax.set_yticks(ind+((width)/2.))

ax.set_yticklabels(labels, rotation='horizontal')

ax.set_xlabel("Count of missing values")

ax.set_title("Number of missing values in each column")

#autolabel(rects)

plt.show()
cols_to_use = ['technical_30', 'technical_20', 'fundamental_11', 'technical_19']

fig = plt.figure(figsize=(8, 20))

plot_count = 0

for col in cols_to_use:

    plot_count += 1

    plt.subplot(4, 1, plot_count)

    plt.scatter(range(df.shape[0]), df[col].values)

    plt.title("Distribution of "+col)

plt.show()
plt.figure(figsize=(8, 5))

plt.scatter(range(df.shape[0]), df.y.values)

plt.show()
fig = plt.figure(figsize=(12, 6))

sns.countplot(x='timestamp', data=df)

plt.show()
print(len(df.id.unique()))
temp_df = df.groupby('id')['y'].agg('mean').reset_index().sort_values(by='y')

temp_df.head()
id_to_use = [1431, 93, 882, 1637, 1118]

fig = plt.figure(figsize=(8, 25))

plot_count = 0

for id_val in id_to_use:

    plot_count += 1

    plt.subplot(5, 1, plot_count)

    temp_df = df.ix[df['id']==id_val,:]

    plt.plot(temp_df.timestamp.values, temp_df.y.values)

    plt.plot(temp_df.timestamp.values, temp_df.y.cumsum())

    plt.title("Asset ID : "+str(id_val))

    

plt.show()
temp_df = df.groupby('id')['y'].agg('mean').reset_index().sort_values(by='y')

temp_df.tail()
id_to_use = [767, 226, 824, 1809, 1089]

fig = plt.figure(figsize=(8, 25))

plot_count = 0

for id_val in id_to_use:

    plot_count += 1

    plt.subplot(5, 1, plot_count)

    temp_df = df.ix[df['id']==id_val,:]

    plt.plot(temp_df.timestamp.values, temp_df.y.values)

    plt.plot(temp_df.timestamp.values, temp_df.y.cumsum())

    plt.title("Asset ID : "+str(id_val))

plt.show()
temp_df = df.groupby('id')['y'].agg('count').reset_index().sort_values(by='y')

temp_df.tail()
id_to_use = [1548, 699, 697, 704, 1066]

fig = plt.figure(figsize=(8, 25))

plot_count = 0

for id_val in id_to_use:

    plot_count += 1

    plt.subplot(5, 1, plot_count)

    temp_df = df.ix[df['id']==id_val,:]

    plt.plot(temp_df.timestamp.values, temp_df.y.values)

    plt.plot(temp_df.timestamp.values, temp_df.y.cumsum())

    plt.title("Asset ID : "+str(id_val))

plt.show()