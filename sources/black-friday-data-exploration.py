
# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
import matplotlib as mlp
import matplotlib.pyplot as plt
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
##reading the data 
df = pd.read_csv('../input/BlackFriday.csv')
df.info()
df.head(5)
## Looks like we have some null/NaN values in the product categories.
## to check which columns have null values.
df.isna().any()
print('Product_Category_2', df['Product_Category_2'].unique())
print('-----------------------------------------------------------------------------------')
print('Product_Category_3', df['Product_Category_3'].unique())
## So all values are in integer range. May be we can assign value zero for the NaN cases
df.fillna(value=0,inplace=True)
## Also looks like the product categories are float while they can be int.
df["Product_Category_2"] = df["Product_Category_2"].astype(int)
df["Product_Category_3"] = df["Product_Category_3"].astype(int)
print('Product_Category_2', df['Product_Category_2'].unique())
print('-----------------------------------------------------------------------------------')
print('Product_Category_3', df['Product_Category_3'].unique())
## We might not need product Id and user Id hence dropping them
df.drop(columns = ["User_ID","Product_ID"],inplace=True)
## need to always remember to use inplace to make the changes in current data frame
sns.countplot(df['Gender'])
sns.countplot(df['Age'])
'''while we could write df.Age, as a best practice and 
to avoid name mismatch with existing attributes/functions of a data frame, I am using []'''
sns.countplot(df['Age'],hue=df['Gender'])
df['combined_G_M'] = df.apply(lambda x:'%s_%s' % (x['Gender'],x['Marital_Status']),axis=1)
print(df['combined_G_M'].unique())
sns.countplot(df['Age'],hue=df['combined_G_M'])
sns.countplot(df['Product_Category_2'],hue=df['combined_G_M'])
sns.countplot(df['Product_Category_3'],hue=df['combined_G_M'])
df.columns
df_by_occupation_and_categories = df.groupby(['Occupation','Product_Category_1']).count().reset_index('Product_Category_1')
fig = plt.figure()
from matplotlib.font_manager import FontProperties
fontP = FontProperties()
fontP.set_size('small')
cmap = plt.get_cmap('terrain')
colors = [cmap(i) for i in np.linspace(0, 1, 20)]
plotFor = "Occupation {0}"
title = 'Product_Category_1'
for i in range(21):
    explode = [0.15 for i in range(len(df_by_occupation_and_categories['Gender'][i].get_values()))]
    a = plt.pie(df_by_occupation_and_categories['Gender'][i].get_values(), radius=1, colors=colors,startangle=90,autopct='%1.1f%%',explode=explode)
    plt.legend(df_by_occupation_and_categories['Product_Category_1'][0].get_values(),loc='upper center',prop=fontP, bbox_to_anchor=(1.2, 1),title=title)
    plt.title(plotFor.format(i))
    plt.show()
## We can use tableau as well to create better charts.

