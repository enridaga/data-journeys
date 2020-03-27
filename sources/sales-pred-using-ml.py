
# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.
sales=pd.read_csv("/kaggle/input/competitive-data-science-predict-future-sales/sales_train.csv")
sample=pd.read_csv("/kaggle/input/competitive-data-science-predict-future-sales/sample_submission.csv")
items=pd.read_csv("/kaggle/input/competitive-data-science-predict-future-sales/items.csv")
shops=pd.read_csv('/kaggle/input/competitive-data-science-predict-future-sales/shops.csv')
item_category=pd.read_csv('/kaggle/input/competitive-data-science-predict-future-sales/item_categories.csv')
test_data=pd.read_csv('/kaggle/input/competitive-data-science-predict-future-sales/test.csv')

test_data=pd.read_csv("/kaggle/input/competitive-data-science-predict-future-sales/test.csv")
test_data
sales.describe

sales
import seaborn as sbn
import matplotlib.pyplot as plt
sbn.heatmap(sales.corr())
by_cat = items.groupby(['item_category_id']).size().sort_values(ascending=False).head(20).plot.bar()
plt.xlabel('Items per category')
plt.ylabel('No. of Times')
total_sale = sales.groupby(['date_block_num'])['item_cnt_day'].sum()
#total_sale.head()
plt.figure(figsize=(8,8))
plt.title('Total sales of company')
plt.xlabel('Time (January 2013 is 0, February 2013 is 1,..., October 2015 is 33)')
plt.ylabel('sales')
plt.plot(total_sale)
plt.figure(figsize=(16,6))
plt.plot(total_sale, color='blue',label='Original')
plt.plot(total_sale.rolling(window= 12,center= False).mean(),label='Rolling Mean')
plt.plot(total_sale.rolling(window=12,center= False).std(),label='Rolling std')
plt.legend()