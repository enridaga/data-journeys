
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
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sb
import seaborn as sns
plt.style.use('seaborn-whitegrid')  

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LinearRegression
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import r2_score
import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


#%matplotlib inline
print(pd.__version__)
print(np.__version__)
%time
df = pd.read_csv("../input/AB_NYC_2019.csv")
df
import pandas as pd
AB_NYC_2019 = pd.read_csv("../input/AB_NYC_2019.csv")
print("кол-во строк:",df.shape[0])
print("кол-во строк:",df.shape[1])
df.info()
df.describe()
df.head()
df.tail()
import matplotlib.pyplot as plt
#%matplotlib inline

df.hist(bins=50, figsize=(20,10))
df["room_type"].value_counts()
#заполним пропущенные значения 0
df = df.fillna(0)
df.isnull().sum()
df.dropna(how='any',inplace=True)
df.info() 
hostname_DF = AB_NYC_2019.loc[AB_NYC_2019.price>1500][['name','host_name', 'price']][:11].set_index('host_name').sort_values(by = 'price', ascending = False)
print(hostname_DF)
hostname_DF = AB_NYC_2019.loc[AB_NYC_2019.price>1500][['host_name', 'price']][:11].set_index('host_name').sort_values(by = 'price', ascending = False).plot(kind = 'bar', figsize = (12,5))
plt.xlabel('host names')
plt.ylabel('price')
print(hostname_DF)
AB_NYC_2019.groupby(['neighbourhood_group'])['price'].mean().plot(kind = 'bar', figsize=(12,5))
AB_NYC_2019.groupby(['neighbourhood_group','room_type'])['price'].mean().sort_values(ascending = False)
plt.figure(figsize=(12,8))
plt.hist(AB_NYC_2019["reviews_per_month"], rwidth=0.5,bins=np.arange(0, AB_NYC_2019["reviews_per_month"].max() +20, 5))
plt.figure(figsize=(12,8))
plt.hist(AB_NYC_2019["availability_365"], rwidth=0.5,bins=np.arange(0, AB_NYC_2019["availability_365"].max() +12, 25))
plt.figure(figsize=(12,8))
ytickrange = np.arange(0, 14000, 500) 
ax = sns.countplot(x='room_type', hue="neighbourhood_group", data=AB_NYC_2019)
ax.set_yticks(ytickrange)
#отбираем количественные признаки
AB_NYC_2019_model = AB_NYC_2019[["price","minimum_nights", "number_of_reviews", "reviews_per_month", "calculated_host_listings_count","availability_365"]]
x=AB_NYC_2019_model[["price","minimum_nights", "number_of_reviews", "reviews_per_month","availability_365"]]
y=AB_NYC_2019_model[["calculated_host_listings_count"]]
x.head()
y
#заполним пропуски 0
df.drop(['host_id','latitude','longitude','neighbourhood','number_of_reviews','reviews_per_month'], axis=1, inplace=True)
#Проверка на изменения
df.head(5)
#Кодирование вводных переменных
def Encode(df):
    for column in df.columns[df.columns.isin(['neighbourhood_group', 'room_type'])]:
        df[column] = df[column].factorize()[0]
    return df

df_en = Encode(df.copy())
df_en.head(15)
#Получаем корреляцию между различными переменными
corr = df_en.corr(method='kendall')
plt.figure(figsize=(18,12))
sns.heatmap(corr, annot=True)
df_en.columns
#Определение независимых переменных и зависимых переменных
x = df_en.iloc[:,[0,1,3,4,5]]
y = df_en['price']
#Получение тестового и тренировочного набора (разделим выборку на 80 и 20%)
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=11)
x_train.head()
y_train.head()
x_train.shape
#разделим выборку на 80 и 20%
x_train, x_test, y_train, y_test = train_test_split(x,y, test_size=0.2, random_state=11)
x_train = x_train.fillna(0)
clf_LR = LogisticRegression(C=1, tol=1e-3, solver="lbfgs")
clf_LR.fit(x_train, y_train)
x_test = x_test.fillna(0)
y_pred_LR = clf_LR.predict(x_test)
cm = confusion_matrix(y_pred_LR, y_test)
print(accuracy_score(y_pred_LR, y_test))
# Confusion matrix
cm
from sklearn.ensemble import RandomForestClassifier

clf_RF = RandomForestClassifier()
clf_RF.fit(x_train, y_train)
# получаем предсказания
y_pred_RF = clf_RF.predict(x_test)
#точность модели
cm = confusion_matrix(y_pred_RF, y_test)
print(accuracy_score(y_pred_RF, y_test))
# Confusion matrix
cm
