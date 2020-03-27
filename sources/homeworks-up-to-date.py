
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
data = pd.read_csv("/kaggle/input/videogamesales/vgsales.csv")
data.info()
data.describe()
data.head(10)
import seaborn as sns
data.corr()#corelation table
import matplotlib.pyplot as plt
f,ax = plt.subplots(figsize = (18, 18))
sns.heatmap(data.corr(), annot =True, linewidths =.5,fmt = '.1f',ax=ax);
na = data['NA_Sales']
eu = data['EU_Sales']
gb = data['Global_Sales']
plt.scatter(na,gb,c='r',alpha = 0.5)
plt.xlabel('North America Sales')
plt.ylabel('Global Sales')
plt.title('Norh America Sales - Global Sales Correlation Chart');
plt.scatter(eu,gb,c='g',alpha = 0.5)
plt.xlabel('Europe Sales')
plt.ylabel('Global Sales')
plt.title('Europe Sales - Global Sales Correlation Chart');
def threetimes (x):
    return x*3
a = threetimes(2)
print(a)
    
def func1():
    def func2():
        a = 2
        b = 8
        return b/a
    return func2()/2
print(func1())
def func3(x, y = 1,z = 1):
    return x*(y+z)
print(func3(2))
print(func3(2,2,2))
def func4(*args):
    for i in args:
        print(i)
func4(1)
func4(5,6,1,4)
def func5(**kwargs):
    for k,v in kwargs.items():
        print(k,':', v)
func5(Movie = 'Avatar', Year = 2009, Genre = 'Fantasy')
threetimes = lambda x: x*3
print(threetimes(2))
import numpy as np
vec = np.arange(1,10,1)
y = map(lambda x: x**2,vec)
print(list(y))
list1 = list(np.arange(1,5))
list2 = list(np.arange(5,9))
zlist = zip(list1,list2)
print(zlist)
z_list = list(zlist)
print(z_list)
unzip = zip(*z_list)
unlist1,unlist2 = list(unzip)
print(unlist1)
print(unlist2)
x1 = [1,2,3]
x2 = [ i + 1 for i in x1]
print(x2)
x1 = [4,8,16]
x2 = [i*2 if i == 8 else i/2 if i < 8 else i*0 for i in x1]
print(x2)
tmark = sum(data['Global_Sales'])/len(data.Global_Sales)
data["earn"] = ["high" if i > tmark else "low" for i in data.Global_Sales]
data.loc[:,["earn","Global_Sales"]]
print(tmark)
print(data['Genre'].value_counts(dropna = False))
#frequency of game genres
data.boxplot(column = 'Rank',by = 'Genre')
data_new = data.head()
data_new
#lets melt
melted = pd.melt(frame = data_new,id_vars = 'Name', value_vars = ['EU_Sales','NA_Sales'])
melted
melted.pivot(index = 'Name',columns = 'variable', values = 'value')
#concatenating
data1 = data.head()
data2= data.tail()
cdr = pd.concat([data1,data2],axis = 0,ignore_index = True)
cdr
data1 = data['NA_Sales'].head()
data2 = data['EU_Sales'].head()
cdc = pd.concat([data1,data2],axis = 1)
cdc
data.dtypes
data[['Name', 'Platform', 'Genre', 'Publisher','earn']] = data[['Name', 'Platform', 'Genre', 'Publisher','earn']].astype('category')
data.Rank = data['Rank'].astype('float')
data.dtypes
data.Rank = data['Rank'].astype('float')
data.info()
data['Year'].value_counts(dropna = False)
data1 = data.copy()
data1['Year'].dropna(inplace = True)
assert data1['Year'].notnull().all()
data1['Year'].value_counts(dropna = False)
