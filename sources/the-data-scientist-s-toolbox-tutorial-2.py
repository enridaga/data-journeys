
from pandas import get_dummies
import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib
import warnings
import scipy
import numpy
import json
import sys
import csv
import os
print('scipy: {}'.format(scipy.__version__))
print('seaborn: {}'.format(sns.__version__))
print('pandas: {}'.format(pd.__version__))
print('numpy: {}'.format(np.__version__))
print('Python: {}'.format(sys.version))
warnings.filterwarnings('ignore')
### precision 2
    hp_train=pd.read_csv('../input/melb_data.csv')
import numpy as np
mylist = [1, 2, 3]
myarray = np.array(mylist)
myarray.shape
myarray.shape
myarray.resize(3, 3)
myarray
np.ones((3, 2))
np.zeros((2, 3))
np.eye(3)
np.diag(myarray)
np.array([1, 2, 3] * 3)
np.repeat([1, 2, 3], 3)
p = np.ones([2, 3], int)
p
np.vstack([p, 2*p])
np.hstack([p, 2*p])
x=np.array([1, 2, 3])
y=np.array([4, 5, 6])
print(x + y) # elementwise addition     [1 2 3] + [4 5 6] = [5  7  9]
print(x - y) # elementwise subtraction  [1 2 3] - [4 5 6] = [-3 -3 -3]
print(x * y) # elementwise multiplication  [1 2 3] * [4 5 6] = [4  10  18]
print(x / y) # elementwise divison         [1 2 3] / [4 5 6] = [0.25  0.4  0.5]
print(x**2) # elementwise power  [1 2 3] ^2 =  [1 4 9]
x.dot(y) # dot product  1*4 + 2*5 + 3*6
z = np.array([y, y**2])
print(len(z)) # number of rows of array
z = np.array([y, y**2])
z
z.shape
z.T
z.T.shape
z.dtype
z = z.astype('f')
z.dtype
myarray = np.array([-4, -2, 1, 3, 5])
myarray.sum()
myarray.max()
myarray.min()
myarray.mean()
myarray.std()
myarray.argmax()
myarray.argmin()
myarray = np.arange(13)**2
myarray
myarray[0], myarray[4], myarray[-1]
myarray[1:5]
myarray[-4:]
myarray[-5::-2]
r = np.arange(36)
r.resize((6, 6))
r
r[2, 2]
r[3, 3:6]
r[:2, :-1]
r[-1, ::2]
r[r > 30]
r[r > 30] = 30
r
r2 = r[:3,:3]
r2
r2[:] = 0
r2
r
r_copy = r.copy()
r_copy
r_copy[:] = 10
print(r_copy, '\n')
print(r)
test = np.random.randint(0, 10, (4,3))
test
for row in test:
    print(row)
for i in range(len(test)):
    print(test[i])
for i, row in enumerate(test):
    print('row', i, 'is', row)
test2 = test**2
test2
for i, j in zip(test, test2):
    print(i,'+',j,'=',i+j)
animals = ['Tiger', 'Bear', 'Moose']
pd.Series(animals)
numbers = [1, 2, 3]
pd.Series(numbers)
animals = ['Tiger', 'Bear', None]
pd.Series(animals)
numbers = [1, 2, None]
pd.Series(numbers)
import numpy as np
np.nan == None
np.nan == np.nan
np.isnan(np.nan)
sports = {'Archery': 'Bhutan',
          'Golf': 'Scotland',
          'Sumo': 'Japan',
          'Taekwondo': 'South Korea'}
s = pd.Series(sports)
s
s.index
s = pd.Series(['Tiger', 'Bear', 'Moose'], index=['India', 'America', 'Canada'])
s
sports = {'Archery': 'Bhutan',
          'Golf': 'Scotland',
          'Sumo': 'Japan',
          'Taekwondo': 'South Korea'}
s = pd.Series(sports, index=['Golf', 'Sumo', 'Hockey'])
s
sports = {'Archery': 'Bhutan',
          'Golf': 'Scotland',
          'Sumo': 'Japan',
          'Taekwondo': 'South Korea'}
s = pd.Series(sports)
s
s.iloc[3]
s.loc['Golf']
s[3]
s['Golf']
sports = {99: 'Bhutan',
          100: 'Scotland',
          101: 'Japan',
          102: 'South Korea'}
s = pd.Series(sports)
s = pd.Series([100.00, 120.00, 101.00, 3.00])
s
total = 0
for item in s:
    total+=item
print(total)
total = np.sum(s)
print(total)
#this creates a big series of random numbers
s = pd.Series(np.random.randint(0,1000,10000))
s.head()
len(s)
### %timeit -n 100
summary = 0
for item in s:
    summary+=item
### %timeit -n 100
summary = np.sum(s)
s+=2 #adds two to each item in s using broadcasting
s.head()
for label, value in s.iteritems():
    s.set_value(label, value+2)
s.head()
### %timeit -n 10
s = pd.Series(np.random.randint(0,1000,100))
for label, value in s.iteritems():
    s.loc[label]= value+2
### %timeit -n 10
s = pd.Series(np.random.randint(0,1000,100))
s+=2

s = pd.Series([1, 2, 3])
s.loc['Animal'] = 'Bears'
s
np.random.binomial(1, 0.5)
np.random.binomial(1000, 0.5)/1000
chance_of_tornado = 0.01/100
np.random.binomial(100000, chance_of_tornado)
chance_of_tornado = 0.01

tornado_events = np.random.binomial(1, chance_of_tornado, 1000000)
    
two_days_in_a_row = 0
for j in range(1,len(tornado_events)-1):
    if tornado_events[j]==1 and tornado_events[j-1]==1:
        two_days_in_a_row+=1

print('{} tornadoes back to back in {} years'.format(two_days_in_a_row, 1000000/365))
np.random.uniform(0, 1)
np.random.normal(0.75)
distribution = np.random.normal(0.75,size=1000)

np.sqrt(np.sum((np.mean(distribution)-distribution)**2)/len(distribution))
np.std(distribution)

purchase_1 = pd.Series({'Name': 'Chris',
                        'Item Purchased': 'Dog Food',
                        'Cost': 22.50})
purchase_2 = pd.Series({'Name': 'Kevyn',
                        'Item Purchased': 'Kitty Litter',
                        'Cost': 2.50})
purchase_3 = pd.Series({'Name': 'Vinod',
                        'Item Purchased': 'Bird Seed',
                        'Cost': 5.00})
df = pd.DataFrame([purchase_1, purchase_2, purchase_3], index=['Store 1', 'Store 1', 'Store 2'])
df.head()
df.loc['Store 2']
type(df.loc['Store 2'])
df.loc['Store 1']
df.loc['Store 1', 'Cost']
df.T
df.T.loc['Cost']
df['Cost']
df.loc['Store 1']['Cost']
df.loc[:,['Name', 'Cost']]
df.drop('Store 1')
df
copy_df = df.copy()
copy_df = copy_df.drop('Store 1')
copy_df
copy_df.drop
del copy_df['Name']
copy_df
df['Location'] = None
df
costs = df['Cost']
costs
costs+=2
costs
df
df = pd.read_csv('../input/melb_data.csv')
df.head()
df.columns
# Querying a DataFrame
df['Price'] > 10000000
only_SalePrice = df.where(df['Price'] > 0)
only_SalePrice.head()
only_SalePrice['Price'].count()
df['Price'].count()
only_SalePrice = only_SalePrice.dropna()
only_SalePrice.head()
only_SalePrice = df[df['Price'] > 0]
only_SalePrice.head()
len(df[(df['Price'] > 0) | (df['Price'] > 0)])
df[(df['Price'] > 0) & (df['Price'] == 0)]
df.head()
df['SalePrice'] = df.index
df = df.set_index('SalePrice')
df.head()

df = df.reset_index()
df.head()
df = pd.read_csv('../input/melb_data.csv')
df.fillna
df = df.fillna(method='ffill')
df.head()
df = pd.DataFrame([{'Name': 'MJ', 'Item Purchased': 'Sponge', 'Cost': 22.50},
                   {'Name': 'Kevyn', 'Item Purchased': 'Kitty Litter', 'Cost': 2.50},
                   {'Name': 'Filip', 'Item Purchased': 'Spoon', 'Cost': 5.00}],
                  index=['Store 1', 'Store 1', 'Store 2'])
df
df['Date'] = ['December 1', 'January 1', 'mid-May']
df
df['Delivered'] = True
df
df['Feedback'] = ['Positive', None, 'Negative']
df
adf = df.reset_index()
adf['Date'] = pd.Series({0: 'December 1', 2: 'mid-May'})
adf
df = pd.read_csv('../input/melb_data.csv')
df.head()
df = df[df['Price']>500000]
df
df = pd.DataFrame(['A+', 'A', 'A-', 'B+', 'B', 'B-', 'C+', 'C', 'C-', 'D+', 'D'],
                  index=['excellent', 'excellent', 'excellent', 'good', 'good', 'good', 'ok', 'ok', 'ok', 'poor', 'poor'])
df.rename(columns={0: 'Grades'}, inplace=True)
df
df['Grades'].astype('category').head()
grades = df['Grades'].astype('category',
                             categories=['D', 'D+', 'C-', 'C', 'C+', 'B-', 'B', 'B+', 'A-', 'A', 'A+'],
                             ordered=True)
grades.head()
grades > 'C'
df.loc[df['Grades'] == 'A+']

df_test = pd.DataFrame({'A': [1, 2, 3], 'B': [1, 4, 7]})
df_test.isin({'A': [1, 3], 'B': [4, 7, 12]})
df.loc[(df['Grades'] == 'A+') & (df['Grades'] == 'D')]


df.loc[df['Grades'] != 'B+']

df_test = pd.DataFrame({'A': [1, 2, 3], 'B': [1, 4, 7]})
df_test.loc[~df_test['A'].isin({'A': [1, 3], 'B': [4, 7, 12]})]
pd.Timestamp('9/1/2016 10:05AM')
pd.Period('1/2016')
pd.Period('3/5/2016')
t1 = pd.Series(list('abc'), [pd.Timestamp('2016-09-01'), pd.Timestamp('2016-09-02'), pd.Timestamp('2016-09-03')])
t1
type(t1.index)
t2 = pd.Series(list('def'), [pd.Period('2016-09'), pd.Period('2016-10'), pd.Period('2016-11')])
t2
type(t2.index)
d1 = ['2 June 2013', 'Aug 29, 2014', '2015-06-26', '7/12/16']
ts3 = pd.DataFrame(np.random.randint(10, 100, (4,2)), index=d1, columns=list('ab'))
ts3
ts3.index = pd.to_datetime(ts3.index)
ts3
pd.to_datetime('4.7.12', dayfirst=True)
pd.Timestamp('9/3/2016')-pd.Timestamp('9/1/2016')
pd.Timestamp('9/3/2016')-pd.Timestamp('9/1/2016')
pd.Timestamp('9/2/2016 8:10AM') + pd.Timedelta('12D 3H')
dates = pd.date_range('10-01-2016', periods=9, freq='2W-SUN')
dates
df.index.ravel