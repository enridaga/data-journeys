
# Import Libraries
import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
### matplotlib inline
import seaborn as sns
# Load Data
train = pd.read_csv('../input/titanic/train.csv')
test = pd.read_csv('../input/titanic/test.csv')
# Concatenate train & test
train_objs_num = len(train)
y = train['Survived']
dataset = pd.concat(objs=[train.drop(columns=['Survived']), test], axis=0)
dataset.info()
total = dataset.isnull().sum().sort_values(ascending=False)
percent = (dataset.isnull().sum()/dataset.isnull().count()).sort_values(ascending=False)
missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
f, ax = plt.subplots(figsize=(15, 6))
plt.xticks(rotation='90')
sns.barplot(x=missing_data.index, y=missing_data['Percent'])
plt.xlabel('Features', fontsize=15)
plt.ylabel('Percent of missing values', fontsize=15)
plt.title('Percent missing data by feature', fontsize=15)
missing_data.head()
dataset.head()
# Plot histogram using seaborn
plt.figure(figsize=(15,8))
sns.distplot(dataset.Age, bins =30)
df = dataset
# Will drop all features with missing values 
df.dropna(inplace = True)
df.isnull().sum()
df1 = dataset
# Will drop the rows only if all of the values in the row are missing
df1.dropna(how = 'all',inplace = True)
df = dataset
# Will drop a feature that has some missing values.
df.dropna(axis = 1,inplace = True)
df = dataset
# Keep only the rows with at least 4 non-na values
df.dropna(thresh = 4,inplace = True)
df = dataset
#for back fill
df.fillna(method='bfill',inplace=True)
#for forward-fill
df.fillna(method='ffill',inplace=True)
# Replace with a constant value
# dataframe.Column_Name.fillna(-99,inplace=True)
df3 = train
df3['Age'].isnull().sum()
df3['Age'].mean()
df3['Age'].replace(np.NaN,df3['Age'].mean()).head(15)
df4 = train
df4['Age'].fillna(df4['Age'].median(),inplace=True)
df4.head()
data_cat=train
data_cat['Embarked'].fillna(data_cat['Embarked'].mode()[0], inplace=True)
data_cat.head()
data_unique = train
data_unique['Cabin'].head(10)
data_unique['Cabin'].fillna('U').head(10)