
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt # this is used for the plot the graph 
import seaborn as sns # used for plot interactive graph.

import matplotlib.pyplot as plt
import plotly.offline as py
py.init_notebook_mode(connected=True)
import plotly.graph_objs as go

import warnings
warnings.filterwarnings('ignore')
from pylab import rcParams
# figure size in inches

### matplotlib inline
## Read file

data = pd.read_csv("../input/googleplaystore.csv")
data.head()
print(data.shape)
#missing data
total = data.isnull().sum().sort_values(ascending=False)
percent = (data.isnull().sum()/data.isnull().count()).sort_values(ascending=False)
missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
missing_data.head(6)
data.dropna(how ='any', inplace = True)
#missing data
total = data.isnull().sum().sort_values(ascending=False)
percent = (data.isnull().sum()/data.isnull().count()).sort_values(ascending=False)
missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
missing_data.head(6)
print(data.shape)
data['Rating'].describe()
# rating distibution 
rcParams['figure.figsize'] = 11.7,8.27
g = sns.kdeplot(data.Rating, color="Red", shade = True)
g.set_xlabel("Rating")
g.set_ylabel("Frequency")
plt.title('Distribution of Rating',size = 20)
print( len(data['Category'].unique()) , "categories")

print("\n", data['Category'].unique())
g = sns.countplot(x="Category",data=data, palette = "Set1")
g.set_xticklabels(g.get_xticklabels(), rotation=90, ha="right")
g 
plt.title('Count of app in each category',size = 20)
g = sns.catplot(x="Category",y="Rating",data=data, kind="box", height = 10 ,
palette = "Set1")
g.despine(left=True)
g.set_xticklabels(rotation=90)
g.set( xticks=range(0,34))
g = g.set_ylabels("Rating")
plt.title('Boxplot of Rating VS Category',size = 20)
data['Reviews'].head()
# convert to int

data['Reviews'] = data['Reviews'].apply(lambda x: int(x))
# rating distibution 
rcParams['figure.figsize'] = 11.7,8.27
g = sns.kdeplot(data.Reviews, color="Green", shade = True)
g.set_xlabel("Reviews")
g.set_ylabel("Frequency")
plt.title('Distribution of Reveiw',size = 20)
data[data.Reviews > 5000000].head()
plt.figure(figsize = (10,10))
g = sns.jointplot(x="Reviews", y="Rating",color = 'orange', data=data,size = 8);
plt.figure(figsize = (10,10))
sns.regplot(x="Reviews", y="Rating", color = 'darkorange',data=data[data['Reviews']<1000000]);
plt.title('Rating VS Reveiws',size = 20)
data['Size'].head()
data['Size'].unique()
len(data[data.Size == 'Varies with device'])
# change it to NA first

data['Size'].replace('Varies with device', np.nan, inplace = True ) 
data.Size = (data.Size.replace(r'[kM]+$', '', regex=True).astype(float) * \
             data.Size.str.extract(r'[\d\.]+([KM]+)', expand=False)
            .fillna(1)
            .replace(['k','M'], [10**3, 10**6]).astype(int))
data['Size'].fillna(data.groupby('Category')['Size'].transform('mean'),inplace = True)
plt.figure(figsize = (10,10))
g = sns.jointplot(x="Size", y="Rating",color = 'orangered', data=data, size = 8);
data['Installs'].head()
data['Installs'].unique()
data.Installs = data.Installs.apply(lambda x: x.replace(',',''))
data.Installs = data.Installs.apply(lambda x: x.replace('+',''))
data.Installs = data.Installs.apply(lambda x: int(x))
data['Installs'].unique()
Sorted_value = sorted(list(data['Installs'].unique()))
data['Installs'].replace(Sorted_value,range(0,len(Sorted_value),1), inplace = True )
data['Installs'].head()
plt.figure(figsize = (10,10))
sns.regplot(x="Installs", y="Rating", color = 'teal',data=data);
plt.title('Rating VS Installs',size = 20)
data['Type'].unique()
# Data to plot
labels =data['Type'].value_counts(sort = True).index
sizes = data['Type'].value_counts(sort = True)


colors = ["palegreen","orangered"]
explode = (0.1,0)  # explode 1st slice
 
rcParams['figure.figsize'] = 8,8
# Plot
plt.pie(sizes, explode=explode, labels=labels, colors=colors,
        autopct='%1.1f%%', shadow=True, startangle=270,)

plt.title('Percent of Free App in store',size = 20)
plt.show()
data['Free'] = data['Type'].map(lambda s :1  if s =='Free' else 0)
data.drop(['Type'], axis=1, inplace=True)
data['Price'].head()
data.Price.unique()
data['Price'].value_counts().head(30)
data.Price = data.Price.apply(lambda x: x.replace('$',''))
data['Price'] = data['Price'].apply(lambda x: float(x))
data['Price'].describe()
data[data['Price'] == 400]
plt.figure(figsize = (10,10))
sns.regplot(x="Price", y="Rating", color = 'darkorange',data=data[data['Reviews']<1000000]);
plt.title('Scatter plot Rating VS Price',size = 20)
data.loc[ data['Price'] == 0, 'PriceBand'] = '0 Free'
data.loc[(data['Price'] > 0) & (data['Price'] <= 0.99), 'PriceBand'] = '1 cheap'
data.loc[(data['Price'] > 0.99) & (data['Price'] <= 2.99), 'PriceBand']   = '2 not cheap'
data.loc[(data['Price'] > 2.99) & (data['Price'] <= 4.99), 'PriceBand']   = '3 normal'
data.loc[(data['Price'] > 4.99) & (data['Price'] <= 14.99), 'PriceBand']   = '4 expensive'
data.loc[(data['Price'] > 14.99) & (data['Price'] <= 29.99), 'PriceBand']   = '5 too expensive'
data.loc[(data['Price'] > 29.99), 'PriceBand']  = '6 FXXXing expensive'

data[['PriceBand', 'Rating']].groupby(['PriceBand'], as_index=False).mean()
g = sns.catplot(x="PriceBand",y="Rating",data=data, kind="boxen", height = 10 ,palette = "Pastel1")
g.despine(left=True)
g.set_xticklabels(rotation=90)
g = g.set_ylabels("Rating")
plt.title('Boxen plot Rating VS PriceBand',size = 20)
import random

def generate_color():
    color = '#{:02x}{:02x}{:02x}'.format(*map(lambda x: random.randint(0, 255), range(3)))
    return color
# Create palette for categories

flatui = []
for i in range(0,len(data['Category'].unique()),1):
    flatui.append(generate_color())
g = sns.catplot(x="PriceBand", y="Rating", hue="Category", kind="swarm", data=data,palette = flatui,size = 10)
g.despine(left=True)
g.set_xticklabels(rotation=90)
plt.title('Category in each Priceband VS Rating',size = 20)
data['Content Rating'].unique()
g = sns.catplot(x="Content Rating",y="Rating",data=data, kind="box", height = 10 ,palette = "Paired")
g.despine(left=True)
g.set_xticklabels(rotation=90)
g = g.set_ylabels("Rating")
plt.title('Box plot Rating VS Content Rating',size = 20)
data[data['Content Rating']=='Unrated']
data = data[data['Content Rating'] != 'Unrated']
data = pd.get_dummies(data, columns= ["Content Rating"])
print( len(data['Genres'].unique()) , "genres")

print("\n", data['Genres'].unique())
data.Genres.value_counts().head(10)
data.Genres.value_counts().tail(10)
data['Genres'] = data['Genres'].str.split(';').str[0]
print( len(data['Genres'].unique()) , "genres")

print("\n", data['Genres'].unique())

data.Genres.value_counts().tail(10)
data['Genres'].replace('Music & Audio', 'Music',inplace = True)
data[['Genres', 'Rating']].groupby(['Genres'], as_index=False).mean().describe()
data[['Genres', 'Rating']].groupby(['Genres'], as_index=False).mean().sort_values('Rating').head(1)
data[['Genres', 'Rating']].groupby(['Genres'], as_index=False).mean().sort_values('Rating').tail(1)
g = sns.catplot(x="Genres",y="Rating",data=data, kind="boxen", height = 10 ,palette = "Paired")
g.despine(left=True)
g.set_xticklabels(rotation=90)
g = g.set_ylabels("Rating")
plt.title('Boxenplot of Rating VS Genres',size = 20)
data['Last Updated'].head()
data['new'] = pd.to_datetime(data['Last Updated'])
data['new'].describe()
# new format of  Last Updated
data['new'].max() 
# Example of finding difference between two dates in pandas
data['new'][0] -  data['new'].max()
data['lastupdate'] = (data['new'] -  data['new'].max()).dt.days
data['lastupdate'].head()
plt.figure(figsize = (10,10))
sns.regplot(x="lastupdate", y="Rating", color = 'lightpink',data=data );
plt.title('Rating  VS Last Update( days ago )',size = 20)

data.head()