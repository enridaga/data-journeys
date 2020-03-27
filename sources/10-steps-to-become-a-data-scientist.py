
import matplotlib.animation as animation
from matplotlib.figure import Figure
import plotly.figure_factory as ff
import matplotlib.pylab as pylab
from ipywidgets import interact
import plotly.graph_objs as go
import plotly.offline as py
from random import randint
from plotly import tools
import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib
import warnings
import string
import numpy
import csv
import os
print(os.listdir("../input/"))
titanic_train=pd.read_csv('../input/train.csv')
titanic_test=pd.read_csv('../input/test.csv')
titanic_train2=pd.read_csv('../input/train.csv',nrows=1000)
 
print("Train: rows:{} columns:{}".format(titanic_train.shape[0], titanic_train.shape[1]))
titanic_train.isna().sum()
total = titanic_train.isnull().sum().sort_values(ascending=False)
percent = (titanic_train.isnull().sum()/titanic_train.isnull().count()).sort_values(ascending=False)
missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
missing_data.head(20)
titanic_train.describe()
titanic_train['Age'].describe()
titanic_train.Age.describe()
titanic_train.columns
titanic_train.head()
titanic_train.sample(5)
titanic_train.sample(frac=0.007)
PassengerId=titanic_train['PassengerId'].copy()

PassengerId.head()
type(PassengerId)
titanic_train=titanic_train.drop('PassengerId',1)
titanic_train.head()
titanic_train=pd.read_csv('../input/train.csv')
titanic_train.tail() 
all_data = pd.concat((titanic_train.loc[:,'Pclass':'Embarked'],
                      titanic_test.loc[:,'Pclass':'Embarked']))
all_data.head()
titanic_train.shape
titanic_test.shape
all_data.shape
titanic_train['Sex'].unique()

titanic_train['Cabin'].unique()

titanic_train['Pclass'].unique()

titanic_train[titanic_train['Age']>70]
titanic_train[titanic_train['Pclass']==1]