
# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA, TruncatedSVD
import matplotlib.patches as mpatches
import time

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
import collections

from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from imblearn.pipeline import make_pipeline as imbalanced_make_pipeline
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import NearMiss
from imblearn.metrics import classification_report_imbalanced
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, accuracy_score, classification_report
from collections import Counter
from sklearn.model_selection import KFold, StratifiedKFold


# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.
dataframe=pd.read_csv('../input/creditcardfraud/creditcard.csv')
dataframe.head(5)
dataframe['Class'].describe()
dataframe.isnull().sum().max()
dataframe.columns
print(' No Fraud' ,round(dataframe['Class'].value_counts()[0]/len(dataframe)*100,2))
print('Fraud' ,round(dataframe['Class'].value_counts()[1]/len(dataframe)*100,2))


color=["blue","green"]
sns.countplot('Class',data=dataframe,palette=color)
plt.title("Non Fraud and  Fraud", fontsize=14)






fig,ax=plt.subplots(1,2,figsize=(18,4))
amount_value=dataframe['Amount'].values
time_value=dataframe['Time'].values

sns.distplot(amount_value,ax=ax[0],color='r')
ax[0].set_title('Amount value analysis', fontsize=14)
ax[0].set_xlim([min(amount_value),max(amount_value)])


sns.distplot(time_value,ax=ax[1],color='b')
ax[1].set_title('Time value analysis', fontsize=14)
ax[1].set_xlim([min(time_value),max(time_value)])

from sklearn.preprocessing import  RobustScaler

rob_scaler=RobustScaler()

dataframe['scaled_amount'] = rob_scaler.fit_transform(dataframe['Amount'].values.reshape(-1,1))
dataframe['scaled_time'] = rob_scaler.fit_transform(dataframe['Time'].values.reshape(-1,1))

dataframe.drop(['Time','Amount'], axis=1, inplace =True)
dataframe.head(10)
scaled_amount = dataframe['scaled_amount']
scaled_time = dataframe['scaled_time']

dataframe.drop(['scaled_amount', 'scaled_time'], axis=1, inplace=True)
dataframe.insert(0, 'scaled_amount', scaled_amount)
dataframe.insert(1, 'scaled_time', scaled_time)

dataframe.head(5)
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedShuffleSplit
print('No Frauds', round(dataframe['Class'].value_counts()[0]/len(dataframe) * 100,2), '% of the dataset')
print('Frauds', round(dataframe['Class'].value_counts()[1]/len(dataframe) * 100,2), '% of the dataset')
X = dataframe.drop('Class', axis=1)
y = dataframe['Class']

sss = StratifiedKFold(n_splits=5, random_state=None, shuffle=False)
for train_index, test_index in sss.split(X, y):
    print("Train:", train_index, "Test:", test_index)
    original_Xtrain, original_Xtest = X.iloc[train_index], X.iloc[test_index]
    original_ytrain, original_ytest = y.iloc[train_index], y.iloc[test_index]
fraud=dataframe.loc[dataframe['Class']==1]
non_fraud=dataframe.loc[dataframe['Class']==0][:492]
new_df=pd.concat([fraud,non_fraud])
new_df=new_df.sample(frac=1,random_state=42)
fraud['Class'].value_counts()
print(new_df['Class'].value_counts()/len(new_df))
sns.countplot('Class',data=new_df,palette=color)
plt.title('Equally Distributed Dataset for Fraud and Non Fraud')
plt.show()
f,(ax1,ax2)=plt.subplots(2,1,figsize=(24,20))


corr=dataframe.corr()
sns.heatmap(corr,cmap='coolwarm_r', annot_kws={'size':20},ax=ax1)


new_corr_df=new_df.corr()
sns.heatmap(new_corr_df,cmap='coolwarm_r', annot_kws={'size':20},ax=ax2)

plt.show()
f,axes=plt.subplots(ncols=5,figsize=(8,8))
sns.boxplot(x='Class',y='V1',data=new_df,palette=color,ax=axes[0])
sns.boxplot(x='Class',y='V10',data=new_df,palette=color,ax=axes[1])
sns.boxplot(x='Class',y='V8',data=new_df,palette=color,ax=axes[2])
sns.boxplot(x='Class',y='V12',data=new_df,palette=color,ax=axes[3])
sns.boxplot(x='Class',y='V14',data=new_df,palette=color,ax=axes[4])
plt.show()

from sklearn.linear_model import LogisticRegression
LR=LogisticRegression()
yhat=LR.fit(x_train,y_train)
