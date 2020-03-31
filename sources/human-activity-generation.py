
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
import numpy as np 
import pandas as pd 

import matplotlib.pyplot as plt
import seaborn as sns
train = pd.read_csv('/kaggle/input/human-activity-recognition-with-smartphones/train.csv')
test = pd.read_csv('/kaggle/input/human-activity-recognition-with-smartphones/test.csv')
print(train.shape, test.shape)
print('Number of duplicates in train : ',sum(train.duplicated()))
print('Number of duplicates in test : ', sum(test.duplicated()))
print('Total number of missing values in train : ', train.isna().values.sum())
print('Total number of missing values in train : ', test.isna().values.sum())
plt.figure(figsize=(10,8))
plt.title('Barplot of Activity')
sns.countplot(train.Activity)
plt.xticks(rotation=90)
train.Activity.value_counts().index
plt.figure(figsize=(10,8))
for i in train.Activity.value_counts().index:
    sns.distplot(train[train['Activity']==i]['tBodyAccMag-mean()'],label=i)
plt.legend()
plt.show()
plt.figure(figsize=(10,8))
sns.boxplot(x='Activity', y='tBodyAccMag-mean()',data=train)

from sklearn.manifold import TSNE
X_for_tsne = train.drop(['subject', 'Activity'], axis=1)
### time
tsne = TSNE(random_state = 42, n_components=2, verbose=1, perplexity=50, n_iter=1000).fit_transform(X_for_tsne)
plt.figure(figsize=(12,8))
sns.scatterplot(x =tsne[:, 0], y = tsne[:, 1], hue = train["Activity"],palette="bright")
X_train = train.drop(['subject', 'Activity'], axis=1)
y_train = train.Activity
X_test = test.drop(['subject', 'Activity'], axis=1)
y_test = test.Activity
print('Training data size : ', X_train.shape)
print('Test data size : ', X_test.shape)
from sklearn. linear_model import LogisticRegression
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report

import warnings
warnings.filterwarnings("ignore")
parameters = {'C':np.arange(10,61,10), 'penalty':['l2','l1']}
lr_classifier = LogisticRegression()
lr_classifier_rs = RandomizedSearchCV(lr_classifier, param_distributions=parameters,random_state = 42)
lr_classifier_rs.fit(X_train, y_train)
y_pred = lr_classifier_rs.predict(X_test)
lr_accuracy = accuracy_score(y_true=y_test, y_pred=y_pred)
print("Accuracy using Logistic Regression : ", lr_accuracy)
labels=np.unique(y_pred)
labels
plt.figure(figsize=(10,10))
sns.heatmap(confusion_matrix(y_test.values,y_pred),annot=True,cmap='Blues',fmt='',xticklabels=labels,yticklabels=labels)
def get_best_randomsearch_results(model):
    print("Best estimator : ", model.best_estimator_)
    print("Best set of parameters : ", model.best_params_)
    print("Best score : ", model.best_score_)
# getting best random search attributes
get_best_randomsearch_results(lr_classifier_rs)
from sklearn.tree import DecisionTreeClassifier
parameters = {'max_depth':np.arange(2,10,2)}
dt_classifier = DecisionTreeClassifier()
dt_classifier_rs = RandomizedSearchCV(dt_classifier,param_distributions=parameters,random_state = 42)
dt_classifier_rs.fit(X_train, y_train)
y_pred = dt_classifier_rs.predict(X_test)
plt.figure(figsize=(10,10))
sns.heatmap(confusion_matrix(y_test.values,y_pred),annot=True,cmap='Blues',fmt='',xticklabels=labels,yticklabels=labels)
# getting best random search attributes
get_best_randomsearch_results(dt_classifier_rs)
from sklearn.ensemble import RandomForestClassifier
params = {'n_estimators': np.arange(20,101,10), 'max_depth':np.arange(2,16,2)}
rf_classifier = RandomForestClassifier()
rf_classifier_rs = RandomizedSearchCV(rf_classifier, param_distributions=params,random_state = 42)
rf_classifier_rs.fit(X_train, y_train)
y_pred = rf_classifier_rs.predict(X_test)
plt.figure(figsize=(10,10))
sns.heatmap(confusion_matrix(y_test.values,y_pred),annot=True,cmap='Blues',fmt='',xticklabels=labels,yticklabels=labels)
# getting best random search attributes
get_best_randomsearch_results(rf_classifier_rs)
