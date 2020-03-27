
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
import pandas as pd
sample_submission = pd.read_csv("../input/cat-in-the-dat/sample_submission.csv")
test = pd.read_csv("../input/cat-in-the-dat/test.csv")
train = pd.read_csv("../input/cat-in-the-dat/train.csv")
# Subset
target = train['target']
train_id = train['id']
test_id = test['id']
train.drop(['target', 'id'], axis=1, inplace=True)
test.drop('id', axis=1, inplace=True)

print(train.shape)
print(test.shape)
%%time
# One Hot Encoding
traintest = pd.concat([train, test])
dummies = pd.get_dummies(traintest, columns=traintest.columns, drop_first=True, sparse=True)
train_ohe = dummies.iloc[:train.shape[0], :]
test_ohe = dummies.iloc[train.shape[0]:, :]

print(train_ohe.shape)
print(test_ohe.shape)
%%time
# to sparse Matrix
train_ohe = train_ohe.sparse.to_coo().tocsr()
test_ohe = test_ohe.sparse.to_coo().tocsr()
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
c_values = np.logspace(-2, 3, 500)
grid = {
    'C': c_values,'solver': ['newton-cg']
}
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=17)
clf = LogisticRegression()
gs = GridSearchCV(clf, grid, scoring='roc_auc', cv=skf)
gs.fit(train_ohe, target)
from sklearn.metrics import roc_auc_score
print("GridSearch roc score:")
roc_auc_score(y_test,gs.predict_proba(X_test)[:,1])
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(train_ohe,target,test_size=0.33, random_state=17)
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegressionCV
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=17)
c_values = np.logspace(-2, 3, 500)
logit_searcher = LogisticRegressionCV(Cs=c_values, cv=skf,verbose=1,n_jobs=-1)
%%time
# fit on full data
logit_searcher.fit(train_ohe, target)
#from sklearn.metrics import roc_auc_score
#y_pred = pd.Series(logit_searcher.predict_proba(X_test)[:,1])
#roc_auc_score(y_test,y_pred)
y_test_pred = logit_searcher.predict_proba(test_ohe)
from sklearn.metrics import roc_auc_score
print("RidgeClassifier roc score:")
roc_auc_score(y_test,logit_searcher.predict_proba(X_test)[:,1])
result = pd.DataFrame()
result['id'] = test_id
result['target'] = y_test_pred[:,1]
result = result.set_index(result.id)
result = result.drop('id', axis = 1)
#result.head()
result.to_csv('result4.csv')