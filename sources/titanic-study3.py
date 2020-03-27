
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
train = pd.read_csv("/kaggle/input/titanic/train.csv")
test = pd.read_csv("/kaggle/input/titanic/test.csv")
gender_submission = pd.read_csv("/kaggle/input/titanic/gender_submission.csv")
y_train = train.Survived.values
train = train.drop("Survived", axis=1)
data = pd.concat([train, test])
data.head()
data.shape
def sep():
    print("-------------------------------------------")
print(data.isnull().sum())
sep()
print(data.nunique())
sep()
print(data.Embarked.value_counts())
data.Age.fillna(data.Age.median(), inplace=True)
data.Fare.fillna(data.Fare.median(), inplace=True)
data.Embarked.fillna(data.Embarked.mode()[0], inplace=True)
data.loc[data.Sex == "male", ["Sex"]] = 0
data.loc[data.Sex == "female", ["Sex"]] = 1
data["FamilySize"] = data.SibSp + data.Parch + 1
data["IsAlone"] = 0
data.loc[(data.SibSp + data.Parch) == 0, ["IsAlone"]] = 1

print(data.isnull().sum())
data.head()
df_Embarked = pd.get_dummies(data.Embarked)
data = pd.concat([data, df_Embarked], axis=1)
data.head()
data.drop(["Name", "Ticket", "Cabin", "Embarked"], axis=1, inplace=True)
data.head()
train_kai = data[:len(train)]
test_kai = data[len(train):]
X_train = train_kai[["Age", "Sex", "Fare", "FamilySize", "Pclass", "C", "Q", "S"]].values
X_test = test_kai[["Age", "Sex", "Fare", "FamilySize", "Pclass", "C", "Q", "S"]].values
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
#clf = LogisticRegression(random_state=0)
clf = RandomForestClassifier(n_estimators=100, max_depth=2, random_state=0)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
y_pred
sub = gender_submission
sub["Survived"] = y_pred
sub.to_csv("submission.csv", index=False)
