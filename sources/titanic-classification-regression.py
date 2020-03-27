
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt;
import seaborn as sns
#%matplotlib inline
train = pd.read_csv('../input/train.csv')
train.tail()
sns.heatmap(train.isnull(),yticklabels=False,cbar=False,cmap='viridis')
sns.set_style('whitegrid')
sns.countplot(x='Survived',data=train,palette='RdBu_r')
sns.set_style('whitegrid')
sns.countplot(x='Survived',hue='Sex',data=train,palette='RdBu_r')
sns.set_style('whitegrid')
sns.countplot(x='Survived',hue='Pclass',data=train,palette='rainbow')
sns.distplot(train['Age'].dropna(),kde=False,color='darkred',bins=30)
train['Age'].hist(bins=30,color='darkred',alpha=0.7)
sns.countplot(x='SibSp',data=train)
train['Fare'].hist(color='green',bins=40,figsize=(8,4))
plt.figure(figsize=(12, 7))
sns.boxplot(x='Pclass',y='Age',data=train,palette='winter')
def impute_age(cols):
    Age = cols[0]
    Pclass = cols[1]
    
    if pd.isnull(Age):

        if Pclass == 1:
            return 37

        elif Pclass == 2:
            return 29

        else:
            return 24

    else:
        return Age
train['Age'] = train[['Age','Pclass']].apply(impute_age,axis=1)
sns.heatmap(train.isnull(),yticklabels=False,cbar=False,cmap='viridis')
train.drop('Cabin',axis=1,inplace=True)
train.head()
train.dropna(inplace=True)
train.info()
sex = pd.get_dummies(train['Sex'],drop_first=True)
embark = pd.get_dummies(train['Embarked'],drop_first=True)
train.drop(['Sex','Embarked','Name','Ticket'],axis=1,inplace=True)
train = pd.concat([train,sex,embark],axis=1)
train.head()
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(train.drop('Survived',axis=1), 
                                                    train['Survived'], test_size=0.20, 
                                                    random_state=101)
from sklearn.linear_model import LogisticRegression
logmodel = LogisticRegression()
logmodel.fit(X_train,y_train)
predictions = logmodel.predict(X_test)
X_test.head()
predictions
from sklearn.metrics import classification_report,confusion_matrix
print(confusion_matrix(y_test,predictions))
print(classification_report(y_test,predictions))
from sklearn.tree import DecisionTreeClassifier
dt_model=DecisionTreeClassifier()
dt_model.fit(X_train,y_train)
dt_pred = dt_model.predict(X_test)
print(confusion_matrix(y_test,dt_pred))
print(classification_report(y_test,dt_pred))
from sklearn.ensemble import RandomForestClassifier
rf= RandomForestClassifier(n_estimators=5)
rf.fit(X_train,y_train)
rf_pre=rf.predict(X_test)
print(confusion_matrix(y_test,rf_pre))
print(classification_report(y_test,rf_pre))
test = pd.read_csv('../input/test.csv')
sns.heatmap(test.isnull())
test.drop('Cabin',axis=1,inplace=True)
test['Fare'].fillna(test['Fare'].mean(), inplace=True)
test.info()
test.head()
test['Age'] = test[['Age','Pclass']].apply(impute_age,axis=1)
sex_test = pd.get_dummies(test['Sex'],drop_first=True)
embark_test= pd.get_dummies(test['Embarked'],drop_first=True)
test.drop(['Sex','Embarked','Name','Ticket'],axis=1,inplace=True)
test = pd.concat([test,sex_test,embark_test],axis=1)
test.head()
test_prediction = rf.predict(test)
test_prediction.shape
test_pred = pd.DataFrame(test_prediction, columns= ['Survived'])
new_test = pd.concat([test, test_pred], axis=1, join='inner')
new_test.head()
df= new_test[['PassengerId' ,'Survived']]
df.to_csv('predictions.csv' , index=False)