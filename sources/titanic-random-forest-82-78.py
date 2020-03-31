
import os

import pandas as pd

import numpy as np

import seaborn as sns

import matplotlib.pyplot as plt

### matplotlib inline
train = pd.read_csv(os.path.join('../input', 'train.csv'))

test = pd.read_csv(os.path.join('../input', 'test.csv'))
train.info()
train.head()
train['Survived'].value_counts(normalize=True)
sns.countplot(train['Survived'])
train['Survived'].groupby(train['Pclass']).mean()
sns.countplot(train['Pclass'], hue=train['Survived'])
train['Name'].head()
train['Name_Title'] = train['Name'].apply(lambda x: x.split(',')[1]).apply(lambda x: x.split()[0])

train['Name_Title'].value_counts()
train['Survived'].groupby(train['Name_Title']).mean()
train['Name_Len'] = train['Name'].apply(lambda x: len(x))

train['Survived'].groupby(pd.qcut(train['Name_Len'],5)).mean()
pd.qcut(train['Name_Len'],5).value_counts()
train['Sex'].value_counts(normalize=True)
train['Survived'].groupby(train['Sex']).mean()
train['Survived'].groupby(train['Age'].isnull()).mean()
train['Survived'].groupby(pd.qcut(train['Age'],5)).mean()
pd.qcut(train['Age'],5).value_counts()
train['Survived'].groupby(train['SibSp']).mean()
train['SibSp'].value_counts()
train['Survived'].groupby(train['Parch']).mean()
train['Parch'].value_counts()
train['Ticket'].head(n=10)
train['Ticket_Len'] = train['Ticket'].apply(lambda x: len(x))
train['Ticket_Len'].value_counts()
train['Ticket_Lett'] = train['Ticket'].apply(lambda x: str(x)[0])
train['Ticket_Lett'].value_counts()
train.groupby(['Ticket_Lett'])['Survived'].mean()
pd.qcut(train['Fare'], 3).value_counts()
train['Survived'].groupby(pd.qcut(train['Fare'], 3)).mean()
pd.crosstab(pd.qcut(train['Fare'], 5), columns=train['Pclass'])
train['Cabin_Letter'] = train['Cabin'].apply(lambda x: str(x)[0])
train['Cabin_Letter'].value_counts()
train['Survived'].groupby(train['Cabin_Letter']).mean()
train['Cabin_num'] = train['Cabin'].apply(lambda x: str(x).split(' ')[-1][1:])

train['Cabin_num'].replace('an', np.NaN, inplace = True)

train['Cabin_num'] = train['Cabin_num'].apply(lambda x: int(x) if not pd.isnull(x) and x != '' else np.NaN)
pd.qcut(train['Cabin_num'],3).value_counts()
train['Survived'].groupby(pd.qcut(train['Cabin_num'], 3)).mean()
train['Survived'].corr(train['Cabin_num'])
train['Embarked'].value_counts()
train['Embarked'].value_counts(normalize=True)
train['Survived'].groupby(train['Embarked']).mean()
sns.countplot(train['Embarked'], hue=train['Pclass'])
def names(train, test):

    for i in [train, test]:

        i['Name_Len'] = i['Name'].apply(lambda x: len(x))

        i['Name_Title'] = i['Name'].apply(lambda x: x.split(',')[1]).apply(lambda x: x.split()[0])

        del i['Name']

    return train, test
def age_impute(train, test):

    for i in [train, test]:

        i['Age_Null_Flag'] = i['Age'].apply(lambda x: 1 if pd.isnull(x) else 0)

        data = train.groupby(['Name_Title', 'Pclass'])['Age']

        i['Age'] = data.transform(lambda x: x.fillna(x.mean()))

    return train, test
def fam_size(train, test):

    for i in [train, test]:

        i['Fam_Size'] = np.where((i['SibSp']+i['Parch']) == 0 , 'Solo',

                           np.where((i['SibSp']+i['Parch']) <= 3,'Nuclear', 'Big'))

        del i['SibSp']

        del i['Parch']

    return train, test
def ticket_grouped(train, test):

    for i in [train, test]:

        i['Ticket_Lett'] = i['Ticket'].apply(lambda x: str(x)[0])

        i['Ticket_Lett'] = i['Ticket_Lett'].apply(lambda x: str(x))

        i['Ticket_Lett'] = np.where((i['Ticket_Lett']).isin(['1', '2', '3', 'S', 'P', 'C', 'A']), i['Ticket_Lett'],

                                   np.where((i['Ticket_Lett']).isin(['W', '4', '7', '6', 'L', '5', '8']),

                                            'Low_ticket', 'Other_ticket'))

        i['Ticket_Len'] = i['Ticket'].apply(lambda x: len(x))

        del i['Ticket']

    return train, test
def cabin(train, test):

    for i in [train, test]:

        i['Cabin_Letter'] = i['Cabin'].apply(lambda x: str(x)[0])

        del i['Cabin']

    return train, test
def cabin_num(train, test):

    for i in [train, test]:

        i['Cabin_num1'] = i['Cabin'].apply(lambda x: str(x).split(' ')[-1][1:])

        i['Cabin_num1'].replace('an', np.NaN, inplace = True)

        i['Cabin_num1'] = i['Cabin_num1'].apply(lambda x: int(x) if not pd.isnull(x) and x != '' else np.NaN)

        i['Cabin_num'] = pd.qcut(train['Cabin_num1'],3)

    train = pd.concat((train, pd.get_dummies(train['Cabin_num'], prefix = 'Cabin_num')), axis = 1)

    test = pd.concat((test, pd.get_dummies(test['Cabin_num'], prefix = 'Cabin_num')), axis = 1)

    del train['Cabin_num']

    del test['Cabin_num']

    del train['Cabin_num1']

    del test['Cabin_num1']

    return train, test
def embarked_impute(train, test):

    for i in [train, test]:

        i['Embarked'] = i['Embarked'].fillna('S')

    return train, test
test['Fare'].fillna(train['Fare'].mean(), inplace = True)
def dummies(train, test, columns = ['Pclass', 'Sex', 'Embarked', 'Ticket_Lett', 'Cabin_Letter', 'Name_Title', 'Fam_Size']):

    for column in columns:

        train[column] = train[column].apply(lambda x: str(x))

        test[column] = test[column].apply(lambda x: str(x))

        good_cols = [column+'_'+i for i in train[column].unique() if i in test[column].unique()]

        train = pd.concat((train, pd.get_dummies(train[column], prefix = column)[good_cols]), axis = 1)

        test = pd.concat((test, pd.get_dummies(test[column], prefix = column)[good_cols]), axis = 1)

        del train[column]

        del test[column]

    return train, test
def drop(train, test, bye = ['PassengerId']):

    for i in [train, test]:

        for z in bye:

            del i[z]

    return train, test
train = pd.read_csv(os.path.join('../input', 'train.csv'))

test = pd.read_csv(os.path.join('../input', 'test.csv'))

train, test = names(train, test)

train, test = age_impute(train, test)

train, test = cabin_num(train, test)

train, test = cabin(train, test)

train, test = embarked_impute(train, test)

train, test = fam_size(train, test)

test['Fare'].fillna(train['Fare'].mean(), inplace = True)

train, test = ticket_grouped(train, test)

train, test = dummies(train, test, columns = ['Pclass', 'Sex', 'Embarked', 'Ticket_Lett',

                                                                     'Cabin_Letter', 'Name_Title', 'Fam_Size'])

train, test = drop(train, test)
print(len(train.columns))
from sklearn.ensemble import RandomForestClassifier



rf = RandomForestClassifier(criterion='gini', 

                             n_estimators=700,

                             min_samples_split=10,

                             min_samples_leaf=1,

                             max_features='auto',

                             oob_score=True,

                             random_state=1,

                             n_jobs=-1)

rf.fit(train.iloc[:, 1:], train.iloc[:, 0])

print("%.4f" % rf.oob_score_)
pd.concat((pd.DataFrame(train.iloc[:, 1:].columns, columns = ['variable']), 

           pd.DataFrame(rf.feature_importances_, columns = ['importance'])), 

          axis = 1).sort_values(by='importance', ascending = False)[:20]
predictions = rf.predict(test)

predictions = pd.DataFrame(predictions, columns=['Survived'])

test = pd.read_csv(os.path.join('../input', 'test.csv'))

predictions = pd.concat((test.iloc[:, 0], predictions), axis = 1)

predictions.to_csv('y_test15.csv', sep=",", index = False)