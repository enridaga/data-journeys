
import numpy as np 

import pandas as pd 



from sklearn import preprocessing

import matplotlib.pyplot as plt 

plt.rc("font", size=14)

import seaborn as sns

sns.set(style="white") #white background style for seaborn plots

sns.set(style="whitegrid", color_codes=True)
# Read CSV train data file into DataFrame

train_df = pd.read_csv("../input/train.csv")



# Read CSV test data file into DataFrame

test_df = pd.read_csv("../input/test.csv")



# preview train data

train_df.head()
print('The number of samples into the train data is {}.'.format(train_df.shape[0]))
# preview test data

test_df.head()
print('The number of samples into the test data is {}.'.format(test_df.shape[0]))
# check missing values in train data

train_df.isnull().sum()
# percent of missing "Age" 

print('Percent of missing "Age" records is %.2f%%' %((train_df['Age'].isnull().sum()/train_df.shape[0])*100))
ax = train_df["Age"].hist(bins=15, density=True, stacked=True, color='teal', alpha=0.6)

train_df["Age"].plot(kind='density', color='teal')

ax.set(xlabel='Age')

plt.xlim(-10,85)

plt.show()
# mean age

print('The mean of "Age" is %.2f' %(train_df["Age"].mean(skipna=True)))

# median age

print('The median of "Age" is %.2f' %(train_df["Age"].median(skipna=True)))
# percent of missing "Cabin" 

print('Percent of missing "Cabin" records is %.2f%%' %((train_df['Cabin'].isnull().sum()/train_df.shape[0])*100))
# percent of missing "Embarked" 

print('Percent of missing "Embarked" records is %.2f%%' %((train_df['Embarked'].isnull().sum()/train_df.shape[0])*100))
print('Boarded passengers grouped by port of embarkation (C = Cherbourg, Q = Queenstown, S = Southampton):')

print(train_df['Embarked'].value_counts())

sns.countplot(x='Embarked', data=train_df, palette='Set2')

plt.show()
print('The most common boarding port of embarkation is %s.' %train_df['Embarked'].value_counts().idxmax())
train_data = train_df.copy()

train_data["Age"].fillna(train_df["Age"].median(skipna=True), inplace=True)

train_data["Embarked"].fillna(train_df['Embarked'].value_counts().idxmax(), inplace=True)

train_data.drop('Cabin', axis=1, inplace=True)
# check missing values in adjusted train data

train_data.isnull().sum()
# preview adjusted train data

train_data.head()
plt.figure(figsize=(15,8))

ax = train_df["Age"].hist(bins=15, density=True, stacked=True, color='teal', alpha=0.6)

train_df["Age"].plot(kind='density', color='teal')

ax = train_data["Age"].hist(bins=15, density=True, stacked=True, color='orange', alpha=0.5)

train_data["Age"].plot(kind='density', color='orange')

ax.legend(['Raw Age', 'Adjusted Age'])

ax.set(xlabel='Age')

plt.xlim(-10,85)

plt.show()
## Create categorical variable for traveling alone

train_data['TravelAlone']=np.where((train_data["SibSp"]+train_data["Parch"])>0, 0, 1)

train_data.drop('SibSp', axis=1, inplace=True)

train_data.drop('Parch', axis=1, inplace=True)
#create categorical variables and drop some variables

training=pd.get_dummies(train_data, columns=["Pclass","Embarked","Sex"])

training.drop('Sex_female', axis=1, inplace=True)

training.drop('PassengerId', axis=1, inplace=True)

training.drop('Name', axis=1, inplace=True)

training.drop('Ticket', axis=1, inplace=True)



final_train = training

final_train.head()
test_df.isnull().sum()
test_data = test_df.copy()

test_data["Age"].fillna(train_df["Age"].median(skipna=True), inplace=True)

test_data["Fare"].fillna(train_df["Fare"].median(skipna=True), inplace=True)

test_data.drop('Cabin', axis=1, inplace=True)



test_data['TravelAlone']=np.where((test_data["SibSp"]+test_data["Parch"])>0, 0, 1)



test_data.drop('SibSp', axis=1, inplace=True)

test_data.drop('Parch', axis=1, inplace=True)



testing = pd.get_dummies(test_data, columns=["Pclass","Embarked","Sex"])

testing.drop('Sex_female', axis=1, inplace=True)

testing.drop('PassengerId', axis=1, inplace=True)

testing.drop('Name', axis=1, inplace=True)

testing.drop('Ticket', axis=1, inplace=True)



final_test = testing

final_test.head()
plt.figure(figsize=(15,8))

ax = sns.kdeplot(final_train["Age"][final_train.Survived == 1], color="darkturquoise", shade=True)

sns.kdeplot(final_train["Age"][final_train.Survived == 0], color="lightcoral", shade=True)

plt.legend(['Survived', 'Died'])

plt.title('Density Plot of Age for Surviving Population and Deceased Population')

ax.set(xlabel='Age')

plt.xlim(-10,85)

plt.show()
plt.figure(figsize=(20,8))

avg_survival_byage = final_train[["Age", "Survived"]].groupby(['Age'], as_index=False).mean()

g = sns.barplot(x='Age', y='Survived', data=avg_survival_byage, color="LightSeaGreen")

plt.show()
final_train['IsMinor']=np.where(final_train['Age']<=16, 1, 0)



final_test['IsMinor']=np.where(final_test['Age']<=16, 1, 0)
plt.figure(figsize=(15,8))

ax = sns.kdeplot(final_train["Fare"][final_train.Survived == 1], color="darkturquoise", shade=True)

sns.kdeplot(final_train["Fare"][final_train.Survived == 0], color="lightcoral", shade=True)

plt.legend(['Survived', 'Died'])

plt.title('Density Plot of Fare for Surviving Population and Deceased Population')

ax.set(xlabel='Fare')

plt.xlim(-20,200)

plt.show()
sns.barplot('Pclass', 'Survived', data=train_df, color="darkturquoise")

plt.show()
sns.barplot('Embarked', 'Survived', data=train_df, color="teal")

plt.show()
sns.barplot('TravelAlone', 'Survived', data=final_train, color="mediumturquoise")

plt.show()
sns.barplot('Sex', 'Survived', data=train_df, color="aquamarine")

plt.show()
from sklearn.linear_model import LogisticRegression

from sklearn.feature_selection import RFE



cols = ["Age","Fare","TravelAlone","Pclass_1","Pclass_2","Embarked_C","Embarked_S","Sex_male","IsMinor"] 

X = final_train[cols]

y = final_train['Survived']

# Build a logreg and compute the feature importances

model = LogisticRegression()

# create the RFE model and select 8 attributes

rfe = RFE(model, 8)

rfe = rfe.fit(X, y)

# summarize the selection of the attributes

print('Selected features: %s' % list(X.columns[rfe.support_]))
from sklearn.feature_selection import RFECV

# Create the RFE object and compute a cross-validated score.

# The "accuracy" scoring is proportional to the number of correct classifications

rfecv = RFECV(estimator=LogisticRegression(), step=1, cv=10, scoring='accuracy')

rfecv.fit(X, y)



print("Optimal number of features: %d" % rfecv.n_features_)

print('Selected features: %s' % list(X.columns[rfecv.support_]))



# Plot number of features VS. cross-validation scores

plt.figure(figsize=(10,6))

plt.xlabel("Number of features selected")

plt.ylabel("Cross validation score (nb of correct classifications)")

plt.plot(range(1, len(rfecv.grid_scores_) + 1), rfecv.grid_scores_)

plt.show()
Selected_features = ['Age', 'TravelAlone', 'Pclass_1', 'Pclass_2', 'Embarked_C', 

                     'Embarked_S', 'Sex_male', 'IsMinor']

X = final_train[Selected_features]



plt.subplots(figsize=(8, 5))

sns.heatmap(X.corr(), annot=True, cmap="RdYlGn")

plt.show()
from sklearn.model_selection import train_test_split, cross_val_score

from sklearn.metrics import accuracy_score, classification_report, precision_score, recall_score 

from sklearn.metrics import confusion_matrix, precision_recall_curve, roc_curve, auc, log_loss



# create X (features) and y (response)

X = final_train[Selected_features]

y = final_train['Survived']



# use train/test split with different random_state values

# we can change the random_state values that changes the accuracy scores

# the scores change a lot, this is why testing scores is a high-variance estimate

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=2)



# check classification scores of logistic regression

logreg = LogisticRegression()

logreg.fit(X_train, y_train)

y_pred = logreg.predict(X_test)

y_pred_proba = logreg.predict_proba(X_test)[:, 1]

[fpr, tpr, thr] = roc_curve(y_test, y_pred_proba)

print('Train/Test split results:')

print(logreg.__class__.__name__+" accuracy is %2.3f" % accuracy_score(y_test, y_pred))

print(logreg.__class__.__name__+" log_loss is %2.3f" % log_loss(y_test, y_pred_proba))

print(logreg.__class__.__name__+" auc is %2.3f" % auc(fpr, tpr))



idx = np.min(np.where(tpr > 0.95)) # index of the first threshold for which the sensibility > 0.95



plt.figure()

plt.plot(fpr, tpr, color='coral', label='ROC curve (area = %0.3f)' % auc(fpr, tpr))

plt.plot([0, 1], [0, 1], 'k--')

plt.plot([0,fpr[idx]], [tpr[idx],tpr[idx]], 'k--', color='blue')

plt.plot([fpr[idx],fpr[idx]], [0,tpr[idx]], 'k--', color='blue')

plt.xlim([0.0, 1.0])

plt.ylim([0.0, 1.05])

plt.xlabel('False Positive Rate (1 - specificity)', fontsize=14)

plt.ylabel('True Positive Rate (recall)', fontsize=14)

plt.title('Receiver operating characteristic (ROC) curve')

plt.legend(loc="lower right")

plt.show()



print("Using a threshold of %.3f " % thr[idx] + "guarantees a sensitivity of %.3f " % tpr[idx] +  

      "and a specificity of %.3f" % (1-fpr[idx]) + 

      ", i.e. a false positive rate of %.2f%%." % (np.array(fpr[idx])*100))
# 10-fold cross-validation logistic regression

logreg = LogisticRegression()

# Use cross_val_score function

# We are passing the entirety of X and y, not X_train or y_train, it takes care of splitting the data

# cv=10 for 10 folds

# scoring = {'accuracy', 'neg_log_loss', 'roc_auc'} for evaluation metric - althought they are many

scores_accuracy = cross_val_score(logreg, X, y, cv=10, scoring='accuracy')

scores_log_loss = cross_val_score(logreg, X, y, cv=10, scoring='neg_log_loss')

scores_auc = cross_val_score(logreg, X, y, cv=10, scoring='roc_auc')

print('K-fold cross-validation results:')

print(logreg.__class__.__name__+" average accuracy is %2.3f" % scores_accuracy.mean())

print(logreg.__class__.__name__+" average log_loss is %2.3f" % -scores_log_loss.mean())

print(logreg.__class__.__name__+" average auc is %2.3f" % scores_auc.mean())
from sklearn.model_selection import cross_validate



scoring = {'accuracy': 'accuracy', 'log_loss': 'neg_log_loss', 'auc': 'roc_auc'}



modelCV = LogisticRegression()



results = cross_validate(modelCV, X, y, cv=10, scoring=list(scoring.values()), 

                         return_train_score=False)



print('K-fold cross-validation results:')

for sc in range(len(scoring)):

    print(modelCV.__class__.__name__+" average %s: %.3f (+/-%.3f)" % (list(scoring.keys())[sc], -results['test_%s' % list(scoring.values())[sc]].mean()

                               if list(scoring.values())[sc]=='neg_log_loss' 

                               else results['test_%s' % list(scoring.values())[sc]].mean(), 

                               results['test_%s' % list(scoring.values())[sc]].std()))
cols = ["Age","Fare","TravelAlone","Pclass_1","Pclass_2","Embarked_C","Embarked_S","Sex_male","IsMinor"]

X = final_train[cols]



scoring = {'accuracy': 'accuracy', 'log_loss': 'neg_log_loss', 'auc': 'roc_auc'}



modelCV = LogisticRegression()



results = cross_validate(modelCV, final_train[cols], y, cv=10, scoring=list(scoring.values()), 

                         return_train_score=False)



print('K-fold cross-validation results:')

for sc in range(len(scoring)):

    print(modelCV.__class__.__name__+" average %s: %.3f (+/-%.3f)" % (list(scoring.keys())[sc], -results['test_%s' % list(scoring.values())[sc]].mean()

                               if list(scoring.values())[sc]=='neg_log_loss' 

                               else results['test_%s' % list(scoring.values())[sc]].mean(), 

                               results['test_%s' % list(scoring.values())[sc]].std()))
from sklearn.model_selection import GridSearchCV



X = final_train[Selected_features]



param_grid = {'C': np.arange(1e-05, 3, 0.1)}

scoring = {'Accuracy': 'accuracy', 'AUC': 'roc_auc', 'Log_loss': 'neg_log_loss'}



gs = GridSearchCV(LogisticRegression(), return_train_score=True,

                  param_grid=param_grid, scoring=scoring, cv=10, refit='Accuracy')



gs.fit(X, y)

results = gs.cv_results_



print('='*20)

print("best params: " + str(gs.best_estimator_))

print("best params: " + str(gs.best_params_))

print('best score:', gs.best_score_)

print('='*20)



plt.figure(figsize=(10, 10))

plt.title("GridSearchCV evaluating using multiple scorers simultaneously",fontsize=16)



plt.xlabel("Inverse of regularization strength: C")

plt.ylabel("Score")

plt.grid()



ax = plt.axes()

ax.set_xlim(0, param_grid['C'].max()) 

ax.set_ylim(0.35, 0.95)



# Get the regular numpy array from the MaskedArray

X_axis = np.array(results['param_C'].data, dtype=float)



for scorer, color in zip(list(scoring.keys()), ['g', 'k', 'b']): 

    for sample, style in (('train', '--'), ('test', '-')):

        sample_score_mean = -results['mean_%s_%s' % (sample, scorer)] if scoring[scorer]=='neg_log_loss' else results['mean_%s_%s' % (sample, scorer)]

        sample_score_std = results['std_%s_%s' % (sample, scorer)]

        ax.fill_between(X_axis, sample_score_mean - sample_score_std,

                        sample_score_mean + sample_score_std,

                        alpha=0.1 if sample == 'test' else 0, color=color)

        ax.plot(X_axis, sample_score_mean, style, color=color,

                alpha=1 if sample == 'test' else 0.7,

                label="%s (%s)" % (scorer, sample))



    best_index = np.nonzero(results['rank_test_%s' % scorer] == 1)[0][0]

    best_score = -results['mean_test_%s' % scorer][best_index] if scoring[scorer]=='neg_log_loss' else results['mean_test_%s' % scorer][best_index]

        

    # Plot a dotted vertical line at the best score for that scorer marked by x

    ax.plot([X_axis[best_index], ] * 2, [0, best_score],

            linestyle='-.', color=color, marker='x', markeredgewidth=3, ms=8)



    # Annotate the best score for that scorer

    ax.annotate("%0.2f" % best_score,

                (X_axis[best_index], best_score + 0.005))



plt.legend(loc="best")

plt.grid('off')

plt.show()
from sklearn.preprocessing import StandardScaler

from sklearn.model_selection import RepeatedStratifiedKFold

from sklearn.pipeline import Pipeline



#Define simple model

###############################################################################

C = np.arange(1e-05, 5.5, 0.1)

scoring = {'Accuracy': 'accuracy', 'AUC': 'roc_auc', 'Log_loss': 'neg_log_loss'}

log_reg = LogisticRegression()



#Simple pre-processing estimators

###############################################################################

std_scale = StandardScaler(with_mean=False, with_std=False)

#std_scale = StandardScaler()



#Defining the CV method: Using the Repeated Stratified K Fold

###############################################################################



n_folds=5

n_repeats=5



rskfold = RepeatedStratifiedKFold(n_splits=n_folds, n_repeats=n_repeats, random_state=2)



#Creating simple pipeline and defining the gridsearch

###############################################################################



log_clf_pipe = Pipeline(steps=[('scale',std_scale), ('clf',log_reg)])



log_clf = GridSearchCV(estimator=log_clf_pipe, cv=rskfold,

              scoring=scoring, return_train_score=True,

              param_grid=dict(clf__C=C), refit='Accuracy')



log_clf.fit(X, y)

results = log_clf.cv_results_



print('='*20)

print("best params: " + str(log_clf.best_estimator_))

print("best params: " + str(log_clf.best_params_))

print('best score:', log_clf.best_score_)

print('='*20)



plt.figure(figsize=(10, 10))

plt.title("GridSearchCV evaluating using multiple scorers simultaneously",fontsize=16)



plt.xlabel("Inverse of regularization strength: C")

plt.ylabel("Score")

plt.grid()



ax = plt.axes()

ax.set_xlim(0, C.max()) 

ax.set_ylim(0.35, 0.95)



# Get the regular numpy array from the MaskedArray

X_axis = np.array(results['param_clf__C'].data, dtype=float)



for scorer, color in zip(list(scoring.keys()), ['g', 'k', 'b']): 

    for sample, style in (('train', '--'), ('test', '-')):

        sample_score_mean = -results['mean_%s_%s' % (sample, scorer)] if scoring[scorer]=='neg_log_loss' else results['mean_%s_%s' % (sample, scorer)]

        sample_score_std = results['std_%s_%s' % (sample, scorer)]

        ax.fill_between(X_axis, sample_score_mean - sample_score_std,

                        sample_score_mean + sample_score_std,

                        alpha=0.1 if sample == 'test' else 0, color=color)

        ax.plot(X_axis, sample_score_mean, style, color=color,

                alpha=1 if sample == 'test' else 0.7,

                label="%s (%s)" % (scorer, sample))



    best_index = np.nonzero(results['rank_test_%s' % scorer] == 1)[0][0]

    best_score = -results['mean_test_%s' % scorer][best_index] if scoring[scorer]=='neg_log_loss' else results['mean_test_%s' % scorer][best_index]

        

    # Plot a dotted vertical line at the best score for that scorer marked by x

    ax.plot([X_axis[best_index], ] * 2, [0, best_score],

            linestyle='-.', color=color, marker='x', markeredgewidth=3, ms=8)



    # Annotate the best score for that scorer

    ax.annotate("%0.2f" % best_score,

                (X_axis[best_index], best_score + 0.005))



plt.legend(loc="best")

plt.grid('off')

plt.show()
final_test['Survived'] = log_clf.predict(final_test[Selected_features])

final_test['PassengerId'] = test_df['PassengerId']



submission = final_test[['PassengerId','Survived']]



submission.to_csv("submission.csv", index=False)



submission.tail()