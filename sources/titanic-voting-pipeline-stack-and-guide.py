
# General Packages
import pandas as pd
import numpy as np
import random as rnd
import os
import re
import itertools
# import multiprocessing

# Visualization
import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')
plt.rcParams['figure.figsize'] = (8, 6)
import scikitplot as skplt

# Supervised Machine Learning Models
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, BaggingClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import Perceptron, SGDClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.neural_network import MLPClassifier
from sklearn import feature_selection
import xgboost as xgb # XGBOOST
from xgboost.sklearn import XGBClassifier # XGBOOST
import hyperopt #CatBoost
from catboost import Pool, CatBoostClassifier #CatBoost
import lightgbm as lgb # Light GBM
import statsmodels.api as sm # Logistic Regression with StatModels

# Unsupervised Models
from sklearn.decomposition import PCA

# Evalaluation
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.model_selection import cross_val_score
from sklearn.metrics import confusion_matrix, roc_curve, auc, accuracy_score

# Grid
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.feature_selection import RFE
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV
import scipy.stats as st

# Pipeline
from sklearn.pipeline import make_pipeline
from sklearn.pipeline import Pipeline

# Esemble Voting
from mlxtend.classifier import EnsembleVoteClassifier
from sklearn import metrics
from sklearn.metrics import classification_report, accuracy_score

# Stacking
from sklearn.preprocessing import StandardScaler
from sklearn import metrics
from matplotlib.colors import ListedColormap

# Warnings
import warnings
warnings.filterwarnings(action='ignore')
warnings.filterwarnings(action='ignore', category=DeprecationWarning)
warnings.filterwarnings(action='ignore', category=FutureWarning)

import time
import datetime
import platform
start = time.time()
print('Version      :', platform.python_version())
print('Compiler     :', platform.python_compiler())
print('Build        :', platform.python_build())

print("\nCurrent date and time using isoformat:")
print(datetime.datetime.now().isoformat())
# Write model submissions?
save = True
n_row = None
debug = False
if debug is True: n_row = 80

# Master Parameters:
n_splits = 2 # Cross Validation Splits
n_iter = 70 # Randomized Search Iterations
scoring = 'accuracy' # Model Selection during Cross-Validation
rstate = 27 # Random State used 
testset_size = 0.35

# Boosting rounds
num_rounds = 800

# Trees Parameters
n_tree_range = st.randint(100, num_rounds)
# Load
train_df = pd.read_csv("../input/train.csv", index_col='PassengerId', nrows=n_row)
test_df = pd.read_csv("../input/test.csv", index_col='PassengerId', nrows=n_row)
# train_df = pd.read_csv("Titanic Support/train.csv", index_col='PassengerId')
# test_df = pd.read_csv("Titanic Support/test.csv", index_col='PassengerId')

# For Pre-Processing, combine train/test to simultaneously apply transformations
Survived = train_df['Survived'].copy()
train_df = train_df.drop('Survived', axis=1).copy()
df = pd.concat([test_df, train_df])
traindex = train_df.index
testdex = test_df.index
del train_df
del test_df

# New Variables engineering, heavily influenced by:
# Kaggle Source- https://www.kaggle.com/arthurtok/introduction-to-ensembling-stacking-in-python
# Family Size
df['FamilySize'] = df['SibSp'] + df['Parch'] + 1
# Name Length
df['Name_length'] = df['Name'].apply(len)
# Is Alone?
df['IsAlone'] = 0
df.loc[df['FamilySize'] == 1, 'IsAlone'] = 1

# Title: (Source)
# Kaggle Source- https://www.kaggle.com/ash316/eda-to-prediction-dietanic
df['Title']=0
df['Title']=df.Name.str.extract('([A-Za-z]+)\.') #lets extract the Salutations
df['Title'].replace(['Mlle','Mme','Ms','Dr','Major','Lady','Countess','Jonkheer','Col','Rev','Capt','Sir','Don'],
                    ['Miss','Miss','Miss','Mr','Mr','Mrs','Mrs','Other','Other','Other','Mr','Mr','Mr'],inplace=True)

# Age
df.loc[(df.Age.isnull())&(df.Title=='Mr'),'Age']= df.Age[df.Title=="Mr"].mean()
df.loc[(df.Age.isnull())&(df.Title=='Mrs'),'Age']= df.Age[df.Title=="Mrs"].mean()
df.loc[(df.Age.isnull())&(df.Title=='Master'),'Age']= df.Age[df.Title=="Master"].mean()
df.loc[(df.Age.isnull())&(df.Title=='Miss'),'Age']= df.Age[df.Title=="Miss"].mean()
df.loc[(df.Age.isnull())&(df.Title=='Other'),'Age']= df.Age[df.Title=="Other"].mean()
df = df.drop('Name', axis=1)

# Fill NA
# Categoricals Variable
df['Embarked'] = df['Embarked'].fillna(df['Embarked'].mode().iloc[0])

# Continuous Variable
df['Fare'] = df['Fare'].fillna(df['Fare'].mean())

## Assign Binary to Sex str
df['Sex'] = df['Sex'].map({'female': 1,
                           'male': 0}).astype(int)
# Title
df['Title'] = df['Title'].map( {'Mr': 0, 'Mrs': 1, 'Miss': 2, 'Master':3, 'Other':4} )#.astype(int)
df['Title'] = df['Title'].fillna(df['Title'].mode().iloc[0])
df['Title'] = df['Title'].astype(int)

# Embarked
df['Embarked'] = df['Embarked'].map( {'Q': 0, 'S': 1, 'C': 2} ).astype(int)

# Get Rid of Ticket and Cabin Variable
df= df.drop(['Ticket', 'Cabin'], axis=1)

categorical_features = ["Pclass","Sex","IsAlone","Title", "Embarked"]
df.head()
# Histogram
pd.concat([df.loc[traindex, :], Survived], axis=1).hist()
plt.tight_layout(pad=0)
plt.show()
# Correlation Plot
f, ax = plt.subplots(figsize=[8,6])
sns.heatmap(pd.concat([df.loc[traindex, :], Survived], axis=1).corr(),
            annot=True, fmt=".2f",cbar_kws={'label': 'Percentage %'},cmap="plasma",ax=ax)
ax.set_title("Correlation Plot")
plt.show()
# Scaling between -1 and 1. Good practice for continuous variables.
from sklearn import preprocessing
continuous_features = ['Fare','Age','Name_length']
for col in continuous_features:
    transf = df[col].values.reshape(-1,1)
    scaler = preprocessing.StandardScaler().fit(transf)
    df[col] = scaler.transform(transf)

# Finish Pre-Processing
# Dummmy Variables (One Hot Encoding)
#df = pd.get_dummies(df, columns=['Embarked','Title','Parch','SibSp','Pclass'], )

# Now that pre-processing is complete, split data into train/test again.
train_df = df.loc[traindex, :]
train_df['Survived'] = Survived
test_df = df.loc[testdex, :]

# Dead Weight
del df
# Depedent and Indepedent Variables
X = train_df.drop(["Survived"] , axis=1)
y = train_df["Survived"]
print("X, Y, Test Shape:",X.shape, y.shape, test_df.shape) # Data Dimensions

# Storage for Model and Results
results = pd.DataFrame(columns=['Model','Para','Test_Score','CV Mean','CV STDEV'])
ensemble_models= {}
print("Depedent Variable Distribution")
print(y.value_counts(normalize=True)*100)
print("0 = Died", "\n1 = Survived")
# Calculating level of imbalance for future models.
imbalance_weight = y.value_counts(normalize=True)[0]/y.value_counts(normalize=True)[1]
print("Imbalance Weight: ",imbalance_weight)
print("Feature Count (With One Hot Encoding):", X.shape[1])
levels = [2,4,6,8,10,12]
for x in levels:
    pca = PCA(n_components=x)
    fit = pca.fit(train_df)

    print(("{} Components \nExplained Variance: {}\n").format(x, fit.explained_variance_ratio_))
    #print(fit.components_)
"""
# Reduce Dimensionality
pca = PCA(n_components=5)
fit = pca.fit(X)

sns.heatmap(pd.concat([pd.DataFrame(fit.transform(X)), Survived],
                      axis=1).corr(), annot=True, fmt=".2f")
# Apply Reduction
X = pd.DataFrame(fit.transform(X))
test_df = pd.DataFrame(fit.transform(test_df))

No longer applied to dataset since performance did not get substancially improved.
Better just use method at an individual basis, through the pipeline method.
"""
# Stratified Train/Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=testset_size, stratify=y,random_state=rstate)
X_train.shape, y_train.shape, X_test.shape, y_test.shape

# Stratified Cross-Validation
cv = StratifiedShuffleSplit(n_splits=n_splits, test_size=0.2, random_state=rstate)
# Compute, Print, and Save Model Evaluation
def save(model, modelname):
    """
    This funciton saves the cross-validation and held-out (Test Set) accuracy scores aswell as their standard deviations
    to the "results" dataframe, then it executes the model prediction on the submission set.
    Finally, it also outputs a confusion matrix of the test set.
    
    Function Arguments:
    model = The Sklearn Randomized/Grid SearchCV model.
    modelname = String of the model name for saving purposes.
    """
    global results
    # Once best model is found, establish more evaluation metrics.
    model.best_estimator_.fit(X_train, y_train)
    scores = cross_val_score(model.best_estimator_, X_train, y_train, cv=5,
                             scoring=scoring, verbose =0)
    CV_scores = scores.mean()
    STDev = scores.std()
    Test_scores = model.score(X_test, y_test)

    # CV and Save scores
    results = results.append({'Model': modelname,'Para': model.best_params_,'Test_Score': Test_scores,
                             'CV Mean':CV_scores, 'CV STDEV': STDev}, ignore_index=True)
    ensemble_models[modelname] = model.best_estimator_
    
    # Print Evaluation
    print("\nEvaluation Method: {}".format(scoring))
    print("Optimal Model Parameters: {}".format(grid.best_params_))
    print("Train CV Accuracy: %0.2f (+/- %0.2f) [%s]" % (CV_scores, STDev, modelname))
    print('Test_Score:', Test_scores)
        
    # Scikit Confusion Matrix
    model.best_estimator_.fit(X_train, y_train)
    pred = model.predict(X_test)
    skplt.metrics.plot_confusion_matrix(y_test, pred, title="{} Confusion Matrix".format(modelname),
                normalize=True,figsize=(6,6),text_fontsize='large')
    plt.show()
    # Colors https://matplotlib.org/examples/color/colormaps_reference.html

def norm_save(model,score, modelname):
    global results
    model.fit(X, y)
    submission = model.predict(test_df)
    df = pd.DataFrame({'PassengerId':test_df.index, 
                           'Survived':submission})
    
    CV_Score = score.mean()
    Test_scores = model.score(X_test, y_test)
    STDev = score.std()
    
    # CV and Save Scores
    Test_Score = model.score(X_test, y_test)
    results = results.append({'Model': modelname,'Para': model,'Test_Score': Test_scores,
                             'CV Mean': CV_Score, 'CV STDEV': STDev}, ignore_index=True)
    ensemble_models[modelname] = model
    
    print("\nEvaluation Method: {}".format(scoring))
    print("Train CV Accuracy: %0.2f (+/- %0.2f) [%s]" % (CV_Score, STDev, modelname))  
    print('Test_Score:', Test_scores)
        
    #Scikit Confusion Matrix
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    skplt.metrics.plot_confusion_matrix(y_test, pred, title="{} Confusion Matrix".format(modelname),
                normalize=True,figsize=(6,6),text_fontsize='large')
    plt.show()
    
# ROC Curve Plot
# http://scikit-plot.readthedocs.io/en/stable/metrics.html
def eval_plot(model):
    skplt.metrics.plot_roc_curve(y_test, model.predict_proba(X_test))
    plt.show()
# Hyper parameters. Since RandomizedSearchCV is used, I use an uniform random interger range for the function to choose from.
param_grid ={'n_neighbors': st.randint(1,40),
             # Increasing this value reduces bias, and increases variance. Don't Overfit!
            'weights':['uniform','distance']
            }
# Hyper-Parameter Tuning with Cross-Validation
grid = RandomizedSearchCV(KNeighborsClassifier(),
                    param_grid, # Hyper Parameters
                    cv=cv, # Cross-Validation splits. Stratified.
                    scoring=scoring, # Best-Validation selection metric.
                    verbose=1, # Quality of Life. Frequency of model updates
                    n_iter=n_iter, # Number of hyperparameter combinations tried.
                    random_state=rstate) # Reproducibility 

# Execute Tuning on entire dataset
grid.fit(X_train, y_train)
save(grid, "KNN")
SGDClassifier().get_params().keys()
param_grid ={'loss':["hinge","log","modified_huber","epsilon_insensitive","squared_epsilon_insensitive"]}

grid = GridSearchCV(SGDClassifier(max_iter=5, tol=None),
                    param_grid,cv=cv, scoring=scoring,
                    verbose=1)

grid.fit(X_train, y_train)
save(grid, "StochasticGradientDescent")
# Helper Function to visualize feature importance
plt.rcParams['figure.figsize'] = (8, 4)
predictors = [x for x in X.columns if x not in ['Survived']]
def feature_imp(model):
    MO = model.fit(X_train, y_train)
    feat_imp = pd.Series(MO.feature_importances_, predictors).sort_values(ascending=False)
    feat_imp.plot(kind='bar', title='Feature Importances')
    plt.ylabel('Feature Importance Score')
DecisionTreeClassifier().get_params().keys()
# Baseline Decision Tree
tree = DecisionTreeClassifier()
print("Mean CV Accuracy:",cross_val_score(tree, X, y, cv=cv, scoring=scoring).mean())
feature_imp(tree)
f, ax = plt.subplots(figsize=(6, 4))
s = sns.heatmap(pd.crosstab(train_df.Sex, train_df.Title),
            annot=True, fmt="d", linewidths=.5, ax=ax,cmap="plasma",
                cbar_kws={'label': 'Count'})
s.set_title('Title Count by Sex Crosstab Heatmap')
s.set_xticklabels(["Mr","Mrs","Miss","Master","Rare"])
plt.show()
# Parameter Tuning
param_grid ={'n_estimators': n_tree_range}

tree = DecisionTreeClassifier()
grid = RandomizedSearchCV(BaggingClassifier(tree),
                    param_grid, cv=cv, scoring=scoring,
                    verbose=1,n_iter=n_iter, random_state=rstate)

grid.fit(X_train, y_train)
save(grid, "Bagger_ensemble")
model = RandomForestClassifier()
print("Mean CV Accuracy:",cross_val_score(model, X, y, cv=cv, scoring=scoring).mean())
feature_imp(model)
RandomForestClassifier().get_params().keys()
param_grid ={
             'max_depth': st.randint(3, 11),
#              'n_estimators': int(num_rounds),
             'max_features':["sqrt", "log2"],
             'max_leaf_nodes':st.randint(6, 10)
            }

model= RandomForestClassifier()

grid = RandomizedSearchCV(model,
                    param_grid, cv=cv,
                    scoring=scoring,
                    verbose=1,n_iter=n_iter, random_state=rstate)

grid.fit(X_train, y_train)
save(grid, "Random_Forest")
feature_imp(grid.best_estimator_)
AdaBoostClassifier().get_params().keys()
param_grid ={'n_estimators':n_tree_range,
            'learning_rate':np.arange(.1, 4, .5)}

grid = RandomizedSearchCV(AdaBoostClassifier(),
                    param_grid,cv=cv, scoring=scoring,
                    verbose=1, n_iter=n_iter, random_state=rstate)

grid.fit(X_train, y_train)
save(grid, "AdaBoost_Ensemble")
feature_imp(grid.best_estimator_)
GradientBoostingClassifier().get_params().keys()
param_grid ={'n_estimators': n_tree_range,
            'loss': ['deviance', 'exponential'],
            'learning_rate': [0.1, 0.01,0.05,0.001],
            'max_depth': np.arange(2, 12, 1)}

grid = RandomizedSearchCV(GradientBoostingClassifier(),
                    param_grid,cv=cv,
                    scoring=scoring,
                    verbose=1, n_iter=n_iter, random_state=rstate)

grid.fit(X_train, y_train)
save(grid, "Gradient_Boosting")
feature_imp(grid.best_estimator_)
XGBClassifier().get_params().keys()
params = {  
    "n_estimators": n_tree_range,
    "max_depth": st.randint(3, 15),
    "learning_rate": [0.05]
}

xgbreg = XGBClassifier(objective= 'binary:logistic', eval_metric="auc",
                       nthreads=2)

grid = RandomizedSearchCV(xgbreg, params, n_jobs=1, verbose=1, n_iter=n_iter,
                          random_state=rstate, scoring=scoring)  
grid.fit(X_train,y_train, verbose=False)
save(grid, "Sci_kit XGB")
feature_imp(grid.best_estimator_)
# What is going on with Age, Sex and Survival?
with sns.axes_style(style='ticks'):
    g = sns.factorplot("Sex", "Age", "Survived", data=train_df, kind="box")
    g.set(ylim=(-1,5))
    g.set_axis_labels("Sex", "Age");
model = XGBClassifier(n_estimators = num_rounds,
                      objective= 'binary:logistic',
                      learning_rate=0.01,
                      scale_pos_weight = imbalance_weight,
                      random_state=rstate,
                      scoring=scoring,
                      verbose_eval=10)

# use early_stopping_rounds to stop the cv when there is no score imporovement
model.fit(X_train,y_train, early_stopping_rounds=50, eval_set=[(X_test,
y_test)], verbose=False)
score = cross_val_score(model, X_train,y_train, cv=cv)
print("\nxgBoost - CV Train : %.2f" % score.mean())
print("xgBoost - Train : %.2f" % metrics.accuracy_score(model.predict(X_train), y_train))
print("xgBoost - Test : %.2f" % metrics.accuracy_score(model.predict(X_test), y_test))
norm_save(model,score, "XGBsklearn")
xgtrain = xgb.DMatrix(X_train, label=y_train)
xgtest = xgb.DMatrix(X_test, label=y_test)
xgtesting = xgb.DMatrix(test_df)

# set xgboost params
param = {'eta': 0.05, 
         'max_depth': 8, 
         'subsample': 0.8, 
         'colsample_bytree': 0.75,
          #'min_child_weight' : 1.5,
          'objective': 'binary:logistic', 
          'eval_metric': 'logloss',
          'scale_pos_weight': imbalance_weight,
          'seed': 23,
          'lambda': 1.5,
          'alpha': .6
        }

clf_xgb_cv = xgb.cv(param, xgtrain, num_rounds, 
                    stratified=True, 
                    nfold=n_splits, 
                    early_stopping_rounds=50,
                    verbose_eval=20)
print("Optimal number of trees/estimators is %i" % clf_xgb_cv.shape[0])

watchlist  = [(xgtrain,'train'),(xgtest,'test')]                
clf_xgb = xgb.train(param, xgtrain,clf_xgb_cv.shape[0], watchlist,verbose_eval=0)

# Best Cutoff Threshold
p = clf_xgb.predict(xgtest)
fpr, tpr, thresholds = metrics.roc_curve(y_test,p)
optimal_idx = np.argmax(tpr - fpr)
optimal_threshold = thresholds[optimal_idx]
print("\nOptimal Threshold: ",optimal_threshold)

# so we'll use 0.5 cutoff to convert probability to class label
y_train_pred = (clf_xgb.predict(xgtrain, ntree_limit=clf_xgb.best_iteration) > optimal_threshold).astype(int)
y_test_pred = (clf_xgb.predict(xgtest, ntree_limit=clf_xgb.best_iteration) > optimal_threshold).astype(int)
score= metrics.accuracy_score(y_test_pred, y_test)
print("\nXGB - CV Train : %.2f" % score)
train_score= metrics.accuracy_score(y_train_pred, y_train)
print("XGB - Train : %.2f" % train_score)

results = results.append({'Model': "BESTXGBOOST",'Para': param,'Test_Score': score,
                             'CV Mean':train_score, 'CV STDEV': None}, ignore_index=True)

xgboostprobpred = clf_xgb.predict(xgtesting, ntree_limit=clf_xgb.best_iteration)
xgboostpred = (xgboostprobpred > optimal_threshold).astype(int)
submission = pd.DataFrame({'PassengerId':test_df.index,'Survived':xgboostpred})
submission.to_csv('BESTXGBOOST.csv',index=False)
f, ax = plt.subplots(figsize=[4,5])
xgb.plot_importance(clf_xgb,max_num_features=50,ax=ax)
plt.title("XGBOOST Feature Importance")
plt.show()
def column_index(df, query_cols):
    cols = df.columns.values
    sidx = np.argsort(cols)
    return sidx[np.searchsorted(cols,query_cols,sorter=sidx)]
categorical_features_pos = column_index(X,categorical_features)
model = CatBoostClassifier(
                           eval_metric='Logloss',
                           iterations = num_rounds,
                           use_best_model=True,
                           od_wait = 100,
                           random_seed=42,
                           logging_level = "Verbose",
                           metric_period = 100
                          )
# from catboost import cv as catcv
# catpool = Pool(X_train,y_train,cat_features=categorical_features_pos)
# cv_data = catcv(catpool,model.get_params(),fold_count=2)
# best_cat_iterations = cv_data['test-Accuracy-mean'].idxmax()
# print("Best Iteration: ",best_cat_iterations)
# print("Best Score: ", cv_data['test-Accuracy-mean'][best_cat_iterations])
model = CatBoostClassifier(eval_metric='Accuracy',
                           iterations = 500,
                           scale_pos_weight= imbalance_weight,
                           random_seed=42,
                           logging_level = "Verbose",
                           metric_period = 100
                          )
model.fit(X,y,cat_features=categorical_features_pos)
model.get_params()
# cat_cv_std = cv_data.loc[cv_data['test-Accuracy-mean'].idxmax(),["train-Accuracy-mean","train-Accuracy-std"]]
# print("Train CV Accuracy: %0.2f (+/- %0.2f)" % (cat_cv_std[0],cat_cv_std[1])) 

# results = results.append({'Model': "Catboost",'Para': model.get_params(),'Test_Score': None,
#                              'CV Mean':cat_cv_std[0], 'CV STDEV': cat_cv_std[1]}, ignore_index=True)

# catprobpred = model.predict_proba(test_df)[:,1]
# catpred = model.predict(test_df).astype(np.int)
# submission = pd.DataFrame({'PassengerId':test_df.index,'Survived':catpred})
# submission.to_csv('catboost.csv',index=False)
lgtrain = lgb.Dataset(X_train, y_train,
                      categorical_feature=categorical_features
                     )

lgvalid = lgb.Dataset(X_test, y_test,
                      categorical_feature=categorical_features
                     )

lgbm_params = {
    'boosting_type': 'gbdt',
    'objective': 'binary',
    'metric': 'auc',
    'max_depth': 12,
    #'num_leaves': 500,
    'learning_rate': 0.01,
    'feature_fraction': 0.80,
    'bagging_fraction': 0.80,
    'bagging_freq': 5,
    'max_bin':300,
    #'verbose': 0,
    #'num_threads': 1,
    'lambda_l2': 1.5,
    #'min_gain_to_split': 0,
    'is_unbalance': True
    #'scale_pos_weight':0.15
}  

modelstart = time.time()
lgb_clf = lgb.train(
    lgbm_params,
    lgtrain,
    num_boost_round=num_rounds,
    valid_sets=[lgtrain,lgvalid],
    valid_names=['train','valid'],
    early_stopping_rounds=50,
    verbose_eval=50
)
print("\nModel Runtime: %0.2f Minutes"%((time.time() - modelstart)/60))
# Best Cutoff Threshold
p = lgb_clf.predict(X_test)
fpr, tpr, thresholds = metrics.roc_curve(y_test,p)
optimal_idx = np.argmax(tpr - fpr)
optimal_threshold = thresholds[optimal_idx]
print("Optimal Threshold: ",optimal_threshold)

# so we'll use 0.5 cutoff to convert probability to class label
y_train_pred = (lgb_clf.predict(X_train) > optimal_threshold).astype(int)
y_test_pred = (lgb_clf.predict(X_test) > optimal_threshold).astype(int)

score= metrics.accuracy_score(y_test_pred, y_test)
print("\nXGB - CV Train : %.2f" % score)
train_score = metrics.accuracy_score(y_train_pred, y_train)
print("XGB - Train : %.2f" % train_score)

# Save
results = results.append({'Model': "LGBM",'Para': lgbm_params,'Test_Score': score,
                             'CV Mean':train_score, 'CV STDEV': None}, ignore_index=True)

lgbmprobpred = lgb_clf.predict(test_df)
lgbmpred = (lgbmprobpred > optimal_threshold).astype(int)
submission = pd.DataFrame({'PassengerId':test_df.index,'Survived':lgbmpred})
submission.to_csv('LGBM.csv',index=False)
f, ax = plt.subplots(figsize=[8,5])
lgb.plot_importance(lgb_clf, max_num_features=50, ax=ax)
plt.title("Light GBM Feature Importance")
plt.show()
# Simple Mode Blend
modeblend = pd.DataFrame([xgboostpred,lgbmpred, lgbmpred]).T
modeblend.head()
modeblend["Mode"] = modeblend.mode(axis=1)
submission = pd.DataFrame({'PassengerId':test_df.index,'Survived':modeblend["Mode"]})
submission.to_csv('Mode_GBM_Blend.csv',index=False)
submission.head()
# Mean Blend
predblend = pd.DataFrame(np.array([xgboostprobpred,lgbmprobpred,lgbmpred])).T
predblend.columns = ["XG","LGBM","CAT"]
predblend["meanblend"] = predblend.mean(axis=1)
predblend["blendout"] = (predblend["meanblend"] > 0.5).astype(int)
submission = pd.DataFrame({'PassengerId':test_df.index,'Survived':predblend["blendout"]})
submission.to_csv('mean_GBM_blend.csv',index=False)
submission.head()
# Stratified Cross-Validation
cv = StratifiedShuffleSplit(n_splits=n_splits, test_size=0.2, random_state=rstate)
model= LogisticRegression()
score = cross_val_score(model, X_train, y_train, cv=cv, scoring='accuracy')
norm_save(LogisticRegression(),score, "Logistic_Regression")
# Simple Linear Regression Model
logit = sm.OLS(y,sm.add_constant(X)).fit() 
logit.summary()
# from pygam import LogisticGAM # Logistic Generalized Additive Model
# gam = LogisticGAM().gridsearch(X_train, y_train)
# # lgampred = gam.predict(test_df)
# # submission = pd.DataFrame({'PassengerId':test_df.index,'Survived':lgampred})
# # submission.to_csv('LGAM.csv',index=False)

# train_score= gam.accuracy(X_train,y_train); print("Train Score: %.2f" % train_score)
# test_score= gam.accuracy(X_test,y_test); print("Test Score: %.2f\n" % test_score)
# results = results.append({'Model': "Logistic GAM",'Para': None,'Test_Score': test_score,
#                              'CV Mean':train_score, 'CV STDEV': None}, ignore_index=True)
# gam.summary()

# from pygam.utils import generate_X_grid
# XX = generate_X_grid(gam)
# fig, axs = plt.subplots(1, 6, figsize= [15,3])
# titles = X.columns[:6]
# for i, ax in enumerate(axs):
#     pdep, confi = gam.partial_dependence(XX, feature=i+1, width=.95)
#     ax.plot(XX[:, i], pdep)
#     ax.plot(XX[:, i], confi[0][:, 0], c='grey', ls='--')
#     ax.plot(XX[:, i], confi[0][:, 1], c='grey', ls='--')
#     ax.set_title(titles[i],fontsize=26)
# plt.tight_layout(pad=0)
# plt.show()

# from pygam.utils import generate_X_grid
# XX = generate_X_grid(gam)
# fig, axs = plt.subplots(1, 5, figsize= [15,3])
# titles = X.columns[6:]
# for i, ax in enumerate(axs):
#     pdep, confi = gam.partial_dependence(XX, feature=i+1, width=.95)
#     ax.plot(XX[:, i], pdep)
#     ax.plot(XX[:, i], confi[0][:, 0], c='grey', ls='--')
#     ax.plot(XX[:, i], confi[0][:, 1], c='grey', ls='--')
#     ax.set_title(titles[i],fontsize=26)
# plt.tight_layout(pad=0)
# plt.show()
MLPClassifier().get_params().keys()
# Start with a RandomSearchCV to efficiently Narrow the Ballpark
param_grid ={'max_iter': np.logspace(1, 5, 10).astype("int32"),
             'hidden_layer_sizes': np.logspace(2, 3, 4).astype("int32"),
             'activation':['identity', 'logistic', 'tanh', 'relu'],
             'learning_rate': ['adaptive'],
             'early_stopping': [True],
             'alpha': np.logspace(2, 3, 4).astype("int32")
            }

model = MLPClassifier()
grid = RandomizedSearchCV(model,
                    param_grid, cv=cv, scoring=scoring,
                    verbose=1, n_iter=n_iter, random_state=rstate)

grid.fit(X_train, y_train)
save(grid, "FFNeural_Net")
LinearSVC().get_params().keys()
# Define Model
model = LinearSVC()

#Fit Model
scores= cross_val_score(model, X_train, y_train, cv=cv, scoring=scoring)
norm_save(model, scores, "LinearSV")
SVC().get_params().keys()
svc = SVC(kernel= 'rbf', probability=True)

model = Pipeline(steps=[('svc', svc)])


param_grid = {'svc__C': st.randint(1,10000),
              'svc__gamma': np.logspace(1, -7, 10)}

grid = RandomizedSearchCV(model, param_grid,
                          cv=cv, verbose=1, scoring=scoring,
                         n_iter=n_iter, random_state=rstate)

grid.fit(X_train, y_train)
save(grid, "SVCrbf")
pca = PCA()
svc = SVC(kernel= 'rbf',probability=True)

model = Pipeline(steps=[('pca',pca),
                        ('svc', svc)])


param_grid = {'svc__C': st.randint(1,10000),
              'svc__gamma': np.logspace(1, -7, 10),
             'pca__n_components': st.randint(1,len(X.columns))}

grid = RandomizedSearchCV(model, param_grid,
                          cv=cv, verbose=1,
                         n_iter=n_iter, random_state=rstate, scoring=scoring)

grid.fit(X_train, y_train)
save(grid, "PCA_SVC")
model = GaussianNB()

score = cross_val_score(model, X_train, y_train, cv=cv, scoring=scoring)
norm_save(model,score, "Gaussian")
results.sort_values(by=["Test_Score"], ascending=False, inplace=True)
results
### Ensemble Voting
ignore = ["Catboost","BESTXGBOOST","LGBM", "Logistic GAM"]
not_proba_list = ['LinearSV','StochasticGradientDescent']
not_proba =  results.query("Model in @not_proba_list")
hard_models = results.query("Model not in @ignore")
prob_models = results.query("Model not in @not_proba_list & Model not in @ignore")

# Alternative Indexing
# [x for x in results.Model if x not in not_proba_list]
# results.loc[~results.Model.isin(["Catboost"]+not_proba_list),:]

# Submission DataFrame for correlation purposes
# Hard Output
test_hard_pred_matrix = pd.DataFrame()
train_hard_pred_matrix = pd.DataFrame()
valid_hard_pred_matrix = pd.DataFrame()

#Soft Output
test_soft_pred_matrix = pd.DataFrame()
train_soft_pred_matrix = pd.DataFrame()
valid_soft_pred_matrix = pd.DataFrame()
# Only Non-Probabilistic
models = list(zip([ensemble_models[x] for x in not_proba.Model], not_proba.Model))
clfs = []

print('5-fold cross validation:\n')
for clf, label in models:
    scores = cross_val_score(clf, X_train, y_train, cv=cv, scoring=scoring, verbose=0)
    print("Train CV Accuracy: %0.2f (+/- %0.2f) [%s]" % (scores.mean(), scores.std(), label))
    md = clf.fit(X_train, y_train)
    valid_hard_pred_matrix = pd.concat((valid_hard_pred_matrix, pd.DataFrame({label: md.predict(X_test)})), axis=1)
    clfs.append(md)
    print("Test Accuracy: %0.2f \n" % (metrics.accuracy_score(clf.predict(X_test), y_test)))
    
    # Model on Full Data
    md = clf.fit(X,y)
    submission = md.predict(test_df)
    df = pd.DataFrame({'PassengerId':test_df.index, 
                           'Survived':submission})
    train_hard_pred_matrix = pd.concat((train_hard_pred_matrix, pd.DataFrame({label: md.predict(X)})), axis=1)
    test_hard_pred_matrix = pd.concat((test_hard_pred_matrix, pd.DataFrame({label: submission})), axis=1)
    
    # Output Submission
    df.to_csv("{}.csv".format(label),header=True,index=False)

del clfs
warnings.filterwarnings(action='ignore', category=DeprecationWarning)
# Only Probabilistic
models = list(zip([ensemble_models[x] for x in prob_models.Model], prob_models.Model))
plt.figure()

print('5-fold cross validation:\n')
clfs = []
for clf, model_label in models:
    scores = cross_val_score(clf, X_train, y_train,cv=5, scoring=scoring, verbose=0)
    print("Train CV Accuracy: %0.2f (+/- %0.2f) [%s]" % (scores.mean(), scores.std(), model_label))
    md = clf.fit(X_train, y_train)
    valid_soft_pred_matrix = pd.concat((valid_soft_pred_matrix, pd.DataFrame({model_label: md.predict_proba(X_test)[:,1]})), axis=1)
    # Add to Roc Curve
    fpr, tpr, _ = roc_curve(y_test, md.predict_proba(X_test)[:,1])
    roc_auc = auc(fpr, tpr)

    print('ROC AUC: %0.2f' % roc_auc)
    plt.plot(fpr, tpr, label='{} ROC curve (area = {:.2})'.format(model_label, roc_auc))
    
    clfs.append(md)
    print("Test Accuracy: %0.2f \n" % (metrics.accuracy_score(clf.predict(X_test), y_test)))
    
    # Model on Full Data
    md = clf.fit(X,y)
    submission = md.predict(test_df)
    df = pd.DataFrame({'PassengerId':test_df.index, 
                           'Survived':submission})
    train_hard_pred_matrix = pd.concat((train_hard_pred_matrix, pd.DataFrame({model_label: md.predict(X)})), axis=1)
    test_hard_pred_matrix = pd.concat((test_hard_pred_matrix, pd.DataFrame({model_label: submission})), axis=1)

    train_soft_pred_matrix = pd.concat((train_soft_pred_matrix, pd.DataFrame({model_label: md.predict_proba(X)[:,1]})), axis=1)
    test_soft_pred_matrix = pd.concat((test_soft_pred_matrix, pd.DataFrame({model_label: md.predict_proba(test_df)[:,1]})), axis=1)
    
    # Output Submission
    df.to_csv("{}.csv".format(model_label),header=True,index=False)

# Plot
plt.plot([0, 1], [0, 1], 'k--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend(loc="lower right")
plt.show()
# Play with Weights
plt.figure()
voters = {}
for x in [2,3,5,7,10]:
    ECH = EnsembleVoteClassifier([ensemble_models.get(key) for key in hard_models.Model[:x]], voting='hard')
    ECS = EnsembleVoteClassifier([ensemble_models.get(key) for key in prob_models.Model[:x]], voting='soft')
    
    print('\n{}-Voting Models: 5-fold cross validation:\n'.format(x))
    for clf, model_label in zip([ECS, ECH], 
                          ['{}-VM-Ensemble Soft Voting'.format(x),
                           '{}-VM-Ensemble Hard Voting'.format(x)]):
        scores = cross_val_score(clf, X_train, y_train, cv=5, scoring='accuracy')
        print("Train CV Accuracy: %0.2f (+/- %0.2f) [%s]" % (scores.mean(), scores.std(), model_label))
        md = clf.fit(X_train, y_train)
        clfs.append(md)        
        
        Test_Score = metrics.accuracy_score(clf.predict(X_test), y_test)
        print("Test Accuracy: %0.2f " % Test_Score)
        
        CV_Score = scores.mean()
        STDev = scores.std()
        
        global results
        results = results.append({'Model': model_label,'Para': clf, 'CV Mean': CV_Score,
                'Test_Score':Test_Score,'CV STDEV': STDev}, ignore_index=True)
        voters[model_label] = clf
        
        # Model on Full Data
        md = clf.fit(X,y)
        submission = md.predict(test_df)
        df = pd.DataFrame({'PassengerId':test_df.index,'Survived':submission})
        df.to_csv("{}.csv".format(model_label),header=True,index=False)
        
        if clf is ECH:
            # Hard Correlation
            train_hard_pred_matrix = pd.concat((train_hard_pred_matrix, pd.DataFrame({model_label: md.predict(X)})), axis=1)
            test_hard_pred_matrix = pd.concat((test_hard_pred_matrix, pd.DataFrame({model_label: submission})), axis=1)
        
        elif clf is ECS:
            # Add to Roc Curve
            fpr, tpr, _ = roc_curve(y_test, md.predict_proba(X_test)[:,1])
            roc_auc = auc(fpr, tpr)
            print('ROC AUC: %0.2f' % roc_auc)
            plt.plot(fpr, tpr, label='{} ROC curve (area = {:.2})'.format(model_label, roc_auc))
            # Soft Correlation
            train_soft_pred_matrix = pd.concat((train_soft_pred_matrix, pd.DataFrame({model_label: md.predict_proba(X)[:,1]})), axis=1)
            test_soft_pred_matrix = pd.concat((test_soft_pred_matrix, pd.DataFrame({model_label: md.predict_proba(test_df)[:,1]})), axis=1)
        
        
# Plot
plt.plot([0, 1], [0, 1], 'k--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Soft Model ROC Curve')
plt.legend(loc="lower right")
plt.show()
voters.get('10-VM-Ensemble Hard Voting')
Xstack = X.copy()
ystack = y.copy()
X_trainstack = X_train.copy()
X_teststack = X_test.copy()
y_trainstack = y_train.copy()
y_teststack = y_test.copy()
from sklearn.model_selection import cross_val_score
# warnings.filterwarnings(action='ignore', category=DeprecationWarning)
num_trees = 10
verbose = True # to print the progress

clfs = [ensemble_models.get('KNN'),
        ensemble_models.get('Sci_kit XGB')]

# Creating train and test sets for blending
dataset_blend_train = np.zeros((X_trainstack.shape[0], len(clfs)))
dataset_blend_test = np.zeros((X_teststack.shape[0], len(clfs)))
dataset_blend_test_df = np.zeros((test_df.shape[0], len(clfs)))

print('5-fold cross validation:')
for i, clf in enumerate(clfs):   
    scores = cross_val_score(clf, X_trainstack, y_trainstack, cv=4, scoring='accuracy')
    print("##### Base Model %0.0f #####" % i)
    print("Train CV Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std()))
    clf.fit(X_trainstack, y_trainstack)   
    print("Train Accuracy: %0.2f " % (metrics.accuracy_score(clf.predict(X_trainstack), y_trainstack)))
    dataset_blend_train[:,i] = clf.predict_proba(X_trainstack)[:, 1]
    dataset_blend_test[:,i] = clf.predict_proba(X_teststack)[:, 1]
    dataset_blend_test_df[:,i] = clf.predict_proba(test_df)[:, 1]
    print("Test Accuracy: %0.2f \n" % (metrics.accuracy_score(clf.predict(X_teststack), y_teststack)))    

print("##### Meta Model #####")
clf = LogisticRegression()
scores = cross_val_score(clf, dataset_blend_train, y_trainstack, cv=4, scoring=scoring)
clf.fit(dataset_blend_train, y_trainstack)
print("Train CV Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std()))
print("Train Accuracy: %0.2f " % (metrics.accuracy_score(clf.predict(dataset_blend_train), y_trainstack)))
print("Test Accuracy: %0.2f " % (metrics.accuracy_score(clf.predict(dataset_blend_test), y_teststack)))

# Correlate Results
#test_hard_pred_matrix = pd.concat((test_hard_pred_matrix, pd.DataFrame({label: clf.predict(dataset_blend_test_df)})), axis=1)
#train_hard_pred_matrix = pd.concat((train_hard_pred_matrix, pd.DataFrame({label: model.predict(dataset_blend_train)})), axis=1)

# Save
if save == True:
    pd.DataFrame({'PassengerId':test_df.index, 
        'Survived':clf.predict(dataset_blend_test_df)}).to_csv(
        "{}.csv".format("Stacked"),header=True,index=False)
score = cross_val_score(clf, X, y, cv=cv, scoring=scoring)
norm_save(clf, score, "stacked")
eval_plot(clf)
# Add Validation Set
# Train Prep
train_hard_pred_matrix = train_hard_pred_matrix.set_index([traindex])
train_hard_pred_matrix = pd.concat([train_hard_pred_matrix, Survived], axis=1)
valid_hard_pred_matrix = valid_hard_pred_matrix.set_index([X_test.index])
valid_hard_pred_matrix = pd.concat([valid_hard_pred_matrix, Survived[X_test.index]], axis=1)

# Soft
train_soft_pred_matrix = train_soft_pred_matrix.set_index([traindex])
train_soft_pred_matrix = pd.concat([train_soft_pred_matrix, Survived], axis=1)
valid_soft_pred_matrix = valid_soft_pred_matrix.set_index([X_test.index])
valid_soft_pred_matrix = pd.concat([valid_soft_pred_matrix, Survived[X_test.index]], axis=1)

# Test Prep
test_hard_pred_matrix = test_hard_pred_matrix.set_index([testdex])
valid_hard_pred_matrix = valid_hard_pred_matrix.set_index([X_test.index])
# Soft
test_soft_pred_matrix = test_soft_pred_matrix.set_index([testdex])
valid_soft_pred_matrix = valid_soft_pred_matrix.set_index([X_test.index])

# OUTPUT SUBMISSION RESULTS
# Test
test_hard_pred_matrix.to_csv("test_hard_pred_matrix.csv")
test_soft_pred_matrix.to_csv("test_soft_pred_matrix.csv")
valid_soft_pred_matrix.to_csv("valid_soft_pred_matrix.csv")

# Train
train_hard_pred_matrix.to_csv("train_hard_pred_matrix.csv")
train_soft_pred_matrix.to_csv("train_soft_pred_matrix.csv")
valid_soft_pred_matrix.to_csv("valid_soft_pred_matrix.csv")
results.sort_values(by=["Test_Score"], ascending=False, inplace=True)
results.to_csv("titanic_clf_results.csv", index=False)
results
valid_soft_pred_matrix.head()
top_models = [x for x in valid_soft_pred_matrix.columns
              if x in list(results.sort_values(by="CV Mean",ascending = False).Model[:15])]
top_models = [x for x in valid_soft_pred_matrix.columns
              if x in list(results.sort_values(by="CV Mean",ascending = False).Model[:15])]
print("Models Chosen: ",top_models)
valid_soft_pred_matrix = valid_soft_pred_matrix[top_models]
test_soft_pred_matrix = test_soft_pred_matrix[top_models]
# Prepare Data
pred_cols = list(valid_soft_pred_matrix.columns)

def stack_features(tempdf):
    tempdf['max'] = np.max(np.array([tempdf[col] for col in pred_cols]),axis=0)
    tempdf['min'] = np.min(np.array([tempdf[col] for col in pred_cols]),axis=0)
    tempdf['avg'] = np.mean(np.array([tempdf[col] for col in pred_cols]),axis=0)
    tempdf['med'] = np.median(np.array([tempdf[col] for col in pred_cols]),axis=0)
    tempdf['std'] = np.std(np.array([tempdf[col] for col in pred_cols]),axis=0)
    
    for p1, p2 in itertools.combinations(pred_cols, 2):
        tempdf['difference_%s__%s'%(p1,p2)] = tempdf[p2] - tempdf[p1]
        tempdf['sums_%s__%s'%(p1,p2)] = tempdf[p2] + tempdf[p1]
    return tempdf
# Create Features
all_soft_preds = pd.concat([valid_soft_pred_matrix, test_soft_pred_matrix],axis=0)
all_soft_preds = stack_features(all_soft_preds)

valid_soft_pred_matrix = all_soft_preds.loc[X_test.index, :]
test_soft_pred_matrix = all_soft_preds.loc[testdex, :]

# Combine with Original Features
X_stack = pd.concat([X_test, valid_soft_pred_matrix], axis=1)
test_stack = pd.concat([test_df, test_soft_pred_matrix],axis=1)
# X_stack = X
# test_stack = test_df

print("Train Shape: ", X_stack.shape)
print("Test Shape: ", test_stack.shape)

# Stratified Train/Test Split
y_stack = y[X_test.index]
X_stack_train, X_stack_test, y_stack_train, y_stack_test = train_test_split(X_stack, y_stack, test_size=.30, stratify=y_stack,random_state=23)
print(X_stack_train.shape, y_stack_train.shape, X_stack_test.shape, y_stack_test.shape)

# Lgbm Dataset Formating
lgtrain = lgb.Dataset(data = X_stack_train,label = y_stack_train.values, categorical_feature = categorical_features)
lgb_results = pd.DataFrame(columns = ["Rounds","Score","STDV", "LB", "Parameters"])
fulllgtrain = lgb.Dataset(data = X_stack,label = y_stack.values, categorical_feature = categorical_features)
# Logistic Stacker
from sklearn.linear_model import LogisticRegression
model = LogisticRegression()
print("Mean CV Accuracy:",cross_val_score(model, X_stack, y_stack, cv=5, scoring="accuracy").mean())

# Fit on Full and Submit..
model.fit(X_stack, y_stack)
logregpred = model.predict(test_stack)
submission = pd.DataFrame({'PassengerId':test_stack.index,'Survived':logregpred})
submission.to_csv('LogRegStack.csv',index=False)
print("Light Gradient Boosting Classifier: ")
lgbm_params =  {
    'task': 'train',
    'boosting_type': 'gbdt',
    'objective': 'binary',
    'metric': ['binary_logloss'],
    "learning_rate": 0.05,
    "num_leaves": 20,
    "max_depth": 5,
#     "feature_fraction": 0.5,
#     "bagging_fraction": 0.5,
#     "reg_alpha": 0.15,
#     "reg_lambda": 0.01,
    'is_unbalance': True
                }
# Find Optimal Parameters / Boosting Rounds
lgb_cv = lgb.cv(
    params = lgbm_params,
    train_set = lgtrain,
    num_boost_round=10000,
    stratified=True,
    nfold = 5,
    verbose_eval=25,
    seed = 23,
    early_stopping_rounds=75,
    categorical_feature= categorical_features)

loss = lgbm_params["metric"][0]
optimal_rounds = np.argmin(lgb_cv[str(loss) + '-mean'])
best_cv_score = round(min(lgb_cv[str(loss) + '-mean']), 7)
best_std_score = round(lgb_cv[str(loss) + '-stdv'][optimal_rounds],7)

print("\nOptimal Round: {}\nOptimal Score: {} + {}".format(
    optimal_rounds,best_cv_score,best_std_score))

lgb_results = lgb_results.append({"Rounds": optimal_rounds,
                          "Score": best_cv_score,
                          "STDV": best_std_score,
                          "LB": None,
                          "Parameters": lgbm_params}, ignore_index=True)
display(lgb_results.sort_values(by="Score",ascending = True))
# Best Parameters
final_model_params = lgb_results.iloc[lgb_results["Score"].idxmin(),:]["Parameters"]
optimal_rounds = lgb_results.iloc[lgb_results["Score"].idxmin(),:]["Rounds"]
print("Parameters for Final Models:\n",final_model_params)
print("Score: {} +/- {}".format(lgb_results.iloc[lgb_results["Score"].idxmin(),:]["Score"],lgb_results.iloc[lgb_results["Score"].idxmin(),:]["STDV"]))
print("Rounds: ", optimal_rounds)
print("Find Optimal Cutoff")
lgb_clf = lgb.train(
    final_model_params,
    lgtrain,
    num_boost_round = optimal_rounds + 1,
    verbose_eval=100,
    categorical_feature= categorical_features)

# Best Cutoff Threshold
p = lgb_clf.predict(X_stack_test)
fpr, tpr, thresholds = metrics.roc_curve(y_stack_test,p)
optimal_idx = np.argmax(tpr - fpr)
optimal_threshold = thresholds[optimal_idx]
print("Optimal Threshold: ",optimal_threshold)

y_stack_train_pred = (lgb_clf.predict(X_stack_train) > optimal_threshold).astype(int)
y_stack_test_pred = (lgb_clf.predict(X_stack_test) > optimal_threshold).astype(int)

score= metrics.accuracy_score(y_stack_test_pred, y_stack_test)
print("\nLGBM STACKER - CV Train : %.2f" % score)
allmodelstart= time.time()
all_feature_importance_df  = pd.DataFrame()

modelstart= time.time()
lgb_clf = lgb.train(
    final_model_params,
    fulllgtrain,
    num_boost_round = optimal_rounds + 1,
    verbose_eval=100,
    categorical_feature= categorical_features)

# Feature Importance
fold_importance_df = pd.DataFrame()
fold_importance_df["feature"] = X_stack.columns
fold_importance_df["importance"] = lgb_clf.feature_importance()
all_feature_importance_df = pd.concat([all_feature_importance_df, fold_importance_df], axis=0)

lgbmprobpred = lgb_clf.predict(test_stack)
lgbmpred = (lgbmprobpred > optimal_threshold).astype(int)
submission = pd.DataFrame({'PassengerId':test_stack.index,'Survived':lgbmpred})
submission.to_csv('LGBM.csv',index=False)
print("Model Runtime: %0.2f Minutes"%((time.time() - modelstart)/60))

cols = all_feature_importance_df[["feature", "importance"]].groupby("feature").mean().sort_values(
    by="importance", ascending=False)[:50].index
best_features = all_feature_importance_df.loc[all_feature_importance_df.feature.isin(cols)]
plt.figure(figsize=(10,8))
sns.barplot(x="importance", y="feature", 
            data=best_features.sort_values(by="importance", ascending=False))
plt.title('LightGBM Features')
plt.tight_layout()
plt.savefig('lgbm_importances.png')
print("\nClassification Report:")
print(classification_report(y_stack_test_pred, y_stack_test))
confusion_matrix(y_stack_test, y_stack_test_pred)

# Matrix
skplt.metrics.plot_confusion_matrix(y_stack_test_pred, y_stack_test, normalize=True)
plt.show()
import time
end = time.time()
print("Notebook took %0.2f minutes to Run"%((end - start)/60))