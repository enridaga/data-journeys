
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
import lightgbm as lgb
from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold
from sklearn.metrics import roc_auc_score
import time
from sklearn import preprocessing
import warnings
import datetime
warnings.filterwarnings("ignore")
import gc
from tqdm import tqdm

from sklearn.svm import SVC
from sklearn.feature_selection import VarianceThreshold

from scipy.stats import describe
#%matplotlib inline

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import mean_squared_error
import xgboost as xgb
# Any results you write to the current directory are saved as output.
import os
print(os.listdir("../input"))
#Loading Train and Test Data
train = pd.read_csv("../input/train.csv")
test = pd.read_csv("../input/test.csv")
print("{} observations and {} features in train set.".format(train.shape[0],train.shape[1]))
print("{} observations and {} features in test set.".format(test.shape[0],test.shape[1]))
train.head()
test.head()
train.target.describe()
plt.figure(figsize=(12, 5))
plt.hist(train.target.values, bins=200)
plt.title('Histogram target counts')
plt.xlabel('Count')
plt.ylabel('Target')
plt.show()
plt.figure(figsize=(12, 5))
plt.hist(train['muggy-smalt-axolotl-pembus'].values, bins=200)
plt.title('Histogram muggy-smalt-axolotl-pembus counts')
plt.xlabel('Value')
plt.ylabel('Count')
plt.show()
plt.figure(figsize=(12, 5))
plt.hist(train['dorky-peach-sheepdog-ordinal'].values, bins=200)
plt.title('Histogram muggy-smalt-axolotl-pembus counts')
plt.xlabel('Value')
plt.ylabel('Count')
plt.show()


plt.figure(figsize=(12, 5))
plt.hist(train['crabby-teal-otter-unsorted'].values, bins=200)
plt.title('Histogram muggy-smalt-axolotl-pembus counts')
plt.xlabel('Value')
plt.ylabel('Count')
plt.show()

plt.figure(figsize=(12, 5))
plt.hist(train['wheezy-copper-turtle-magic'].values, bins=1000)
plt.title('Histogram muggy-smalt-axolotl-pembus counts')
plt.xlabel('Value')
plt.ylabel('Count')
plt.show()

train.describe()
test.describe()
def normal(train, test):
    print('Scaling with StandardScaler\n')
    len_train = len(train)

    traintest = pd.concat([train,test], axis=0, ignore_index=True).reset_index(drop=True)
    
    scaler = StandardScaler()
    cols = [c for c in train.columns if c not in ['id', 'target', 'wheezy-copper-turtle-magic']]
    traintest[cols] = scaler.fit_transform(traintest[cols])
    traintest['wheezy-copper-turtle-magic'] = traintest['wheezy-copper-turtle-magic'].astype('category')
    train = traintest[:len_train].reset_index(drop=True)
    test = traintest[len_train:].reset_index(drop=True)

    return train, test
%%time
train, test = normal(train, test)

%%time
featues_to_use = [c for c in train.columns if c not in ['id', 'target']]
target = train['target']
#train = train[featues_to_use]
#test = test[featues_to_use]
#classifier = LogisticRegression(C=1, solver='sag')
#cv_score = np.mean(cross_val_score(classifier, train, target, cv=3, scoring='roc_auc'))
#print(cv_score)
%%time
folds = KFold(n_splits=10, shuffle=True, random_state=137)
oof = np.zeros(train.shape[0])
pred = 0

for fold_, (trn_idx, val_idx) in enumerate(folds.split(train.values, target.values)):
    print("Fold {}".format(fold_+1))
    x_train, y_train = train.iloc[trn_idx][featues_to_use], target.iloc[trn_idx]
    x_val, y_val = train.iloc[val_idx][featues_to_use], target.iloc[val_idx]
    classifier = LogisticRegression(C=1, solver='sag')
    classifier.fit(x_train, y_train)
    val_pred = classifier.predict_proba(x_val)[:,1]
    oof[val_idx] = val_pred
    pred += classifier.predict_proba(test[featues_to_use])[:,1]/10
    print(roc_auc_score(y_val, val_pred))
    
print(roc_auc_score(target.values, oof))
%%time

NFOLDS = 25
NVALUES = 512

cols = [c for c in train.columns if c not in ['id', 'target', 'wheezy-copper-turtle-magic']]

folds = StratifiedKFold(n_splits=NFOLDS, shuffle=True, random_state=137)
oof_lr = np.zeros(train.shape[0])
pred_lr = np.zeros(test.shape[0])

for fold_, (trn_idx, val_idx) in enumerate(folds.split(train.values, target.values)):
    print("Fold {}".format(fold_+1))
    x_train = train.iloc[trn_idx]
    x_val, y_val = train.iloc[val_idx], target.iloc[val_idx]
    
    
    for i in tqdm(range(NVALUES)):
        # ONLY TRAIN WITH DATA WHERE WHEEZY EQUALS I
        x_train_2 = x_train[x_train['wheezy-copper-turtle-magic']==i]
        x_val_2 = x_val[x_val['wheezy-copper-turtle-magic']==i]
        test_2 = test[test['wheezy-copper-turtle-magic']==i]
        idx1 = x_train_2.index; idx2 = x_val_2.index; idx3 = test_2.index
        x_train_2.reset_index(drop=True,inplace=True)
        x_val_2.reset_index(drop=True,inplace=True)
        test_2.reset_index(drop=True,inplace=True)
        clf = LogisticRegression(solver='liblinear',penalty='l1',C=0.05)
        y_train =x_train_2['target']
        clf.fit(x_train_2[cols],y_train)
        
        oof_lr[idx2] = clf.predict_proba(x_val_2[cols])[:,1]
        pred_lr[idx3] += clf.predict_proba(test_2[cols])[:,1] / NFOLDS

    oof_lr_val = oof_lr[val_idx]

    
    print(roc_auc_score(y_val, oof_lr_val))
    
print(roc_auc_score(target, oof_lr))

print(roc_auc_score(target, oof_lr))
# INITIALIZE VARIABLES
oof_svm = np.zeros(len(train))
preds_svm = np.zeros(len(test))
cols = [c for c in train.columns if c not in ['id', 'target', 'wheezy-copper-turtle-magic']]

# BUILD 512 SEPARATE NON-LINEAR MODELS
for i in range(512):
    
    # EXTRACT SUBSET OF DATASET WHERE WHEEZY-MAGIC EQUALS I
    train2 = train[train['wheezy-copper-turtle-magic']==i]
    test2 = test[test['wheezy-copper-turtle-magic']==i]
    idx1 = train2.index; idx2 = test2.index
    train2.reset_index(drop=True,inplace=True)
    
    # FEATURE SELECTION (USE APPROX 40 OF 255 FEATURES)
    sel = VarianceThreshold(threshold=1.5).fit(train2[cols])
    train3 = sel.transform(train2[cols])
    test3 = sel.transform(test2[cols])
        
    # STRATIFIED K FOLD (Using splits=25 scores 0.002 better but is slower)
    skf = StratifiedKFold(n_splits=11, random_state=42)
    for train_index, test_index in skf.split(train3, train2['target']):
        
        # MODEL WITH SUPPORT VECTOR MACHINE
        clf = SVC(probability=True,kernel='poly',degree=4,gamma='auto')
        clf.fit(train3[train_index,:],train2.loc[train_index]['target'])
        oof_svm[idx1[test_index]] = clf.predict_proba(train3[test_index,:])[:,1]
        preds_svm[idx2] += clf.predict_proba(test3)[:,1] / skf.n_splits
        
    #if i%10==0: print(i)
        
# PRINT VALIDATION CV AUC
auc = roc_auc_score(train['target'],oof_svm)
print('CV score =',round(auc,5))
roc_auc_score(train['target'],0.6*oof_svm+0.4*oof_lr)
'''%%time

param = {
    'bagging_freq': 3,
    'bagging_fraction': 0.8,
    'boost_from_average':'false',
    'boost': 'gbdt',
    'feature_fraction': 0.9,
    'learning_rate': 0.05,
    'max_depth': 10,  
    'metric':'auc',
    'min_data_in_leaf': 82,
    'min_sum_hessian_in_leaf': 10.0,
    'num_leaves': 10,
    'objective': 'binary', 
    'verbosity': 1
}

folds = KFold(n_splits=10, shuffle=True, random_state=137)
oof_lgb = np.zeros(train.shape[0])
pred_lgb = 0

for fold_, (trn_idx, val_idx) in enumerate(folds.split(train.values, target.values)):
    print("Fold {}".format(fold_+1))
    x_train, y_train = train.iloc[trn_idx][featues_to_use], target.iloc[trn_idx]
    x_val, y_val = train.iloc[val_idx][featues_to_use], target.iloc[val_idx]
    trn_data = lgb.Dataset(x_train, label=y_train)
    val_data = lgb.Dataset(x_val, label=y_val)
    classifier = lgb.train(param, trn_data, 1000000, valid_sets = [trn_data, val_data], verbose_eval=200, early_stopping_rounds = 300)

    val_pred = classifier.predict(x_val, num_iteration=classifier.best_iteration)
    oof_lgb[val_idx] = val_pred
    pred_lgb += classifier.predict(test[featues_to_use], num_iteration=classifier.best_iteration)/10
    print(roc_auc_score(y_val, val_pred))'''
%%time
submission = pd.read_csv('../input/sample_submission.csv')


'''submission['target'] = pred_lr
submission.to_csv('submission_0.csv', index=False)
submission['target'] = 0.9*pred_lr + 0.1*pred_lgb
submission.to_csv('submission.csv', index=False)
submission['target'] = 0.8*pred_lr + 0.2*pred_lgb
submission.to_csv('submission_2.csv', index=False)'''


submission['target'] = 0.6*preds_svm + 0.4*pred_lr
submission.to_csv('submission_3.csv', index=False)

