
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
import operator
#import gc
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import describe
#%matplotlib inline

from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
import lightgbm as lgb
import xgboost as xgb
print(os.listdir("../input"))
train_df = pd.read_csv('../input/train.csv')
train_df.head()
train_df.shape
train_df.info()
train_df.isnull().values.any()
test_df = pd.read_csv('../input/test.csv')
test_df.head()
test_df.shape
test_df.info()
test_df.isnull().values.any()
train_df_describe = train_df.describe()
train_df_describe
test_df_describe = test_df.describe()
test_df_describe
plt.figure(figsize=(12, 5))
plt.hist(train_df.target.values, bins=100)
plt.title('Histogram target counts')
plt.xlabel('Count')
plt.ylabel('Target')
plt.show()
plt.figure(figsize=(12, 5))
plt.hist(np.log(1+train_df.target.values), bins=100)
plt.title('Histogram target counts')
plt.xlabel('Count')
plt.ylabel('Log 1+Target')
plt.show()
sns.set_style("whitegrid")
ax = sns.violinplot(x=np.log(1+train_df.target.values))
plt.show()
train_log_target = train_df[['target']]
train_log_target['target'] = np.log(1+train_df['target'].values)
train_log_target.describe()
constant_train = train_df.loc[:, (train_df == train_df.iloc[0]).all()].columns.tolist()
constant_test = test_df.loc[:, (test_df == test_df.iloc[0]).all()].columns.tolist()
print('Number of constant columns in the train set:', len(constant_train))
print('Number of constant columns in the test set:', len(constant_test))
columns_to_use = test_df.columns.tolist()
del columns_to_use[0] # Remove 'ID'
columns_to_use = [x for x in columns_to_use if x not in constant_train] #Remove all 0 columns
len(columns_to_use)
describe(train_df[columns_to_use].values, axis=None)
plt.figure(figsize=(12, 5))
plt.hist(train_df[columns_to_use].values.flatten(), bins=50)
plt.title('Histogram all train counts')
plt.xlabel('Count')
plt.ylabel('Value')
plt.show()
plt.figure(figsize=(12, 5))
plt.hist(np.log(train_df[columns_to_use].values.flatten()+1), bins=50)
plt.title('Log Histogram all train counts')
plt.xlabel('Count')
plt.ylabel('Log value')
plt.show()
sns.set_style("whitegrid")
ax = sns.violinplot(x=np.log(train_df[columns_to_use].values.flatten()+1))
plt.show()
train_nz = np.log(train_df[columns_to_use].values.flatten()+1)
train_nz = train_nz[np.nonzero(train_nz)]
plt.figure(figsize=(12, 5))
plt.hist(train_nz, bins=50)
plt.title('Log Histogram nonzero train counts')
plt.xlabel('Count')
plt.ylabel('Log value')
plt.show()
sns.set_style("whitegrid")
ax = sns.violinplot(x=train_nz)
plt.show()
describe(train_nz)
test_nz = np.log(test_df[columns_to_use].values.flatten()+1)
test_nz = test_nz[np.nonzero(test_nz)]
plt.figure(figsize=(12, 5))
plt.hist(test_nz, bins=50)
plt.title('Log Histogram nonzero test counts')
plt.xlabel('Count')
plt.ylabel('Log value')
plt.show()
sns.set_style("whitegrid")
ax = sns.violinplot(x=test_nz)
plt.show()
describe(test_nz)
train_df[columns_to_use].values.flatten().shape
((train_df[columns_to_use].values.flatten())==0).mean()
train_zeros = pd.DataFrame({'Percentile':((train_df[columns_to_use].values)==0).mean(axis=0),
                           'Column' : columns_to_use})
train_zeros.head()
describe(train_zeros.Percentile.values)
plt.figure(figsize=(12, 5))
plt.hist(train_zeros.Percentile.values, bins=50)
plt.title('Histogram percentage zeros train counts')
plt.xlabel('Count')
plt.ylabel('Value')
plt.show()
describe(np.log(train_df[columns_to_use].values+1), axis=None)
describe(test_df[columns_to_use].values, axis=None)
describe(np.log(test_df[columns_to_use].values+1), axis=None)
test_zeros = pd.DataFrame({'Percentile':(np.log(1+test_df[columns_to_use].values)==0).mean(axis=0),
                           'Column' : columns_to_use})
test_zeros.head()
describe(test_zeros.Percentile.values)
y = np.log(1+train_df.target.values)
y.shape
y
train = lgb.Dataset(train_df[columns_to_use],y ,feature_name = "auto")
params = {'boosting_type': 'gbdt', 
          'objective': 'regression', 
          'metric': 'rmse', 
          'learning_rate': 0.01, 
          'num_leaves': 100, 
          'feature_fraction': 0.4, 
          'bagging_fraction': 0.6, 
          'max_depth': 5, 
          'min_child_weight': 10}


clf = lgb.train(params,
        train,
        num_boost_round = 400,
        verbose_eval=True)

preds = clf.predict(test_df[columns_to_use])
preds
sample_submission = pd.read_csv("../input/sample_submission.csv")
sample_submission.target = np.exp(preds)-1
sample_submission.to_csv('simple_lgbm_1.csv', index=False)
sample_submission.head()
nr_splits = 5
random_state = 1054

y_oof = np.zeros((y.shape[0]))
total_preds = 0


kf = KFold(n_splits=nr_splits, shuffle=True, random_state=random_state)
for i, (train_index, val_index) in enumerate(kf.split(y)):
    print('Fitting fold', i+1, 'out of', nr_splits)
    X_train, X_val  = train_df[columns_to_use].iloc[train_index], train_df[columns_to_use].iloc[val_index]
    y_train, y_val = y[train_index], y[val_index]
    
    train = lgb.Dataset(X_train,y_train ,feature_name = "auto")
    val = lgb.Dataset(X_val ,y_val ,feature_name = "auto")
    clf = lgb.train(params,train,num_boost_round = 400,verbose_eval=True)
    
    total_preds += clf.predict(test_df[columns_to_use])/nr_splits
    pred_oof = clf.predict(X_val)
    y_oof[val_index] = pred_oof
    print('Fold error', np.sqrt(mean_squared_error(y_val, pred_oof)))

print('Total error', np.sqrt(mean_squared_error(y, y_oof)))
params['max_depth'] = 4

y_oof_2 = np.zeros((y.shape[0]))
total_preds_2 = 0


kf = KFold(n_splits=nr_splits, shuffle=True, random_state=random_state)
for i, (train_index, val_index) in enumerate(kf.split(y)):
    print('Fitting fold', i+1, 'out of', nr_splits)
    X_train, X_val  = train_df[columns_to_use].iloc[train_index], train_df[columns_to_use].iloc[val_index]
    y_train, y_val = y[train_index], y[val_index]
    
    train = lgb.Dataset(X_train,y_train ,feature_name = "auto")
    val = lgb.Dataset(X_val ,y_val ,feature_name = "auto")
    clf = lgb.train(params,train,num_boost_round = 400,verbose_eval=True)
    
    total_preds_2 += clf.predict(test_df[columns_to_use])/nr_splits
    pred_oof = clf.predict(X_val)
    y_oof_2[val_index] = pred_oof
    print('Fold error', np.sqrt(mean_squared_error(y_val, pred_oof)))

print('Total error', np.sqrt(mean_squared_error(y, y_oof_2)))
params['max_depth'] = 6

y_oof_3 = np.zeros((y.shape[0]))
total_preds_3 = 0


kf = KFold(n_splits=nr_splits, shuffle=True, random_state=random_state)
for i, (train_index, val_index) in enumerate(kf.split(y)):
    print('Fitting fold', i+1, 'out of', nr_splits)
    X_train, X_val  = train_df[columns_to_use].iloc[train_index], train_df[columns_to_use].iloc[val_index]
    y_train, y_val = y[train_index], y[val_index]
    
    train = lgb.Dataset(X_train,y_train ,feature_name = "auto")
    val = lgb.Dataset(X_val ,y_val ,feature_name = "auto")
    clf = lgb.train(params,train,num_boost_round = 400,verbose_eval=True)
    
    total_preds_3 += clf.predict(test_df[columns_to_use])/nr_splits
    pred_oof = clf.predict(X_val)
    y_oof_3[val_index] = pred_oof
    print('Fold error', np.sqrt(mean_squared_error(y_val, pred_oof)))

print('Total error', np.sqrt(mean_squared_error(y, y_oof_3)))
params['max_depth'] = 7

y_oof_4 = np.zeros((y.shape[0]))
total_preds_4 = 0


kf = KFold(n_splits=nr_splits, shuffle=True, random_state=random_state)
for i, (train_index, val_index) in enumerate(kf.split(y)):
    print('Fitting fold', i+1, 'out of', nr_splits)
    X_train, X_val  = train_df[columns_to_use].iloc[train_index], train_df[columns_to_use].iloc[val_index]
    y_train, y_val = y[train_index], y[val_index]
    
    train = lgb.Dataset(X_train,y_train ,feature_name = "auto")
    val = lgb.Dataset(X_val ,y_val ,feature_name = "auto")
    clf = lgb.train(params,train,num_boost_round = 400,verbose_eval=True)
    
    total_preds_4 += clf.predict(test_df[columns_to_use])/nr_splits
    pred_oof = clf.predict(X_val)
    y_oof_4[val_index] = pred_oof
    print('Fold error', np.sqrt(mean_squared_error(y_val, pred_oof)))

print('Total error', np.sqrt(mean_squared_error(y, y_oof_4)))
params['max_depth'] = 8

y_oof_5 = np.zeros((y.shape[0]))
total_preds_5 = 0


kf = KFold(n_splits=nr_splits, shuffle=True, random_state=random_state)
for i, (train_index, val_index) in enumerate(kf.split(y)):
    print('Fitting fold', i+1, 'out of', nr_splits)
    X_train, X_val  = train_df[columns_to_use].iloc[train_index], train_df[columns_to_use].iloc[val_index]
    y_train, y_val = y[train_index], y[val_index]
    
    train = lgb.Dataset(X_train,y_train ,feature_name = "auto")
    val = lgb.Dataset(X_val ,y_val ,feature_name = "auto")
    clf = lgb.train(params,train,num_boost_round = 400,verbose_eval=True)
    
    total_preds_5 += clf.predict(test_df[columns_to_use])/nr_splits
    pred_oof = clf.predict(X_val)
    y_oof_5[val_index] = pred_oof
    print('Fold error', np.sqrt(mean_squared_error(y_val, pred_oof)))

print('Total error', np.sqrt(mean_squared_error(y, y_oof_5)))
params['max_depth'] = 10

y_oof_6 = np.zeros((y.shape[0]))
total_preds_6 = 0


kf = KFold(n_splits=nr_splits, shuffle=True, random_state=random_state)
for i, (train_index, val_index) in enumerate(kf.split(y)):
    print('Fitting fold', i+1, 'out of', nr_splits)
    X_train, X_val  = train_df[columns_to_use].iloc[train_index], train_df[columns_to_use].iloc[val_index]
    y_train, y_val = y[train_index], y[val_index]
    
    train = lgb.Dataset(X_train,y_train ,feature_name = "auto")
    val = lgb.Dataset(X_val ,y_val ,feature_name = "auto")
    clf = lgb.train(params,train,num_boost_round = 400,verbose_eval=True)
    
    total_preds_6 += clf.predict(test_df[columns_to_use])/nr_splits
    pred_oof = clf.predict(X_val)
    y_oof_6[val_index] = pred_oof
    print('Fold error', np.sqrt(mean_squared_error(y_val, pred_oof)))

print('Total error', np.sqrt(mean_squared_error(y, y_oof_6)))
params['max_depth'] = 12

y_oof_7 = np.zeros((y.shape[0]))
total_preds_7 = 0


kf = KFold(n_splits=nr_splits, shuffle=True, random_state=random_state)
for i, (train_index, val_index) in enumerate(kf.split(y)):
    print('Fitting fold', i+1, 'out of', nr_splits)
    X_train, X_val  = train_df[columns_to_use].iloc[train_index], train_df[columns_to_use].iloc[val_index]
    y_train, y_val = y[train_index], y[val_index]
    
    train = lgb.Dataset(X_train,y_train ,feature_name = "auto")
    val = lgb.Dataset(X_val ,y_val ,feature_name = "auto")
    clf = lgb.train(params,train,num_boost_round = 400,verbose_eval=True)
    
    total_preds_7 += clf.predict(test_df[columns_to_use])/nr_splits
    pred_oof = clf.predict(X_val)
    y_oof_7[val_index] = pred_oof
    print('Fold error', np.sqrt(mean_squared_error(y_val, pred_oof)))

print('Total error', np.sqrt(mean_squared_error(y, y_oof_7)))
print('Total error', np.sqrt(mean_squared_error(y, 1.4*(1.6*y_oof_7-0.6*y_oof_6)-0.4*y_oof_5)))
print('Total error', np.sqrt(mean_squared_error(y, -0.5*y_oof-0.5*y_oof_2-y_oof_3
                                                +3*y_oof_4)))
print('Total error', np.sqrt(mean_squared_error(y, 0.75*(1.4*(1.6*y_oof_7-0.6*y_oof_6)-0.4*y_oof_5)+
                                                0.25*(-0.5*y_oof-0.5*y_oof_2-y_oof_3
                                                +3*y_oof_4))))
sub_preds = (0.75*(1.4*(1.6*total_preds_7-0.6*total_preds_6)-0.4*total_preds_5)+
                                                0.25*(-0.5*total_preds-0.5*total_preds_2-total_preds_3
                                                +3*total_preds_4))
#sub_preds = (-0.5*total_preds-0.5*total_preds_2-total_preds_3+3*total_preds_4)
sample_submission.target = np.exp(sub_preds)-1
sample_submission.to_csv('blended_submission_2.csv', index=False)
sample_submission.head()
params = {'objective': 'reg:linear', 
          'eval_metric': 'rmse',
          'eta': 0.01,
          'max_depth': 10, 
          'subsample': 0.6, 
          'colsample_bytree': 0.6,
          'alpha':0.001,
          'random_state': 42, 
          'silent': True}

y_oof_8 = np.zeros((y.shape[0]))
total_preds_8 = 0

dtest = xgb.DMatrix(test_df[columns_to_use])

kf = KFold(n_splits=nr_splits, shuffle=True, random_state=random_state)
for i, (train_index, val_index) in enumerate(kf.split(y)):
    print('Fitting fold', i+1, 'out of', nr_splits)
    X_train, X_val  = train_df[columns_to_use].iloc[train_index], train_df[columns_to_use].iloc[val_index]
    y_train, y_val = y[train_index], y[val_index]
    
    
    train = xgb.DMatrix(X_train, y_train)
    val = xgb.DMatrix(X_val, y_val)
    
    watchlist = [(train, 'train'), (val, 'val')]
    
    clf = xgb.train(params, train, 1000, watchlist, 
                          maximize=False, early_stopping_rounds = 60, verbose_eval=100)

    
    total_preds_8 += clf.predict(dtest, ntree_limit=clf.best_ntree_limit)/nr_splits
    pred_oof = clf.predict(val, ntree_limit=clf.best_ntree_limit)
    y_oof_8[val_index] = pred_oof
    print('Fold error', np.sqrt(mean_squared_error(y_val, pred_oof)))

print('Total error', np.sqrt(mean_squared_error(y, y_oof_8)))
print('Total error', np.sqrt(mean_squared_error(y, 0.7*(0.75*(1.4*(1.6*y_oof_7-0.6*y_oof_6)-0.4*y_oof_5)+
                                                0.25*(-0.5*y_oof-0.5*y_oof_2-y_oof_3
                                                +3*y_oof_4))+0.3*y_oof_8)))
sub_preds = (0.7*(0.75*(1.4*(1.6*total_preds_7-0.6*total_preds_6)-0.4*total_preds_5)+
                                                0.25*(-0.5*total_preds-0.5*total_preds_2-total_preds_3
                                                +3*total_preds_4))+0.3*total_preds_8)
#sub_preds = (-0.5*total_preds-0.5*total_preds_2-total_preds_3+3*total_preds_4)
sample_submission.target = np.exp(sub_preds)-1
sample_submission.to_csv('blended_submission_3.csv', index=False)
sample_submission.head()
feature_importances = clf.get_fscore()
importance = sorted(feature_importances.items(), key=operator.itemgetter(1))
best_2500 = importance[::-1][:2500]
best_2500 =[ x[0] for x in best_2500]

params = {'objective': 'reg:linear', 
          'eval_metric': 'rmse',
          'eta': 0.01,
          'max_depth': 10, 
          'subsample': 0.6, 
          'colsample_bytree': 0.6,
          'alpha':0.001,
          'random_state': 42, 
          'silent': True}

y_oof_9 = np.zeros((y.shape[0]))
total_preds_9 = 0

dtest = xgb.DMatrix(test_df[best_2500])

kf = KFold(n_splits=nr_splits, shuffle=True, random_state=random_state)
for i, (train_index, val_index) in enumerate(kf.split(y)):
    print('Fitting fold', i+1, 'out of', nr_splits)
    X_train, X_val  = train_df[best_2500].iloc[train_index], train_df[best_2500].iloc[val_index]
    y_train, y_val = y[train_index], y[val_index]
    
    
    train = xgb.DMatrix(X_train, y_train)
    val = xgb.DMatrix(X_val, y_val)
    
    watchlist = [(train, 'train'), (val, 'val')]
    
    clf = xgb.train(params, train, 1500, watchlist, 
                          maximize=False, early_stopping_rounds = 60, verbose_eval=100)

    
    total_preds_9 += clf.predict(dtest, ntree_limit=clf.best_ntree_limit)/nr_splits
    pred_oof = clf.predict(val, ntree_limit=clf.best_ntree_limit)
    y_oof_9[val_index] = pred_oof
    print('Fold error', np.sqrt(mean_squared_error(y_val, pred_oof)))

print('Total error', np.sqrt(mean_squared_error(y, y_oof_9)))
print('Total error', np.sqrt(mean_squared_error(y, 0.5*y_oof_9+0.5*(0.7*(0.75*(1.4*(1.6*y_oof_7-0.6*y_oof_6)-0.4*y_oof_5)+
                                                0.25*(-0.5*y_oof-0.5*y_oof_2-y_oof_3
                                                +3*y_oof_4))+0.3*y_oof_8))))

sub_preds = 0.5*total_preds_9+0.5*(0.7*(0.75*(1.4*(1.6*total_preds_7-0.6*total_preds_6)-0.4*total_preds_5)+
                                                0.25*(-0.5*total_preds-0.5*total_preds_2-total_preds_3
                                                +3*total_preds_4))+0.3*total_preds_8)
#sub_preds = (-0.5*total_preds-0.5*total_preds_2-total_preds_3+3*total_preds_4)
sample_submission.target = np.exp(sub_preds)-1
sample_submission.to_csv('blended_submission_4.csv', index=False)
sample_submission.head()