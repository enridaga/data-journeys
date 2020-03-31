
import numpy as np
import pandas as pd
import time
import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.metrics import mean_squared_error
from sklearn.metrics import log_loss
### %time
df_train = pd.read_csv('../input/predicting-outliers-to-improve-your-score/train_clean.csv')
df_test = pd.read_csv('../input/predicting-outliers-to-improve-your-score/test_clean.csv')
df_train = df_train[df_train['outliers'] == 0]
target = df_train['target']
del df_train['target']
features = [c for c in df_train.columns if c not in ['card_id', 'first_active_month','outliers']]
categorical_feats = [c for c in features if 'feature_' in c]
param = {'objective':'regression',
         'num_leaves': 31,
         'min_data_in_leaf': 25,
         'max_depth': 7,
         'learning_rate': 0.01,
         'lambda_l1':0.13,
         "boosting": "gbdt",
         "feature_fraction":0.85,
         'bagging_freq':8,
         "bagging_fraction": 0.9 ,
         "metric": 'rmse',
         "verbosity": -1,
         "random_state": 2333}
### %time
folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=2333)
oof = np.zeros(len(df_train))
predictions = np.zeros(len(df_test))
feature_importance_df = pd.DataFrame()

for fold_, (trn_idx, val_idx) in enumerate(folds.split(df_train,df_train['outliers'].values)):
    print("fold {}".format(fold_))
    trn_data = lgb.Dataset(df_train.iloc[trn_idx][features], label=target.iloc[trn_idx])#, categorical_feature=categorical_feats)
    val_data = lgb.Dataset(df_train.iloc[val_idx][features], label=target.iloc[val_idx])#, categorical_feature=categorical_feats)

    num_round = 10000
    clf = lgb.train(param, trn_data, num_round, valid_sets = [trn_data, val_data], verbose_eval= 100, early_stopping_rounds = 200)
    oof[val_idx] = clf.predict(df_train.iloc[val_idx][features], num_iteration=clf.best_iteration)
    
    fold_importance_df = pd.DataFrame()
    fold_importance_df["Feature"] = features
    fold_importance_df["importance"] = clf.feature_importance()
    fold_importance_df["fold"] = fold_ + 1
    feature_importance_df = pd.concat([feature_importance_df, fold_importance_df], axis=0)
    
    predictions += clf.predict(df_test[features], num_iteration=clf.best_iteration) / folds.n_splits

print("CV score: {:<8.5f}".format(mean_squared_error(oof, target)**0.5))
model_without_outliers = pd.DataFrame({"card_id":df_test["card_id"].values})
model_without_outliers["target"] = predictions
### %time
df_train = pd.read_csv('../input/predicting-outliers-to-improve-your-score/train_clean.csv')
df_test = pd.read_csv('../input/predicting-outliers-to-improve-your-score/test_clean.csv')
target = df_train['outliers']
del df_train['outliers']
del df_train['target']
features = [c for c in df_train.columns if c not in ['card_id', 'first_active_month']]
categorical_feats = [c for c in features if 'feature_' in c]
param = {'num_leaves': 31,
         'min_data_in_leaf': 30, 
         'objective':'binary',
         'max_depth': 6,
         'learning_rate': 0.01,
         "boosting": "rf",
         "feature_fraction": 0.9,
         "bagging_freq": 1,
         "bagging_fraction": 0.9 ,
         "bagging_seed": 11,
         "metric": 'binary_logloss',
         "lambda_l1": 0.1,
         "verbosity": -1,
         "random_state": 2333}
### %time
folds = KFold(n_splits=5, shuffle=True, random_state=15)
oof = np.zeros(len(df_train))
predictions = np.zeros(len(df_test))
feature_importance_df = pd.DataFrame()

start = time.time()


for fold_, (trn_idx, val_idx) in enumerate(folds.split(df_train.values, target.values)):
    print("fold n°{}".format(fold_))
    trn_data = lgb.Dataset(df_train.iloc[trn_idx][features], label=target.iloc[trn_idx], categorical_feature=categorical_feats)
    val_data = lgb.Dataset(df_train.iloc[val_idx][features], label=target.iloc[val_idx], categorical_feature=categorical_feats)

    num_round = 10000
    clf = lgb.train(param, trn_data, num_round, valid_sets = [trn_data, val_data], verbose_eval=100, early_stopping_rounds = 200)
    oof[val_idx] = clf.predict(df_train.iloc[val_idx][features], num_iteration=clf.best_iteration)
    
    fold_importance_df = pd.DataFrame()
    fold_importance_df["feature"] = features
    fold_importance_df["importance"] = clf.feature_importance()
    fold_importance_df["fold"] = fold_ + 1
    feature_importance_df = pd.concat([feature_importance_df, fold_importance_df], axis=0)
    
    predictions += clf.predict(df_test[features], num_iteration=clf.best_iteration) / folds.n_splits

print("CV score: {:<8.5f}".format(log_loss(target, oof)))
### 'target' is the probability of whether an observation is an outlier
df_outlier_prob = pd.DataFrame({"card_id":df_test["card_id"].values})
df_outlier_prob["target"] = predictions
df_outlier_prob.head()
# if the test set has the same ratio of outliers as training set, 
# then the numbuer of outliers in test is about: (1.06% outliers in training set)
123623*0.0106
# In case missing some predictable outlier, we choose top 25000 with highest outliers likelyhood.
outlier_id = pd.DataFrame(df_outlier_prob.sort_values(by='target',ascending = False).head(25000)['card_id'])
best_submission = pd.read_csv('../input/predicting-outliers-to-improve-your-score/3.695.csv')
most_likely_liers = best_submission.merge(outlier_id,how='right')
most_likely_liers.head()
### %time
for card_id in most_likely_liers['card_id']:
    model_without_outliers.loc[model_without_outliers['card_id']==card_id,'target']\
    = most_likely_liers.loc[most_likely_liers['card_id']==card_id,'target'].values
model_without_outliers.to_csv("combining_submission.csv", index=False)