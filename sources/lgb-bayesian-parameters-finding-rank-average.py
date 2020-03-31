
### pip install bayesian-optimization
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from scipy.stats import rankdata
import lightgbm as lgb
from sklearn import metrics
import gc
import warnings

pd.set_option('display.max_columns', 200)
train_df = pd.read_csv('../input/train.csv')

test_df = pd.read_csv('../input/test.csv')
train_df.head()
test_df.head()
target = 'target'
predictors = train_df.columns.values.tolist()[2:]
train_df.target.value_counts()
bayesian_tr_index, bayesian_val_index  = list(StratifiedKFold(n_splits=2, shuffle=True, random_state=1).split(train_df, train_df.target.values))[0]
def LGB_bayesian(
    num_leaves,  # int
    min_data_in_leaf,  # int
    learning_rate,
    min_sum_hessian_in_leaf,    # int  
    feature_fraction,
    lambda_l1,
    lambda_l2,
    min_gain_to_split,
    max_depth):
    
    # LightGBM expects next three parameters need to be integer. So we make them integer
    num_leaves = int(num_leaves)
    min_data_in_leaf = int(min_data_in_leaf)
    max_depth = int(max_depth)

    assert type(num_leaves) == int
    assert type(min_data_in_leaf) == int
    assert type(max_depth) == int

    param = {
        'num_leaves': num_leaves,
        'max_bin': 63,
        'min_data_in_leaf': min_data_in_leaf,
        'learning_rate': learning_rate,
        'min_sum_hessian_in_leaf': min_sum_hessian_in_leaf,
        'bagging_fraction': 1.0,
        'bagging_freq': 5,
        'feature_fraction': feature_fraction,
        'lambda_l1': lambda_l1,
        'lambda_l2': lambda_l2,
        'min_gain_to_split': min_gain_to_split,
        'max_depth': max_depth,
        'save_binary': True, 
        'seed': 1337,
        'feature_fraction_seed': 1337,
        'bagging_seed': 1337,
        'drop_seed': 1337,
        'data_random_seed': 1337,
        'objective': 'binary',
        'boosting_type': 'gbdt',
        'verbose': 1,
        'metric': 'auc',
        'is_unbalance': True,
        'boost_from_average': False,   

    }    
    
    
    xg_train = lgb.Dataset(train_df.iloc[bayesian_tr_index][predictors].values,
                           label=train_df.iloc[bayesian_tr_index][target].values,
                           feature_name=predictors,
                           free_raw_data = False
                           )
    xg_valid = lgb.Dataset(train_df.iloc[bayesian_val_index][predictors].values,
                           label=train_df.iloc[bayesian_val_index][target].values,
                           feature_name=predictors,
                           free_raw_data = False
                           )   

    num_round = 5000
    clf = lgb.train(param, xg_train, num_round, valid_sets = [xg_valid], verbose_eval=250, early_stopping_rounds = 50)
    
    predictions = clf.predict(train_df.iloc[bayesian_val_index][predictors].values, num_iteration=clf.best_iteration)   
    
    score = metrics.roc_auc_score(train_df.iloc[bayesian_val_index][target].values, predictions)
    
    return score
# Bounded region of parameter space
bounds_LGB = {
    'num_leaves': (5, 20), 
    'min_data_in_leaf': (5, 20),  
    'learning_rate': (0.01, 0.3),
    'min_sum_hessian_in_leaf': (0.00001, 0.01),    
    'feature_fraction': (0.05, 0.5),
    'lambda_l1': (0, 5.0), 
    'lambda_l2': (0, 5.0), 
    'min_gain_to_split': (0, 1.0),
    'max_depth':(3,15),
}
from bayes_opt import BayesianOptimization
LGB_BO = BayesianOptimization(LGB_bayesian, bounds_LGB, random_state=13)
print(LGB_BO.space.keys)
init_points = 5
n_iter = 5
print('-' * 130)

with warnings.catch_warnings():
    warnings.filterwarnings('ignore')
    LGB_BO.maximize(init_points=init_points, n_iter=n_iter, acq='ucb', xi=0.0, alpha=1e-6)
LGB_BO.max['target']
LGB_BO.max['params']
# parameters from version 2 of
#https://www.kaggle.com/fayzur/customer-transaction-prediction?scriptVersionId=10522231

LGB_BO.probe(
    params={'feature_fraction': 0.1403, 
            'lambda_l1': 4.218, 
            'lambda_l2': 1.734, 
            'learning_rate': 0.07, 
            'max_depth': 14, 
            'min_data_in_leaf': 17, 
            'min_gain_to_split': 0.1501, 
            'min_sum_hessian_in_leaf': 0.000446, 
            'num_leaves': 6},
    lazy=True, # 
)
LGB_BO.maximize(init_points=0, n_iter=0) # remember no init_points or n_iter
for i, res in enumerate(LGB_BO.res):
    print("Iteration {}: \n\t{}".format(i, res))
LGB_BO.max['target']
LGB_BO.max['params']
param_lgb = {
        'num_leaves': int(LGB_BO.max['params']['num_leaves']), # remember to int here
        'max_bin': 63,
        'min_data_in_leaf': int(LGB_BO.max['params']['min_data_in_leaf']), # remember to int here
        'learning_rate': LGB_BO.max['params']['learning_rate'],
        'min_sum_hessian_in_leaf': LGB_BO.max['params']['min_sum_hessian_in_leaf'],
        'bagging_fraction': 1.0, 
        'bagging_freq': 5, 
        'feature_fraction': LGB_BO.max['params']['feature_fraction'],
        'lambda_l1': LGB_BO.max['params']['lambda_l1'],
        'lambda_l2': LGB_BO.max['params']['lambda_l2'],
        'min_gain_to_split': LGB_BO.max['params']['min_gain_to_split'],
        'max_depth': int(LGB_BO.max['params']['max_depth']), # remember to int here
        'save_binary': True,
        'seed': 1337,
        'feature_fraction_seed': 1337,
        'bagging_seed': 1337,
        'drop_seed': 1337,
        'data_random_seed': 1337,
        'objective': 'binary',
        'boosting_type': 'gbdt',
        'verbose': 1,
        'metric': 'auc',
        'is_unbalance': True,
        'boost_from_average': False,
    }
nfold = 5
gc.collect()
skf = StratifiedKFold(n_splits=nfold, shuffle=True, random_state=2019)
oof = np.zeros(len(train_df))
predictions = np.zeros((len(test_df),nfold))

i = 1
for train_index, valid_index in skf.split(train_df, train_df.target.values):
    print("\nfold {}".format(i))
    xg_train = lgb.Dataset(train_df.iloc[train_index][predictors].values,
                           label=train_df.iloc[train_index][target].values,
                           feature_name=predictors,
                           free_raw_data = False
                           )
    xg_valid = lgb.Dataset(train_df.iloc[valid_index][predictors].values,
                           label=train_df.iloc[valid_index][target].values,
                           feature_name=predictors,
                           free_raw_data = False
                           )   

    
    clf = lgb.train(param_lgb, xg_train, 5000, valid_sets = [xg_valid], verbose_eval=250, early_stopping_rounds = 50)
    oof[valid_index] = clf.predict(train_df.iloc[valid_index][predictors].values, num_iteration=clf.best_iteration) 
    
    predictions[:,i-1] += clf.predict(test_df[predictors], num_iteration=clf.best_iteration)
    i = i + 1

print("\n\nCV AUC: {:<0.2f}".format(metrics.roc_auc_score(train_df.target.values, oof)))
predictions
print("Rank averaging on", nfold, "fold predictions")
rank_predictions = np.zeros((predictions.shape[0],1))
for i in range(nfold):
    rank_predictions[:, 0] = np.add(rank_predictions[:, 0], rankdata(predictions[:, i].reshape(-1,1))/rank_predictions.shape[0]) 

rank_predictions /= nfold
sub_df = pd.DataFrame({"ID_code": test_df.ID_code.values})
sub_df["target"] = rank_predictions
sub_df[:10]
sub_df.to_csv("Customer_Transaction_rank_predictions.csv", index=False)
