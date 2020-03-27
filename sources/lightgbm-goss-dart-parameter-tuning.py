
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns
import lightgbm as lgb
from sklearn.model_selection import KFold
import warnings
import time
import sys
import datetime
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import BayesianRidge
warnings.simplefilter(action='ignore', category=FutureWarning)

from contextlib import contextmanager
from pandas.core.common import SettingWithCopyWarning
import gc
from tqdm import tqdm_notebook
warnings.simplefilter(action='ignore', category=SettingWithCopyWarning)
warnings.simplefilter(action='ignore', category=FutureWarning)
FEATS_EXCLUDED = ['first_active_month', 'target', 'card_id', 'outliers',
                  'hist_purchase_date_max', 'hist_purchase_date_min', 'hist_card_id_size',
                  'new_purchase_date_max', 'new_purchase_date_min', 'new_card_id_size',
                  'OOF_PRED', 'month_0']
@contextmanager
def timer(title):
    t0 = time.time()
    yield
    print("{} - done in {:.0f}s".format(title, time.time() - t0))

# rmse
def rmse(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))

# One-hot encoding for categorical columns with get_dummies
def one_hot_encoder(df, nan_as_category = True):
    original_columns = list(df.columns)
    categorical_columns = [col for col in df.columns if df[col].dtype == 'object']
    df = pd.get_dummies(df, columns= categorical_columns, dummy_na= nan_as_category)
    new_columns = [c for c in df.columns if c not in original_columns]
    return df, new_columns
    
# Display/plot feature importance
def display_importances(feature_importance_df_):
    cols = feature_importance_df_[["feature", "importance"]].groupby("feature").mean().sort_values(by="importance", ascending=False)[:40].index
    best_features = feature_importance_df_.loc[feature_importance_df_.feature.isin(cols)]

    plt.figure(figsize=(8, 10))
    sns.barplot(x="importance", y="feature", data=best_features.sort_values(by="importance", ascending=False))
    plt.title('LightGBM Features (avg over folds)')
    plt.tight_layout()
    plt.savefig('lgbm_importances.png')
    
def reduce_mem_usage(df, verbose=True):
    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
    start_mem = df.memory_usage().sum() / 1024**2    
    for col in df.columns:
        col_type = df[col].dtypes
        if col_type in numerics:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)  
            else:
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)    
    end_mem = df.memory_usage().sum() / 1024**2
    if verbose: print('Mem. usage decreased to {:5.2f} Mb ({:.1f}% reduction)'.format(end_mem, 100 * (start_mem - end_mem) / start_mem))
    return df
%%time
# load csv
def load_data():
    train_df = pd.read_csv('../input/elo-blending/train_feature.csv')
    test_df = pd.read_csv('../input/elo-blending/test_feature.csv')
    display(train_df.head())
    display(test_df.head())
    print(train_df.shape,test_df.shape)
    train = pd.read_csv('../input/elo-merchant-category-recommendation/train.csv', index_col=['card_id'])
    test = pd.read_csv('../input/elo-merchant-category-recommendation/test.csv', index_col=['card_id'])
    train_df['card_id'] = train.index
    test_df['card_id'] = test.index
    train_df.index = train_df['card_id']
    test_df.index = test_df['card_id']
    display(train_df.head())
    display(test.head())
    print(train.shape,test.shape)
    del train,test    
    return (train_df,test_df)

print(gc.collect())
# train_df,test_df = load_data()
boosting = ["goss","dart"]
boosting[0],boosting[1]
%%time

boosting = ["goss","dart"]
# LightGBM GBDT with KFold or Stratified KFold
def kfold_lightgbm(train_df, test_df, num_folds, stratified = False, boosting = boosting[0]):
    print("Starting LightGBM. Train shape: {}, test shape: {}".format(train_df.shape, test_df.shape))

    # Cross validation model
    if stratified:
        folds = StratifiedKFold(n_splits= num_folds, shuffle=True, random_state=326)
    else:
        folds = KFold(n_splits= num_folds, shuffle=True, random_state=2045)

    # Create arrays and dataframes to store results
    oof_preds = np.zeros(train_df.shape[0])
    sub_preds = np.zeros(test_df.shape[0])
    feature_importance_df = pd.DataFrame()
    feats = [f for f in train_df.columns if f not in FEATS_EXCLUDED]

    # k-fold
    for n_fold, (train_idx, valid_idx) in enumerate(folds.split(train_df[feats], train_df['outliers'])):
        train_x, train_y = train_df[feats].iloc[train_idx], train_df['target'].iloc[train_idx]
        valid_x, valid_y = train_df[feats].iloc[valid_idx], train_df['target'].iloc[valid_idx]

        # set data structure
        lgb_train = lgb.Dataset(train_x,label=train_y,free_raw_data=False)
        lgb_test = lgb.Dataset(valid_x,label=valid_y,free_raw_data=False)

        # params optimized by optuna
        params ={
                        'task': 'train',
                        'boosting': 'goss',
                        'objective': 'regression',
                        'metric': 'rmse',
                        'learning_rate': 0.005,
                        'subsample': 0.9855232997390695,
                        'max_depth': 8,
                        'top_rate': 0.9064148448434349,
                        'num_leaves': 87,
                        'min_child_weight': 41.9612869171337,
                        'other_rate': 0.0721768246018207,
                        'reg_alpha': 9.677537745007898,
                        'colsample_bytree': 0.5665320670155495,
                        'min_split_gain': 9.820197773625843,
                        'reg_lambda': 8.2532317400459,
                        'min_data_in_leaf': 21,
                        'verbose': -1,
                        'seed':int(2**n_fold),
                        'bagging_seed':int(2**n_fold),
                        'drop_seed':int(2**n_fold)
                        }

        reg = lgb.train(
                        params,
                        lgb_train,
                        valid_sets=[lgb_train, lgb_test],
                        valid_names=['train', 'test'],
                        num_boost_round=10000,
                        early_stopping_rounds= 200,
                        verbose_eval=100
                        )

        oof_preds[valid_idx] = reg.predict(valid_x, num_iteration=reg.best_iteration)
        sub_preds += reg.predict(test_df[feats], num_iteration=reg.best_iteration) / folds.n_splits

        fold_importance_df = pd.DataFrame()
        fold_importance_df["feature"] = feats
        fold_importance_df["importance"] = np.log1p(reg.feature_importance(importance_type='gain', iteration=reg.best_iteration))
        fold_importance_df["fold"] = n_fold + 1
        feature_importance_df = pd.concat([feature_importance_df, fold_importance_df], axis=0)
        print('Fold %2d RMSE : %.6f' % (n_fold + 1, rmse(valid_y, oof_preds[valid_idx])))
        del reg, train_x, train_y, valid_x, valid_y
        gc.collect()

    # display importances
    display_importances(feature_importance_df)
    
        # save submission file
    submission = pd.read_csv("../input/elo-merchant-category-recommendation/sample_submission.csv")
    submission['target'] = sub_preds
    submission.to_csv(boosting+".csv", index=False)
    display(submission.head())
    return (submission)
%%time
train_df,test_df = load_data()
print(gc.collect())
submission = kfold_lightgbm(train_df, test_df, num_folds=7, stratified=False, boosting=boosting[0])
submission1 = kfold_lightgbm(train_df, test_df, num_folds=7, stratified=False, boosting=boosting[1])
final = pd.read_csv("../input/elo-merchant-category-recommendation/sample_submission.csv")
final['target'] = submission['target'] * 0.5 + submission1['target'] * 0.5
final.to_csv("blend.csv",index = False)  