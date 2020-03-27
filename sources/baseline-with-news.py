
import gc
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
from sklearn.metrics import log_loss
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
import seaborn as sns
import matplotlib.pyplot as plt
from kaggle.competitions import twosigmanews
pd.set_option('max_columns', 50)
env = twosigmanews.make_env()
market_train, news_train = env.get_training_data()
start = datetime(2009, 1, 1, 0, 0, 0).date()
market_train = market_train.loc[market_train['time'].dt.date >= start].reset_index(drop=True)
news_train = news_train.loc[news_train['time'].dt.date >= start].reset_index(drop=True)
market_train.head(3)
news_train.head(3)
def preprocess_news(news_train):
    drop_list = [
        'audiences', 'subjects', 'assetName',
        'headline', 'firstCreated', 'sourceTimestamp',
    ]
    news_train.drop(drop_list, axis=1, inplace=True)
    
    # Factorize categorical columns
    for col in ['headlineTag', 'provider', 'sourceId']:
        news_train[col], uniques = pd.factorize(news_train[col])
        del uniques
    
    # Remove {} and '' from assetCodes column
    news_train['assetCodes'] = news_train['assetCodes'].apply(lambda x: x[1:-1].replace("'", ""))
    return news_train

news_train = preprocess_news(news_train)
def unstack_asset_codes(news_train):
    codes = []
    indexes = []
    for i, values in news_train['assetCodes'].iteritems():
        explode = values.split(", ")
        codes.extend(explode)
        repeat_index = [int(i)]*len(explode)
        indexes.extend(repeat_index)
    index_df = pd.DataFrame({'news_index': indexes, 'assetCode': codes})
    del codes, indexes
    gc.collect()
    return index_df

index_df = unstack_asset_codes(news_train)
index_df.head()
def merge_news_on_index(news_train, index_df):
    news_train['news_index'] = news_train.index.copy()

    # Merge news on unstacked assets
    news_unstack = index_df.merge(news_train, how='left', on='news_index')
    news_unstack.drop(['news_index', 'assetCodes'], axis=1, inplace=True)
    return news_unstack

news_unstack = merge_news_on_index(news_train, index_df)
del news_train, index_df
gc.collect()
news_unstack.head(3)
def group_news(news_frame):
    news_frame['date'] = news_frame.time.dt.date  # Add date column
    
    aggregations = ['mean']
    gp = news_frame.groupby(['assetCode', 'date']).agg(aggregations)
    gp.columns = pd.Index(["{}_{}".format(e[0], e[1]) for e in gp.columns.tolist()])
    gp.reset_index(inplace=True)
    # Set datatype to float32
    float_cols = {c: 'float32' for c in gp.columns if c not in ['assetCode', 'date']}
    return gp.astype(float_cols)

news_agg = group_news(news_unstack)
del news_unstack; gc.collect()
news_agg.head(3)
market_train['date'] = market_train.time.dt.date
df = market_train.merge(news_agg, how='left', on=['assetCode', 'date'])
del market_train, news_agg
gc.collect()
df.head(3)
def custom_metric(date, pred_proba, num_target, universe):
    y = pred_proba*2 - 1
    r = num_target.clip(-1,1) # get rid of outliers
    x = y * r * universe
    result = pd.DataFrame({'day' : date, 'x' : x})
    x_t = result.groupby('day').sum().values
    return np.mean(x_t) / np.std(x_t)
date = df.date
num_target = df.returnsOpenNextMktres10.astype('float32')
bin_target = (df.returnsOpenNextMktres10 >= 0).astype('int8')
universe = df.universe.astype('int8')
# Drop columns that are not features
df.drop(['returnsOpenNextMktres10', 'date', 'universe', 'assetCode', 'assetName', 'time'], 
        axis=1, inplace=True)
df = df.astype('float32')  # Set all remaining columns to float32 datatype
gc.collect()
train_index, test_index = train_test_split(df.index.values, test_size=0.1, shuffle=False)
def evaluate_model(df, target, train_index, test_index, params):
    params['n_jobs'] = 2  # Use 2 cores/threads
    #model = XGBClassifier(**params)
    model = LGBMClassifier(**params)
    model.fit(df.iloc[train_index], target.iloc[train_index])
    return log_loss(target.iloc[test_index], model.predict_proba(df.iloc[test_index]))
param_grid = {
    'learning_rate': [0.15, 0.1, 0.05, 0.02, 0.01],
    'num_leaves': [i for i in range(12, 90, 6)],
    'n_estimators': [50, 200, 400, 600, 800],
    'min_child_samples': [i for i in range(10, 100, 10)],
    'colsample_bytree': [0.8, 0.9, 0.95, 1],
    'subsample': [0.8, 0.9, 0.95, 1],
    'reg_alpha': [0.1, 0.2, 0.4, 0.6, 0.8],
    'reg_lambda': [0.1, 0.2, 0.4, 0.6, 0.8],
}

best_eval_score = 0
for i in range(100):  # Hundred runs
    params = {k: np.random.choice(v) for k, v in param_grid.items()}
    score = evaluate_model(df, bin_target, train_index, test_index, params)
    if score < best_eval_score or best_eval_score == 0:
        best_eval_score = score
        best_params = params
print("Best evaluation logloss", best_eval_score)
# Train model with full data
clf = LGBMClassifier(**best_params)
clf.fit(df, bin_target)
def write_submission(model, env):
    days = env.get_prediction_days()
    for (market_obs_df, news_obs_df, predictions_template_df) in days:
        news_obs_df = preprocess_news(news_obs_df)
        # Unstack news
        index_df = unstack_asset_codes(news_obs_df)
        news_unstack = merge_news_on_index(news_obs_df, index_df)
        # Group and and get aggregations (mean)
        news_obs_agg = group_news(news_unstack)

        # Join market and news frames
        market_obs_df['date'] = market_obs_df.time.dt.date
        obs_df = market_obs_df.merge(news_obs_agg, how='left', on=['assetCode', 'date'])
        del market_obs_df, news_obs_agg, news_obs_df, news_unstack, index_df
        gc.collect()
        obs_df = obs_df[obs_df.assetCode.isin(predictions_template_df.assetCode)]
        
        # Drop cols that are not features
        feats = [c for c in obs_df.columns if c not in ['date', 'assetCode', 'assetName', 'time']]

        preds = model.predict_proba(obs_df[feats])[:, 1] * 2 - 1
        sub = pd.DataFrame({'assetCode': obs_df['assetCode'], 'confidence': preds})
        predictions_template_df = predictions_template_df.merge(sub, how='left').drop(
            'confidenceValue', axis=1).fillna(0).rename(columns={'confidence':'confidenceValue'})
        
        env.predict(predictions_template_df)
        del obs_df, predictions_template_df, preds, sub
        gc.collect()
    env.write_submission_file()
    
write_submission(clf, env)
feat_importance = pd.DataFrame()
feat_importance["feature"] = df.columns
feat_importance["gain"] = clf.booster_.feature_importance(importance_type='gain')
feat_importance.sort_values(by='gain', ascending=False, inplace=True)
plt.figure(figsize=(8,10))
ax = sns.barplot(y="feature", x="gain", data=feat_importance)