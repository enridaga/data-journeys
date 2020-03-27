
# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.
def degToCompass(num):
    val=int((num/22.5)+.5)
    arr=[i for i in range(0,16)]
    return arr[(val % 16)]
building_df = pd.read_csv("../input/ashrae-energy-prediction/building_metadata.csv")
weather_train = pd.read_csv("../input/ashrae-energy-prediction/weather_train.csv")
train = pd.read_csv("../input/ashrae-energy-prediction/train.csv")

train = train.merge(building_df, left_on = "building_id", right_on = "building_id", how = "left")
train = train.merge(weather_train, left_on = ["site_id", "timestamp"], right_on = ["site_id", "timestamp"])
del weather_train
train["timestamp"] = pd.to_datetime(train["timestamp"])
train["weekday"] = train["timestamp"].dt.weekday
train["hour"] = train["timestamp"].dt.hour
train["weekday"] = train['weekday'].astype(np.uint8)
train["hour"] = train['hour'].astype(np.uint8)
train['year_built'] = train['year_built']-1900
train['square_feet'] = np.log(train['square_feet'])
def average_imputation(df, column_name):
    imputation = df.groupby(['timestamp'])[column_name].mean()
    
    df.loc[df[column_name].isnull(), column_name] = df[df[column_name].isnull()][[column_name]].apply(lambda x: imputation[df['timestamp'][x.index]].values)
    del imputation
    return df
train = average_imputation(train, 'wind_speed')
train = average_imputation(train, 'wind_direction')

beaufort = [(0, 0, 0.3), (1, 0.3, 1.6), (2, 1.6, 3.4), (3, 3.4, 5.5), (4, 5.5, 8), (5, 8, 10.8), (6, 10.8, 13.9), 
          (7, 13.9, 17.2), (8, 17.2, 20.8), (9, 20.8, 24.5), (10, 24.5, 28.5), (11, 28.5, 33), (12, 33, 200)]

for item in beaufort:
    train.loc[(train['wind_speed']>=item[1]) & (train['wind_speed']<item[2]), 'beaufort_scale'] = item[0]
del train["timestamp"]
train['wind_direction'] = train['wind_direction'].apply(degToCompass)
train['beaufort_scale'] = train['beaufort_scale'].astype(np.uint8)
train["wind_direction"] = train['wind_direction'].astype(np.uint8)
train["meter"] = train['meter'].astype(np.uint8)
train["site_id"] = train['site_id'].astype(np.uint8)
from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()
train["primary_use"] = le.fit_transform(train["primary_use"])

categoricals = ["site_id", "building_id", "primary_use", "hour", "weekday", "meter",  "wind_direction"]
drop_cols = ["sea_level_pressure", "wind_speed"]

numericals = ["square_feet", "year_built", "air_temperature", "cloud_coverage",
              "dew_temperature", 'precip_depth_1_hr', 'floor_count', 'beaufort_scale']

feat_cols = categoricals + numericals
target = np.log1p(train["meter_reading"])

del train["meter_reading"] 

train = train.drop(drop_cols, axis = 1)
from sklearn.metrics import mean_squared_error
import lightgbm as lgb
from sklearn.model_selection import KFold, StratifiedKFold
from tqdm import tqdm


params = {
            'boosting_type': 'gbdt',
            'objective': 'regression',
            'metric': {'rmse'},
            'subsample': 0.25,
            'subsample_freq': 1,
            'learning_rate': 0.4,
            'num_leaves': 20,
            'feature_fraction': 0.9,
            'lambda_l1': 1,  
            'lambda_l2': 1
            }

folds = 4
seed = 666

kf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)

models = []
for train_index, val_index in kf.split(train, train['building_id']):
    train_X = train[feat_cols].iloc[train_index]
    val_X = train[feat_cols].iloc[val_index]
    train_y = target.iloc[train_index]
    val_y = target.iloc[val_index]
    lgb_train = lgb.Dataset(train_X, train_y, categorical_feature=categoricals)
    lgb_eval = lgb.Dataset(val_X, val_y, categorical_feature=categoricals)
    gbm = lgb.train(params,
                lgb_train,
                num_boost_round=500,
                valid_sets=(lgb_train, lgb_eval),
                early_stopping_rounds=100,
                verbose_eval = 100)
    models.append(gbm)
import gc
del train, train_X, val_X, lgb_train, lgb_eval, train_y, val_y, target
gc.collect()
test = pd.read_csv("../input/ashrae-energy-prediction/test.csv")
test = test.merge(building_df, left_on = "building_id", right_on = "building_id", how = "left")
del building_df
gc.collect()
test["primary_use"] = le.transform(test["primary_use"])

weather_test = pd.read_csv("../input/ashrae-energy-prediction/weather_test.csv")

test = test.merge(weather_test, left_on = ["site_id", "timestamp"], right_on = ["site_id", "timestamp"], how = "left")
del weather_test
gc.collect()
test["timestamp"] = pd.to_datetime(test["timestamp"])
test["hour"] = test["timestamp"].dt.hour
test["weekday"] = test["timestamp"].dt.weekday
test["weekday"] = test['weekday'].astype(np.uint8)
test["hour"] = test['hour'].astype(np.uint8)
test['year_built'] = test['year_built']-1900
test['square_feet'] = np.log(test['square_feet'])

test = average_imputation(test, 'wind_speed')
test = average_imputation(test, 'wind_direction')

for item in beaufort:
    test.loc[(test['wind_speed']>=item[1]) & (test['wind_speed']<item[2]), 'beaufort_scale'] = item[0]
test['wind_direction'] = test['wind_direction'].apply(degToCompass)

test['wind_direction'] = test['wind_direction'].apply(degToCompass)
test['beaufort_scale'] = test['beaufort_scale'].astype(np.uint8)
test["wind_direction"] = test['wind_direction'].astype(np.uint8)
test["meter"] = test['meter'].astype(np.uint8)
test["site_id"] = test['site_id'].astype(np.uint8)

test = test[feat_cols]
i=0
res=[]
step_size = 50000
for j in tqdm(range(int(np.ceil(test.shape[0]/50000)))):
    res.append(np.expm1(sum([model.predict(test.iloc[i:i+step_size]) for model in models])/folds))
    i+=step_size
res = np.concatenate(res)
submission = pd.read_csv('/kaggle/input/ashrae-energy-prediction/sample_submission.csv')
submission['meter_reading'] = res
submission.loc[submission['meter_reading']<0, 'meter_reading'] = 0
submission.to_csv('submission.csv', index=False)
submission