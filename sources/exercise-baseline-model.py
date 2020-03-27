
# Set up code checking
from learntools.core import binder
binder.bind(globals())
from learntools.feature_engineering.ex1 import *

import pandas as pd

click_data = pd.read_csv('../input/feature-engineering-data/train_sample.csv',
                         parse_dates=['click_time'])
click_data.head(10)
# Add new columns for timestamp features day, hour, minute, and second
clicks = click_data.copy()
clicks['day'] = clicks['click_time'].dt.day.astype('uint8')
# Fill in the rest
clicks['hour'] = clicks.click_time.dt.hour.astype('uint8')
clicks['minute'] = clicks.click_time.dt.minute.astype('uint8')
clicks['second'] = clicks.click_time.dt.second.astype('uint8')

q_1.check()
# Uncomment these if you need guidance
# q_1.hint()
# q_1.solution()
from sklearn.preprocessing import LabelEncoder

cat_features = ['ip', 'app', 'device', 'os', 'channel']
encoder = LabelEncoder()
# Create new columns in clicks using preprocessing.LabelEncoder()
for feature in cat_features:
    encoder.fit(clicks[feature])
    clicks[feature+"_labels"] =  encoder.transform(clicks[feature])

q_2.check()
# Uncomment these if you need guidance
# q_2.hint()
# q_2.solution()
clicks.head()
q_3.solution()
 q_4.solution()
feature_cols = ['day', 'hour', 'minute', 'second', 
                'ip_labels', 'app_labels', 'device_labels',
                'os_labels', 'channel_labels']

valid_fraction = 0.1
clicks_srt = clicks.sort_values('click_time')
valid_rows = int(len(clicks_srt) * valid_fraction)
train = clicks_srt[:-valid_rows * 2]
# valid size == test size, last two sections of the data
valid = clicks_srt[-valid_rows * 2:-valid_rows]
test = clicks_srt[-valid_rows:]
import lightgbm as lgb

dtrain = lgb.Dataset(train[feature_cols], label=train['is_attributed'])
dvalid = lgb.Dataset(valid[feature_cols], label=valid['is_attributed'])
dtest = lgb.Dataset(test[feature_cols], label=test['is_attributed'])

param = {'num_leaves': 64, 'objective': 'binary'}
param['metric'] = 'auc'
num_round = 1000
bst = lgb.train(param, dtrain, num_round, valid_sets=[dvalid], early_stopping_rounds=10)
from sklearn import metrics

ypred = bst.predict(test[feature_cols])
score = metrics.roc_auc_score(test['is_attributed'], ypred)
print(f"Test score: {score}")