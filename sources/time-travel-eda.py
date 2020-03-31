
import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

### matplotlib inline
train = pd.read_csv('../input/act_train.csv', parse_dates=['date'])

test = pd.read_csv('../input/act_test.csv', parse_dates=['date'])

ppl = pd.read_csv('../input/people.csv', parse_dates=['date'])



df_train = pd.merge(train, ppl, on='people_id')

df_test = pd.merge(test, ppl, on='people_id')

del train, test, ppl
for d in ['date_x', 'date_y']:

    print('Start of ' + d + ': ' + str(df_train[d].min().date()))

    print('  End of ' + d + ': ' + str(df_train[d].max().date()))

    print('Range of ' + d + ': ' + str(df_train[d].max() - df_train[d].min()) + '\n')
date_x = pd.DataFrame()

date_x['Class probability'] = df_train.groupby('date_x')['outcome'].mean()

date_x['Frequency'] = df_train.groupby('date_x')['outcome'].size()

date_x.plot(secondary_y='Frequency', figsize=(20, 10))
date_y = pd.DataFrame()

date_y['Class probability'] = df_train.groupby('date_y')['outcome'].mean()

date_y['Frequency'] = df_train.groupby('date_y')['outcome'].size()

# We need to split it into multiple graphs since the time-scale is too long to show well on one graph

i = int(len(date_y) / 3)

date_y[:i].plot(secondary_y='Frequency', figsize=(20, 5), title='date_y Year 1')

date_y[i:2*i].plot(secondary_y='Frequency', figsize=(20, 5), title='date_y Year 2')

date_y[2*i:].plot(secondary_y='Frequency', figsize=(20, 5), title='date_y Year 3')
date_x_freq = pd.DataFrame()

date_x_freq['Training set'] = df_train.groupby('date_x')['activity_id'].count()

date_x_freq['Testing set'] = df_test.groupby('date_x')['activity_id'].count()

date_x_freq.plot(secondary_y='Testing set', figsize=(20, 8), 

                 title='Comparison of date_x distribution between training/testing set')

date_y_freq = pd.DataFrame()

date_y_freq['Training set'] = df_train.groupby('date_y')['activity_id'].count()

date_y_freq['Testing set'] = df_test.groupby('date_y')['activity_id'].count()

date_y_freq[:i].plot(secondary_y='Testing set', figsize=(20, 8), 

                 title='Comparison of date_y distribution between training/testing set (first year)')

date_y_freq[2*i:].plot(secondary_y='Testing set', figsize=(20, 8), 

                 title='Comparison of date_y distribution between training/testing set (last year)')
print('Correlation of date_x distribution in training/testing sets: ' + str(np.corrcoef(date_x_freq.T)[0,1]))

print('Correlation of date_y distribution in training/testing sets: ' + str(np.corrcoef(date_y_freq.fillna(0).T)[0,1]))
print('date_y correlation in year 1: ' + str(np.corrcoef(date_y_freq[:i].fillna(0).T)[0,1]))

print('date_y correlation in year 2: ' + str(np.corrcoef(date_y_freq[i:2*i].fillna(0).T)[0,1]))

print('date_y correlation in year 3: ' + str(np.corrcoef(date_y_freq[2*i:].fillna(0).T)[0,1]))
from sklearn.metrics import roc_auc_score

features = pd.DataFrame()

features['date_x_prob'] = df_train.groupby('date_x')['outcome'].transform('mean')

features['date_y_prob'] = df_train.groupby('date_y')['outcome'].transform('mean')

features['date_x_count'] = df_train.groupby('date_x')['outcome'].transform('count')

features['date_y_count'] = df_train.groupby('date_y')['outcome'].transform('count')

_=[print(f.ljust(12) + ' AUC: ' + str(round(roc_auc_score(df_train['outcome'], features[f]), 6))) for f in features.columns]