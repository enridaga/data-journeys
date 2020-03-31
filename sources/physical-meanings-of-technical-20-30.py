
#load libs

import kagglegym

import numpy as np

import pandas as pd

from sklearn.ensemble import ExtraTreesRegressor

from sklearn.linear_model import LinearRegression, BayesianRidge



import matplotlib.pyplot as plt

import seaborn as sns

### matplotlib inline



env = kagglegym.make()

o = env.reset()

col = [c for c in o.train.columns if '_' in c]



full_df = pd.read_hdf('../input/train.h5')

def get_reward(y_true, y_fit):

    R2 = 1 - np.sum((y_true - y_fit)**2) / np.sum((y_true - np.mean(y_true))**2)

    R = np.sign(R2) * np.sqrt(abs(R2))

    return(R)

pd.options.mode.chained_assignment = None  # default='warn'
id = 1561 #val_set.id.sample().values[0]

print(id)

temp = full_df[full_df.id==id]

temp['feature'] = temp['technical_20']

temp['feature'] = temp['feature'] * 4

temp[['y', 'feature']].iloc[:100,:].plot(marker='.')
temp['y_cum'] = temp.y.cumsum()

temp[['y_cum', 'feature']].iloc[:100,:].plot(marker='.')
temp[['y_cum', 'technical_30']].iloc[:100,:].plot(marker='.')
print(np.correlate(temp['y'], temp['technical_20']))

print(np.correlate(temp['y'], temp['technical_30']))
train_data = o.train

cols = 'technical_20'



low_y_cut = -0.086093

high_y_cut = 0.093497



y_values_within = ((train_data['y'] > low_y_cut) & (train_data['y'] <high_y_cut))



train_cut = train_data.loc[y_values_within,:]



# Fill missing values

mean_vals = train_cut.mean()

train_cut.fillna(mean_vals,inplace=True)



x_train = train_cut[cols]

y = train_cut["y"]



lr_model = LinearRegression()

lr_model.fit(np.array(x_train.values).reshape(-1,1),y.values)



val_set = full_df[full_df.timestamp>=906]

x_val = val_set[cols].fillna(mean_vals[cols])

pred = lr_model.predict(x_val.values.reshape(-1, 1))

re = get_reward(val_set['y'], pred)

print("Public score: {}".format(re))

print("learned parameter of the model: {}".format(lr_model.coef_))
train_cut = train_data.loc[y_values_within,:]

x_train = train_cut.groupby('id')[cols].shift(-1).fillna(mean_vals[cols])

y = train_cut["y"]



lr_model2 = LinearRegression()

lr_model2.fit(np.array(x_train.values).reshape(-1,1),y.values)



x_val = val_set.groupby('id')[cols].shift(-1).fillna(mean_vals[cols])

pred = lr_model2.predict(x_val.values.reshape(-1, 1))

re = get_reward(val_set['y'], pred)

print("Public score: {}".format(re))

print("learned parameter of the model: {}".format(lr_model2.coef_))
temp = full_df.copy()

temp['feature'] = temp['technical_20'] - temp['technical_30']

n = 5

for i in range(1, n+1):

    temp['fea_%d'%i] = temp.groupby('id')['y'].shift(i)

temp = temp.filter(regex='^fea').dropna()

estimator = LinearRegression()

estimator.fit(temp.filter(regex='^fea_').values, temp.feature.values)

#temp['pred_f'] = eastimator.predict(temp.filter(regex='^fea_').fillna(mean_vals).values)

print(estimator.intercept_, estimator.coef_)
id = 142 #full_df.id.sample().values[0]

print(id)

temp = full_df[full_df.id==id]

temp['feature'] = temp['technical_20'] - temp['technical_30']

mean_vals = temp.y.mean()

for i in range(1, n+1):

    temp['fea_%d'%i] = temp['y'].shift(i)

temp['y_2'] = estimator.predict(temp.filter(regex='^fea_').fillna(mean_vals).values)

temp[['y_2', 'feature']].iloc[:100,:].plot(marker='.')
temp = full_df.copy()

temp['feature'] = temp['technical_20'] - temp['technical_30']

n = 20

for i in range(1, n+1):

    temp['fea_%d'%i] = temp.groupby('id')['y'].shift(i)

temp = temp.filter(regex='^fea').dropna()

estimator = LinearRegression()

estimator.fit(temp.filter(regex='^fea_').values, temp.feature.values)

#temp['pred_f'] = eastimator.predict(temp.filter(regex='^fea_').fillna(mean_vals).values)

print(estimator.intercept_, estimator.coef_)



id = 142 #full_df.id.sample().values[0]

temp = full_df[full_df.id==id]

temp['feature'] = temp['technical_20'] - temp['technical_30']

mean_vals = temp.y.mean()

for i in range(1, n+1):

    temp['fea_%d'%i] = temp['y'].shift(i)

temp['y_2'] = estimator.predict(temp.filter(regex='^fea_').fillna(mean_vals).values)

temp[['y_2', 'feature']].iloc[:100,:].plot(marker='.')
for i in range(len(estimator.coef_)-1):

    print(estimator.coef_[i+1]/estimator.coef_[i])
temp['feature_'] = 0.07 * temp['fea_1'] + 0.92 * temp['feature'].shift(1)

temp[['feature_', 'feature']].iloc[:100,:].plot(marker='.')
id = 12 #full_df.id.sample().values[0]

temp = full_df[full_df.id==id]

temp['feature'] = temp['technical_20'] - temp['technical_30']

temp['feature_'] = 0.07 * temp['y'].shift(1) + 0.92 * temp['feature'].shift(1)

temp[['feature_', 'feature']].iloc[:100,:].plot(marker='.')
