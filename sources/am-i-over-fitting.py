
# Import all the necessary packages 

import kagglegym

import numpy as np

import pandas as pd

from sklearn.ensemble import ExtraTreesRegressor

from sklearn.linear_model import LinearRegression, Ridge

import math

import matplotlib.pyplot as plt



#%matplotlib inline
# Read the full data set stored as HDF5 file

full_df = pd.read_hdf('../input/train.h5')
# A custom function to compute the R score

def get_reward(y_true, y_fit):

    R2 = 1 - np.sum((y_true - y_fit)**2) / np.sum((y_true - np.mean(y_true))**2)

    R = np.sign(R2) * math.sqrt(abs(R2))

    return(R)
# Some pre-processing as seen from most of the public scripts.

# The "environment" is our interface for code competitions

env = kagglegym.make()



# We get our initial observation by calling "reset"

observation = env.reset()

target_var = 'y'



# Get the train dataframe

train = observation.train

mean_values = train.median(axis=0)

train.fillna(mean_values, inplace=True)



# Observed with histograns:

low_y_cut = -0.086093

high_y_cut = 0.093497



y_is_above_cut = (train.y > high_y_cut)

y_is_below_cut = (train.y < low_y_cut)

y_is_within_cut = (~y_is_above_cut & ~y_is_below_cut)
### https://www.kaggle.com/ymcdull/two-sigma-financial-modeling/ridge-lb-0-0100659/run/545100

# The "environment" is our interface for code competitions

env = kagglegym.make()



# We get our initial observation by calling "reset"

observation = env.reset()



# cols_to_use for ridge model

#cols_to_use = ['technical_30', 'technical_20', 'fundamental_11']

cols_to_use = ['technical_20']



# model build

model = Ridge()

model.fit(np.array(train.loc[y_is_within_cut, cols_to_use].values), train.loc[y_is_within_cut, target_var])



# getting the y mean dict for averaging

ymean_dict = dict(train.groupby(["id"])["y"].mean())



# weighted average of model & mean

def get_weighted_y(series):

    id, y = series["id"], series["y"]

    return 0.95 * y + 0.05 * ymean_dict[id] if id in ymean_dict else y



y_actual_list = []

y_pred_list = []

r1_overall_reward_list = []

ts_list = []

while True:

    timestamp = observation.features["timestamp"][0]

    actual_y = list(full_df[full_df["timestamp"] == timestamp]["y"].values)

    observation.features.fillna(mean_values, inplace=True)

    test_x = np.array(observation.features[cols_to_use].values)

    observation.target.y = model.predict(test_x).clip(low_y_cut, high_y_cut)

    

    ## weighted y using average value

    observation.target.y = observation.target.apply(get_weighted_y, axis = 1)

    target = observation.target

    observation, reward, done, info = env.step(target)

    

    if timestamp % 100 == 0:

        print("Timestamp #{}".format(timestamp))

    

    pred_y = list(target.y.values)

    y_actual_list.extend(actual_y)

    y_pred_list.extend(pred_y)

    overall_reward = get_reward(np.array(y_actual_list), np.array(y_pred_list))

    r1_overall_reward_list.append(overall_reward)

    ts_list.append(timestamp)

    if done:

        break

    

print(info)
fig = plt.figure(figsize=(12, 6))

plt.plot(ts_list, r1_overall_reward_list, c='blue')

plt.plot(ts_list, [0]*len(ts_list), c='red')

plt.title("Cumulative R value change for Univariate Ridge (technical_20)")

plt.ylim([-0.04,0.04])

plt.xlim([850, 1850])

plt.show()
### https://www.kaggle.com/ymcdull/two-sigma-financial-modeling/ridge-lb-0-0100659/run/545100

# The "environment" is our interface for code competitions

env = kagglegym.make()



# We get our initial observation by calling "reset"

observation = env.reset()



# cols_to_use for ridge model

#cols_to_use = ['technical_30', 'technical_20', 'fundamental_11']

cols_to_use = ['technical_30', 'technical_20']



# model build

model = Ridge()

model.fit(np.array(train.loc[y_is_within_cut, cols_to_use].values), train.loc[y_is_within_cut, target_var])



# getting the y mean dict for averaging

ymean_dict = dict(train.groupby(["id"])["y"].mean())



# weighted average of model & mean

def get_weighted_y(series):

    id, y = series["id"], series["y"]

    return 0.95 * y + 0.05 * ymean_dict[id] if id in ymean_dict else y



y_actual_list = []

y_pred_list = []

r2_overall_reward_list = []

ts_list = []

while True:

    timestamp = observation.features["timestamp"][0]

    actual_y = list(full_df[full_df["timestamp"] == timestamp]["y"].values)

    observation.features.fillna(mean_values, inplace=True)

    test_x = np.array(observation.features[cols_to_use].values)

    observation.target.y = model.predict(test_x).clip(low_y_cut, high_y_cut)

    

    ## weighted y using average value

    observation.target.y = observation.target.apply(get_weighted_y, axis = 1)

    target = observation.target

    observation, reward, done, info = env.step(target)

    

    if timestamp % 100 == 0:

        print("Timestamp #{}".format(timestamp))

    

    pred_y = list(target.y.values)

    y_actual_list.extend(actual_y)

    y_pred_list.extend(pred_y)

    overall_reward = get_reward(np.array(y_actual_list), np.array(y_pred_list))

    r2_overall_reward_list.append(overall_reward)

    ts_list.append(timestamp)

    if done:

        break

    

print(info)
fig, ax = plt.subplots(figsize=(12, 6))

ax.plot(ts_list, r2_overall_reward_list, c='green', label='ridge-2')

ax.plot(ts_list, r1_overall_reward_list, c='blue', label='ridge-1')

ax.plot(ts_list, [0]*len(ts_list), c='red', label='zero line')

ax.legend(loc='lower right')

ax.set_ylim([-0.04,0.04])

ax.set_xlim([850, 1850])

plt.title("Cumulative R value change for Bivariate Ridge")

plt.show()
### https://www.kaggle.com/ymcdull/two-sigma-financial-modeling/ridge-lb-0-0100659/run/545100

# The "environment" is our interface for code competitions

env = kagglegym.make()



# We get our initial observation by calling "reset"

observation = env.reset()



# cols_to_use for ridge model

cols_to_use = ['technical_30', 'technical_20', 'fundamental_11']



# model build

model = Ridge()

model.fit(np.array(train.loc[y_is_within_cut, cols_to_use].values), train.loc[y_is_within_cut, target_var])



# getting the y mean dict for averaging

ymean_dict = dict(train.groupby(["id"])["y"].mean())



# weighted average of model & mean

def get_weighted_y(series):

    id, y = series["id"], series["y"]

    return 0.95 * y + 0.05 * ymean_dict[id] if id in ymean_dict else y



y_actual_list = []

y_pred_list = []

r3_overall_reward_list = []

ts_list = []

while True:

    timestamp = observation.features["timestamp"][0]

    actual_y = list(full_df[full_df["timestamp"] == timestamp]["y"].values)

    observation.features.fillna(mean_values, inplace=True)

    test_x = np.array(observation.features[cols_to_use].values)

    observation.target.y = model.predict(test_x).clip(low_y_cut, high_y_cut)

    

    ## weighted y using average value

    observation.target.y = observation.target.apply(get_weighted_y, axis = 1)

    target = observation.target

    observation, reward, done, info = env.step(target)

    

    if timestamp % 100 == 0:

        print("Timestamp #{}".format(timestamp))

    

    pred_y = list(target.y.values)

    y_actual_list.extend(actual_y)

    y_pred_list.extend(pred_y)

    overall_reward = get_reward(np.array(y_actual_list), np.array(y_pred_list))

    r3_overall_reward_list.append(overall_reward)

    ts_list.append(timestamp)

    if done:

        break

    

print(info)
fig, ax = plt.subplots(figsize=(12, 6))

ax.plot(ts_list, r3_overall_reward_list, c='brown', label='ridge-3')

ax.plot(ts_list, r2_overall_reward_list, c='green', label='ridge-2')

ax.plot(ts_list, [0]*len(ts_list), c='red', label='zero line')

ax.legend(loc='lower right')

ax.set_ylim([-0.04,0.04])

ax.set_xlim([850, 1850])

plt.title("Cumulative R value change for Trivariate Ridge")

plt.show()
env = kagglegym.make()

o = env.reset()



excl = [env.ID_COL_NAME, env.SAMPLE_COL_NAME, env.TARGET_COL_NAME, env.TIME_COL_NAME]

col = [c for c in o.train.columns if c not in excl]



train = o.train[col]

d_mean= train.median(axis=0)



train = o.train[col]

n = train.isnull().sum(axis=1)

for c in train.columns:

    train[c + '_nan_'] = pd.isnull(train[c])

    d_mean[c + '_nan_'] = 0

train = train.fillna(d_mean)

train['znull'] = n



print("Building ET..")

model_et = ExtraTreesRegressor(n_estimators=25, max_depth=4, n_jobs=-1, random_state=123, verbose=0)

model_et.fit(train, o.train['y'])

env = kagglegym.make()

o = env.reset()



#https://www.kaggle.com/ymcdull/two-sigma-financial-modeling/ridge-lb-0-0100659

ymean_dict = dict(o.train.groupby(["id"])["y"].median())

def get_weighted_y(series):

    id, y = series["id"], series["y"]

    return 0.95 * y + 0.05 * ymean_dict[id] if id in ymean_dict else y



y_actual_list = []

y_pred_list = []

et_overall_reward_list = []

ts_list = []

while True:

    timestamp = o.features["timestamp"][0]

    actual_y = list(full_df[full_df["timestamp"] == timestamp]["y"].values)

    

    test = o.features[col]

    n = test.isnull().sum(axis=1)

    for c in test.columns:

        test[c + '_nan_'] = pd.isnull(test[c])

    test = test.fillna(d_mean)

    test['znull'] = n

    

    pred = o.target

    pred['y'] = model_et.predict(test).clip(low_y_cut, high_y_cut) 

    pred['y'] = pred.apply(get_weighted_y, axis = 1)

    o, reward, done, info = env.step(pred[['id','y']])

    

    pred_y = list(pred.y.values)

    y_actual_list.extend(actual_y)

    y_pred_list.extend(pred_y)

    overall_reward = get_reward(np.array(y_actual_list), np.array(y_pred_list))

    et_overall_reward_list.append(overall_reward)

    ts_list.append(timestamp)

    

    if timestamp % 100 == 0:

        print("Timestamp #{}".format(timestamp))

    

    if done:

        print(info)

        break
fig, ax = plt.subplots(figsize=(12, 6))

ax.plot(ts_list, et_overall_reward_list, c='blue', label='Extra Trees')

ax.plot(ts_list, r1_overall_reward_list, c='green', label='Ridge-1')

ax.plot(ts_list, [0]*len(ts_list), c='red', label='zero line')

ax.legend(loc='lower right')

ax.set_ylim([-0.04,0.04])

ax.set_xlim([850, 1850])

plt.title("Cumulative R value change for Extra Trees")



plt.show()