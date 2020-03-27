
import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns

color = sns.color_palette()



#%matplotlib inline



pd.options.mode.chained_assignment = None

pd.options.display.max_columns = 999
from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))
train_df = pd.read_csv("../input/train_2016.csv", parse_dates=["transactiondate"])

train_df.shape
train_df.head()
plt.figure(figsize=(8,6))

plt.scatter(range(train_df.shape[0]), np.sort(train_df.logerror.values))

plt.xlabel('index', fontsize=12)

plt.ylabel('logerror', fontsize=12)

plt.show()
ulimit = np.percentile(train_df.logerror.values, 99)

llimit = np.percentile(train_df.logerror.values, 1)

train_df['logerror'].ix[train_df['logerror']>ulimit] = ulimit

train_df['logerror'].ix[train_df['logerror']<llimit] = llimit



plt.figure(figsize=(12,8))

sns.distplot(train_df.logerror.values, bins=50, kde=False)

plt.xlabel('logerror', fontsize=12)

plt.show()
train_df['transaction_month'] = train_df['transactiondate'].dt.month



cnt_srs = train_df['transaction_month'].value_counts()

plt.figure(figsize=(12,6))

sns.barplot(cnt_srs.index, cnt_srs.values, alpha=0.8, color=color[3])

plt.xticks(rotation='vertical')

plt.xlabel('Month of transaction', fontsize=12)

plt.ylabel('Number of Occurrences', fontsize=12)

plt.show()
(train_df['parcelid'].value_counts().reset_index())['parcelid'].value_counts()
prop_df = pd.read_csv("../input/properties_2016.csv")

prop_df.shape
prop_df.head()
missing_df = prop_df.isnull().sum(axis=0).reset_index()

missing_df.columns = ['column_name', 'missing_count']

missing_df = missing_df.ix[missing_df['missing_count']>0]

missing_df = missing_df.sort_values(by='missing_count')



ind = np.arange(missing_df.shape[0])

width = 0.9

fig, ax = plt.subplots(figsize=(12,18))

rects = ax.barh(ind, missing_df.missing_count.values, color='blue')

ax.set_yticks(ind)

ax.set_yticklabels(missing_df.column_name.values, rotation='horizontal')

ax.set_xlabel("Count of missing values")

ax.set_title("Number of missing values in each column")

plt.show()
plt.figure(figsize=(12,12))

sns.jointplot(x=prop_df.latitude.values, y=prop_df.longitude.values, size=10)

plt.ylabel('Longitude', fontsize=12)

plt.xlabel('Latitude', fontsize=12)

plt.show()
train_df = pd.merge(train_df, prop_df, on='parcelid', how='left')

train_df.head()
pd.options.display.max_rows = 65



dtype_df = train_df.dtypes.reset_index()

dtype_df.columns = ["Count", "Column Type"]

dtype_df
dtype_df.groupby("Column Type").aggregate('count').reset_index()
missing_df = train_df.isnull().sum(axis=0).reset_index()

missing_df.columns = ['column_name', 'missing_count']

missing_df['missing_ratio'] = missing_df['missing_count'] / train_df.shape[0]

missing_df.ix[missing_df['missing_ratio']>0.999]
# Let us just impute the missing values with mean values to compute correlation coefficients #

mean_values = train_df.mean(axis=0)

train_df_new = train_df.fillna(mean_values, inplace=True)



# Now let us look at the correlation coefficient of each of these variables #

x_cols = [col for col in train_df_new.columns if col not in ['logerror'] if train_df_new[col].dtype=='float64']



labels = []

values = []

for col in x_cols:

    labels.append(col)

    values.append(np.corrcoef(train_df_new[col].values, train_df_new.logerror.values)[0,1])

corr_df = pd.DataFrame({'col_labels':labels, 'corr_values':values})

corr_df = corr_df.sort_values(by='corr_values')

    

ind = np.arange(len(labels))

width = 0.9

fig, ax = plt.subplots(figsize=(12,40))

rects = ax.barh(ind, np.array(corr_df.corr_values.values), color='y')

ax.set_yticks(ind)

ax.set_yticklabels(corr_df.col_labels.values, rotation='horizontal')

ax.set_xlabel("Correlation coefficient")

ax.set_title("Correlation coefficient of the variables")

#autolabel(rects)

plt.show()
corr_zero_cols = ['assessmentyear', 'storytypeid', 'pooltypeid2', 'pooltypeid7', 'pooltypeid10', 'poolcnt', 'decktypeid', 'buildingclasstypeid']

for col in corr_zero_cols:

    print(col, len(train_df_new[col].unique()))
corr_df_sel = corr_df.ix[(corr_df['corr_values']>0.02) | (corr_df['corr_values'] < -0.01)]

corr_df_sel
cols_to_use = corr_df_sel.col_labels.tolist()



temp_df = train_df[cols_to_use]

corrmat = temp_df.corr(method='spearman')

f, ax = plt.subplots(figsize=(8, 8))



# Draw the heatmap using seaborn

sns.heatmap(corrmat, vmax=1., square=True)

plt.title("Important variables correlation map", fontsize=15)

plt.show()
col = "finishedsquarefeet12"

ulimit = np.percentile(train_df[col].values, 99.5)

llimit = np.percentile(train_df[col].values, 0.5)

train_df[col].ix[train_df[col]>ulimit] = ulimit

train_df[col].ix[train_df[col]<llimit] = llimit



plt.figure(figsize=(12,12))

sns.jointplot(x=train_df.finishedsquarefeet12.values, y=train_df.logerror.values, size=10, color=color[4])

plt.ylabel('Log Error', fontsize=12)

plt.xlabel('Finished Square Feet 12', fontsize=12)

plt.title("Finished square feet 12 Vs Log error", fontsize=15)

plt.show()
col = "calculatedfinishedsquarefeet"

ulimit = np.percentile(train_df[col].values, 99.5)

llimit = np.percentile(train_df[col].values, 0.5)

train_df[col].ix[train_df[col]>ulimit] = ulimit

train_df[col].ix[train_df[col]<llimit] = llimit



plt.figure(figsize=(12,12))

sns.jointplot(x=train_df.calculatedfinishedsquarefeet.values, y=train_df.logerror.values, size=10, color=color[5])

plt.ylabel('Log Error', fontsize=12)

plt.xlabel('Calculated finished square feet', fontsize=12)

plt.title("Calculated finished square feet Vs Log error", fontsize=15)

plt.show()
plt.figure(figsize=(12,8))

sns.countplot(x="bathroomcnt", data=train_df)

plt.ylabel('Count', fontsize=12)

plt.xlabel('Bathroom', fontsize=12)

plt.xticks(rotation='vertical')

plt.title("Frequency of Bathroom count", fontsize=15)

plt.show()
plt.figure(figsize=(12,8))

sns.boxplot(x="bathroomcnt", y="logerror", data=train_df)

plt.ylabel('Log error', fontsize=12)

plt.xlabel('Bathroom Count', fontsize=12)

plt.xticks(rotation='vertical')

plt.title("How log error changes with bathroom count?", fontsize=15)

plt.show()
plt.figure(figsize=(12,8))

sns.countplot(x="bedroomcnt", data=train_df)

plt.ylabel('Frequency', fontsize=12)

plt.xlabel('Bedroom Count', fontsize=12)

plt.xticks(rotation='vertical')

plt.title("Frequency of Bedroom count", fontsize=15)

plt.show()
train_df['bedroomcnt'].ix[train_df['bedroomcnt']>7] = 7

plt.figure(figsize=(12,8))

sns.violinplot(x='bedroomcnt', y='logerror', data=train_df)

plt.xlabel('Bedroom count', fontsize=12)

plt.ylabel('Log Error', fontsize=12)

plt.show()
col = "taxamount"

ulimit = np.percentile(train_df[col].values, 99.5)

llimit = np.percentile(train_df[col].values, 0.5)

train_df[col].ix[train_df[col]>ulimit] = ulimit

train_df[col].ix[train_df[col]<llimit] = llimit



plt.figure(figsize=(12,12))

sns.jointplot(x=train_df['taxamount'].values, y=train_df['logerror'].values, size=10, color='g')

plt.ylabel('Log Error', fontsize=12)

plt.xlabel('Tax Amount', fontsize=12)

plt.title("Tax Amount Vs Log error", fontsize=15)

plt.show()
from ggplot import *

ggplot(aes(x='yearbuilt', y='logerror'), data=train_df) + \

    geom_point(color='steelblue', size=1) + \

    stat_smooth()
ggplot(aes(x='latitude', y='longitude', color='logerror'), data=train_df) + \

    geom_point() + \

    scale_color_gradient(low = 'red', high = 'blue')
ggplot(aes(x='finishedsquarefeet12', y='taxamount', color='logerror'), data=train_df) + \

    geom_point(alpha=0.7) + \

    scale_color_gradient(low = 'pink', high = 'blue')
ggplot(aes(x='finishedsquarefeet12', y='taxamount', color='logerror'), data=train_df) + \

    geom_now_its_art()
train_y = train_df['logerror'].values

cat_cols = ["hashottuborspa", "propertycountylandusecode", "propertyzoningdesc", "fireplaceflag", "taxdelinquencyflag"]

train_df = train_df.drop(['parcelid', 'logerror', 'transactiondate', 'transaction_month']+cat_cols, axis=1)

feat_names = train_df.columns.values



from sklearn import ensemble

model = ensemble.ExtraTreesRegressor(n_estimators=25, max_depth=30, max_features=0.3, n_jobs=-1, random_state=0)

model.fit(train_df, train_y)



## plot the importances ##

importances = model.feature_importances_

std = np.std([tree.feature_importances_ for tree in model.estimators_], axis=0)

indices = np.argsort(importances)[::-1][:20]



plt.figure(figsize=(12,12))

plt.title("Feature importances")

plt.bar(range(len(indices)), importances[indices], color="r", yerr=std[indices], align="center")

plt.xticks(range(len(indices)), feat_names[indices], rotation='vertical')

plt.xlim([-1, len(indices)])

plt.show()
import xgboost as xgb

xgb_params = {

    'eta': 0.05,

    'max_depth': 8,

    'subsample': 0.7,

    'colsample_bytree': 0.7,

    'objective': 'reg:linear',

    'silent': 1,

    'seed' : 0

}

dtrain = xgb.DMatrix(train_df, train_y, feature_names=train_df.columns.values)

model = xgb.train(dict(xgb_params, silent=0), dtrain, num_boost_round=50)



# plot the important features #

fig, ax = plt.subplots(figsize=(12,18))

xgb.plot_importance(model, max_num_features=50, height=0.8, ax=ax)

plt.show()