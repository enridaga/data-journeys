
import pandas as pd #Analysis 
import matplotlib.pyplot as plt #Visulization
import seaborn as sns #Visulization
import numpy as np #Analysis 
from scipy.stats import norm #Analysis 
from sklearn.preprocessing import StandardScaler #Analysis 
from scipy import stats #Analysis 
import warnings 
warnings.filterwarnings('ignore')
### matplotlib inline
import plotly.offline as py
py.init_notebook_mode(connected=True)
import plotly.graph_objs as go
import plotly.tools as tls
import plotly.figure_factory as ff
from sklearn.model_selection import KFold
from sklearn.metrics import mean_absolute_error
import gc
#If you run all dataset, you change debug False
debug = False
if debug == True:
    df_train = pd.read_csv('../input/train_V2.csv', nrows=10000)
    df_test  = pd.read_csv('../input/test_V2.csv')
else:
    df_train = pd.read_csv('../input/train_V2.csv')
    df_test  = pd.read_csv('../input/test_V2.csv')
print("Train : ",df_train.shape)
print("Test : ",df_test.shape)
df_train.head()
# Memory saving function credit to https://www.kaggle.com/gemartin/load-data-reduce-memory-usage
def reduce_mem_usage(df):
    """ iterate through all the columns of a dataframe and modify the data type
        to reduce memory usage.        
    """
    #start_mem = df.memory_usage().sum() / 1024**2
    #print('Memory usage of dataframe is {:.2f} MB'.format(start_mem))

    for col in df.columns:
        col_type = df[col].dtype

        if col_type != object:
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

    #end_mem = df.memory_usage().sum() / 1024**2
    #print('Memory usage after optimization is: {:.2f} MB'.format(end_mem))
    #print('Decreased by {:.1f}%'.format(100 * (start_mem - end_mem) / start_mem))

    return df
df_train = reduce_mem_usage(df_train)
df_test = reduce_mem_usage(df_test)
df_train[df_train['groupId']=='4d4b580de459be']
len(df_train[df_train['matchId']=='a10357fd1a4a91'])
temp = df_train[df_train['matchId']=='a10357fd1a4a91']['groupId'].value_counts().sort_values(ascending=False)
#print("Total number of states : ",len(temp))
trace = go.Bar(
    x = temp.index,
    y = (temp)
)
data = [trace]
layout = go.Layout(
    title = "GroupId of Match Id: a10357fd1a4a91",
    xaxis=dict(
        title='groupId',
        tickfont=dict(
            size=14,
            color='rgb(107, 107, 107)'
        )
    ),
    yaxis=dict(
        title='Count of groupId of type of MatchId a10357fd1a4a91',
        titlefont=dict(
            size=16,
            color='rgb(107, 107, 107)'
        ),
        tickfont=dict(
            size=14,
            color='rgb(107, 107, 107)'
        )
)
)
fig = go.Figure(data=data, layout=layout)
py.iplot(fig, filename='schoolStateNames')
temp = df_train['assists'].value_counts().sort_values(ascending=False)

#print("Total number of states : ",len(temp))
trace = go.Bar(
    x = temp.index,
    y = (temp)
)
data = [trace]
layout = go.Layout(
    title = "",
    xaxis=dict(
        title='assists',
        tickfont=dict(
            size=14,
            color='rgb(107, 107, 107)'
        )
    ),
    yaxis=dict(
        title='Count of assists',
        titlefont=dict(
            size=16,
            color='rgb(107, 107, 107)'
        ),
        tickfont=dict(
            size=14,
            color='rgb(107, 107, 107)'
        )
)
)
fig = go.Figure(data=data, layout=layout)
py.iplot(fig, filename='schoolStateNames')
temp = df_train['kills'].value_counts().sort_values(ascending=False)

#print("Total number of states : ",len(temp))
trace = go.Bar(
    x = temp.index,
    y = (temp)
)
data = [trace]
layout = go.Layout(
    title = "",
    xaxis=dict(
        title='kills',
        tickfont=dict(
            size=14,
            color='rgb(107, 107, 107)'
        )
    ),
    yaxis=dict(
        title='Count of kills',
        titlefont=dict(
            size=16,
            color='rgb(107, 107, 107)'
        ),
        tickfont=dict(
            size=14,
            color='rgb(107, 107, 107)'
        )
)
)
fig = go.Figure(data=data, layout=layout)
py.iplot(fig, filename='schoolStateNames')
temp = df_train['killStreaks'].value_counts().sort_values(ascending=False)

#print("Total number of states : ",len(temp))
trace = go.Bar(
    x = temp.index,
    y = (temp)
)
data = [trace]
layout = go.Layout(
    title = "",
    xaxis=dict(
        title='killStreaks',
        tickfont=dict(
            size=14,
            color='rgb(107, 107, 107)'
        )
    ),
    yaxis=dict(
        title='Count of killStreaks',
        titlefont=dict(
            size=16,
            color='rgb(107, 107, 107)'
        ),
        tickfont=dict(
            size=14,
            color='rgb(107, 107, 107)'
        )
)
)
fig = go.Figure(data=data, layout=layout)
py.iplot(fig, filename='schoolStateNames')
temp = df_train['roadKills'].value_counts().sort_values(ascending=False)

#print("Total number of states : ",len(temp))
trace = go.Bar(
    x = temp.index,
    y = (temp)
)
data = [trace]
layout = go.Layout(
    title = "",
    xaxis=dict(
        title='roadKills',
        tickfont=dict(
            size=14,
            color='rgb(107, 107, 107)'
        )
    ),
    yaxis=dict(
        title='Count of roadKills',
        titlefont=dict(
            size=16,
            color='rgb(107, 107, 107)'
        ),
        tickfont=dict(
            size=14,
            color='rgb(107, 107, 107)'
        )
)
)
fig = go.Figure(data=data, layout=layout)
py.iplot(fig, filename='schoolStateNames')
df_train['roadKills'].value_counts()
temp = df_train['teamKills'].value_counts().sort_values(ascending=False)

#print("Total number of states : ",len(temp))
trace = go.Bar(
    x = temp.index,
    y = (temp),
)
data = [trace]
layout = go.Layout(
    title = "",
    xaxis=dict(
        title='teamKills',
        tickfont=dict(
            size=14,
            color='rgb(107, 107, 107)'
        )
    ),
    yaxis=dict(
        title='Count of teamKills',
        titlefont=dict(
            size=16,
            color='rgb(107, 107, 107)'
        ),
        tickfont=dict(
            size=14,
            color='rgb(107, 107, 107)'
        )
)
)
fig = go.Figure(data=data, layout=layout)
py.iplot(fig, filename='schoolStateNames')
df_train['teamKills'].value_counts()
#histogram
f, ax = plt.subplots(figsize=(18, 8))
sns.distplot(df_train['longestKill'])
temp = df_train['weaponsAcquired'].value_counts().sort_values(ascending=False)

#print("Total number of states : ",len(temp))
trace = go.Bar(
    x = temp.index,
    y = (temp),
)
data = [trace]
layout = go.Layout(
    title = "",
    xaxis=dict(
        title='weaponsAcquired',
        tickfont=dict(
            size=14,
            color='rgb(107, 107, 107)'
        )
    ),
    yaxis=dict(
        title='Count of weaponsAcquired',
        titlefont=dict(
            size=16,
            color='rgb(107, 107, 107)'
        ),
        tickfont=dict(
            size=14,
            color='rgb(107, 107, 107)'
        )
)
)
fig = go.Figure(data=data, layout=layout)
py.iplot(fig, filename='schoolStateNames')
temp = df_train['headshotKills'].value_counts().sort_values(ascending=False)

#print("Total number of states : ",len(temp))
trace = go.Bar(
    x = temp.index,
    y = (temp),
)
data = [trace]
layout = go.Layout(
    title = "",
    xaxis=dict(
        title='headshotKills',
        tickfont=dict(
            size=14,
            color='rgb(107, 107, 107)'
        )
    ),
    yaxis=dict(
        title='Count of headshotKills',
        titlefont=dict(
            size=16,
            color='rgb(107, 107, 107)'
        ),
        tickfont=dict(
            size=14,
            color='rgb(107, 107, 107)'
        )
)
)
fig = go.Figure(data=data, layout=layout)
py.iplot(fig, filename='schoolStateNames')
df_train['headshotKills'].value_counts()
temp = df_train['DBNOs'].value_counts().sort_values(ascending=False)

#print("Total number of states : ",len(temp))
trace = go.Bar(
    x = temp.index,
    y = (temp),
)
data = [trace]
layout = go.Layout(
    title = "",
    xaxis=dict(
        title='DBNOs',
        tickfont=dict(
            size=14,
            color='rgb(107, 107, 107)'
        )
    ),
    yaxis=dict(
        title='Count of DBNOs',
        titlefont=dict(
            size=16,
            color='rgb(107, 107, 107)'
        ),
        tickfont=dict(
            size=14,
            color='rgb(107, 107, 107)'
        )
)
)
fig = go.Figure(data=data, layout=layout)
py.iplot(fig, filename='schoolStateNames')
temp = df_train['boosts'].value_counts().sort_values(ascending=False)

#print("Total number of states : ",len(temp))
trace = go.Bar(
    x = temp.index,
    y = (temp),
)
data = [trace]
layout = go.Layout(
    title = "",
    xaxis=dict(
        title='boosts',
        tickfont=dict(
            size=14,
            color='rgb(107, 107, 107)'
        )
    ),
    yaxis=dict(
        title='Count of boosts',
        titlefont=dict(
            size=16,
            color='rgb(107, 107, 107)'
        ),
        tickfont=dict(
            size=14,
            color='rgb(107, 107, 107)'
        )
)
)
fig = go.Figure(data=data, layout=layout)
py.iplot(fig, filename='schoolStateNames')
temp = df_train['heals'].value_counts().sort_values(ascending=False)

#print("Total number of states : ",len(temp))
trace = go.Bar(
    x = temp.index,
    y = (temp),
)
data = [trace]
layout = go.Layout(
    title = "",
    xaxis=dict(
        title='heals',
        tickfont=dict(
            size=14,
            color='rgb(107, 107, 107)'
        )
    ),
    yaxis=dict(
        title='Count of heals',
        titlefont=dict(
            size=16,
            color='rgb(107, 107, 107)'
        ),
        tickfont=dict(
            size=14,
            color='rgb(107, 107, 107)'
        )
)
)
fig = go.Figure(data=data, layout=layout)
py.iplot(fig, filename='schoolStateNames')
#histogram
f, ax = plt.subplots(figsize=(8, 6))
sns.distplot(df_train['damageDealt'])
f, ax = plt.subplots(figsize=(8, 6))
df_train['revives'].value_counts().sort_values(ascending=False).plot.bar()
plt.show()
#histogram
f, ax = plt.subplots(figsize=(18, 8))
sns.distplot(df_train['walkDistance'])
#histogram
f, ax = plt.subplots(figsize=(18, 8))
sns.distplot(df_train['rideDistance'])
temp = df_train['vehicleDestroys'].value_counts().sort_values(ascending=False)

#print("Total number of states : ",len(temp))
trace = go.Bar(
    x = temp.index,
    y = (temp),
)
data = [trace]
layout = go.Layout(
    title = "",
    xaxis=dict(
        title='vehicleDestroys',
        tickfont=dict(
            size=14,
            color='rgb(107, 107, 107)'
        )
    ),
    yaxis=dict(
        title='Count of vehicleDestroys',
        titlefont=dict(
            size=16,
            color='rgb(107, 107, 107)'
        ),
        tickfont=dict(
            size=14,
            color='rgb(107, 107, 107)'
        )
)
)
fig = go.Figure(data=data, layout=layout)
py.iplot(fig, filename='schoolStateNames')
df_train['vehicleDestroys'].value_counts()
temp = df_train['weaponsAcquired'].value_counts().sort_values(ascending=False)

#print("Total number of states : ",len(temp))
trace = go.Bar(
    x = temp.index,
    y = (temp),
)
data = [trace]
layout = go.Layout(
    title = "",
    xaxis=dict(
        title='weaponsAcquired',
        tickfont=dict(
            size=14,
            color='rgb(107, 107, 107)'
        )
    ),
    yaxis=dict(
        title='Count of weaponsAcquired',
        titlefont=dict(
            size=16,
            color='rgb(107, 107, 107)'
        ),
        tickfont=dict(
            size=14,
            color='rgb(107, 107, 107)'
        )
)
)
fig = go.Figure(data=data, layout=layout)
py.iplot(fig, filename='schoolStateNames')
#missing data
total = df_train.isnull().sum().sort_values(ascending=False)
percent = (df_train.isnull().sum()/df_train.isnull().count()).sort_values(ascending=False)
missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
missing_data.head(20)

#histogram
#missing_data = missing_data.head(20)
percent_data = percent.head(20)
percent_data.plot(kind="bar", figsize = (8,6), fontsize = 10)
plt.xlabel("Columns", fontsize = 20)
plt.ylabel("Count", fontsize = 20)
plt.title("Total Missing Value (%) in Train", fontsize = 20)
#missing data
total = df_test.isnull().sum().sort_values(ascending=False)
percent = (df_test.isnull().sum()/df_test.isnull().count()).sort_values(ascending=False)
missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
missing_data.head(20)

#histogram
#missing_data = missing_data.head(20)
percent_data = percent.head(20)
percent_data.plot(kind="bar", figsize = (8,6), fontsize = 10)
plt.xlabel("Columns", fontsize = 20)
plt.ylabel("Count", fontsize = 20)
plt.title("Total Missing Value (%) in Test", fontsize = 20)
#winPlacePerc correlation matrix
k = 10 #number of variables for heatmap
corrmat = df_train.corr() 
cols = corrmat.nlargest(k, 'winPlacePerc').index # nlargest : Return this many descending sorted values
cm = np.corrcoef(df_train[cols].values.T) # correlation 
sns.set(font_scale=1.25)
f, ax = plt.subplots(figsize=(8, 6))
hm = sns.heatmap(cm, cbar=True, annot=True, square=True, fmt='.2f', annot_kws={'size': 8}, yticklabels=cols.values, xticklabels=cols.values)
plt.show()
df_train.plot(x="walkDistance",y="winPlacePerc", kind="scatter", figsize = (8,6))
f, ax = plt.subplots(figsize=(8, 6))
fig = sns.boxplot(x='boosts', y="winPlacePerc", data=df_train)
fig.axis(ymin=0, ymax=1);
df_train.plot(x="weaponsAcquired",y="winPlacePerc", kind="scatter", figsize = (8,6))
df_train.plot(x="damageDealt",y="winPlacePerc", kind="scatter", figsize = (8,6))
df_train.plot(x="heals",y="winPlacePerc", kind="scatter", figsize = (8,6))
df_train.plot(x="longestKill",y="winPlacePerc", kind="scatter", figsize = (8,6))
df_train.plot(x="kills",y="winPlacePerc", kind="scatter", figsize = (8,6))
f, ax = plt.subplots(figsize=(8, 6))
fig = sns.boxplot(x='killStreaks', y="winPlacePerc", data=df_train)
fig.axis(ymin=0, ymax=1);
f, ax = plt.subplots(figsize=(8, 6))
fig = sns.boxplot(x='assists', y="winPlacePerc", data=df_train)
fig.axis(ymin=0, ymax=1);
df_train = df_train[df_train['Id']!='f70c74418bb064']
headshot = df_train[['kills','winPlacePerc','headshotKills']]
headshot['headshotrate'] = headshot['kills'] / headshot['headshotKills']
headshot.corr()
del headshot
df_train['headshotrate'] = df_train['kills']/df_train['headshotKills']
df_test['headshotrate'] = df_test['kills']/df_test['headshotKills']
killStreak = df_train[['kills','winPlacePerc','killStreaks']]
killStreak['killStreakrate'] = killStreak['killStreaks']/killStreak['kills']
killStreak.corr()
healthitems = df_train[['heals','winPlacePerc','boosts']]
healthitems['healthitems'] = healthitems['heals'] + healthitems['boosts']
healthitems.corr()
del healthitems
kills = df_train[['assists','winPlacePerc','kills']]
kills['kills_assists'] = (kills['kills'] + kills['assists'])
kills.corr()
del df_train,df_test;
gc.collect()
def feature_engineering(is_train=True,debug=True):
    test_idx = None
    if is_train: 
        print("processing train.csv")
        if debug == True:
            df = pd.read_csv('../input/train_V2.csv', nrows=10000)
        else:
            df = pd.read_csv('../input/train_V2.csv')           

        df = df[df['maxPlace'] > 1]
    else:
        print("processing test.csv")
        df = pd.read_csv('../input/test_V2.csv')
        test_idx = df.Id
    
    # df = reduce_mem_usage(df)
    #df['totalDistance'] = df['rideDistance'] + df["walkDistance"] + df["swimDistance"]
    
    # df = df[:100]
    
    print("remove some columns")
    target = 'winPlacePerc'

    print("Adding Features")
 
    df['headshotrate'] = df['kills']/df['headshotKills']
    df['killStreakrate'] = df['killStreaks']/df['kills']
    df['healthitems'] = df['heals'] + df['boosts']
    df['totalDistance'] = df['rideDistance'] + df["walkDistance"] + df["swimDistance"]
    df['killPlace_over_maxPlace'] = df['killPlace'] / df['maxPlace']
    df['headshotKills_over_kills'] = df['headshotKills'] / df['kills']
    df['distance_over_weapons'] = df['totalDistance'] / df['weaponsAcquired']
    df['walkDistance_over_heals'] = df['walkDistance'] / df['heals']
    df['walkDistance_over_kills'] = df['walkDistance'] / df['kills']
    df['killsPerWalkDistance'] = df['kills'] / df['walkDistance']
    df["skill"] = df["headshotKills"] + df["roadKills"]

    df[df == np.Inf] = np.NaN
    df[df == np.NINF] = np.NaN
    
    print("Removing Na's From DF")
    df.fillna(0, inplace=True)

    
    features = list(df.columns)
    features.remove("Id")
    features.remove("matchId")
    features.remove("groupId")
    features.remove("matchType")
    
    # matchType = pd.get_dummies(df['matchType'])
    # df = df.join(matchType)    
    
    y = None
    
    
    if is_train: 
        print("get target")
        y = np.array(df.groupby(['matchId','groupId'])[target].agg('mean'), dtype=np.float64)
        features.remove(target)

    print("get group mean feature")
    agg = df.groupby(['matchId','groupId'])[features].agg('mean')
    agg_rank = agg.groupby('matchId')[features].rank(pct=True).reset_index()
    
    if is_train: df_out = agg.reset_index()[['matchId','groupId']]
    else: df_out = df[['matchId','groupId']]

    df_out = df_out.merge(agg.reset_index(), suffixes=["", ""], how='left', on=['matchId', 'groupId'])
    df_out = df_out.merge(agg_rank, suffixes=["_mean", "_mean_rank"], how='left', on=['matchId', 'groupId'])
    
    # print("get group sum feature")
    # agg = df.groupby(['matchId','groupId'])[features].agg('sum')
    # agg_rank = agg.groupby('matchId')[features].rank(pct=True).reset_index()
    # df_out = df_out.merge(agg.reset_index(), suffixes=["", ""], how='left', on=['matchId', 'groupId'])
    # df_out = df_out.merge(agg_rank, suffixes=["_sum", "_sum_rank"], how='left', on=['matchId', 'groupId'])
    
    # print("get group sum feature")
    # agg = df.groupby(['matchId','groupId'])[features].agg('sum')
    # agg_rank = agg.groupby('matchId')[features].agg('sum')
    # df_out = df_out.merge(agg.reset_index(), suffixes=["", ""], how='left', on=['matchId', 'groupId'])
    # df_out = df_out.merge(agg_rank.reset_index(), suffixes=["_sum", "_sum_pct"], how='left', on=['matchId', 'groupId'])
    
    print("get group max feature")
    agg = df.groupby(['matchId','groupId'])[features].agg('max')
    agg_rank = agg.groupby('matchId')[features].rank(pct=True).reset_index()
    df_out = df_out.merge(agg.reset_index(), suffixes=["", ""], how='left', on=['matchId', 'groupId'])
    df_out = df_out.merge(agg_rank, suffixes=["_max", "_max_rank"], how='left', on=['matchId', 'groupId'])
    
    print("get group min feature")
    agg = df.groupby(['matchId','groupId'])[features].agg('min')
    agg_rank = agg.groupby('matchId')[features].rank(pct=True).reset_index()
    df_out = df_out.merge(agg.reset_index(), suffixes=["", ""], how='left', on=['matchId', 'groupId'])
    df_out = df_out.merge(agg_rank, suffixes=["_min", "_min_rank"], how='left', on=['matchId', 'groupId'])
    
    print("get group size feature")
    agg = df.groupby(['matchId','groupId']).size().reset_index(name='group_size')
    df_out = df_out.merge(agg, how='left', on=['matchId', 'groupId'])
    
    print("get match mean feature")
    agg = df.groupby(['matchId'])[features].agg('mean').reset_index()
    df_out = df_out.merge(agg, suffixes=["", "_match_mean"], how='left', on=['matchId'])
    
    # print("get match type feature")
    # agg = df.groupby(['matchId'])[matchType.columns].agg('mean').reset_index()
    # df_out = df_out.merge(agg, suffixes=["", "_match_type"], how='left', on=['matchId'])
    
    print("get match size feature")
    agg = df.groupby(['matchId']).size().reset_index(name='match_size')
    df_out = df_out.merge(agg, how='left', on=['matchId'])
    
    df_out.drop(["matchId", "groupId"], axis=1, inplace=True)

    X = df_out
    
    feature_names = list(df_out.columns)

    del df, df_out, agg, agg_rank
    gc.collect()

    return X, y, feature_names, test_idx
x_train, y_train, train_columns, _ = feature_engineering(True,False)
x_test, _, _ , test_idx = feature_engineering(False,False)
x_train.shape
x_train = reduce_mem_usage(x_train)
x_test = reduce_mem_usage(x_test)
import os
import time
import gc
import warnings
warnings.filterwarnings("ignore")
# data manipulation

# model
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
import lightgbm as lgb
#excluded_features = []
#use_cols = [col for col in df_train.columns if col not in excluded_features]

train_index = round(int(x_train.shape[0]*0.7))
dev_X = x_train[:train_index] 
val_X = x_train[train_index:]
dev_y = y_train[:train_index] 
val_y = y_train[train_index:] 
gc.collect();

# custom function to run light gbm model
def run_lgb(train_X, train_y, val_X, val_y, x_test):
    params = {"objective" : "regression", "metric" : "mae", 'n_estimators':20000, 'early_stopping_rounds':200,
              "num_leaves" : 31, "learning_rate" : 0.05, "bagging_fraction" : 0.7,
               "bagging_seed" : 0, "num_threads" : 4,"colsample_bytree" : 0.7
             }
    
    lgtrain = lgb.Dataset(train_X, label=train_y)
    lgval = lgb.Dataset(val_X, label=val_y)
    model = lgb.train(params, lgtrain, valid_sets=[lgtrain, lgval], early_stopping_rounds=200, verbose_eval=1000)
    
    pred_test_y = model.predict(x_test, num_iteration=model.best_iteration)
    return pred_test_y, model

# Training the model #
pred_test, model = run_lgb(dev_X, dev_y, val_X, val_y, x_test)
del dev_X, dev_y, val_X, val_y, x_test;
gc.collect()
df_sub = pd.read_csv("../input/sample_submission_V2.csv")
df_test = pd.read_csv("../input/test_V2.csv")
df_sub['winPlacePerc'] = pred_test
# Restore some columns
df_sub = df_sub.merge(df_test[["Id", "matchId", "groupId", "maxPlace", "numGroups"]], on="Id", how="left")

# Sort, rank, and assign adjusted ratio
df_sub_group = df_sub.groupby(["matchId", "groupId"]).first().reset_index()
df_sub_group["rank"] = df_sub_group.groupby(["matchId"])["winPlacePerc"].rank()
df_sub_group = df_sub_group.merge(
    df_sub_group.groupby("matchId")["rank"].max().to_frame("max_rank").reset_index(), 
    on="matchId", how="left")
df_sub_group["adjusted_perc"] = (df_sub_group["rank"] - 1) / (df_sub_group["numGroups"] - 1)

df_sub = df_sub.merge(df_sub_group[["adjusted_perc", "matchId", "groupId"]], on=["matchId", "groupId"], how="left")
df_sub["winPlacePerc"] = df_sub["adjusted_perc"]

# Deal with edge cases
df_sub.loc[df_sub.maxPlace == 0, "winPlacePerc"] = 0
df_sub.loc[df_sub.maxPlace == 1, "winPlacePerc"] = 1

# Align with maxPlace
# Credit: https://www.kaggle.com/anycode/simple-nn-baseline-4
subset = df_sub.loc[df_sub.maxPlace > 1]
gap = 1.0 / (subset.maxPlace.values - 1)
new_perc = np.around(subset.winPlacePerc.values / gap) * gap
df_sub.loc[df_sub.maxPlace > 1, "winPlacePerc"] = new_perc

# Edge case
df_sub.loc[(df_sub.maxPlace > 1) & (df_sub.numGroups == 1), "winPlacePerc"] = 0
assert df_sub["winPlacePerc"].isnull().sum() == 0

df_sub[["Id", "winPlacePerc"]].to_csv("submission_adjusted.csv", index=False)