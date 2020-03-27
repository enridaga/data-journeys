
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

from IPython.display import HTML
HTML('<iframe width="1100" height="619" src="https://www.youtube.com/embed/45Da3eqQKXQ" frameborder="0" allow="accelerometer; autoplay; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>')

import os
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# matplotlib and seaborn for plotting
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.patches as patches

from plotly import tools, subplots
import plotly.offline as py
py.init_notebook_mode(connected=True)
import plotly.graph_objs as go
import plotly.express as px
pd.set_option('max_columns', 100)


py.init_notebook_mode(connected=True)
from plotly.offline import init_notebook_mode, iplot


import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

from tqdm import tqdm_notebook

from IPython.display import HTML

#%matplotlib inline
plt.rc('figure', figsize=(15.0, 8.0))
import os
print(os.listdir("../input/data-science-bowl-2019/"))
%%time
root = '../input/data-science-bowl-2019/'

# Only load those columns in order to save space
keep_cols = ['event_id', 'game_session', 'installation_id', 'event_count', 'event_code', 'title', 'game_time', 'type', 'world']
train = pd.read_csv(root + 'train.csv',usecols=keep_cols)
test = pd.read_csv(root + 'test.csv', usecols=keep_cols)

train_labels = pd.read_csv(root + 'train_labels.csv')
specs = pd.read_csv(root + 'specs.csv')
sample_submission = pd.read_csv(root + 'sample_submission.csv')
print('Size of train data', train.shape)
print('Size of train_labels data', train_labels.shape)
print('Size of specs data', specs.shape)
print('Size of test data', test.shape)
train.head()
train_labels.head()
specs.head()
train.dtypes.value_counts()
train.select_dtypes('object').apply(pd.Series.nunique, axis = 0)
train_labels.dtypes.value_counts()
train_labels.select_dtypes('object').apply(pd.Series.nunique, axis = 0)
specs.dtypes.value_counts()
specs.select_dtypes('object').apply(pd.Series.nunique, axis = 0)
total = train.isnull().sum().sort_values(ascending = False)
percent = (train.isnull().sum()/train.isnull().count()*100).sort_values(ascending = False)
missing__train_data  = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
missing__train_data.head(10)
total = train_labels.isnull().sum().sort_values(ascending = False)
percent = (train_labels.isnull().sum()/train_labels.isnull().count()*100).sort_values(ascending = False)
missing__train_data  = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
missing__train_data.head(10)
total = specs.isnull().sum().sort_values(ascending = False)
percent = (specs.isnull().sum()/specs.isnull().count()*100).sort_values(ascending = False)
missing__train_data  = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
missing__train_data.head(10)
corrs = train.corr()
corrs
plt.figure(figsize = (20, 8))

# Heatmap of correlations
sns.heatmap(corrs, cmap = plt.cm.RdYlBu_r, vmin = -0.25, annot = True, vmax = 0.6)
plt.title('Correlation Heatmap');
corrs2 = train_labels.corr()
corrs2
plt.figure(figsize = (20, 8))

# Heatmap of correlations
sns.heatmap(corrs2, cmap = plt.cm.RdYlBu_r, vmin = -0.25, annot = True, vmax = 0.6)
plt.title('Correlation Heatmap');
plt.figure(figsize=(8, 6))
sns.countplot(x="accuracy_group",data=train_labels, order = train_labels['accuracy_group'].value_counts().index)
plt.title('Accuracy Group Count Column')
plt.tight_layout()
plt.show()
train_labels.groupby('accuracy_group')['game_session'].count() \
    .plot(kind='barh', figsize=(15, 5), title='Target (accuracy group)')
plt.show()
train.head()

palete = sns.color_palette(n_colors=10)
train.groupby('installation_id') \
    .count()['event_id'] \
    .apply(np.log1p) \
    .plot(kind='hist',
          bins=40,
          color=palete[1],
         figsize=(15, 5),
         title='Log(Count) of Observations by installation_id')
plt.show()
train.groupby('title')['event_id'] \
    .count() \
    .sort_values() \
    .plot(kind='barh',
          title='Count of Observation by Game/Video title',
         color=palete[1],
         figsize=(15, 15))
plt.show()
train.groupby('world')['event_id'] \
    .count() \
    .sort_values() \
    .plot(kind='bar',
          figsize=(15, 4),
          title='Count by World',
          color=palete[1])
plt.show()
def group_and_reduce(df):
    # group1 and group2 are intermediary "game session" groups,
    # which are reduced to one record by game session. group1 takes
    # the max value of game_time (final game time in a session) and 
    # of event_count (total number of events happened in the session).
    # group2 takes the total number of event_code of each type
    group1 = df.drop(columns=['event_id', 'event_code']).groupby(
        ['game_session', 'installation_id', 'title', 'type', 'world']
    ).max().reset_index()

    group2 = pd.get_dummies(
        df[['installation_id', 'event_code']], 
        columns=['event_code']
    ).groupby(['installation_id']).sum()

    # group3, group4 and group5 are grouped by installation_id 
    # and reduced using summation and other summary stats
    group3 = pd.get_dummies(
        group1.drop(columns=['game_session', 'event_count', 'game_time']),
        columns=['title', 'type', 'world']
    ).groupby(['installation_id']).sum()

    group4 = group1[
        ['installation_id', 'event_count', 'game_time']
    ].groupby(
        ['installation_id']
    ).agg([np.sum, np.mean, np.std])

    return group2.join(group3).join(group4)
%%time
train_small = group_and_reduce(train)
test_small = group_and_reduce(test)

print(train_small.shape)
train_small.head()
%%time
from sklearn.model_selection import KFold
small_labels = train_labels[['installation_id', 'accuracy_group']].set_index('installation_id')
train_joined = train_small.join(small_labels).dropna()
kf = KFold(n_splits=5, random_state=2019)
X = train_joined.drop(columns='accuracy_group').values
y = train_joined['accuracy_group'].values.astype(np.int32)
y_pred = np.zeros((len(test_small), 4))
for train, test in kf.split(X):
    x_train, x_val, y_train, y_val = X[train], X[test], y[train], y[test]
    train_set = lgb.Dataset(x_train, y_train)
    val_set = lgb.Dataset(x_val, y_val)

    params = {
        'learning_rate': 0.01,
        'bagging_fraction': 0.9,
        'feature_fraction': 0.9,
        'num_leaves': 50,
        'lambda_l1': 0.1,
        'lambda_l2': 1,
        'metric': 'multiclass',
        'objective': 'multiclass',
        'num_classes': 4,
        'random_state': 2019
    }

    model = lgb.train(params, train_set, num_boost_round=5000, early_stopping_rounds=50, valid_sets=[train_set, val_set], verbose_eval=50)
    y_pred += model.predict(test_small)
%%time
y_pred = y_pred.argmax(axis=1)
test_small['accuracy_group'] = y_pred
test_small[['accuracy_group']].to_csv('submission.csv')
%%time
val_pred = model.predict(x_val).argmax(axis=1)
print(classification_report(y_val, val_pred))

HTML('<iframe width="1106" height="622" src="https://www.youtube.com/embed/1ejHigxuR2Q" frameborder="0" allow="accelerometer; autoplay; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>')