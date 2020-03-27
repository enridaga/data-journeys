
import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

from sklearn.preprocessing import Imputer

from sklearn.preprocessing import PolynomialFeatures

from sklearn.preprocessing import StandardScaler

from sklearn.feature_selection import VarianceThreshold

from sklearn.feature_selection import SelectFromModel

from sklearn.utils import shuffle

from sklearn.ensemble import RandomForestClassifier



pd.set_option('display.max_columns', 100)
train = pd.read_csv('../input/train.csv')

test = pd.read_csv('../input/test.csv')
train.head()
train.tail()
train.shape
train.drop_duplicates()

train.shape
test.shape
train.info()
data = []

for f in train.columns:

    # Defining the role

    if f == 'target':

        role = 'target'

    elif f == 'id':

        role = 'id'

    else:

        role = 'input'

         

    # Defining the level

    if 'bin' in f or f == 'target':

        level = 'binary'

    elif 'cat' in f or f == 'id':

        level = 'nominal'

    elif train[f].dtype == float:

        level = 'interval'

    elif train[f].dtype == int:

        level = 'ordinal'

        

    # Initialize keep to True for all variables except for id

    keep = True

    if f == 'id':

        keep = False

    

    # Defining the data type 

    dtype = train[f].dtype

    

    # Creating a Dict that contains all the metadata for the variable

    f_dict = {

        'varname': f,

        'role': role,

        'level': level,

        'keep': keep,

        'dtype': dtype

    }

    data.append(f_dict)

    

meta = pd.DataFrame(data, columns=['varname', 'role', 'level', 'keep', 'dtype'])

meta.set_index('varname', inplace=True)
meta
meta[(meta.level == 'nominal') & (meta.keep)].index
pd.DataFrame({'count' : meta.groupby(['role', 'level'])['role'].size()}).reset_index()
v = meta[(meta.level == 'interval') & (meta.keep)].index

train[v].describe()
v = meta[(meta.level == 'ordinal') & (meta.keep)].index

train[v].describe()
v = meta[(meta.level == 'binary') & (meta.keep)].index

train[v].describe()
desired_apriori=0.10



# Get the indices per target value

idx_0 = train[train.target == 0].index

idx_1 = train[train.target == 1].index



# Get original number of records per target value

nb_0 = len(train.loc[idx_0])

nb_1 = len(train.loc[idx_1])



# Calculate the undersampling rate and resulting number of records with target=0

undersampling_rate = ((1-desired_apriori)*nb_1)/(nb_0*desired_apriori)

undersampled_nb_0 = int(undersampling_rate*nb_0)

print('Rate to undersample records with target=0: {}'.format(undersampling_rate))

print('Number of records with target=0 after undersampling: {}'.format(undersampled_nb_0))



# Randomly select records with target=0 to get at the desired a priori

undersampled_idx = shuffle(idx_0, random_state=37, n_samples=undersampled_nb_0)



# Construct list with remaining indices

idx_list = list(undersampled_idx) + list(idx_1)



# Return undersample data frame

train = train.loc[idx_list].reset_index(drop=True)
vars_with_missing = []



for f in train.columns:

    missings = train[train[f] == -1][f].count()

    if missings > 0:

        vars_with_missing.append(f)

        missings_perc = missings/train.shape[0]

        

        print('Variable {} has {} records ({:.2%}) with missing values'.format(f, missings, missings_perc))

        

print('In total, there are {} variables with missing values'.format(len(vars_with_missing)))
# Dropping the variables with too many missing values

vars_to_drop = ['ps_car_03_cat', 'ps_car_05_cat']

train.drop(vars_to_drop, inplace=True, axis=1)

meta.loc[(vars_to_drop),'keep'] = False  # Updating the meta



# Imputing with the mean or mode

mean_imp = Imputer(missing_values=-1, strategy='mean', axis=0)

mode_imp = Imputer(missing_values=-1, strategy='most_frequent', axis=0)

train['ps_reg_03'] = mean_imp.fit_transform(train[['ps_reg_03']]).ravel()

train['ps_car_12'] = mean_imp.fit_transform(train[['ps_car_12']]).ravel()

train['ps_car_14'] = mean_imp.fit_transform(train[['ps_car_14']]).ravel()

train['ps_car_11'] = mode_imp.fit_transform(train[['ps_car_11']]).ravel()
v = meta[(meta.level == 'nominal') & (meta.keep)].index



for f in v:

    dist_values = train[f].value_counts().shape[0]

    print('Variable {} has {} distinct values'.format(f, dist_values))
# Script by https://www.kaggle.com/ogrellier

# Code: https://www.kaggle.com/ogrellier/python-target-encoding-for-categorical-features

def add_noise(series, noise_level):

    return series * (1 + noise_level * np.random.randn(len(series)))



def target_encode(trn_series=None, 

                  tst_series=None, 

                  target=None, 

                  min_samples_leaf=1, 

                  smoothing=1,

                  noise_level=0):

    """

    Smoothing is computed like in the following paper by Daniele Micci-Barreca

    https://kaggle2.blob.core.windows.net/forum-message-attachments/225952/7441/high%20cardinality%20categoricals.pdf

    trn_series : training categorical feature as a pd.Series

    tst_series : test categorical feature as a pd.Series

    target : target data as a pd.Series

    min_samples_leaf (int) : minimum samples to take category average into account

    smoothing (int) : smoothing effect to balance categorical average vs prior  

    """ 

    assert len(trn_series) == len(target)

    assert trn_series.name == tst_series.name

    temp = pd.concat([trn_series, target], axis=1)

    # Compute target mean 

    averages = temp.groupby(by=trn_series.name)[target.name].agg(["mean", "count"])

    # Compute smoothing

    smoothing = 1 / (1 + np.exp(-(averages["count"] - min_samples_leaf) / smoothing))

    # Apply average function to all target data

    prior = target.mean()

    # The bigger the count the less full_avg is taken into account

    averages[target.name] = prior * (1 - smoothing) + averages["mean"] * smoothing

    averages.drop(["mean", "count"], axis=1, inplace=True)

    # Apply averages to trn and tst series

    ft_trn_series = pd.merge(

        trn_series.to_frame(trn_series.name),

        averages.reset_index().rename(columns={'index': target.name, target.name: 'average'}),

        on=trn_series.name,

        how='left')['average'].rename(trn_series.name + '_mean').fillna(prior)

    # pd.merge does not keep the index so restore it

    ft_trn_series.index = trn_series.index 

    ft_tst_series = pd.merge(

        tst_series.to_frame(tst_series.name),

        averages.reset_index().rename(columns={'index': target.name, target.name: 'average'}),

        on=tst_series.name,

        how='left')['average'].rename(trn_series.name + '_mean').fillna(prior)

    # pd.merge does not keep the index so restore it

    ft_tst_series.index = tst_series.index

    return add_noise(ft_trn_series, noise_level), add_noise(ft_tst_series, noise_level)
train_encoded, test_encoded = target_encode(train["ps_car_11_cat"], 

                             test["ps_car_11_cat"], 

                             target=train.target, 

                             min_samples_leaf=100,

                             smoothing=10,

                             noise_level=0.01)

    

train['ps_car_11_cat_te'] = train_encoded

train.drop('ps_car_11_cat', axis=1, inplace=True)

meta.loc['ps_car_11_cat','keep'] = False  # Updating the meta

test['ps_car_11_cat_te'] = test_encoded

test.drop('ps_car_11_cat', axis=1, inplace=True)
v = meta[(meta.level == 'nominal') & (meta.keep)].index



for f in v:

    plt.figure()

    fig, ax = plt.subplots(figsize=(20,10))

    # Calculate the percentage of target=1 per category value

    cat_perc = train[[f, 'target']].groupby([f],as_index=False).mean()

    cat_perc.sort_values(by='target', ascending=False, inplace=True)

    # Bar plot

    # Order the bars descending on target mean

    sns.barplot(ax=ax, x=f, y='target', data=cat_perc, order=cat_perc[f])

    plt.ylabel('% target', fontsize=18)

    plt.xlabel(f, fontsize=18)

    plt.tick_params(axis='both', which='major', labelsize=18)

    plt.show();
def corr_heatmap(v):

    correlations = train[v].corr()



    # Create color map ranging between two colors

    cmap = sns.diverging_palette(220, 10, as_cmap=True)



    fig, ax = plt.subplots(figsize=(10,10))

    sns.heatmap(correlations, cmap=cmap, vmax=1.0, center=0, fmt='.2f',

                square=True, linewidths=.5, annot=True, cbar_kws={"shrink": .75})

    plt.show();

    

v = meta[(meta.level == 'interval') & (meta.keep)].index

corr_heatmap(v)
s = train.sample(frac=0.1)
sns.lmplot(x='ps_reg_02', y='ps_reg_03', data=s, hue='target', palette='Set1', scatter_kws={'alpha':0.3})

plt.show()
sns.lmplot(x='ps_car_12', y='ps_car_13', data=s, hue='target', palette='Set1', scatter_kws={'alpha':0.3})

plt.show()
sns.lmplot(x='ps_car_12', y='ps_car_14', data=s, hue='target', palette='Set1', scatter_kws={'alpha':0.3})

plt.show()
sns.lmplot(x='ps_car_15', y='ps_car_13', data=s, hue='target', palette='Set1', scatter_kws={'alpha':0.3})

plt.show()
v = meta[(meta.level == 'ordinal') & (meta.keep)].index

corr_heatmap(v)
v = meta[(meta.level == 'nominal') & (meta.keep)].index

print('Before dummification we have {} variables in train'.format(train.shape[1]))

train = pd.get_dummies(train, columns=v, drop_first=True)

print('After dummification we have {} variables in train'.format(train.shape[1]))
v = meta[(meta.level == 'interval') & (meta.keep)].index

poly = PolynomialFeatures(degree=2, interaction_only=False, include_bias=False)

interactions = pd.DataFrame(data=poly.fit_transform(train[v]), columns=poly.get_feature_names(v))

interactions.drop(v, axis=1, inplace=True)  # Remove the original columns

# Concat the interaction variables to the train data

print('Before creating interactions we have {} variables in train'.format(train.shape[1]))

train = pd.concat([train, interactions], axis=1)

print('After creating interactions we have {} variables in train'.format(train.shape[1]))
selector = VarianceThreshold(threshold=.01)

selector.fit(train.drop(['id', 'target'], axis=1)) # Fit to train without id and target variables



f = np.vectorize(lambda x : not x) # Function to toggle boolean array elements



v = train.drop(['id', 'target'], axis=1).columns[f(selector.get_support())]

print('{} variables have too low variance.'.format(len(v)))

print('These variables are {}'.format(list(v)))
X_train = train.drop(['id', 'target'], axis=1)

y_train = train['target']



feat_labels = X_train.columns



rf = RandomForestClassifier(n_estimators=1000, random_state=0, n_jobs=-1)



rf.fit(X_train, y_train)

importances = rf.feature_importances_



indices = np.argsort(rf.feature_importances_)[::-1]



for f in range(X_train.shape[1]):

    print("%2d) %-*s %f" % (f + 1, 30,feat_labels[indices[f]], importances[indices[f]]))

sfm = SelectFromModel(rf, threshold='median', prefit=True)

print('Number of features before selection: {}'.format(X_train.shape[1]))

n_features = sfm.transform(X_train).shape[1]

print('Number of features after selection: {}'.format(n_features))

selected_vars = list(feat_labels[sfm.get_support()])

train = train[selected_vars + ['target']]
scaler = StandardScaler()

scaler.fit_transform(train.drop(['target'], axis=1))