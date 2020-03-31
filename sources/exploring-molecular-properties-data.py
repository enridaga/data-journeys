
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
import matplotlib.pylab as plt
import seaborn as sns
from sklearn import metrics
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
plt.style.use('ggplot')
color_pal = [x['color'] for x in plt.rcParams['axes.prop_cycle']]
# Show how the files appear in the input folder
### ls -GFlash --color ../input
train_df = pd.read_csv('../input/train.csv')
test_df = pd.read_csv('../input/test.csv')
train_df.head()
print('The training set has shape {}'.format(train_df.shape))
print('The test set has shape {}'.format(test_df.shape))
# Distribution of the target
train_df['scalar_coupling_constant'].plot(kind='hist', figsize=(20, 5), bins=1000, title='Distribution of the target scalar coupling constant')
plt.show()
# Number of of atoms in molecule
fig, ax = plt.subplots(1, 2)
train_df.groupby('molecule_name').count().sort_values('id')['id'].plot(kind='hist',
                                                                       bins=25,
                                                                       color=color_pal[6],
                                                                      figsize=(20, 5),
                                                                      title='# of Atoms in Molecule (Train Set)',
                                                                      ax=ax[0])
test_df.groupby('molecule_name').count().sort_values('id')['id'].plot(kind='hist',
                                                                       bins=25,
                                                                       color=color_pal[2],
                                                                      figsize=(20, 5),
                                                                      title='# of Atoms in Molecule (Test Set)',
                                                                     ax=ax[1])
plt.show()
ss = pd.read_csv('../input/sample_submission.csv')
ss.head()
test_df.head()
train_df.head(1)
###  cat ../input/structures/dsgdb9nsd_000001.xyz
structures = pd.read_csv('../input/structures.csv')
structures.head()
# 3D Plot!
from mpl_toolkits.mplot3d import Axes3D
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
example = structures.loc[structures['molecule_name'] == 'dsgdb9nsd_000001']
ax.scatter(xs=example['x'], ys=example['y'], zs=example['z'], s=100)
plt.suptitle('dsgdb9nsd_000001')
plt.show()
dm = pd.read_csv('../input/dipole_moments.csv')
dm.head()
mst = pd.read_csv('../input/magnetic_shielding_tensors.csv')
mul = pd.read_csv('../input/mulliken_charges.csv')
pote = pd.read_csv('../input/potential_energy.csv')
scc = pd.read_csv('../input/scalar_coupling_contributions.csv')
mst.head()
mul.head()
# Plot the distribution of mulliken_charges
mul['mulliken_charge'].plot(kind='hist', figsize=(15, 5), bins=500, title='Distribution of Mulliken Charges')
plt.show()
pote.head()
# Plot the distribution of potential_energy
pote['potential_energy'].plot(kind='hist',
                              figsize=(15, 5),
                              bins=500,
                              title='Distribution of Potential Energy',
                              color='b')
plt.show()
scc.head()
scc.groupby('type').count()['molecule_name'].sort_values().plot(kind='barh',
                                                                color='grey',
                                                               figsize=(15, 5),
                                                               title='Count of Coupling Type in Train Set')
plt.show()
fig, ax = plt.subplots(2, 2, figsize=(20, 10))
scc['fc'].plot(kind='hist', ax=ax.flat[0], bins=500, title='Fermi Contact contribution', color=color_pal[0])
scc['sd'].plot(kind='hist', ax=ax.flat[1], bins=500, title='Spin-dipolar contribution', color=color_pal[1])
scc['pso'].plot(kind='hist', ax=ax.flat[2], bins=500, title='Paramagnetic spin-orbit contribution', color=color_pal[2])
scc['dso'].plot(kind='hist', ax=ax.flat[3], bins=500, title='Diamagnetic spin-orbit contribution', color=color_pal[3])
plt.show()
scc = scc.merge(train_df)
# Downsample to speed up plot time.
sns.pairplot(data=scc.sample(5000), hue='type', vars=['fc','sd','pso','dso','scalar_coupling_constant'])
plt.show()
atom_count_dict = structures.groupby('molecule_name').count()['atom_index'].to_dict()
train_df['atom_count'] = train_df['molecule_name'].map(atom_count_dict)
test_df['atom_count'] = test_df['molecule_name'].map(atom_count_dict)
train_df.sample(1000).plot(x='atom_count',
                           y='scalar_coupling_constant',
                           kind='scatter',
                           color=color_pal[0],
                           figsize=(20, 5),
                           alpha=0.5)
plt.show()
train_df.groupby('type')['scalar_coupling_constant'].mean().plot(kind='barh',
                                                                 figsize=(15, 5),
                                                                title='Average Scalar Coupling Constant by Type')
plt.show()
# THIS IS A MODEL!!! This is a model??
type_mean_dict = train_df.groupby('type')['scalar_coupling_constant'].mean().to_dict()
test_df['scalar_coupling_constant'] = test_df['type'].map(type_mean_dict)
test_df[['id','scalar_coupling_constant']].to_csv('super_simple_submission.csv', index=False)
def metric(df, preds):
    df["prediction"] = preds
    maes = []
    for t in df.type.unique():
        y_true = df[df.type==t].scalar_coupling_constant.values
        y_pred = df[df.type==t].prediction.values
        mae = np.log(metrics.mean_absolute_error(y_true, y_pred))
        maes.append(mae)
    return np.mean(maes)
def map_atom_info(df, atom_idx):
    df = pd.merge(df, structures, how = 'left',
                  left_on  = ['molecule_name', f'atom_index_{atom_idx}'],
                  right_on = ['molecule_name',  'atom_index'])
    
    df = df.drop('atom_index', axis=1)
    df = df.rename(columns={'atom': f'atom_{atom_idx}',
                            'x': f'x_{atom_idx}',
                            'y': f'y_{atom_idx}',
                            'z': f'z_{atom_idx}'})
    return df

train_df = map_atom_info(train_df, 0)
train_df = map_atom_info(train_df, 1)

test_df = map_atom_info(test_df, 0)
test_df = map_atom_info(test_df, 1)
# https://www.kaggle.com/seriousran/just-speed-up-calculate-distance-from-benchmark
train_p_0 = train_df[['x_0', 'y_0', 'z_0']].values
train_p_1 = train_df[['x_1', 'y_1', 'z_1']].values
test_p_0 = test_df[['x_0', 'y_0', 'z_0']].values
test_p_1 = test_df[['x_1', 'y_1', 'z_1']].values

train_df['dist'] = np.linalg.norm(train_p_0 - train_p_1, axis=1)
test_df['dist'] = np.linalg.norm(test_p_0 - test_p_1, axis=1)
# make categorical variables
atom_map = {'H': 0,
            'C': 1,
            'N': 2}
train_df['atom_0_cat'] = train_df['atom_0'].map(atom_map).astype('int')
train_df['atom_1_cat'] = train_df['atom_1'].map(atom_map).astype('int')
test_df['atom_0_cat'] = test_df['atom_0'].map(atom_map).astype('int')
test_df['atom_1_cat'] = test_df['atom_1'].map(atom_map).astype('int')
# One Hot Encode the Type
train_df = pd.concat([train_df, pd.get_dummies(train_df['type'])], axis=1)
test_df = pd.concat([test_df, pd.get_dummies(test_df['type'])], axis=1)
color_index = 0
axes_index = 0
fig, axes = plt.subplots(8, 1, figsize=(20, 20), sharex=True)
for mtype, d in train_df.groupby('type'):
    d['dist'].plot(kind='hist',
                  bins=1000,
                  title='Distribution of Distance Feature for {}'.format(mtype),
                  color=color_pal[color_index],
                  ax=axes[axes_index])
    if color_index == 6:
        color_index = 0
    else:
        color_index += 1
    axes_index += 1
plt.show()
train_df['dist_to_type_mean'] = train_df['dist'] / train_df.groupby('type')['dist'].transform('mean')
test_df['dist_to_type_mean'] = test_df['dist'] / test_df.groupby('type')['dist'].transform('mean')
# Configurables
FEATURES = ['atom_index_0', 'atom_index_1',
            'atom_0_cat',
            'x_0', 'y_0', 'z_0',
            'atom_1_cat', 
            'x_1', 'y_1', 'z_1', 'dist', 'dist_to_type_mean',
            'atom_count',
            '1JHC', '1JHN', '2JHC', '2JHH', '2JHN', '3JHC', '3JHH', '3JHN'
           ]
TARGET = 'scalar_coupling_constant'
CAT_FEATS = ['atom_0','atom_1']
N_ESTIMATORS = 2000
VERBOSE = 500
EARLY_STOPPING_ROUNDS = 200
RANDOM_STATE = 529

X = train_df[FEATURES]
X_test = test_df[FEATURES]
y = train_df[TARGET]
from sklearn.model_selection import KFold
from sklearn.metrics import mean_absolute_error
import lightgbm as lgb

lgb_params = {'num_leaves': 128,
              'min_child_samples': 64,
              'objective': 'regression',
              'max_depth': 6,
              'learning_rate': 0.1,
              "boosting_type": "gbdt",
              "subsample_freq": 1,
              "subsample": 0.9,
              "bagging_seed": 11,
              "metric": 'mae',
              "verbosity": -1,
              'reg_alpha': 0.1,
              'reg_lambda': 0.4,
              'colsample_bytree': 1.0
         }

RUN_LGB = False

if RUN_LGB:
    n_fold = 5
    folds = KFold(n_splits=n_fold, shuffle=True, random_state=RANDOM_STATE)

    # Setup arrays for storing results
    oof = np.zeros(len(X))
    prediction = np.zeros(len(X_test))
    scores = []
    feature_importance = pd.DataFrame()

    # Train the model
    for fold_n, (train_idx, valid_idx) in enumerate(folds.split(X)):
        X_train, X_valid = X.iloc[train_idx], X.iloc[valid_idx]
        y_train, y_valid = y.iloc[train_idx], y.iloc[valid_idx]
        model = lgb.LGBMRegressor(**lgb_params, n_estimators = N_ESTIMATORS, n_jobs = -1)
        model.fit(X_train, y_train,
                  eval_set=[(X_train, y_train), (X_valid, y_valid)],
                  eval_metric='mae',
                  verbose=VERBOSE,
                  early_stopping_rounds=EARLY_STOPPING_ROUNDS)

        y_pred_valid = model.predict(X_valid)
        y_pred = model.predict(X_test, num_iteration=model.best_iteration_)

        # feature importance
        fold_importance = pd.DataFrame()
        fold_importance["feature"] = FEATURES
        fold_importance["importance"] = model.feature_importances_
        fold_importance["fold"] = fold_n + 1
        feature_importance = pd.concat([feature_importance, fold_importance], axis=0)

        prediction /= folds.n_splits
        scores.append(mean_absolute_error(y_valid, y_pred_valid))
        print('CV mean score: {0:.4f}, std: {1:.4f}.'.format(np.mean(scores), np.std(scores)))
        oof[valid_idx] = y_pred_valid.reshape(-1,)
        scores.append(mean_absolute_error(y_valid, y_pred_valid))
        prediction += y_pred
if RUN_LGB:
    # Save Prediction and name appropriately
    submission_csv_name = 'submission_lgb_{}folds_{}CV.csv'.format(n_fold, np.mean(scores))
    oof_csv_name = 'oof_lgb_{}folds_{}CV.csv'.format(n_fold, np.mean(scores))
    fi_csv_name = 'fi_lgb_{}folds_{}CV.csv'.format(n_fold, np.mean(scores))

    print('Saving LGB Submission as:')
    print(submission_csv_name)
    ss = pd.read_csv('../input/sample_submission.csv')
    ss['scalar_coupling_constant'] = prediction
    ss.to_csv(submission_csv_name, index=False)
    ss.head()
    # OOF
    oof_df = train_df[['id','molecule_name','scalar_coupling_constant']].copy()
    oof_df['oof_pred'] = oof
    oof_df.to_csv(oof_csv_name, index=False)
    # Feature Importance
    feature_importance.to_csv(fi_csv_name, index=False)
if RUN_LGB:
    # Plot feature importance as done in https://www.kaggle.com/artgor/artgor-utils
    feature_importance["importance"] /= folds.n_splits
    cols = feature_importance[["feature", "importance"]].groupby("feature").mean().sort_values(
        by="importance", ascending=False)[:50].index

    best_features = feature_importance.loc[feature_importance.feature.isin(cols)]

    plt.figure(figsize=(15, 20));
    ax = sns.barplot(x="importance",
                y="feature",
                hue='fold',
                data=best_features.sort_values(by="importance", ascending=False));
    plt.title('LGB Features (avg over folds)');
from catboost import Pool, cv

RUN_CATBOOST_CV = False

if RUN_CATBOOST_CV:
    labels = train_df['scalar_coupling_constant'].values
    cat_features = ['type','atom_count','atom_0','atom_1']
    cv_data = train_df[['type','atom_count','atom_0','atom_1',
                        'x_0','y_0','z_0','x_1','y_1','z_1','dist']]
    cv_dataset = Pool(data=cv_data,
                      label=labels,
                      cat_features=cat_features)

    ITERATIONS = 100000
    params = {"iterations": ITERATIONS,
              "learning_rate" : 0.02,
              "depth": 7,
              "loss_function": "MAE",
              "verbose": False,
              "task_type" : "GPU"}

    scores = cv(cv_dataset,
                params,
                fold_count=5, 
                plot="True")
    
    scores['iterations'] = scores['iterations'].astype('int')
    scores.set_index('iterations')[['test-MAE-mean','train-MAE-mean']].plot(figsize=(15, 5), title='CV (MAE) Score by iteration (5 Folds)')
from catboost import CatBoostRegressor, Pool

ITERATIONS = 200000

FEATURES = [#'atom_index_0',
            'atom_index_1',
            'atom_0',
            'x_0', 'y_0', 'z_0',
            'atom_1', 
            'x_1', 'y_1', 'z_1',
            'dist', 'dist_to_type_mean',
            'atom_count',
            'type']
TARGET = 'scalar_coupling_constant'
CAT_FEATS = ['atom_0','atom_1','type']

train_dataset = Pool(data=train_df[FEATURES],
                  label=train_df['scalar_coupling_constant'].values,
                  cat_features=CAT_FEATS)

cb_model = CatBoostRegressor(iterations=ITERATIONS,
                             learning_rate=0.2,
                             depth=7,
                             eval_metric='MAE',
                             random_seed = 529,
                             task_type="GPU")

# Fit the model
cb_model.fit(train_dataset, verbose=1000)

# Predict
test_data = test_df[FEATURES]

test_dataset = Pool(data=test_data,
                    cat_features=CAT_FEATS)

ss = pd.read_csv('../input/sample_submission.csv')
ss['scalar_coupling_constant'] = cb_model.predict(test_dataset)
ss.to_csv('basline_catboost_submission.csv', index=False)