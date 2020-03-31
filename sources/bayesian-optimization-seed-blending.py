
import numpy as np
import pandas as pd

import random
random.seed(1029)
np.random.seed(1029)

import os
import copy
import matplotlib.pyplot as plt
### matplotlib inline
from tqdm import tqdm_notebook
from sklearn.preprocessing import StandardScaler
from sklearn.svm import NuSVR, SVR
from sklearn.metrics import mean_absolute_error
pd.options.display.precision = 15
from collections import defaultdict
import lightgbm as lgb
import xgboost as xgb
import catboost as cat
import time
from collections import Counter
import datetime
from catboost import CatBoostRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold, KFold, RepeatedKFold, GroupKFold, GridSearchCV, train_test_split, TimeSeriesSplit, RepeatedStratifiedKFold
from sklearn import metrics
from sklearn.metrics import classification_report, confusion_matrix
from sklearn import linear_model
import gc
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")
from bayes_opt import BayesianOptimization
import eli5
import shap
from IPython.display import HTML
import json
import altair as alt
from category_encoders.ordinal import OrdinalEncoder
import networkx as nx
import matplotlib.pyplot as plt
### matplotlib inline
from typing import List

import os
import time
import datetime
import json
import gc
from numba import jit

from functools import partial
import scipy as sp

import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm_notebook

import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostRegressor, CatBoostClassifier
from sklearn import metrics
from typing import Any
from itertools import product
pd.set_option('max_rows', 500)
import re
from tqdm import tqdm
from joblib import Parallel, delayed
class OptimizedRounder(object):
    """
    An optimizer for rounding thresholds
    to maximize Quadratic Weighted Kappa (QWK) score
    # https://www.kaggle.com/naveenasaithambi/optimizedrounder-improved
    """
    def __init__(self):
        self.coef_ = 0

    def _kappa_loss(self, coef, X, y):
        """
        Get loss according to
        using current coefficients
        
        :param coef: A list of coefficients that will be used for rounding
        :param X: The raw predictions
        :param y: The ground truth labels
        """
        X_p = pd.cut(X, [-np.inf] + list(np.sort(coef)) + [np.inf], labels = [0, 1, 2, 3])

        return -qwk(y, X_p)

    def fit(self, X, y):
        """
        Optimize rounding thresholds
        
        :param X: The raw predictions
        :param y: The ground truth labels
        """
        loss_partial = partial(self._kappa_loss, X=X, y=y)
        initial_coef = [0.5, 1.5, 2.5]
        self.coef_ = sp.optimize.minimize(loss_partial, initial_coef, method='nelder-mead')

    def predict(self, X, coef):
        """
        Make predictions with specified thresholds
        
        :param X: The raw predictions
        :param coef: A list of coefficients that will be used for rounding
        """
        return pd.cut(X, [-np.inf] + list(np.sort(coef)) + [np.inf], labels = [0, 1, 2, 3])


    def coefficients(self):
        """
        Return the optimized coefficients
        """
        return self.coef_['x']
from sklearn.base import BaseEstimator, TransformerMixin
@jit
def qwk(a1, a2):
    """
    Source: https://www.kaggle.com/c/data-science-bowl-2019/discussion/114133#latest-660168

    :param a1:
    :param a2:
    :param max_rat:
    :return:
    """
    max_rat = 3
    a1 = np.asarray(a1, dtype=int)
    a2 = np.asarray(a2, dtype=int)

    hist1 = np.zeros((max_rat + 1, ))
    hist2 = np.zeros((max_rat + 1, ))

    o = 0
    for k in range(a1.shape[0]):
        i, j = a1[k], a2[k]
        hist1[i] += 1
        hist2[j] += 1
        o +=  (i - j) * (i - j)

    e = 0
    for i in range(max_rat + 1):
        for j in range(max_rat + 1):
            e += hist1[i] * hist2[j] * (i - j) * (i - j)

    e = e / a1.shape[0]

    return 1 - o / e


def eval_qwk_lgb(y_true, y_pred):
    """
    Fast cappa eval function for lgb.
    """

    y_pred = y_pred.reshape(len(np.unique(y_true)), -1).argmax(axis=0)
    return 'cappa', qwk(y_true, y_pred), True


def eval_qwk_lgb_regr(y_true, y_pred):
    """
    Fast cappa eval function for lgb.
    """
    y_pred[y_pred <= 1.12232214] = 0
    y_pred[np.where(np.logical_and(y_pred > 1.12232214, y_pred <= 1.73925866))] = 1
    y_pred[np.where(np.logical_and(y_pred > 1.73925866, y_pred <= 2.22506454))] = 2
    y_pred[y_pred > 2.22506454] = 3

    # y_pred = y_pred.reshape(len(np.unique(y_true)), -1).argmax(axis=0)

    return 'cappa', qwk(y_true, y_pred), True


class LGBWrapper_regr(object):
    """
    A wrapper for lightgbm model so that we will have a single api for various models.
    """

    def __init__(self):
        self.model = lgb.LGBMRegressor()

    def fit(self, X_train, y_train, X_valid=None, y_valid=None, X_holdout=None, y_holdout=None, params=None):
        if params['objective'] == 'regression':
            eval_metric = eval_qwk_lgb_regr
        else:
            eval_metric = 'auc'

        eval_set = [(X_train, y_train)]
        eval_names = ['train']
        self.model = self.model.set_params(**params)

        if X_valid is not None:
            eval_set.append((X_valid, y_valid))
            eval_names.append('valid')

        if X_holdout is not None:
            eval_set.append((X_holdout, y_holdout))
            eval_names.append('holdout')

        if 'cat_cols' in params.keys():
            cat_cols = [col for col in params['cat_cols'] if col in X_train.columns]
            if len(cat_cols) > 0:
                categorical_columns = params['cat_cols']
            else:
                categorical_columns = 'auto'
        else:
            categorical_columns = 'auto'

        self.model.fit(X=X_train, y=y_train,
                       eval_set=eval_set, eval_names=eval_names, eval_metric=eval_metric,
                       verbose=params['verbose'], early_stopping_rounds=params['early_stopping_rounds'],
                       categorical_feature=categorical_columns)

        self.best_score_ = self.model.best_score_
        self.feature_importances_ = self.model.feature_importances_

    def predict(self, X_test):
        return self.model.predict(X_test, num_iteration=self.model.best_iteration_)

    
def eval_qwk_xgb(y_pred, y_true):
    """
    Fast cappa eval function for xgb.
    """
    # print('y_true', y_true)
    # print('y_pred', y_pred)
    y_true = y_true.get_label()
    y_pred = y_pred.argmax(axis=1)
    return 'cappa', -qwk(y_true, y_pred)


class LGBWrapper(object):
    """
    A wrapper for lightgbm model so that we will have a single api for various models.
    """

    def __init__(self):
        self.model = lgb.LGBMClassifier()

    def fit(self, X_train, y_train, X_valid=None, y_valid=None, X_holdout=None, y_holdout=None, params=None):

        eval_set = [(X_train, y_train)]
        eval_names = ['train']
        self.model = self.model.set_params(**params)

        if X_valid is not None:
            eval_set.append((X_valid, y_valid))
            eval_names.append('valid')

        if X_holdout is not None:
            eval_set.append((X_holdout, y_holdout))
            eval_names.append('holdout')

        if 'cat_cols' in params.keys():
            cat_cols = [col for col in params['cat_cols'] if col in X_train.columns]
            if len(cat_cols) > 0:
                categorical_columns = params['cat_cols']
            else:
                categorical_columns = 'auto'
        else:
            categorical_columns = 'auto'

        self.model.fit(X=X_train, y=y_train,
                       eval_set=eval_set, eval_names=eval_names, eval_metric=eval_qwk_lgb,
                       verbose=params['verbose'], early_stopping_rounds=params['early_stopping_rounds'],
                       categorical_feature=categorical_columns)

        self.best_score_ = self.model.best_score_
        self.feature_importances_ = self.model.feature_importances_

    def predict_proba(self, X_test):
        if self.model.objective == 'binary':
            return self.model.predict_proba(X_test, num_iteration=self.model.best_iteration_)[:, 1]
        else:
            return self.model.predict_proba(X_test, num_iteration=self.model.best_iteration_)


class MainTransformer(BaseEstimator, TransformerMixin):

    def __init__(self, convert_cyclical: bool = False, create_interactions: bool = False, n_interactions: int = 20):
        """
        Main transformer for the data. Can be used for processing on the whole data.

        :param convert_cyclical: convert cyclical features into continuous
        :param create_interactions: create interactions between features
        """

        self.convert_cyclical = convert_cyclical
        self.create_interactions = create_interactions
        self.feats_for_interaction = None
        self.n_interactions = n_interactions

    def fit(self, X, y=None):

        if self.create_interactions:
            self.feats_for_interaction = [col for col in X.columns if 'sum' in col
                                          or 'mean' in col or 'max' in col or 'std' in col
                                          or 'attempt' in col]
            self.feats_for_interaction1 = np.random.choice(self.feats_for_interaction, self.n_interactions)
            self.feats_for_interaction2 = np.random.choice(self.feats_for_interaction, self.n_interactions)

        return self

    def transform(self, X, y=None):
        data = copy.deepcopy(X)
        if self.create_interactions:
            for col1 in self.feats_for_interaction1:
                for col2 in self.feats_for_interaction2:
                    data[f'{col1}_int_{col2}'] = data[col1] * data[col2]

        if self.convert_cyclical:
            data['timestampHour'] = np.sin(2 * np.pi * data['timestampHour'] / 23.0)
            data['timestampMonth'] = np.sin(2 * np.pi * data['timestampMonth'] / 23.0)
            data['timestampWeek'] = np.sin(2 * np.pi * data['timestampWeek'] / 23.0)
            data['timestampMinute'] = np.sin(2 * np.pi * data['timestampMinute'] / 23.0)

#         data['installation_session_count'] = data.groupby(['installation_id'])['Clip'].transform('count')
#         data['installation_duration_mean'] = data.groupby(['installation_id'])['duration_mean'].transform('mean')
#         data['installation_title_nunique'] = data.groupby(['installation_id'])['session_title'].transform('nunique')

#         data['sum_event_code_count'] = data[['2000', '3010', '3110', '4070', '4090', '4030', '4035', '4021', '4020', '4010', '2080', '2083', '2040', '2020', '2030', '3021', '3121', '2050', '3020', '3120', '2060', '2070', '4031', '4025', '5000', '5010', '2081', '2025', '4022', '2035', '4040', '4100', '2010', '4110', '4045', '4095', '4220', '2075', '4230', '4235', '4080', '4050']].sum(axis=1)

        # data['installation_event_code_count_mean'] = data.groupby(['installation_id'])['sum_event_code_count'].transform('mean')

        return data

    def fit_transform(self, X, y=None, **fit_params):
        data = copy.deepcopy(X)
        self.fit(data)
        return self.transform(data)


class FeatureTransformer(BaseEstimator, TransformerMixin):

    def __init__(self, main_cat_features: list = None, num_cols: list = None):
        """

        :param main_cat_features:
        :param num_cols:
        """
        self.main_cat_features = main_cat_features
        self.num_cols = num_cols

    def fit(self, X, y=None):

#         self.num_cols = [col for col in X.columns if 'sum' in col or 'mean' in col or 'max' in col or 'std' in col
#                          or 'attempt' in col]
        

        return self

    def transform(self, X, y=None):
        data = copy.deepcopy(X)
#         for col in self.num_cols:
#             data[f'{col}_to_mean'] = data[col] / data.groupby('installation_id')[col].transform('mean')
#             data[f'{col}_to_std'] = data[col] / data.groupby('installation_id')[col].transform('std')

        return data

    def fit_transform(self, X, y=None, **fit_params):
        data = copy.deepcopy(X)
        self.fit(data)
        return self.transform(data)
    
    
class RegressorModel(object):
    """
    A wrapper class for classification models.
    It can be used for training and prediction.
    Can plot feature importance and training progress (if relevant for model).

    """

    def __init__(self, columns: list = None, model_wrapper=None):
        """

        :param original_columns:
        :param model_wrapper:
        """
        self.columns = columns
        self.model_wrapper = model_wrapper
        self.result_dict = {}
        self.train_one_fold = False
        self.preprocesser = None

    def fit(self, X: pd.DataFrame, y,
            X_holdout: pd.DataFrame = None, y_holdout=None,
            folds=None,
            params: dict = None,
            eval_metric='rmse',
            cols_to_drop: list = None,
            preprocesser=None,
            transformers: dict = None,
            adversarial: bool = False,
            plot: bool = True):
        """
        Training the model.

        :param X: training data
        :param y: training target
        :param X_holdout: holdout data
        :param y_holdout: holdout target
        :param folds: folds to split the data. If not defined, then model will be trained on the whole X
        :param params: training parameters
        :param eval_metric: metric for validataion
        :param cols_to_drop: list of columns to drop (for example ID)
        :param preprocesser: preprocesser class
        :param transformers: transformer to use on folds
        :param adversarial
        :return:
        """

        if folds is None:
            folds = KFold(n_splits=3, random_state=42)
            self.train_one_fold = True

        self.columns = X.columns if self.columns is None else self.columns
        self.feature_importances = pd.DataFrame(columns=['feature', 'importance'])
        self.trained_transformers = {k: [] for k in transformers}
        self.transformers = transformers
        self.models = []
        self.folds_dict = {}
        self.eval_metric = eval_metric
        n_target = 1
        self.oof = np.zeros((len(X), n_target))
        self.n_target = n_target

        X = X[self.columns]
        if X_holdout is not None:
            X_holdout = X_holdout[self.columns]

        if preprocesser is not None:
            self.preprocesser = preprocesser
            self.preprocesser.fit(X, y)
            X = self.preprocesser.transform(X, y)
            self.columns = X.columns.tolist()
            if X_holdout is not None:
                X_holdout = self.preprocesser.transform(X_holdout)

        for fold_n, (train_index, valid_index) in enumerate(folds.split(X, y, X['installation_id'])):

            if X_holdout is not None:
                X_hold = X_holdout.copy()
            else:
                X_hold = None
            self.folds_dict[fold_n] = {}
            if params['verbose']:
                print(f'Fold {fold_n + 1} started at {time.ctime()}')
            self.folds_dict[fold_n] = {}

            X_train, X_valid = X.iloc[train_index], X.iloc[valid_index]
            y_train, y_valid = y.iloc[train_index], y.iloc[valid_index]
            if self.train_one_fold:
                X_train = X[self.original_columns]
                y_train = y
                X_valid = None
                y_valid = None

            datasets = {'X_train': X_train, 'X_valid': X_valid, 'X_holdout': X_hold, 'y_train': y_train}
            X_train, X_valid, X_hold = self.transform_(datasets, cols_to_drop)

            self.folds_dict[fold_n]['columns'] = X_train.columns.tolist()

            model = copy.deepcopy(self.model_wrapper)

            if adversarial:
                X_new1 = X_train.copy()
                if X_valid is not None:
                    X_new2 = X_valid.copy()
                elif X_holdout is not None:
                    X_new2 = X_holdout.copy()
                X_new = pd.concat([X_new1, X_new2], axis=0)
                y_new = np.hstack((np.zeros((X_new1.shape[0])), np.ones((X_new2.shape[0]))))
                X_train, X_valid, y_train, y_valid = train_test_split(X_new, y_new)

            model.fit(X_train, y_train, X_valid, y_valid, X_hold, y_holdout, params=params)

            self.folds_dict[fold_n]['scores'] = model.best_score_
            if self.oof.shape[0] != len(X):
                self.oof = np.zeros((X.shape[0], self.oof.shape[1]))
            if not adversarial:
                self.oof[valid_index] = model.predict(X_valid).reshape(-1, n_target)

            fold_importance = pd.DataFrame(list(zip(X_train.columns, model.feature_importances_)),
                                           columns=['feature', 'importance'])
            self.feature_importances = self.feature_importances.append(fold_importance)
            self.models.append(model)

        self.feature_importances['importance'] = self.feature_importances['importance'].astype(int)

        # if params['verbose']:
        self.calc_scores_()

        if plot:
            # print(classification_report(y, self.oof.argmax(1)))
            fig, ax = plt.subplots(figsize=(16, 12))
            plt.subplot(2, 2, 1)
            self.plot_feature_importance(top_n=20)
            plt.subplot(2, 2, 2)
            self.plot_metric()
            plt.subplot(2, 2, 3)
            plt.hist(y.values.reshape(-1, 1) - self.oof)
            plt.title('Distribution of errors')
            plt.subplot(2, 2, 4)
            plt.hist(self.oof)
            plt.title('Distribution of oof predictions');

    def transform_(self, datasets, cols_to_drop):
        for name, transformer in self.transformers.items():
            transformer.fit(datasets['X_train'], datasets['y_train'])
            datasets['X_train'] = transformer.transform(datasets['X_train'])
            if datasets['X_valid'] is not None:
                datasets['X_valid'] = transformer.transform(datasets['X_valid'])
            if datasets['X_holdout'] is not None:
                datasets['X_holdout'] = transformer.transform(datasets['X_holdout'])
            self.trained_transformers[name].append(transformer)
        if cols_to_drop is not None:
            cols_to_drop = [col for col in cols_to_drop if col in datasets['X_train'].columns]

            datasets['X_train'] = datasets['X_train'].drop(cols_to_drop, axis=1)
            if datasets['X_valid'] is not None:
                datasets['X_valid'] = datasets['X_valid'].drop(cols_to_drop, axis=1)
            if datasets['X_holdout'] is not None:
                datasets['X_holdout'] = datasets['X_holdout'].drop(cols_to_drop, axis=1)
        self.cols_to_drop = cols_to_drop

        return datasets['X_train'], datasets['X_valid'], datasets['X_holdout']

    def calc_scores_(self):
#         print()
        datasets = [k for k, v in [v['scores'] for k, v in self.folds_dict.items()][0].items() if len(v) > 0]
        self.scores = {}
        for d in datasets:
            scores = [v['scores'][d][self.eval_metric] for k, v in self.folds_dict.items()]
#             print(f"CV mean score on {d}: {np.mean(scores):.4f} +/- {np.std(scores):.4f} std.")
            self.scores[d] = np.mean(scores)

    def predict(self, X_test, averaging: str = 'usual'):
        """
        Make prediction

        :param X_test:
        :param averaging: method of averaging
        :return:
        """
        full_prediction = np.zeros((X_test.shape[0], self.oof.shape[1]))
        if self.preprocesser is not None:
            X_test = self.preprocesser.transform(X_test)
        for i in range(len(self.models)):
            X_t = X_test.copy()
            for name, transformers in self.trained_transformers.items():
                X_t = transformers[i].transform(X_t)

            if self.cols_to_drop is not None:
                cols_to_drop = [col for col in self.cols_to_drop if col in X_t.columns]
                X_t = X_t.drop(cols_to_drop, axis=1)
            y_pred = self.models[i].predict(X_t[self.folds_dict[i]['columns']]).reshape(-1, full_prediction.shape[1])

            # if case transformation changes the number of the rows
            if full_prediction.shape[0] != len(y_pred):
                full_prediction = np.zeros((y_pred.shape[0], self.oof.shape[1]))

            if averaging == 'usual':
                full_prediction += y_pred
            elif averaging == 'rank':
                full_prediction += pd.Series(y_pred).rank().values

        return full_prediction / len(self.models)

    def plot_feature_importance(self, drop_null_importance: bool = True, top_n: int = 10):
        """
        Plot default feature importance.

        :param drop_null_importance: drop columns with null feature importance
        :param top_n: show top n columns
        :return:
        """

        top_feats = self.get_top_features(drop_null_importance, top_n)
        feature_importances = self.feature_importances.loc[self.feature_importances['feature'].isin(top_feats)]
        feature_importances['feature'] = feature_importances['feature'].astype(str)
        top_feats = [str(i) for i in top_feats]
        sns.barplot(data=feature_importances, x='importance', y='feature', orient='h', order=top_feats)
        plt.title('Feature importances')

    def get_top_features(self, drop_null_importance: bool = True, top_n: int = 10):
        """
        Get top features by importance.

        :param drop_null_importance:
        :param top_n:
        :return:
        """
        grouped_feats = self.feature_importances.groupby(['feature'])['importance'].mean()
        if drop_null_importance:
            grouped_feats = grouped_feats[grouped_feats != 0]
        return list(grouped_feats.sort_values(ascending=False).index)[:top_n]

    def plot_metric(self):
        """
        Plot training progress.
        Inspired by `plot_metric` from https://lightgbm.readthedocs.io/en/latest/_modules/lightgbm/plotting.html

        :return:
        """
        full_evals_results = pd.DataFrame()
        for model in self.models:
            evals_result = pd.DataFrame()
            for k in model.model.evals_result_.keys():
                evals_result[k] = model.model.evals_result_[k][self.eval_metric]
            evals_result = evals_result.reset_index().rename(columns={'index': 'iteration'})
            full_evals_results = full_evals_results.append(evals_result)

        full_evals_results = full_evals_results.melt(id_vars=['iteration']).rename(columns={'value': self.eval_metric,
                                                                                            'variable': 'dataset'})
        sns.lineplot(data=full_evals_results, x='iteration', y=self.eval_metric, hue='dataset')
        plt.title('Training progress')
def read_data():
    print('Reading train.csv file....')
    train = pd.read_csv('/kaggle/input/data-science-bowl-2019/train.csv')
    print('Training.csv file have {} rows and {} columns'.format(train.shape[0], train.shape[1]))

    print('Reading test.csv file....')
    test = pd.read_csv('/kaggle/input/data-science-bowl-2019/test.csv')
    print('Test.csv file have {} rows and {} columns'.format(test.shape[0], test.shape[1]))

    print('Reading train_labels.csv file....')
    train_labels = pd.read_csv('/kaggle/input/data-science-bowl-2019/train_labels.csv')
    print('Train_labels.csv file have {} rows and {} columns'.format(train_labels.shape[0], train_labels.shape[1]))

    print('Reading specs.csv file....')
    specs = pd.read_csv('/kaggle/input/data-science-bowl-2019/specs.csv')
    print('Specs.csv file have {} rows and {} columns'.format(specs.shape[0], specs.shape[1]))

    print('Reading sample_submission.csv file....')
    sample_submission = pd.read_csv('/kaggle/input/data-science-bowl-2019/sample_submission.csv')
    print('Sample_submission.csv file have {} rows and {} columns'.format(sample_submission.shape[0], sample_submission.shape[1]))
    return train, test, train_labels, specs, sample_submission

def encode_title(train, test, train_labels):
    # encode title
    train['title_event_code'] = list(map(lambda x, y: str(x) + '_' + str(y), train['title'], train['event_code']))
    test['title_event_code'] = list(map(lambda x, y: str(x) + '_' + str(y), test['title'], test['event_code']))
    all_title_event_code = list(set(train["title_event_code"].unique()).union(test["title_event_code"].unique()))
    # make a list with all the unique 'titles' from the train and test set
    list_of_user_activities = list(set(train['title'].unique()).union(set(test['title'].unique())))
    # make a list with all the unique 'event_code' from the train and test set
    list_of_event_code = list(set(train['event_code'].unique()).union(set(test['event_code'].unique())))
    list_of_event_id = list(set(train['event_id'].unique()).union(set(test['event_id'].unique())))
    # make a list with all the unique worlds from the train and test set
    list_of_worlds = list(set(train['world'].unique()).union(set(test['world'].unique())))
    # create a dictionary numerating the titles
    activities_map = dict(zip(list_of_user_activities, np.arange(len(list_of_user_activities))))
    activities_labels = dict(zip(np.arange(len(list_of_user_activities)), list_of_user_activities))
    activities_world = dict(zip(list_of_worlds, np.arange(len(list_of_worlds))))
    assess_titles = list(set(train[train['type'] == 'Assessment']['title'].value_counts().index).union(set(test[test['type'] == 'Assessment']['title'].value_counts().index)))
    # replace the text titles with the number titles from the dict
    train['title'] = train['title'].map(activities_map)
    test['title'] = test['title'].map(activities_map)
    train['world'] = train['world'].map(activities_world)
    test['world'] = test['world'].map(activities_world)
    train_labels['title'] = train_labels['title'].map(activities_map)
    win_code = dict(zip(activities_map.values(), (4100*np.ones(len(activities_map))).astype('int')))
    # then, it set one element, the 'Bird Measurer (Assessment)' as 4110, 10 more than the rest
    win_code[activities_map['Bird Measurer (Assessment)']] = 4110
    # convert text into datetime
    train['timestamp'] = pd.to_datetime(train['timestamp'])
    test['timestamp'] = pd.to_datetime(test['timestamp'])
    
    
    return train, test, train_labels, win_code, list_of_user_activities, list_of_event_code, activities_labels, assess_titles, list_of_event_id, all_title_event_code

def get_data(user_sample, test_set=False):
    '''
    The user_sample is a DataFrame from train or test where the only one 
    installation_id is filtered
    And the test_set parameter is related with the labels processing, that is only requered
    if test_set=False
    '''
    # Constants and parameters declaration
    last_activity = 0
    
    user_activities_count = {'Clip':0, 'Activity': 0, 'Assessment': 0, 'Game':0}
    
    # new features: time spent in each activity
    last_session_time_sec = 0
    accuracy_groups = {0:0, 1:0, 2:0, 3:0}
    all_assessments = []
    accumulated_accuracy_group = 0
    accumulated_accuracy = 0
    accumulated_correct_attempts = 0 
    accumulated_uncorrect_attempts = 0
    accumulated_actions = 0
    counter = 0
    time_first_activity = float(user_sample['timestamp'].values[0])
    durations = []
    last_accuracy_title = {'acc_' + title: -1 for title in assess_titles}
    event_code_count: Dict[str, int] = {ev: 0 for ev in list_of_event_code}
    event_id_count: Dict[str, int] = {eve: 0 for eve in list_of_event_id}
    title_count: Dict[str, int] = {eve: 0 for eve in activities_labels.values()} 
    title_event_code_count: Dict[str, int] = {t_eve: 0 for t_eve in all_title_event_code}
    
    # itarates through each session of one instalation_id
    for i, session in user_sample.groupby('game_session', sort=False):
        # i = game_session_id
        # session is a DataFrame that contain only one game_session
        
        # get some sessions information
        session_type = session['type'].iloc[0]
        session_title = session['title'].iloc[0]
        session_title_text = activities_labels[session_title]
                    
            
        # for each assessment, and only this kind off session, the features below are processed
        # and a register are generated
        if (session_type == 'Assessment') & (test_set or len(session)>1):
            # search for event_code 4100, that represents the assessments trial
            all_attempts = session.query(f'event_code == {win_code[session_title]}')
            # then, check the numbers of wins and the number of losses
            true_attempts = all_attempts['event_data'].str.contains('true').sum()
            false_attempts = all_attempts['event_data'].str.contains('false').sum()
            # copy a dict to use as feature template, it's initialized with some itens: 
            # {'Clip':0, 'Activity': 0, 'Assessment': 0, 'Game':0}
            features = user_activities_count.copy()
            features.update(last_accuracy_title.copy())
            features.update(event_code_count.copy())
            features.update(event_id_count.copy())
            features.update(title_count.copy())
            features.update(title_event_code_count.copy())
            features.update(last_accuracy_title.copy())
            
            # get installation_id for aggregated features
            features['installation_id'] = session['installation_id'].iloc[-1]
            # add title as feature, remembering that title represents the name of the game
            features['session_title'] = session['title'].iloc[0]
            # the 4 lines below add the feature of the history of the trials of this player
            # this is based on the all time attempts so far, at the moment of this assessment
            features['accumulated_correct_attempts'] = accumulated_correct_attempts
            features['accumulated_uncorrect_attempts'] = accumulated_uncorrect_attempts
            accumulated_correct_attempts += true_attempts 
            accumulated_uncorrect_attempts += false_attempts
            # the time spent in the app so far
            if durations == []:
                features['duration_mean'] = 0
            else:
                features['duration_mean'] = np.mean(durations)
            durations.append((session.iloc[-1, 2] - session.iloc[0, 2] ).seconds)
            # the accurace is the all time wins divided by the all time attempts
            features['accumulated_accuracy'] = accumulated_accuracy/counter if counter > 0 else 0
            accuracy = true_attempts/(true_attempts+false_attempts) if (true_attempts+false_attempts) != 0 else 0
            accumulated_accuracy += accuracy
            last_accuracy_title['acc_' + session_title_text] = accuracy
            # a feature of the current accuracy categorized
            # it is a counter of how many times this player was in each accuracy group
            if accuracy == 0:
                features['accuracy_group'] = 0
            elif accuracy == 1:
                features['accuracy_group'] = 3
            elif accuracy == 0.5:
                features['accuracy_group'] = 2
            else:
                features['accuracy_group'] = 1
            features.update(accuracy_groups)
            accuracy_groups[features['accuracy_group']] += 1
            # mean of the all accuracy groups of this player
            features['accumulated_accuracy_group'] = accumulated_accuracy_group/counter if counter > 0 else 0
            accumulated_accuracy_group += features['accuracy_group']
            # how many actions the player has done so far, it is initialized as 0 and updated some lines below
            features['accumulated_actions'] = accumulated_actions
            
            # there are some conditions to allow this features to be inserted in the datasets
            # if it's a test set, all sessions belong to the final dataset
            # it it's a train, needs to be passed throught this clausule: session.query(f'event_code == {win_code[session_title]}')
            # that means, must exist an event_code 4100 or 4110
            if test_set:
                all_assessments.append(features)
            elif true_attempts+false_attempts > 0:
                all_assessments.append(features)
                
            counter += 1
        
        # this piece counts how many actions was made in each event_code so far
        def update_counters(counter: dict, col: str):
                num_of_session_count = Counter(session[col])
                for k in num_of_session_count.keys():
                    x = k
                    if col == 'title':
                        x = activities_labels[k]
                    counter[x] += num_of_session_count[k]
                return counter
            
        event_code_count = update_counters(event_code_count, "event_code")
        event_id_count = update_counters(event_id_count, "event_id")
        title_count = update_counters(title_count, 'title')
        title_event_code_count = update_counters(title_event_code_count, 'title_event_code')

        # counts how many actions the player has done so far, used in the feature of the same name
        accumulated_actions += len(session)
        if last_activity != session_type:
            user_activities_count[session_type] += 1
            last_activitiy = session_type 
                        
    # if it't the test_set, only the last assessment must be predicted, the previous are scraped
    if test_set:
        return all_assessments[-1]
    # in the train_set, all assessments goes to the dataset
    return all_assessments

def get_train_and_test(train, test):
    compiled_train = []
    compiled_test = []
    for i, (ins_id, user_sample) in tqdm(enumerate(train.groupby('installation_id', sort = False)), total = 17000):
        compiled_train += get_data(user_sample)
    for ins_id, user_sample in tqdm(test.groupby('installation_id', sort = False), total = 1000):
        test_data = get_data(user_sample, test_set = True)
        compiled_test.append(test_data)
    reduce_train = pd.DataFrame(compiled_train)
    reduce_test = pd.DataFrame(compiled_test)
    categoricals = ['session_title']
    return reduce_train, reduce_test, categoricals

# read data
train, test, train_labels, specs, sample_submission = read_data()
# get usefull dict with maping encode
train, test, train_labels, win_code, list_of_user_activities, list_of_event_code, activities_labels, assess_titles, list_of_event_id, all_title_event_code = encode_title(train, test, train_labels)
# tranform function to get the train and test set
reduce_train, reduce_test, categoricals = get_train_and_test(train, test)
def preprocess(reduce_train, reduce_test):
    for df in [reduce_train, reduce_test]:
        df['installation_session_count'] = df.groupby(['installation_id'])['Clip'].transform('count')
        df['installation_duration_mean'] = df.groupby(['installation_id'])['duration_mean'].transform('mean')
        #df['installation_duration_std'] = df.groupby(['installation_id'])['duration_mean'].transform('std')
        df['installation_title_nunique'] = df.groupby(['installation_id'])['session_title'].transform('nunique')
        
        df['sum_event_code_count'] = df[[2050, 4100, 4230, 5000, 4235, 2060, 4110, 5010, 2070, 2075, 2080, 2081, 2083, 3110, 4010, 3120, 3121, 4020, 4021, 
                                        4022, 4025, 4030, 4031, 3010, 4035, 4040, 3020, 3021, 4045, 2000, 4050, 2010, 2020, 4070, 2025, 2030, 4080, 2035, 
                                        2040, 4090, 4220, 4095]].sum(axis = 1)
        
        df['installation_event_code_count_mean'] = df.groupby(['installation_id'])['sum_event_code_count'].transform('mean')
        #df['installation_event_code_count_std'] = df.groupby(['installation_id'])['sum_event_code_count'].transform('std')
        
    features = reduce_train.loc[(reduce_train.sum(axis=1) != 0), (reduce_train.sum(axis=0) != 0)].columns # delete useless columns
    features = [x for x in features if x not in ['accuracy_group', 'installation_id']] + ['acc_' + title for title in assess_titles]
   
    return reduce_train, reduce_test, features

# call feature engineering function
reduce_train, reduce_test, features = preprocess(reduce_train, reduce_test)
all_cols = ['Clip', 'Activity', 'Assessment', 'Game', 'acc_Bird Measurer (Assessment)', 'acc_Mushroom Sorter (Assessment)', 'acc_Chest Sorter (Assessment)', 'acc_Cauldron Filler (Assessment)', 'acc_Cart Balancer (Assessment)', 2050, 4100, 4230, 5000, 4235, 2060, 4110, 5010, 2070, 2075, 2080, 2081, 2083, 3110, 4010, 3120, 3121, 4020, 4021, 4022, 4025, 4030, 4031, 3010, 4035, 4040, 3020, 3021, 4045, 2000, 4050, 2010, 2020, 4070, 2025, 2030, 4080, 2035, 2040, 4090, 4220, 4095, '29f54413', 'c54cf6c5', '0086365d', 'd122731b', '5154fc30', '8d7e386c', '4074bac2', 'e5734469', 'b7530680', 'd3640339', 'd06f75b5', 'f3cd5473', 'a7640a16', 'bd612267', '3d8c61b0', '6cf7d25c', '1af8be29', '7372e1a5', 'df4940d3', 'bb3e370b', 'a16a373e', '4a4c3d21', 'b7dc8128', '25fa8af4', '1cc7cfca', '56817e2b', 'e64e2cfd', '27253bdc', '5e812b27', '65abac75', '3dcdda7f', 'f56e0afc', 'acf5c23f', 'e57dd7af', '38074c54', '56cd3b43', '070a5291', 'b2dba42b', '88d4a5be', '222660ff', '9ee1c98c', '31973d56', 'ca11f653', 'b120f2ac', 'cc5087a3', 'ea296733', '363c86c9', '8ac7cce4', '1575e76c', '565a3990', 'a2df0760', 'a1bbe385', '262136f4', '3ddc79c3', 'a8a78786', '7d093bf9', '2dc29e21', '30614231', '4d911100', '85de926c', '6c930e6e', 'de26c3a6', 'a0faea5d', '5d042115', 'bcceccc6', '28a4eb9a', '85d1b0de', 'e5c9df6f', '76babcde', 'e4d32835', '69fdac0a', 'abc5811c', 'df4fe8b6', '3323d7e9', '250513af', '9b4001e4', '77261ab5', '3bb91dda', '5be391b5', '56bcd38d', '29a42aea', '9ed8f6da', 'a52b92d5', 'bbfe0445', '736f9581', '30df3273', '86c924c4', 'b74258a0', '4a09ace1', 'ab3136ba', '1beb320a', '598f4598', '461eace6', '5e3ea25a', '51102b85', '6aeafed4', 'f71c4741', '6f4bd64e', 'eb2c19cd', '587b5989', 'c952eb01', '73757a5e', '6f4adc4b', '828e68f9', '895865f3', '86ba578b', 'beb0a7b9', '9c5ef70c', '3babcb9b', 'c58186bf', '0d1da71f', '93edfe2e', 'a1e4395d', '3bfd1a65', '7dfe6d8a', 'ad2fc29c', 'c1cac9a2', '5348fd84', '47026d5f', '532a2afb', '74e5f8a7', '37c53127', '5dc079d8', 'b2e5b0f1', '907a054b', '160654fd', '763fc34e', '99ea62f3', '36fa3ebe', '832735e1', '71fe8f75', '55115cbd', 'a76029ee', 'd2e9262e', '46b50ba8', '562cec5f', '71e712d8', '9b23e8ee', '6077cc36', '9de5e594', '45d01abe', '4e5fc6f5', 'f50fc6c1', '3ee399c3', 'dcaede90', '44cb4907', '90d848e0', '1f19558b', '67439901', '67aa2ada', '756e5507', 'ab4ec3a4', 'e79f3763', 'c7fe2a55', 'cb6010f8', 'ad148f58', '8fee50e2', '499edb7c', '17ca3959', '7423acbc', '16dffff1', '9ce586dd', '363d3849', '9e6b7fb5', '5e109ec3', 'd45ed6a1', 'e694a35b', '3d0b9317', '6bf9e3e1', 'fcfdffb6', '04df9b66', '003cd2ee', '0413e89d', '5f5b2617', '13f56524', '1bb5fbdb', '5290eab1', '37ee8496', '1340b8d7', '84b0e0c8', '3d63345e', 'b80e5e84', '7961e599', '5c2f29ca', '51311d7a', '5a848010', '4c2ec19f', '6088b756', '857f21c0', 'b1d5101d', 'a29c5338', 'd88ca108', '28f975ea', '5de79a6a', '6d90d394', '2dcad279', '2fb91ec1', '792530f8', 'b5053438', '8af75982', 'd88e8f25', '65a38bf7', '84538528', 'e7e44842', '8d748b58', '16667cc5', '9e4c8c7b', '923afab1', '3edf6747', '3bf1cf26', '7da34a02', '0db6d71d', '6f8106d9', '3393b68b', '46cd75b4', '1325467d', '7d5c30a2', '9d4e7b25', '7ad3efc6', 'f28c589a', '1375ccb7', 'f806dc10', '01ca3a3c', '3b2048ee', '2b058fe3', '1996c610', '28ed704e', 'f7e47413', '884228c8', '3bb91ced', 'd185d3ea', '2c4e6db0', 'ec138c1c', 'a5e9da97', '33505eae', '1b54d27f', '6c517a88', '4b5efe37', '795e4a37', '26fd2d99', '77ead60d', 'd9c005dd', 'a8cc6fec', 'cdd22e43', 'c51d8688', 'a8876db3', '2230fab4', 'e7561dd2', '4bb2f698', 'dcb1663e', '15ba1109', 'c7128948', '4ef8cdd3', '26a5a3dd', '8d84fa81', 'c0415e5c', '731c0cbe', '6f445b57', 'ac92046e', '3dfd4aa4', '2b9272f4', '4901243f', '7525289a', '804ee27f', '0a08139c', '63f13dd7', 'a6d66e51', '7ec0c298', '5f0eb72c', '9d29771f', '8f094001', 'd2278a3b', '8b757ab8', 'd3268efa', '15f99afc', 'e3ff61fb', '91561152', 'ea321fb1', '3a4be871', 'ecc36b7f', '89aace00', 'b88f38da', '3afb49e6', '47f43a44', '9e34ea74', '90efca10', 'dcb55a27', 'a44b10dc', 'e720d930', 'd02b7a8e', '47efca07', '2ec694de', '1c178d24', 'e4f1efe6', '392e14df', 'f6947f54', 'e04fb33d', 'e9c52111', 'd51b1749', '7f0836bf', 'e37a2b78', '5859dfb6', 'b012cd7f', 'cfbd47c8', '0d18d96c', 'fd20ea40', '9b01374f', 'd3f1e122', '15eb4a7d', '119b5b02', '3ccd3f02', 'bdf49a58', '49ed92e9', '7fd1ac25', 'f54238ee', '611485c5', '58a0de5c', 'c2baf0bd', '7040c096', '7ab78247', 'ecc6157f', 'c7f7f0e1', 'a8efe47b', '5c3d2b2f', 'daac11b0', '19967db1', 'd38c2fd7', '4d6737eb', '90ea0bac', 'e080a381', '05ad839b', 'f32856e4', '29bdd9ba', 'f93fc684', '37937459', '709b1251', 'cb1178ad', 'a1192f43', '14de4c5d', '15a43e5b', '022b4259', '6043a2b4', '9554a50b', '53c6e11a', '37db1c2f', '5b49460a', '0330ab6a', '7cf1bc53', 'bd701df8', '87d743c1', '3afde5dd', 'db02c830', '77c76bc5', '92687c59', 'cf82af56', '2a512369', '0ce40006', 'bc8f2793', 'cf7638f3', 'c189aaf2', '1cf54632', '08fd73f3', 'bfc77bd6', '06372577', 'c277e121', 'a5be6304', 'a592d54e', '2a444e03', '48349b14', 'fbaf3456', 'c74f40cd', '02a42007', 'ecaab346', '28520915', 'f5b8c21a', '93b353f2', '99abe2bb', '00c73085', 'c6971acf', 'd2659ab4', '17113b36', '155f62a4', '08ff79ad', '83c6c409', 'Dino Drink', 'Lifting Heavy Things', 'Magma Peak - Level 1', 'Balancing Act', "Pirate's Tale", 'Bird Measurer (Assessment)', 'Treasure Map', 'Happy Camel', 'Costume Box', 'Tree Top City - Level 3', 'Sandcastle Builder (Activity)', 'Cart Balancer (Assessment)', 'Ordering Spheres', 'Crystals Rule', 'Leaf Leader', 'All Star Sorting', 'Chest Sorter (Assessment)', 'Bug Measurer (Activity)', 'Fireworks (Activity)', 'Watering Hole (Activity)', 'Pan Balance', 'Flower Waterer (Activity)', 'Mushroom Sorter (Assessment)', 'Cauldron Filler (Assessment)', 'Bottle Filler (Activity)', 'Scrub-A-Dub', 'Chicken Balancer (Activity)', 'Rulers', 'Honey Cake', 'Heavy, Heavier, Heaviest', 'Air Show', 'Crystal Caves - Level 2', 'Tree Top City - Level 2', 'Dino Dive', 'Welcome to Lost Lagoon!', 'Tree Top City - Level 1', 'Bubble Bath', 'Crystal Caves - Level 1', 'Slop Problem', '12 Monkeys', 'Crystal Caves - Level 3', 'Egg Dropper (Activity)', 'Chow Time', 'Magma Peak - Level 2', 'Costume Box_2000', 'Balancing Act_2000', 'Bubble Bath_4095', 'Air Show_3020', 'Mushroom Sorter (Assessment)_2000', 'Scrub-A-Dub_4090', 'Pan Balance_3110', 'Cauldron Filler (Assessment)_2010', 'Chest Sorter (Assessment)_4020', 'Cauldron Filler (Assessment)_4070', 'Cart Balancer (Assessment)_2030', 'Happy Camel_3021', 'Air Show_3010', 'Dino Dive_3021', 'Leaf Leader_2030', 'Sandcastle Builder (Activity)_3010', 'Chest Sorter (Assessment)_4025', 'Happy Camel_3121', 'Cauldron Filler (Assessment)_3020', 'Magma Peak - Level 1_2000', 'Scrub-A-Dub_2040', 'Crystals Rule_2020', 'Dino Drink_4070', 'Air Show_4090', "Pirate's Tale_2000", 'Pan Balance_4080', 'All Star Sorting_4090', 'Watering Hole (Activity)_3110', 'Bottle Filler (Activity)_4030', 'Cart Balancer (Assessment)_3020', 'Dino Drink_4020', 'Bird Measurer (Assessment)_4035', 'Dino Dive_4080', 'Magma Peak - Level 2_2000', 'Bubble Bath_2020', 'Bug Measurer (Activity)_2000', 'Bottle Filler (Activity)_4090', 'Pan Balance_3021', 'Cart Balancer (Assessment)_4035', 'Chest Sorter (Assessment)_3110', 'Sandcastle Builder (Activity)_3110', 'Dino Dive_3020', 'Chest Sorter (Assessment)_2020', 'Air Show_3021', 'Scrub-A-Dub_3020', 'Cauldron Filler (Assessment)_3010', 'Crystal Caves - Level 3_2000', 'Bird Measurer (Assessment)_4090', 'Mushroom Sorter (Assessment)_4025', 'Happy Camel_2080', 'Sandcastle Builder (Activity)_4035', 'Watering Hole (Activity)_4090', 'Bubble Bath_4045', 'Bubble Bath_4010', 'Dino Dive_4010', 'Happy Camel_4010', 'Happy Camel_3010', 'Cauldron Filler (Assessment)_4040', 'Mushroom Sorter (Assessment)_4070', 'Chow Time_3020', 'Dino Drink_3110', 'Mushroom Sorter (Assessment)_4035', 'Dino Drink_3120', 'Mushroom Sorter (Assessment)_2025', 'Bubble Bath_3120', 'Dino Dive_4020', 'Leaf Leader_4070', 'Bottle Filler (Activity)_3110', 'Dino Dive_2030', 'Dino Drink_4030', 'Scrub-A-Dub_2081', 'Happy Camel_2020', 'Leaf Leader_4090', 'Fireworks (Activity)_4080', 'Fireworks (Activity)_4090', 'Pan Balance_2010', 'Sandcastle Builder (Activity)_4090', 'Crystals Rule_4020', 'Leaf Leader_3121', 'Welcome to Lost Lagoon!_2000', 'Happy Camel_4020', 'Cauldron Filler (Assessment)_4090', 'Crystals Rule_4010', 'Sandcastle Builder (Activity)_4030', 'Sandcastle Builder (Activity)_4021', 'Scrub-A-Dub_3120', 'Leaf Leader_2075', 'Scrub-A-Dub_4070', 'Watering Hole (Activity)_5010', 'Bottle Filler (Activity)_3010', 'Pan Balance_4025', 'Flower Waterer (Activity)_2000', 'Happy Camel_4095', 'Dino Drink_4080', 'Mushroom Sorter (Assessment)_3021', 'Cart Balancer (Assessment)_2010', 'Dino Dive_4090', 'Bubble Bath_4235', 'Watering Hole (Activity)_4021', 'Fireworks (Activity)_4020', 'Crystals Rule_3010', 'Chicken Balancer (Activity)_4022', 'Watering Hole (Activity)_4020', 'Bubble Bath_3021', 'Sandcastle Builder (Activity)_2000', 'Bird Measurer (Assessment)_3121', 'Pan Balance_2020', 'Flower Waterer (Activity)_4090', 'Cauldron Filler (Assessment)_4035', '12 Monkeys_2000', 'Scrub-A-Dub_2083', 'All Star Sorting_3020', 'Scrub-A-Dub_3121', 'Egg Dropper (Activity)_4090', 'Bubble Bath_2000', 'All Star Sorting_4095', 'Scrub-A-Dub_4080', 'Dino Dive_3010', 'Crystals Rule_3021', 'Bird Measurer (Assessment)_4100', 'Cauldron Filler (Assessment)_2000', 'Chow Time_4010', 'Sandcastle Builder (Activity)_4020', 'Egg Dropper (Activity)_4080', 'Cauldron Filler (Assessment)_2020', 'Leaf Leader_3020', 'Ordering Spheres_2000', 'Happy Camel_2083', 'Crystals Rule_3110', 'Fireworks (Activity)_3010', 'Cart Balancer (Assessment)_4100', 'Chest Sorter (Assessment)_3120', 'Cart Balancer (Assessment)_2000', 'Leaf Leader_4095', 'All Star Sorting_3120', 'Air Show_2000', 'Cart Balancer (Assessment)_3120', 'Chest Sorter (Assessment)_2030', 'Mushroom Sorter (Assessment)_2030', 'Happy Camel_4035', 'Happy Camel_2030', 'Chicken Balancer (Activity)_4020', 'Dino Dive_4070', 'Tree Top City - Level 1_2000', 'Leaf Leader_4010', 'All Star Sorting_2025', 'Bottle Filler (Activity)_2010', 'Treasure Map_2000', 'Cart Balancer (Assessment)_3110', 'Air Show_4080', 'Chest Sorter (Assessment)_3020', 'All Star Sorting_3010', 'Pan Balance_4020', 'Watering Hole (Activity)_5000', 'Mushroom Sorter (Assessment)_4020', 'Cart Balancer (Assessment)_2020', 'Tree Top City - Level 2_2000', 'Bottle Filler (Activity)_2030', 'Cauldron Filler (Assessment)_4020', 'Air Show_4110', 'Crystals Rule_4090', 'Dino Dive_3110', 'Dino Drink_4090', 'Scrub-A-Dub_2000', 'Egg Dropper (Activity)_2000', 'Crystals Rule_2010', 'Pan Balance_4035', 'All Star Sorting_4030', 'Happy Camel_3120', 'Happy Camel_4090', 'Chest Sorter (Assessment)_4090', 'Flower Waterer (Activity)_3110', 'Chicken Balancer (Activity)_4090', 'Bubble Bath_4220', 'Chow Time_4035', 'Pan Balance_4070', 'All Star Sorting_2020', 'Mushroom Sorter (Assessment)_3010', 'Mushroom Sorter (Assessment)_4100', 'Dino Drink_3121', 'Dino Drink_2060', 'Leaf Leader_4020', 'All Star Sorting_2030', 'Happy Camel_4080', 'Dino Dive_2060', 'Cart Balancer (Assessment)_4070', 'Bubble Bath_3110', 'Chow Time_4095', 'Leaf Leader_2020', 'Bird Measurer (Assessment)_3020', 'Dino Drink_3020', 'Happy Camel_4040', 'Crystals Rule_4070', 'Chicken Balancer (Activity)_3110', 'Leaf Leader_2000', 'Bubble Bath_3020', 'Leaf Leader_2070', 'Bird Measurer (Assessment)_2020', 'Egg Dropper (Activity)_3110', 'Air Show_3120', 'Crystal Caves - Level 1_2000', 'Chow Time_3010', 'Flower Waterer (Activity)_4070', 'Flower Waterer (Activity)_3010', 'Watering Hole (Activity)_2010', 'Leaf Leader_3021', 'Bug Measurer (Activity)_3110', 'Sandcastle Builder (Activity)_4080', 'Dino Drink_2020', 'Air Show_2075', 'Crystals Rule_3120', 'Air Show_3110', 'Cart Balancer (Assessment)_4040', 'Chest Sorter (Assessment)_3010', 'Leaf Leader_2060', 'Bubble Bath_4090', 'Cauldron Filler (Assessment)_4100', 'Chow Time_4030', 'Dino Drink_4031', 'Crystals Rule_2000', 'Air Show_2030', 'All Star Sorting_2000', 'Pan Balance_4010', 'Dino Drink_2070', 'Heavy, Heavier, Heaviest_2000', 'Bottle Filler (Activity)_4020', 'Egg Dropper (Activity)_2020', 'Bottle Filler (Activity)_4070', 'Watering Hole (Activity)_4025', 'All Star Sorting_4070', 'Mushroom Sorter (Assessment)_3121', 'Crystals Rule_3121', 'Dino Drink_2030', 'Chest Sorter (Assessment)_4070', 'Cauldron Filler (Assessment)_2030', 'All Star Sorting_3121', 'Fireworks (Activity)_2000', 'Dino Drink_4010', 'Mushroom Sorter (Assessment)_4090', 'Chow Time_2020', 'Pan Balance_3010', 'Air Show_2070', 'Bubble Bath_4070', 'Leaf Leader_3010', 'Bird Measurer (Assessment)_2010', 'Cauldron Filler (Assessment)_3021', 'Chest Sorter (Assessment)_4035', 'Scrub-A-Dub_4020', 'Bottle Filler (Activity)_2000', 'Bird Measurer (Assessment)_4040', 'Mushroom Sorter (Assessment)_3110', 'Chest Sorter (Assessment)_2000', 'All Star Sorting_4035', 'Cauldron Filler (Assessment)_4025', 'Scrub-A-Dub_4010', 'Chicken Balancer (Activity)_2000', 'Cauldron Filler (Assessment)_3121', 'Flower Waterer (Activity)_4080', 'Crystals Rule_2030', 'Chest Sorter (Assessment)_4080', 'Leaf Leader_3110', 'Cauldron Filler (Assessment)_4080', 'Chow Time_2000', 'Sandcastle Builder (Activity)_4070', 'Bird Measurer (Assessment)_3120', 'Dino Drink_2000', 'Dino Dive_3121', 'Pan Balance_3120', 'Dino Drink_3010', 'Mushroom Sorter (Assessment)_2020', 'Egg Dropper (Activity)_4025', 'Air Show_4020', 'Air Show_2060', 'Pan Balance_4030', 'Mushroom Sorter (Assessment)_2035', 'Bug Measurer (Activity)_4070', 'Egg Dropper (Activity)_4070', 'Bubble Bath_4230', 'Crystals Rule_4050', 'Pan Balance_3020', 'Chow Time_4080', 'Bird Measurer (Assessment)_4025', 'Air Show_3121', 'Chow Time_4020', 'Mushroom Sorter (Assessment)_3120', 'Bird Measurer (Assessment)_3110', 'Happy Camel_3020', 'Flower Waterer (Activity)_4025', 'Chicken Balancer (Activity)_4080', 'Rulers_2000', 'Flower Waterer (Activity)_4030', 'Tree Top City - Level 3_2000', 'Bubble Bath_2080', 'Bird Measurer (Assessment)_2000', 'Dino Dive_2020', 'Scrub-A-Dub_3010', 'Watering Hole (Activity)_2000', 'Bubble Bath_2025', 'Sandcastle Builder (Activity)_2010', 'Cart Balancer (Assessment)_4090', 'Cauldron Filler (Assessment)_3110', 'Air Show_2020', 'Chest Sorter (Assessment)_4100', 'Happy Camel_4030', 'Chest Sorter (Assessment)_3121', 'Bug Measurer (Activity)_4090', 'Fireworks (Activity)_4070', 'Mushroom Sorter (Assessment)_2010', 'Chicken Balancer (Activity)_4035', 'Bottle Filler (Activity)_2020', 'Happy Camel_2081', 'Air Show_4070', 'Pan Balance_2000', 'Bottle Filler (Activity)_4080', 'Slop Problem_2000', 'Bug Measurer (Activity)_4030', 'Dino Dive_3120', 'Dino Drink_3021', 'Lifting Heavy Things_2000', 'Chest Sorter (Assessment)_2010', 'Mushroom Sorter (Assessment)_4080', 'Bug Measurer (Activity)_4035', 'Chicken Balancer (Activity)_4070', 'Cauldron Filler (Assessment)_4030', 'Mushroom Sorter (Assessment)_4040', 'Chow Time_4090', 'Bubble Bath_2035', 'Bird Measurer (Assessment)_3010', 'Bubble Bath_3121', 'Dino Dive_2070', 'Chow Time_4070', 'Dino Dive_2000', 'Pan Balance_2030', 'Cart Balancer (Assessment)_4080', 'Air Show_4100', 'Cart Balancer (Assessment)_3021', 'Leaf Leader_4080', 'All Star Sorting_3110', 'Happy Camel_2000', 'Bubble Bath_4020', 'Bubble Bath_2083', 'Happy Camel_4045', 'Bubble Bath_3010', 'Bug Measurer (Activity)_3010', 'Bubble Bath_2030', 'Chicken Balancer (Activity)_4030', 'Chow Time_2030', 'Bug Measurer (Activity)_4080', 'Watering Hole (Activity)_4070', 'Bird Measurer (Assessment)_4030', 'Chow Time_3120', 'Mushroom Sorter (Assessment)_3020', 'Leaf Leader_3120', 'Chest Sorter (Assessment)_4040', 'Chow Time_3110', 'Chicken Balancer (Activity)_3010', 'Bubble Bath_4080', 'Happy Camel_4070', 'Cart Balancer (Assessment)_4030', 'Pan Balance_4100', 'All Star Sorting_4080', 'Cart Balancer (Assessment)_3121', 'Bird Measurer (Assessment)_4020', 'Bottle Filler (Activity)_4035', 'All Star Sorting_3021', 'Honey Cake_2000', 'All Star Sorting_4020', 'All Star Sorting_4010', 'Crystal Caves - Level 2_2000', 'Chest Sorter (Assessment)_3021', 'Chow Time_3021', 'Mushroom Sorter (Assessment)_4030', 'Egg Dropper (Activity)_4020', 'Cart Balancer (Assessment)_4020', 'Air Show_4010', 'Cart Balancer (Assessment)_3010', 'Pan Balance_3121', 'Crystals Rule_3020', 'Bird Measurer (Assessment)_3021', 'Scrub-A-Dub_2080', 'Bird Measurer (Assessment)_2030', 'Flower Waterer (Activity)_4020', 'Bird Measurer (Assessment)_4070', 'Happy Camel_3110', 'Bird Measurer (Assessment)_4080', 'Egg Dropper (Activity)_3010', 'Cauldron Filler (Assessment)_3120', 'Chest Sorter (Assessment)_4030', 'Scrub-A-Dub_2030', 'Bug Measurer (Activity)_4025', 'Bubble Bath_4040', 'Scrub-A-Dub_3021', 'Fireworks (Activity)_4030', 'Fireworks (Activity)_3110', 'Scrub-A-Dub_3110', 'Chow Time_3121', 'Watering Hole (Activity)_3010', 'Flower Waterer (Activity)_4022', 'Scrub-A-Dub_2050', 'Dino Drink_2075', 'Pan Balance_4090', 'Bird Measurer (Assessment)_4110', 'Scrub-A-Dub_2020', 'installation_id', 'session_title', 'accumulated_correct_attempts', 'accumulated_uncorrect_attempts', 'duration_mean', 'accumulated_accuracy', 'accuracy_group', 0, 1, 2, 3, 'accumulated_accuracy_group', 'accumulated_actions', 'installation_session_count', 'installation_duration_mean', 'installation_title_nunique', 'sum_event_code_count', 'installation_event_code_count_mean']

reduce_train = reduce_train[all_cols]
reduce_test = reduce_test[all_cols]
y = reduce_train['accuracy_group']

cols_to_drop = ['game_session', 'installation_id', 'timestamp', 'accuracy_group', 'timestampDate']

n_fold = 5
folds = GroupKFold(n_splits=n_fold)
def LGB_bayesian(max_depth,
                 lambda_l1,
                 lambda_l2,
                 bagging_fraction,
                 bagging_freq,
                 colsample_bytree,
                 learning_rate):
    
    params = {
        'boosting_type': 'gbdt',
        'metric': 'rmse',
        'objective': 'regression',
        'eval_metric': 'cappa',
        'n_jobs': -1,
        'seed': 42,
        'early_stopping_rounds': 100,
        'n_estimators': 2000,
        'learning_rate': learning_rate,
        'max_depth': int(max_depth),
        'lambda_l1': lambda_l1,
        'lambda_l2': lambda_l2,
        'bagging_fraction': bagging_fraction,
        'bagging_freq': int(bagging_freq),
        'colsample_bytree': colsample_bytree,
        'verbose': 0
    }
    
    mt = MainTransformer()
    ft = FeatureTransformer()
    transformers = {'ft': ft}
    model = RegressorModel(model_wrapper=LGBWrapper_regr())
    model.fit(X=reduce_train, 
              y=y, 
              folds=folds, 
              params=params, 
              preprocesser=mt, 
              transformers=transformers,
              eval_metric='cappa', 
              cols_to_drop=cols_to_drop,
              plot=False)
    
    return model.scores['valid']
gc.collect()
init_points = 16
n_iter = 16
bounds_LGB = {
    'max_depth': (8, 11),
    'lambda_l1': (0, 5),
    'lambda_l2': (0, 5),
    'bagging_fraction': (0.4, 0.6),
    'bagging_freq': (1, 10),
    'colsample_bytree': (0.4, 0.6),
    'learning_rate': (0.05, 0.1)
}

LGB_BO = BayesianOptimization(LGB_bayesian, bounds_LGB, random_state=1029)

with warnings.catch_warnings():
    warnings.filterwarnings('ignore')
    LGB_BO.maximize(init_points=init_points, n_iter=n_iter, acq='ucb', xi=0.0, alpha=1e-6)
params = {
    'boosting_type': 'gbdt',
    'metric': 'rmse',
    'objective': 'regression',
    'eval_metric': 'cappa',
    'n_jobs': -1,
    'seed': 42,
    'early_stopping_rounds': 100,
    'n_estimators': 2000,
    'learning_rate': LGB_BO.max['params']['learning_rate'],
    'max_depth': int(LGB_BO.max['params']['max_depth']),
    'lambda_l1': LGB_BO.max['params']['lambda_l1'],
    'lambda_l2': LGB_BO.max['params']['lambda_l2'],
    'bagging_fraction': LGB_BO.max['params']['bagging_fraction'],
    'bagging_freq': int(LGB_BO.max['params']['bagging_freq']),
    'colsample_bytree': LGB_BO.max['params']['colsample_bytree'],
    'verbose': 100
}

mt = MainTransformer()
ft = FeatureTransformer()
transformers = {'ft': ft}
regressor_model = RegressorModel(model_wrapper=LGBWrapper_regr())
regressor_model.fit(X=reduce_train, 
                    y=y, 
                    folds=folds, 
                    params=params, 
                    preprocesser=mt, 
                    transformers=transformers,
                    eval_metric='cappa', 
                    cols_to_drop=cols_to_drop)

preds_1 = regressor_model.predict(reduce_test)
w_1 = LGB_BO.max['target']
del bounds_LGB, LGB_BO, params, mt, ft, transformers, regressor_model
gc.collect()
bounds_LGB = {
    'max_depth': (11, 14),
    'lambda_l1': (0, 10),
    'lambda_l2': (0, 10),
    'bagging_fraction': (0.7, 1),
    'bagging_freq': (1, 10),
    'colsample_bytree': (0.7, 1),
    'learning_rate': (0.08, 0.2)
}

LGB_BO = BayesianOptimization(LGB_bayesian, bounds_LGB, random_state=1030)

with warnings.catch_warnings():
    warnings.filterwarnings('ignore')
    LGB_BO.maximize(init_points=init_points, n_iter=n_iter, acq='ucb', xi=0.0, alpha=1e-6)
params = {
    'boosting_type': 'gbdt',
    'metric': 'rmse',
    'objective': 'regression',
    'eval_metric': 'cappa',
    'n_jobs': -1,
    'seed': 42,
    'early_stopping_rounds': 100,
    'n_estimators': 2000,
    'learning_rate': LGB_BO.max['params']['learning_rate'],
    'max_depth': int(LGB_BO.max['params']['max_depth']),
    'lambda_l1': LGB_BO.max['params']['lambda_l1'],
    'lambda_l2': LGB_BO.max['params']['lambda_l2'],
    'bagging_fraction': LGB_BO.max['params']['bagging_fraction'],
    'bagging_freq': int(LGB_BO.max['params']['bagging_freq']),
    'colsample_bytree': LGB_BO.max['params']['colsample_bytree'],
    'verbose': 100
}

mt = MainTransformer()
ft = FeatureTransformer()
transformers = {'ft': ft}
regressor_model = RegressorModel(model_wrapper=LGBWrapper_regr())
regressor_model.fit(X=reduce_train, 
                    y=y, 
                    folds=folds, 
                    params=params, 
                    preprocesser=mt, 
                    transformers=transformers,
                    eval_metric='cappa', 
                    cols_to_drop=cols_to_drop)

preds_2 = regressor_model.predict(reduce_test)
w_2 = LGB_BO.max['target']
del bounds_LGB, LGB_BO, params, mt, ft, transformers, regressor_model
gc.collect()
preds = (w_1/(w_1+w_2)) * preds_1 + (w_2/(w_1+w_2)) * preds_2

del preds_1, preds_2
gc.collect()
coefficients = [1.12232214, 1.73925866, 2.22506454]
preds[preds <= coefficients[0]] = 0
preds[np.where(np.logical_and(preds > coefficients[0], preds <= coefficients[1]))] = 1
preds[np.where(np.logical_and(preds > coefficients[1], preds <= coefficients[2]))] = 2
preds[preds > coefficients[2]] = 3
sample_submission['accuracy_group'] = preds.astype(int)
sample_submission.to_csv('submission.csv', index=False)
sample_submission['accuracy_group'].value_counts(normalize=True)
