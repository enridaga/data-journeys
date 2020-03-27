
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
print(os.listdir("../input"))
import warnings
warnings.filterwarnings("ignore")
train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')
test_ID = test['ID']
y_train = train['target']
y_train = np.log1p(y_train)
train.drop("ID", axis = 1, inplace = True)
train.drop("target", axis = 1, inplace = True)
test.drop("ID", axis = 1, inplace = True)
cols_with_onlyone_val = train.columns[train.nunique() == 1]
train.drop(cols_with_onlyone_val.values, axis=1, inplace=True)
test.drop(cols_with_onlyone_val.values, axis=1, inplace=True)
NUM_OF_DECIMALS = 32
train = train.round(NUM_OF_DECIMALS)
test = test.round(NUM_OF_DECIMALS)
colsToRemove = []
columns = train.columns
for i in range(len(columns)-1):
    v = train[columns[i]].values
    dupCols = []
    for j in range(i + 1,len(columns)):
        if np.array_equal(v, train[columns[j]].values):
            colsToRemove.append(columns[j])
train.drop(colsToRemove, axis=1, inplace=True) 
test.drop(colsToRemove, axis=1, inplace=True) 
train.shape
from sklearn import model_selection
from sklearn import ensemble
NUM_OF_FEATURES = 1000
def rmsle(y, pred):
    return np.sqrt(np.mean(np.power(y - pred, 2)))

x1, x2, y1, y2 = model_selection.train_test_split(
    train, y_train.values, test_size=0.20, random_state=5)
model = ensemble.RandomForestRegressor(n_jobs=-1, random_state=7)
model.fit(x1, y1)
print(rmsle(y2, model.predict(x2)))

col = pd.DataFrame({'importance': model.feature_importances_, 'feature': train.columns}).sort_values(
    by=['importance'], ascending=[False])[:NUM_OF_FEATURES]['feature'].values
train = train[col]
test = test[col]
train.shape
from scipy.stats import ks_2samp
THRESHOLD_P_VALUE = 0.01 #need tuned
THRESHOLD_STATISTIC = 0.3 #need tuned
diff_cols = []
for col in train.columns:
    statistic, pvalue = ks_2samp(train[col].values, test[col].values)
    if pvalue <= THRESHOLD_P_VALUE and np.abs(statistic) > THRESHOLD_STATISTIC:
        diff_cols.append(col)
for col in diff_cols:
    if col in train.columns:
        train.drop(col, axis=1, inplace=True)
        test.drop(col, axis=1, inplace=True)
train.shape
from sklearn import random_projection
ntrain = len(train)
ntest = len(test)
tmp = pd.concat([train,test])#RandomProjection
weight = ((train != 0).sum()/len(train)).values
tmp_train = train[train!=0]
tmp_test = test[test!=0]
train["weight_count"] = (tmp_train*weight).sum(axis=1)
test["weight_count"] = (tmp_test*weight).sum(axis=1)
train["count_not0"] = (train != 0).sum(axis=1)
test["count_not0"] = (test != 0).sum(axis=1)
train["sum"] = train.sum(axis=1)
test["sum"] = test.sum(axis=1)
train["var"] = tmp_train.var(axis=1)
test["var"] = tmp_test.var(axis=1)
train["median"] = tmp_train.median(axis=1)
test["median"] = tmp_test.median(axis=1)
train["mean"] = tmp_train.mean(axis=1)
test["mean"] = tmp_test.mean(axis=1)
train["std"] = tmp_train.std(axis=1)
test["std"] = tmp_test.std(axis=1)
train["max"] = tmp_train.max(axis=1)
test["max"] = tmp_test.max(axis=1)
train["min"] = tmp_train.min(axis=1)
test["min"] = tmp_test.min(axis=1)
train["skew"] = tmp_train.skew(axis=1)
test["skew"] = tmp_test.skew(axis=1)
train["kurtosis"] = tmp_train.kurtosis(axis=1)
test["kurtosis"] = tmp_test.kurtosis(axis=1)
del(tmp_train)
del(tmp_test)
NUM_OF_COM = 100 #need tuned
transformer = random_projection.SparseRandomProjection(n_components = NUM_OF_COM)
RP = transformer.fit_transform(tmp)
rp = pd.DataFrame(RP)
columns = ["RandomProjection{}".format(i) for i in range(NUM_OF_COM)]
rp.columns = columns

rp_train = rp[:ntrain]
rp_test = rp[ntrain:]
rp_test.index = test.index

#concat RandomProjection and raw data
train = pd.concat([train,rp_train],axis=1)
test = pd.concat([test,rp_test],axis=1)

del(rp_train)
del(rp_test)
train.shape
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.base import BaseEstimator, TransformerMixin, RegressorMixin, clone
from sklearn.metrics import mean_squared_error
import xgboost as xgb
import lightgbm as lgb
#define evaluation method for a given model. we use k-fold cross validation on the training set. 
#the loss function is root mean square logarithm error between target and prediction
#note: train and y_train are feeded as global variables
NUM_FOLDS = 5 #need tuned
def rmsle_cv(model):
    kf = KFold(NUM_FOLDS, shuffle=True, random_state=42).get_n_splits(train.values)
    rmse= np.sqrt(-cross_val_score(model, train, y_train, scoring="neg_mean_squared_error", cv = kf))
    return(rmse)
#ensemble method: model averaging
class AveragingModels(BaseEstimator, RegressorMixin, TransformerMixin):
    def __init__(self, models):
        self.models = models
        
    # we define clones of the original models to fit the data in
    # the reason of clone is avoiding affect the original base models
    def fit(self, X, y):
        self.models_ = [clone(x) for x in self.models]  
        # Train cloned base models
        for model in self.models_:
            model.fit(X, y)
        return self
    
    #Now we do the predictions for cloned models and average them
    def predict(self, X):
        predictions = np.column_stack([ model.predict(X) for model in self.models_ ])
        return np.mean(predictions, axis=1)

model_xgb = xgb.XGBRegressor(colsample_bytree=0.055, colsample_bylevel =0.5, 
                             gamma=1.5, learning_rate=0.02, max_depth=32, 
                             objective='reg:linear',booster='gbtree',
                             min_child_weight=57, n_estimators=1000, reg_alpha=0, 
                             reg_lambda = 0,eval_metric = 'rmse', subsample=0.7, 
                             silent=1, n_jobs = -1, early_stopping_rounds = 14,
                             random_state =7, nthread = -1)
model_lgb = lgb.LGBMRegressor(objective='regression',num_leaves=144,
                              learning_rate=0.005, n_estimators=720, max_depth=13,
                              metric='rmse',is_training_metric=True,
                              max_bin = 55, bagging_fraction = 0.8,verbose=-1,
                              bagging_freq = 5, feature_fraction = 0.9) 
score = rmsle_cv(model_xgb)
print("Xgboost score: {:.4f} ({:.4f})\n".format(score.mean(), score.std()))
score = rmsle_cv(model_lgb)
print("LGBM score: {:.4f} ({:.4f})\n" .format(score.mean(), score.std()))
averaged_models = AveragingModels(models = (model_xgb, model_lgb))
score = rmsle_cv(averaged_models)
print("averaged score: {:.4f} ({:.4f})\n" .format(score.mean(), score.std()))
averaged_models.fit(train.values, y_train)
pred = np.expm1(averaged_models.predict(test.values))
ensemble = pred
sub = pd.DataFrame()
sub['ID'] = test_ID
sub['target'] = ensemble
sub.to_csv('submission.csv',index=False)

#Xgboost score: 1.3582 (0.0640)
#LGBM score: 1.3437 (0.0519)
#averaged score: 1.3431 (0.0586)

#Xgboost score: 1.3566 (0.0525)
#LGBM score: 1.3477 (0.0497)
#averaged score: 1.3438 (0.0516)

#Xgboost score: 1.3540 (0.0621)
#LGBM score: 1.3463 (0.0485)
#averaged score: 1.3423 (0.0556)