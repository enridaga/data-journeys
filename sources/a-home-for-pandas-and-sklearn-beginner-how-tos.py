
# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
import pandas as pd

import numpy as np

import itertools

import matplotlib.pyplot as plt



#prep

from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV

from sklearn.preprocessing import LabelEncoder, StandardScaler, MaxAbsScaler, QuantileTransformer



#models

from sklearn.linear_model import LogisticRegression, LogisticRegressionCV, LinearRegression, Ridge, RidgeCV

from sklearn.tree import DecisionTreeRegressor

from sklearn.ensemble import RandomForestRegressor



#validation libraries

from sklearn.cross_validation import KFold, StratifiedKFold

from IPython.display import display

from sklearn import metrics





### matplotlib inline
train_df = pd.read_csv('../input/train.csv')

df_test = pd.read_csv('../input/test.csv')
# here's one sample

sample_dict = [

    {'label': 'house i would like', 'sqft':5000},

    {'label': 'house i would hate','sqft':500},

    {'label': 'house i live in', 'sqft':800}

]

pd.DataFrame(sample_dict)
train_df.shape
train_df.describe()
train_df.info()
train_df.head(2)
train_df['SalePrice'].head(5)
train_df[['SalePrice']].head(5)
train_df.as_matrix()
pd.DataFrame(train_df.as_matrix()).head()
train_df[['SalePrice','LotShape']].head(4)
train_df.iloc[range(3),]
train_df[train_df['SalePrice']>200000].head(3)
train_df[train_df['LotShape'].isin(['Reg','IR1'])].head(3)
print('this many columns:%d ' % len(train_df.columns))

train_df.columns
train_df.columns = ['Id', 'MSSubClass', 'MSZoning', 'LotFrontage', 'LotArea', 'Street',

       'Alley', 'LotShape', 'LandContour', 'Utilities', 'LotConfig',

       'LandSlope', 'Neighborhood', 'Condition1', 'Condition2', 'BldgType',

       'HouseStyle', 'OverallQual', 'OverallCond', 'YearBuilt', 'YearRemodAdd',

       'RoofStyle', 'RoofMatl', 'Exterior1st', 'Exterior2nd', 'MasVnrType',

       'MasVnrArea', 'ExterQual', 'ExterCond', 'Foundation', 'BsmtQual',

       'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinSF1',

       'BsmtFinType2', 'BsmtFinSF2', 'BsmtUnfSF', 'TotalBsmtSF', 'Heating',

       'HeatingQC', 'CentralAir', 'Electrical', 'FirsstFlrSF', 'SecondFlrSF',

       'LowQualFinSF', 'GrLivArea', 'BsmtFullBath', 'BsmtHalfBath', 'FullBath',

       'HalfBath', 'BedroomAbvGr', 'KitchenAbvGr', 'KitchenQual',

       'TotRmsAbvGrd', 'Functional', 'Fireplaces', 'FireplaceQu', 'GarageType',

       'GarageYrBlt', 'GarageFinish', 'GarageCars', 'GarageArea', 'GarageQual',

       'GarageCond', 'PavedDrive', 'WoodDeckSF', 'OpenPorchSF',

       'EnclosedPorch', 'ThreeSsnPorch', 'ScreenPorch', 'PoolArea', 'PoolQC',

       'Fence', 'MiscFeature', 'MiscVal', 'MoSold', 'YrSold', 'SaleType',

       'SaleCondition', 'SalePrice']
train_df[train_df['Alley'].isnull()].head(10)
train_df['Alley'].fillna(0, inplace=True)
na_totals = train_df.isnull().sum().sort_values(ascending=False)

na_totals[na_totals>0]
train_df.fillna(0, inplace=True)
numeric_cols = [x for x in train_df.columns if ('Area' in x) | ('SF' in x)] + ['SalePrice','LotFrontage','MiscVal','EnclosedPorch','ThreeSsnPorch','ScreenPorch','OverallQual','OverallCond','YearBuilt']



for col in numeric_cols:

    train_df[col] = train_df[col].astype(float)

numeric_cols
categorical_cols = [x for x in train_df.columns if x not in numeric_cols]



for col in categorical_cols:

    train_df[col] = train_df[col].astype('category')
## Applying an element wise function

train_df['LogSalePrice'] = train_df['SalePrice'].map(lambda x : np.log(x)) 



#Vectorized log function acting on a vector

# then assigning all the values at once

train_df['LogSalePrice'] = np.log(train_df['SalePrice'])
train_df['SalePrice'].hist()
train_df['LogSalePrice'].hist()
# element wise function to transform

train_df['above_200k'] = train_df['SalePrice'].map(lambda x : 1 if x > 200000 else 0) 

train_df['above_200k'] = train_df['above_200k'].astype('category')
# manually assign the values to your new field, section by section

# with row filtering

train_df.loc[train_df['SalePrice']>200000,'above_200k'] = 1

train_df.loc[train_df['SalePrice']<=200000,'above_200k'] = 0

train_df['above_200k'] = train_df['above_200k'].astype('category')
train_df['LivArea_Total'] = train_df['GrLivArea'] + train_df['GarageArea'] + train_df['PoolArea']

train_df[['LivArea_Total','GrLivArea','GarageArea','PoolArea']].head()
## concatenating two different fields together in the same row

train_df['Lot_desc'] = train_df.apply(lambda val : val['MSZoning'] + val['LotShape'], axis=1)

train_df[['Lot_desc','MSZoning','LotShape']].head()
train_df['LotArea_norm'] = train_df['LotArea']



ss = StandardScaler()

mas = MaxAbsScaler()

qs = QuantileTransformer()



train_df['LotArea_norm'] = ss.fit_transform(train_df[['LotArea']])

train_df['LotArea_mas'] = mas.fit_transform(train_df[['LotArea']])

train_df['LotArea_qs'] = qs.fit_transform(train_df[['LotArea']])





train_df[['LotArea_norm','LotArea_mas','LotArea_qs', 'LotArea']].head(5)
small_df = train_df[['MSZoning','SalePrice']].copy()

small_df['MSZoning'] = small_df['MSZoning'].astype('category')

small_df.head()

pd.get_dummies(small_df).head(5)
small_df = train_df[['MSSubClass','SalePrice']].copy()

small_df['MSSubClass'] = small_df['MSSubClass'].astype('category')

small_df.head()

le = LabelEncoder()

trf_MSSubClass = le.fit_transform(small_df['MSSubClass'])

trf_MSSubClass
le.classes_
le.inverse_transform(trf_MSSubClass)
feature_cols = [col for col in train_df.columns if 'Price' not in col]
y = train_df['LogSalePrice']

X = train_df[feature_cols]

print(y.head(2),'\n\n', X.head(2))
X_numerical = pd.get_dummies(X)

X_numerical.head(5)
import patsy

formula = 'LogSalePrice ~ %s' % (' + '.join(feature_cols)) 

y, X = patsy.dmatrices(formula, train_df, return_type='dataframe')

print(y.head(2),'\n\n', X.head(2))
def split_vals(a,n): return a[:n], a[n:]

n_valid = 170

n_trn = len(y)-n_valid

X_train, X_valid = split_vals(X, n_trn)

y_train, y_valid = split_vals(y, n_trn)



print(X_train.shape, X_valid.shape, y_train.shape, y_valid.shape)

X_train, X_valid, y_train, y_valid = train_test_split(X,y, test_size=0.2)

print(X_train.shape, X_valid.shape, y_train.shape, y_valid.shape)

lm = LinearRegression()

lm.fit(X_train,y_train)
lm.score(X_train,y_train)
lm.score(X_valid,y_valid)
y_pred = lm.predict(X_valid)

rmse = np.sqrt(metrics.mean_squared_error(y_pred, y_valid))

rmse
rdgCV = RidgeCV(alphas=[0.01,0.1,1,10,100,1000], cv=5)

rdgCV.fit(X_train,y_train)
print(rdgCV.alpha_)
rdg = Ridge(alpha=10)

rdg.fit(X_train, y_train)

rdg.score(X_valid, y_valid)
y_pred = rdg.predict(X_valid)

rmse = np.sqrt(metrics.mean_squared_error(y_pred, y_valid))

rmse
rfr = RandomForestRegressor(n_jobs=-1, n_estimators=100)

rfr.fit(X,y)
rfr.score(X_valid,y_valid)
y_pred = rfr.predict(X_valid)

rmse = np.sqrt(metrics.mean_squared_error(y_pred, y_valid))

rmse
import patsy



train_df['above_200k'] = train_df['above_200k'].astype(float)

formula = 'above_200k ~ %s' % (' + '.join(feature_cols)) 

y_cls, X_cls = patsy.dmatrices(formula, train_df, return_type='dataframe')

print(y.head(2),'\n\n', X.head(2))





X_cls_train, X_cls_valid, y_cls_train, y_cls_valid = train_test_split(X_cls,y_cls, test_size=0.2)

print(X_cls_train.shape, X_cls_valid.shape, y_cls_train.shape, y_valid.shape)
lgm = LogisticRegression()

lgm.fit(X_cls_train,y_cls_train)
lgm.score(X_cls_valid,y_cls_valid)
y_cls_pred = lgm.predict(X_cls_valid)



def plot_confusion_matrix(cm, classes,

                          normalize=False,

                          title='Confusion matrix',

                          cmap=plt.cm.Blues):

    """

    This function prints and plots the confusion matrix.

    Normalization can be applied by setting `normalize=True`.

    """

    if normalize:

        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

        print("Normalized confusion matrix")

    else:

        print('Confusion matrix, without normalization')



    print(cm)



    plt.imshow(cm, interpolation='nearest', cmap=cmap)

    plt.title(title)

    plt.colorbar()

    tick_marks = np.arange(len(classes))

    plt.xticks(tick_marks, classes, rotation=45)

    plt.yticks(tick_marks, classes)



    fmt = '.2f' if normalize else 'd'

    thresh = cm.max() / 2.

    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):

        plt.text(j, i, format(cm[i, j], fmt),

                 horizontalalignment="center",

                 color="white" if cm[i, j] > thresh else "black")



    plt.tight_layout()

    plt.ylabel('True label')

    plt.xlabel('Predicted label')

    

    

cnf_matrix = metrics.confusion_matrix(y_cls_valid, y_cls_pred)

np.set_printoptions(precision=2)



# Plot non-normalized confusion matrix

plt.figure()

plot_confusion_matrix(cnf_matrix, classes=['above_200k', 'below_200k'],

                      title='Confusion matrix, without normalization')

rfr = RandomForestRegressor(n_jobs=-1)
params = {

    'max_features': [0.25, 0.5, 0.7],

    'max_depth' : [ 2,5,10,20]

}

gs = GridSearchCV(cv=5, param_grid=params, estimator=rfr, verbose=0)

gs.fit(X_train,y_train.LogSalePrice.ravel())

print(gs.best_params_, gs.best_score_)
params = {

    'max_features': [0.25, 0.5, 0.7],

    'max_depth' : [ 2,5,10,20]

}

rs = RandomizedSearchCV(cv=5, param_distributions=params, estimator=rfr, verbose=0)

rs.fit(X_train,y_train.LogSalePrice.ravel())

print(rs.best_params_, rs.best_score_)
print(lm.coef_)

print(rdgCV.coef_)

print(lgm.coef_)
rfr = RandomForestRegressor(n_jobs=-1, n_estimators=100)

rfr.fit(X_train,y_train)



y_lm_pred = lm.predict(X_train)

y_rdgCV_pred = rdgCV.predict(X_train)

y_rfr_pred = rfr.predict(X_train)



print('-----training score ---')

print(lm.score(X_train, y_train))

print(rdgCV.score(X_train, y_train))

print(rfr.score(X_train, y_train))

print('----Validation score ---')

print(lm.score(X_valid, y_valid))

print(rdgCV.score(X_valid, y_valid))

print(rfr.score(X_valid, y_valid))
y_cls_train['above_200k'].values
y_lgm_p = lgm.predict(X_cls_train)

y_lgm_lpr = lgm.predict_log_proba(X_cls_train)

y_lgm_pr = lgm.predict_proba(X_cls_train)



y_lgm_lpr[:,0]

y_lgm_pr[:,0]

y_lgm_pr[:,1]

pd.DataFrame({'true': y_cls_train['above_200k'].values,

              'predict':y_lgm_p, 

              'log_prob_0':y_lgm_lpr[:,0],

              'log_prob_1':y_lgm_lpr[:,1],

              'prob_0': y_lgm_pr[:,0],

              'prob_1': y_lgm_pr[:,1]

             }).head(20)
from sklearn.metrics import accuracy_score, auc, classification_report, \

confusion_matrix, f1_score, log_loss, precision_recall_curve, roc_auc_score, roc_curve



print('Log Loss: ', log_loss(y_lgm_p, y_cls_train))

print('Accuracy_score: ', accuracy_score(y_lgm_p, y_cls_train))

print('confusion_matrix: ', confusion_matrix(y_lgm_p, y_cls_train))

print('Classification_Report: ', classification_report(y_lgm_p, y_cls_train))