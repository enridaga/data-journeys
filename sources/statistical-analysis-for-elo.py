
from sklearn import model_selection, preprocessing, metrics
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
import matplotlib.pylab as pylab
import matplotlib.pyplot as plt
from pandas import get_dummies
import lightgbm as lgb
import matplotlib as mpl
import xgboost as xgb
import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib
import warnings
import sklearn
import scipy
import numpy
import json
import sys
import csv
import os
print('matplotlib: {}'.format(matplotlib.__version__))
print('sklearn: {}'.format(sklearn.__version__))
print('scipy: {}'.format(scipy.__version__))
print('seaborn: {}'.format(sns.__version__))
print('pandas: {}'.format(pd.__version__))
print('numpy: {}'.format(np.__version__))
print('Python: {}'.format(sys.version))

sns.set(style='white', context='notebook', palette='deep')
warnings.filterwarnings('ignore')
warnings.simplefilter(action='ignore', category=FutureWarning)
sns.set_style('white')
pd.set_option('display.max_columns', 500)
### matplotlib inline
import os
print(os.listdir("../input"))

main_train = pd.read_csv('../input/train.csv', parse_dates=["first_active_month"] )
main_test = pd.read_csv('../input/test.csv' ,parse_dates=["first_active_month"] )
main_merchants=pd.read_csv('../input/merchants.csv')
main_new_merchant_transactions=pd.read_csv('../input/new_merchant_transactions.csv')
main_historical_transactions = pd.read_csv("../input/historical_transactions.csv")
sample_submission = pd.read_csv("../input/sample_submission.csv")
sample_submission.shape
sample_submission.head()
print(main_train.info())
print(main_test.info())
print("Shape of train set                 : ",main_train.shape)
print("Shape of test set                  : ",main_test.shape)
print("Shape of historical_transactions   : ",main_historical_transactions.shape)
print("Shape of merchants                 : ",main_merchants.shape)
print("Shape of new_merchant_transactions : ",main_new_merchant_transactions.shape)

data_dictionary_train=pd.read_excel('../input/Data_Dictionary.xlsx',sheet_name='train')
data_dictionary_history=pd.read_excel('../input/Data_Dictionary.xlsx',sheet_name='history')
data_dictionary_new_merchant_period=pd.read_excel('../input/Data_Dictionary.xlsx',sheet_name='new_merchant_period')
data_dictionary_merchant=pd.read_excel('../input/Data_Dictionary.xlsx',sheet_name='merchant')
main_train.tail()
data_dictionary_train.head(10)
# what we know about train:
main_train.tail()
main_train.describe()
print('----- test set--------')
print(main_test.head(5))
main_test.info()
main_test.describe()
data_dictionary_history.head(10)
# what we know about history:
main_historical_transactions.head()
main_historical_transactions.shape
main_merchants.head()
data_dictionary_merchant.head(30)
# what we know about merchant:
main_new_merchant_transactions.head()
data_dictionary_new_merchant_period.head(10)
# what we know about new_merchant_period:
def check_missing_data(df):
    flag=df.isna().sum().any()
    if flag==True:
        total = df.isnull().sum()
        percent = (df.isnull().sum())/(df.isnull().count()*100)
        output = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
        data_type = []
        # written by MJ Bahmani
        for col in df.columns:
            dtype = str(df[col].dtype)
            data_type.append(dtype)
        output['Types'] = data_type
        return(np.transpose(output))
    else:
        return(False)
print ('for train :',check_missing_data(main_train))
print ('for test:',check_missing_data(main_test))
# remove rows that have NA's
print('Before Droping',main_train.shape)
main_train = main_train.dropna()
print('After Droping',main_train.shape)
main_train.columns
main_train["target"].hist();
main_train[main_train["target"]<-29].count()
# histograms
main_train.hist(figsize=(15,20))
plt.figure()
f,ax=plt.subplots(1,2,figsize=(20,10))
main_train[main_train['feature_3']==0].target.plot.hist(ax=ax[0],bins=20,edgecolor='black',color='red')
ax[0].set_title('feature_3= 0')
x1=list(range(0,85,5))
ax[0].set_xticks(x1)
main_train[main_train['feature_3']==1].target.plot.hist(ax=ax[1],color='green',bins=20,edgecolor='black')
ax[1].set_title('feature_3= 1')
x2=list(range(0,85,5))
ax[1].set_xticks(x2)
plt.show()
f,ax=plt.subplots(1,2,figsize=(18,8))
main_train['feature_3'].value_counts().plot.pie(explode=[0,0.1],autopct='%1.1f%%',ax=ax[0],shadow=True)
ax[0].set_title('feature_3')
ax[0].set_ylabel('')
sns.countplot('feature_3',data=main_train,ax=ax[1])
ax[1].set_title('feature_3')
plt.show()
f,ax=plt.subplots(1,2,figsize=(18,8))
main_train[['feature_3','feature_2']].groupby(['feature_3']).mean().plot.bar(ax=ax[0])
ax[0].set_title('Survived vs feature_2')
sns.countplot('feature_3',hue='feature_2',data=main_train,ax=ax[1])
ax[1].set_title('feature_3:feature')
plt.show()
sns.distplot(main_train['target'])
sns.violinplot(data=main_train, x="feature_1", y='target')
plt.figure(figsize=(8,6))
plt.scatter(range(main_train.shape[0]), np.sort(main_train['target'].values),marker='o',c='green')
plt.xlabel('index', fontsize=12)
plt.ylabel('Loyalty Score', fontsize=12)
plt.title('Explore: Target')
plt.show();
# Modify the graph above by assigning each species an individual color.
g = sns.FacetGrid(main_train, hue="feature_3", col="feature_2", margin_titles=True,
                  palette={1:"blue", 0:"red"} )
g=g.map(plt.scatter, "first_active_month", "target",edgecolor="w").add_legend();
sns.boxplot(x="feature_3", y="feature_2", data=main_test )
plt.show()
main_train.where(main_train ['target']==1).count()
main_train[main_train['target']<-32].head(5)
main_train[main_train['target']==1].head(5)
main_train.feature_1.unique()
main_train.feature_2.unique()
main_train.feature_3.unique()
main_train.first_active_month.unique()
df_train=main_train
df_test=main_test
df_train["year"] = main_train["first_active_month"].dt.year
df_test["year"] = main_test["first_active_month"].dt.year
df_train["month"] = main_train["first_active_month"].dt.month
df_test["month"] = main_test["first_active_month"].dt.month
x_train = df_train.drop(["target","card_id","first_active_month"],axis=1)
x_test = df_test.drop(["card_id","first_active_month"],axis=1)
y_train = df_train["target"]
df_train = df_train.sample(frac=1, random_state = 7)

Trn_x,val_x,Trn_y,val_y = train_test_split(x_train,y_train,test_size =0.1,random_state = 7)
trn_x , test_x, trn_y, test_y = train_test_split(Trn_x , Trn_y, test_size =0.1, random_state = 7)
# rmse
def rmse(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))
# converting into xgb DMatrix
Train = xgb.DMatrix(trn_x,label = trn_y)
Validation = xgb.DMatrix(val_x, label = val_y)
Test = xgb.DMatrix(test_x)
params = {"booster":"gbtree","eta":0.1,'min_split_loss':0,'max_depth':6,
         'min_child_weight':1, 'max_delta_step':0,'subsample':1,'colsample_bytree':1,
         'colsample_bylevel':1,'reg_lambda':1,'reg_alpha':0,
         'grow_policy':'depthwise','max_leaves':0,'objective':'reg:linear','eval_metric':'rmse',
         'seed':7}
history ={}  # This will record rmse score of training and test set
eval_list =[(Train,"Training"),(Validation,"Validation")]
clf = xgb.train(params, Train, num_boost_round=119, evals=eval_list, obj=None, feval=None, maximize=False, 
          early_stopping_rounds=40, evals_result=history);
prediction = clf.predict(xgb.DMatrix(x_test))
submission = pd.DataFrame({
        "card_id": main_test["card_id"].values,
        "target": np.ravel(prediction)
    })