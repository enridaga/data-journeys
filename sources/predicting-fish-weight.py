
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
#%matplotlib inline
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn import metrics
import statsmodels.formula.api as smf
# UDF to deal with outliers
# UDF to get summary of categorical variables
def conti_summ(x):
    return pd.Series([x.count(),x.isnull().sum(),x.sum(),x.mean(),x.var(),x.quantile(0.01),x.quantile(0.05),x.quantile(0.10),
                     x.quantile(0.25),x.quantile(0.50),x.quantile(0.75),x.quantile(0.90),x.quantile(0.95),x.quantile(0.99),
                     x.max()],
                    index=['Count','Null','Sum','Mean','Var','Q1','Q5','Q10','Q25','Q50','Q75',
                           'Q90','Q95','Q99','Max'])
# UDF to create dummy variables
def dummy_vars(df,colname):
    dummies = pd.get_dummies(df[colname],prefix=colname,drop_first=True)
    df.drop(colname,inplace=True,axis=1)
    df = pd.concat([df,dummies],axis=1)
    return df
fish = pd.read_csv('../input/fish-market/Fish.csv')
fish.head()
fish.shape
fish.dtypes
fish['Species'].value_counts()
fish.isnull().sum()  # No missing values
sns.countplot(fish['Species'])
plt.show()
fish_group = fish.groupby('Species').mean()
fish_group
fish.columns
fish_continuous = fish.loc[:,fish.dtypes=='float64']
fish_continuous.head()
fish_categorical = fish[['Species']]
fish_categorical.head()
fish_continuous.apply(lambda x: conti_summ(x)).T
# cliping the outliers
fish_continuous = fish_continuous.apply(lambda x: x.clip(lower=x.quantile(0.05),upper=x.quantile(0.95)))
fish_continuous.apply(lambda x: conti_summ(x)).T
fish_categorical = dummy_vars(fish_categorical,['Species'])
fish_categorical.head()
fish_final = pd.concat([fish_continuous,fish_categorical],axis=1)
fish_final.head()
fish_final.shape
fish_final.isnull().sum()
plt.figure(figsize=(10,8))
sns.heatmap(fish_final.corr(),annot=True,fmt='.2f')
plt.show()
sns.distplot(fish_final['Weight'])
plt.show()
fish_final['Weight'].skew()
# Doing log transformation to make 'Weight' variable normally distributed
fish_final['ln_Weight'] = np.log(fish_final['Weight'])
sns.distplot(fish_final['ln_Weight'])
plt.show()
np.log(fish_final['Weight']).skew()
train_x,test_x,train_y,test_y = train_test_split(fish_final[fish_final.columns.difference(['Weight','ln_Weight'])],
                                                 fish_final['ln_Weight'],random_state=123,test_size=0.3)
train,test = train_test_split(fish_final,random_state=123,test_size=0.3)
print('No.of observations in train',train.shape)
print('No.of observations in test',test.shape)
train.columns
fish_log = smf.ols('''ln_Weight ~ Length1 + Length2 + Length3 + Height + Width + Species_Parkki + Species_Perch + 
                      Species_Pike + Species_Roach + Species_Smelt + Species_Whitefish''',train).fit()
print(fish_log.summary())
model_param = '''ln_Weight ~ Length1   + Height   + Width +
                     Species_Smelt '''
from patsy import dmatrices
from statsmodels.stats.outliers_influence import variance_inflation_factor
Y,X = dmatrices(model_param,train,return_type='dataframe')
vif=pd.DataFrame()
vif['Features'] = X.columns
vif['Vif_value'] = [variance_inflation_factor(X.values,i) for i in range(X.shape[1])]
vif.sort_values(by='Vif_value',ascending=False)
fish_log_2 = smf.ols('''ln_Weight ~ Length1   + Height  + 
                     Species_Smelt ''',train).fit()
print(fish_log_2.summary())
np.mean(np.abs(train['ln_Weight'] - fish_log_2.predict(train)/train['ln_Weight']))
np.mean(np.abs(test['ln_Weight'] - fish_log_2.predict(test)/test['ln_Weight']))
train['Predicted'] = np.exp(fish_log_2.predict(train))
train.head()
test['Predicted'] = np.exp(fish_log_2.predict(test))
test.head()
from sklearn.ensemble import RandomForestRegressor
fish_random = RandomForestRegressor()
fish_random
param_grid = {'max_depth':np.arange(3,11),'max_features':np.arange(1,7),'n_estimators':[60,80,100,150,200]}

from sklearn.model_selection import GridSearchCV
fish_grid = GridSearchCV(fish_random,param_grid=param_grid,verbose=True,cv=5,n_jobs=-1)
fish_grid.fit(train_x,train_y)
fish_grid.best_params_
# Running random forest model with best parameters
fish_random = RandomForestRegressor(max_depth= 8 ,max_features = 2,n_estimators =100,oob_score=True)
fish_random.fit(train_x,train_y)
metrics.mean_squared_error(fish_random.predict(train_x),train_y)
metrics.mean_squared_error(fish_random.predict(test_x),test_y)
