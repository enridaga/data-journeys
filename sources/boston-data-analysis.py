
from sklearn.datasets import load_boston
import pandas as pd
from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity = "all"
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
 
ds = load_boston()
ds.data.shape 
 
print(ds.feature_names)

 
data_f = pd.DataFrame(data=ds.data,columns=ds.feature_names)
data_t = pd.DataFrame(data=ds.target,columns=['target'])
data_f.info();
data_t.info();
 pd.isnull(data_f).any()
 pd.isnull(data_t).any()
plt.figure(figsize=(10,5))
sns.set(rc={'axes.facecolor':'grey', 'figure.facecolor':'grey'})
sns.distplot(data_t['target'], bins = 50);
 
data_f['RM'].mean()
data_f['RAD'].value_counts()
data_t["target"].corr(data_f["RM"])
data_all = data_f.copy()
data_all["PRICE"] = data_t["target"]
#data_all.corr()
#creating filter to remove symmetric in the heat map
filter = np.zeros_like(data_all.corr())
tri_indice = np.triu_indices_from(filter)
filter[tri_indice]=True
filter
plt.figure(figsize=(15,15));
sns.set_style('dark');
sns.heatmap(data_all.corr(),annot=True,mask=filter);
plt.show();

print(data_f['NOX'].corr(data_f['DIS']))
plt.scatter(x=data_f['DIS'], y=data_f['NOX'],alpha=0.6);
plt.xlabel('Distance from Employment Center');
plt.ylabel('Measure of pollution');

sns.pairplot(data_all,kind='reg', plot_kws = {'line_kws' : {'color':'red'}})
data_all["INDUS"].corr(data_all["PRICE"])
#Split the data into test and train
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(data_f, data_t,test_size=0.25) # meaning 20% will be test data.

from sklearn.linear_model import LinearRegression
lr = LinearRegression()
lr.fit(X_train, y_train)
print("===================================================")
print(" r squared of train : ", lr.score(X_train,y_train))
print("  r squared of test : ", lr.score(X_test,y_test))
print ("        Y Intercept : " , lr.intercept_)
print("===================================================")
lr.coef_.shape
pd.DataFrame(data = lr.coef_,columns=X_train.columns);

data_all['PRICE'].skew() #Zero value means no skewness
price_log = np.log(data_t['target'])
price_log.skew() #clearly the results are near to zero unlike previous 1.108
fig, ax =plt.subplots(1,2)
sns.distplot(data_t['target'], ax=ax[0])
sns.distplot(np.log(data_t['target']), ax=ax[1])
fig.show();
 

data_f_withlog = data_f.copy()
data_f_withlog["LOG_PRICE"] = price_log
sns.lmplot(x="LSTAT",y="PRICE",data=data_all,size=7,scatter_kws={'alpha':0.6},line_kws={'color':'darkred'}  );
sns.lmplot(x="LSTAT",y="LOG_PRICE",data=data_f_withlog,size=7,scatter_kws={'alpha':0.6},line_kws={'color':'darkred'} );
 
#test train using log of the price variable
X_train, X_test, y_train, y_test = train_test_split(data_f, np.log(data_t),test_size=0.25);
lr = LinearRegression();
lr.fit(X_train, y_train);
print("===================================================")
print(" r squared of train : ", lr.score(X_train,y_train));
print("  r squared of test : ", lr.score(X_test,y_test));
print ("        Y Intercept : " , lr.intercept_);
print("===================================================")


#We get better R squared value thn before
X_train
import statsmodels.api as sm
x_include_const = sm.add_constant(X_train);
model = sm.OLS(y_train, x_include_const);
output = model.fit();
print("======================= Parameters ============================")
print(output.params);
print("======================= P Values ============================")
print(round(output.pvalues,3));
from statsmodels.stats.outliers_influence import variance_inflation_factor
vif = [variance_inflation_factor(exog = x_include_const.values, exog_idx=i) for i in range(x_include_const.shape[1])]
pd.DataFrame({'coef_name':x_include_const.columns,'vif' : np.around(vif,4 )})
#With INDUS and PRICE column 
x_include_const = sm.add_constant(X_train);
model = sm.OLS(y_train, x_include_const);
output = model.fit();
coef = pd.DataFrame({'params':output.params,'pvalues':round(output.pvalues,3)})
print("BIC: ",output.bic)
print("R Squared: ",output.rsquared)
print("===================== WITHOUT INDUS AND AGE FEATURES=====================")
#without INDUS and PRICE column
x_exclude = x_include_const.drop(['INDUS','AGE'], axis = 1)
model_exclude = sm.OLS(y_train, x_exclude);
output_exclude = model_exclude.fit();
coef_exclude = pd.DataFrame({'params':output_exclude.params,'pvalues':round(output_exclude.pvalues,3)})
print("BIC: ",output_exclude.bic)
print("R Squared: ",output_exclude.rsquared)