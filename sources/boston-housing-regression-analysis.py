
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
df = pd.read_csv("/kaggle/input/boston-housing/Boston Housing.csv")
df.head()
df.describe()
df = df.drop(["ZN", "Chas"], axis = 1)
df.describe()
#checking columns / rows
df.shape
#lets see if we have any null values in our data
df.isnull().sum()
#moving to our next step to treat Outliers
#lets visualize the data through box plot
for i in df.columns:
  sns.boxplot(y=i, data=df)
  plt.tight_layout(pad=0.4)
  plt.show()

df2 = df.copy()
#Lets check the outliers using for loop and removing the outliers
for i in df2.columns:
  df2.sort_values(by=i, ascending=True, na_position='last') #sorting is required before percentile
  q1, q3 = np.percentile(df2[i], [25,75])
  iqr = q3-q1
  upper_bound = q3+(1.5 * iqr)
  lower_bound = q1-(1.5 * iqr)
  mean = df2[i].mean()
  df2.loc[df2[i]< lower_bound, [i]] = mean
  df2.loc[df2[i]> upper_bound, [i]] = mean
df2.shape
df2.describe()
#importing dataset
X = df2.iloc[:, :-1] #independent variable
y = df2.iloc[:, 11] #dependent variable
import statsmodels.api as sm
X = df2.iloc[:, :-1] #independent variable
y = df2.iloc[:, 11] #dependent variable
X = sm.add_constant(X) # adding a constant

model = sm.OLS(y, X).fit()
predictions = model.predict(X) 

print_model = model.summary()
print(print_model)
#lets make the correlation matrix
corr_data = df2.corr()
corr_data.style.background_gradient(cmap="coolwarm")
#as per the above correlation matrix we can see that Tax & RAD are highly correlated,
#as per the observation of data Rad is more important variable in predicting the Medv so I am dropping Tax here
df2 = df2.drop(["Tax"], axis = 1)
df2.shape
#lets make the correlation matrix
corr_data = df2.corr()
corr_data.style.background_gradient(cmap="coolwarm")
#now after dropping TAX we have noticed that CRIM and RAD are highly correlated and as per the observation of data it is suggested to drop RAD rather than CRIM in respect to MEDV
df2 = df2.drop(["Rad"], axis=1)
#lets make the correlation matrix again
corr_data = df2.corr()
corr_data.style.background_gradient(cmap="coolwarm")
#now after dropping RAD we have noticed that NOX and INDUS are highly correlated and as per the observation of data it is suggested to drop INDUS rather than NOX in respect to MEDV
df2 = df2.drop(["Nox"], axis=1)
#lets make the correlation matrix again
corr_data = df2.corr()
corr_data.style.background_gradient(cmap="coolwarm")
#till now we have removed the highly correlated data that might impact our predictions of MEDV. 
#Lets see the correlation of MEDV with other variables and remove the less correlated variables (using pearson method here)
from scipy.stats import pearsonr
for i in df2.columns:
  corr, p_val = pearsonr(df2[i], df2["Medv"])
  print (i, corr)
#from above pearson methond I have concluded that B is least correlated and have least impact on MEDV so removing same
df2 = df2.drop(["B"], axis= 1)
df2.describe()
#splitting data into train test
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size = 0.2, random_state = 0)
from sklearn.linear_model import LinearRegression
lr = LinearRegression()
lr.fit(X_train, y_train)
y_pred = lr.predict(X_test)
y_compare = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})
from sklearn.model_selection import cross_val_score as cvs
accuracy = cvs(lr, X_train, y_train, scoring='r2', cv=5)
print (accuracy.mean())
y_compare.head() #Comparison b/w Actual & Predicted
#fitting polynomial regression.....WHEN WE USE POLYNOMIAL REGRESSION WE HAVE TO FIT DATASET IN LINEAR REGRESSION FIRST
from sklearn.preprocessing import PolynomialFeatures
polyReg = PolynomialFeatures(degree=2, include_bias=False)
X_train_poly = polyReg.fit_transform(X_train)
X_test_poly = polyReg.fit_transform(X_test)
poly = LinearRegression()
poly.fit(X_train_poly, y_train)
y_pred = poly.predict(X_test_poly)
y_compare_poly = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})
accuracy = cvs(poly, X_train_poly, y_train, scoring='r2', cv=5)
print (accuracy.mean())
y_compare.head()
from sklearn.svm import SVR
svr = SVR (kernel = 'rbf', gamma = 'scale')
svr.fit(X_train, y_train)
y_pred = svr.predict(X_test)
y_compare_svr = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})
accuracy = cvs(svr, X_train, y_train, scoring='r2', cv=5)
print (accuracy.mean())
y_compare.head()
from sklearn.tree import DecisionTreeRegressor
dt = DecisionTreeRegressor (random_state = 0)
dt.fit(X_train, y_train)
y_pred = dt.predict(X_test)
y_compare_dt = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})
accuracy = cvs(dt, X_train, y_train, scoring='r2', cv=5)
print (accuracy.mean())
y_compare_dt.head()
from sklearn.ensemble import RandomForestRegressor
RF = RandomForestRegressor(n_estimators = 160, random_state = 0)
RF.fit(X_train, y_train)
y_pred = dt.predict(X_test)
y_compare_RF = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})
accuracy = cvs(RF, X_train, y_train, scoring='r2', cv=5)
print (accuracy.mean())
y_compare_RF.head()
from sklearn.neighbors import KNeighborsRegressor
KNN = KNeighborsRegressor(n_neighbors = 4)
KNN.fit(X_train, y_train)
y_pred = KNN.predict(X_test)
y_compare_KNN = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})
accuracy = cvs(KNN, X_train, y_train, scoring='r2', cv=5)
print (accuracy.mean())
y_compare_KNN.head()
fig, ax = plt.subplots(nrows=1, ncols=6, figsize=(25,4))
ax = ax.flatten()
y_compare.head(10).plot(kind='bar', title='Linear Regression', grid='True', ax=ax[0])
y_compare_dt.head(10).plot(kind='bar', title='Decission Tree', grid='True', ax=ax[1])
y_compare_KNN.head(10).plot(kind='bar', title='KNN', grid='True', ax=ax[2])
y_compare_RF.head(10).plot(kind='bar', title='Random Forest', grid='True', ax=ax[3])
y_compare_svr.head(10).plot(kind='bar', title='SVR', grid='True', ax=ax[4])
y_compare_poly.head(10).plot(kind='bar', title='Poly', grid='True', ax=ax[5])
print('According to R squared scorring method we got below scores for out machine learning models:')
modelNames = ['Linear', 'Polynomial', 'Support Vector', 'Random Forrest', 'K-Nearest Neighbour', 'Decission Tree']
modelRegressors = [lr, poly, svr, RF, KNN, dt]
models = pd.DataFrame({'modelNames' : modelNames, 'modelRegressors' : modelRegressors})
counter=0
score=[]
for i in models['modelRegressors']:
  if i is poly:
    accuracy = cvs(i, X_train_poly, y_train, scoring='r2', cv=5)
    print('Accuracy of %s Regression model is %.2f' %(models.iloc[counter,0],accuracy.mean()))
    score.append(accuracy.mean())
  else:
    accuracy = cvs(i, X_train, y_train, scoring='r2', cv=5)
    print('Accuracy of %s Regression model is %.2f' %(models.iloc[counter,0],accuracy.mean()))
    score.append(accuracy.mean())
  counter+=1
print('According to Mean Absolute Error scorring method we got below scores for out machine learning models:')
modelNames = ['Linear', 'Polynomial', 'Support Vector', 'Random Forrest', 'K-Nearest Neighbour', 'Decission Tree']
modelRegressors = [lr, poly, svr, RF, KNN, dt]
models = pd.DataFrame({'modelNames' : modelNames, 'modelRegressors' : modelRegressors})
counter=0
score=[]
for i in models['modelRegressors']:
  if i is poly:
    accuracy = cvs(i, X_train_poly, y_train, scoring='neg_mean_absolute_error', cv=5)
    print('Accuracy of %s Regression model is %.2f' %(models.iloc[counter,0],accuracy.mean()))
    score.append(accuracy.mean())
  else:
    accuracy = cvs(i, X_train, y_train, scoring='neg_mean_absolute_error', cv=5)
    print('Accuracy of %s Regression model is %.2f' %(models.iloc[counter,0],accuracy.mean()))
    score.append(accuracy.mean())
  counter+=1
print('According to Mean Squared Error scorring method we got below scores for out machine learning models:')
modelNames = ['Linear', 'Polynomial', 'Support Vector', 'Random Forrest', 'K-Nearest Neighbour', 'Decission Tree']
modelRegressors = [lr, poly, svr, RF, KNN, dt]
models = pd.DataFrame({'modelNames' : modelNames, 'modelRegressors' : modelRegressors})
counter=0
score=[]
for i in models['modelRegressors']:
  if i is poly:
    accuracy = cvs(i, X_train_poly, y_train, scoring='neg_mean_squared_error', cv=5)
    print('Accuracy of %s Regression model is %.2f' %(models.iloc[counter,0],accuracy.mean()))
    score.append(accuracy.mean())
  else:
    accuracy = cvs(i, X_train, y_train, scoring='neg_mean_squared_error', cv=5)
    print('Accuracy of %s Regression model is %.2f' %(models.iloc[counter,0],accuracy.mean()))
    score.append(accuracy.mean())
  counter+=1