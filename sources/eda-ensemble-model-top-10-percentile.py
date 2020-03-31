
import pylab

import calendar

import numpy as np

import pandas as pd

import seaborn as sn

from scipy import stats

import missingno as msno

from datetime import datetime

import matplotlib.pyplot as plt

import warnings

pd.options.mode.chained_assignment = None

warnings.filterwarnings("ignore", category=DeprecationWarning)

### matplotlib inline
dailyData = pd.read_csv("../input/train.csv")
dailyData.shape
dailyData.head(2)
dailyData.dtypes
dailyData["date"] = dailyData.datetime.apply(lambda x : x.split()[0])

dailyData["hour"] = dailyData.datetime.apply(lambda x : x.split()[1].split(":")[0])

dailyData["weekday"] = dailyData.date.apply(lambda dateString : calendar.day_name[datetime.strptime(dateString,"%Y-%m-%d").weekday()])

dailyData["month"] = dailyData.date.apply(lambda dateString : calendar.month_name[datetime.strptime(dateString,"%Y-%m-%d").month])

dailyData["season"] = dailyData.season.map({1: "Spring", 2 : "Summer", 3 : "Fall", 4 :"Winter" })

dailyData["weather"] = dailyData.weather.map({1: " Clear + Few clouds + Partly cloudy + Partly cloudy",\

                                        2 : " Mist + Cloudy, Mist + Broken clouds, Mist + Few clouds, Mist ", \

                                        3 : " Light Snow, Light Rain + Thunderstorm + Scattered clouds, Light Rain + Scattered clouds", \

                                        4 :" Heavy Rain + Ice Pallets + Thunderstorm + Mist, Snow + Fog " })
categoryVariableList = ["hour","weekday","month","season","weather","holiday","workingday"]

for var in categoryVariableList:

    dailyData[var] = dailyData[var].astype("category")
dailyData  = dailyData.drop(["datetime"],axis=1)
dataTypeDf = pd.DataFrame(dailyData.dtypes.value_counts()).reset_index().rename(columns={"index":"variableType",0:"count"})

fig,ax = plt.subplots()

fig.set_size_inches(12,5)

sn.barplot(data=dataTypeDf,x="variableType",y="count",ax=ax)

ax.set(xlabel='variableTypeariable Type', ylabel='Count',title="Variables DataType Count")
msno.matrix(dailyData,figsize=(12,5))
fig, axes = plt.subplots(nrows=2,ncols=2)

fig.set_size_inches(12, 10)

sn.boxplot(data=dailyData,y="count",orient="v",ax=axes[0][0])

sn.boxplot(data=dailyData,y="count",x="season",orient="v",ax=axes[0][1])

sn.boxplot(data=dailyData,y="count",x="hour",orient="v",ax=axes[1][0])

sn.boxplot(data=dailyData,y="count",x="workingday",orient="v",ax=axes[1][1])



axes[0][0].set(ylabel='Count',title="Box Plot On Count")

axes[0][1].set(xlabel='Season', ylabel='Count',title="Box Plot On Count Across Season")

axes[1][0].set(xlabel='Hour Of The Day', ylabel='Count',title="Box Plot On Count Across Hour Of The Day")

axes[1][1].set(xlabel='Working Day', ylabel='Count',title="Box Plot On Count Across Working Day")
dailyDataWithoutOutliers = dailyData[np.abs(dailyData["count"]-dailyData["count"].mean())<=(3*dailyData["count"].std())] 
print ("Shape Of The Before Ouliers: ",dailyData.shape)

print ("Shape Of The After Ouliers: ",dailyDataWithoutOutliers.shape)
corrMatt = dailyData[["temp","atemp","casual","registered","humidity","windspeed","count"]].corr()

mask = np.array(corrMatt)

mask[np.tril_indices_from(mask)] = False

fig,ax= plt.subplots()

fig.set_size_inches(20,10)

sn.heatmap(corrMatt, mask=mask,vmax=.8, square=True,annot=True)
fig,(ax1,ax2,ax3) = plt.subplots(ncols=3)

fig.set_size_inches(12, 5)

sn.regplot(x="temp", y="count", data=dailyData,ax=ax1)

sn.regplot(x="windspeed", y="count", data=dailyData,ax=ax2)

sn.regplot(x="humidity", y="count", data=dailyData,ax=ax3)
fig,axes = plt.subplots(ncols=2,nrows=2)

fig.set_size_inches(12, 10)

sn.distplot(dailyData["count"],ax=axes[0][0])

stats.probplot(dailyData["count"], dist='norm', fit=True, plot=axes[0][1])

sn.distplot(np.log(dailyDataWithoutOutliers["count"]),ax=axes[1][0])

stats.probplot(np.log1p(dailyDataWithoutOutliers["count"]), dist='norm', fit=True, plot=axes[1][1])
fig,(ax1,ax2,ax3,ax4)= plt.subplots(nrows=4)

fig.set_size_inches(12,20)

sortOrder = ["January","February","March","April","May","June","July","August","September","October","November","December"]

hueOrder = ["Sunday","Monday","Tuesday","Wednesday","Thursday","Friday","Saturday"]



monthAggregated = pd.DataFrame(dailyData.groupby("month")["count"].mean()).reset_index()

monthSorted = monthAggregated.sort_values(by="count",ascending=False)

sn.barplot(data=monthSorted,x="month",y="count",ax=ax1,order=sortOrder)

ax1.set(xlabel='Month', ylabel='Avearage Count',title="Average Count By Month")



hourAggregated = pd.DataFrame(dailyData.groupby(["hour","season"],sort=True)["count"].mean()).reset_index()

sn.pointplot(x=hourAggregated["hour"], y=hourAggregated["count"],hue=hourAggregated["season"], data=hourAggregated, join=True,ax=ax2)

ax2.set(xlabel='Hour Of The Day', ylabel='Users Count',title="Average Users Count By Hour Of The Day Across Season",label='big')



hourAggregated = pd.DataFrame(dailyData.groupby(["hour","weekday"],sort=True)["count"].mean()).reset_index()

sn.pointplot(x=hourAggregated["hour"], y=hourAggregated["count"],hue=hourAggregated["weekday"],hue_order=hueOrder, data=hourAggregated, join=True,ax=ax3)

ax3.set(xlabel='Hour Of The Day', ylabel='Users Count',title="Average Users Count By Hour Of The Day Across Weekdays",label='big')



hourTransformed = pd.melt(dailyData[["hour","casual","registered"]], id_vars=['hour'], value_vars=['casual', 'registered'])

hourAggregated = pd.DataFrame(hourTransformed.groupby(["hour","variable"],sort=True)["value"].mean()).reset_index()

sn.pointplot(x=hourAggregated["hour"], y=hourAggregated["value"],hue=hourAggregated["variable"],hue_order=["casual","registered"], data=hourAggregated, join=True,ax=ax4)

ax4.set(xlabel='Hour Of The Day', ylabel='Users Count',title="Average Users Count By Hour Of The Day Across User Type",label='big')
dataTrain = pd.read_csv("../input/train.csv")

dataTest = pd.read_csv("../input/test.csv")
data = dataTrain.append(dataTest)

data.reset_index(inplace=True)

data.drop('index',inplace=True,axis=1)
data["date"] = data.datetime.apply(lambda x : x.split()[0])

data["hour"] = data.datetime.apply(lambda x : x.split()[1].split(":")[0]).astype("int")

data["year"] = data.datetime.apply(lambda x : x.split()[0].split("-")[0])

data["weekday"] = data.date.apply(lambda dateString : datetime.strptime(dateString,"%Y-%m-%d").weekday())

data["month"] = data.date.apply(lambda dateString : datetime.strptime(dateString,"%Y-%m-%d").month)
from sklearn.ensemble import RandomForestRegressor



dataWind0 = data[data["windspeed"]==0]

dataWindNot0 = data[data["windspeed"]!=0]

rfModel_wind = RandomForestRegressor()

windColumns = ["season","weather","humidity","month","temp","year","atemp"]

rfModel_wind.fit(dataWindNot0[windColumns], dataWindNot0["windspeed"])



wind0Values = rfModel_wind.predict(X= dataWind0[windColumns])

dataWind0["windspeed"] = wind0Values

data = dataWindNot0.append(dataWind0)

data.reset_index(inplace=True)

data.drop('index',inplace=True,axis=1)
categoricalFeatureNames = ["season","holiday","workingday","weather","weekday","month","year","hour"]

numericalFeatureNames = ["temp","humidity","windspeed","atemp"]

dropFeatures = ['casual',"count","datetime","date","registered"]
for var in categoricalFeatureNames:

    data[var] = data[var].astype("category")
dataTrain = data[pd.notnull(data['count'])].sort_values(by=["datetime"])

dataTest = data[~pd.notnull(data['count'])].sort_values(by=["datetime"])

datetimecol = dataTest["datetime"]

yLabels = dataTrain["count"]

yLablesRegistered = dataTrain["registered"]

yLablesCasual = dataTrain["casual"]
dataTrain  = dataTrain.drop(dropFeatures,axis=1)

dataTest  = dataTest.drop(dropFeatures,axis=1)
def rmsle(y, y_,convertExp=True):

    if convertExp:

        y = np.exp(y),

        y_ = np.exp(y_)

    log1 = np.nan_to_num(np.array([np.log(v + 1) for v in y]))

    log2 = np.nan_to_num(np.array([np.log(v + 1) for v in y_]))

    calc = (log1 - log2) ** 2

    return np.sqrt(np.mean(calc))
from sklearn.linear_model import LinearRegression,Ridge,Lasso

from sklearn.model_selection import GridSearchCV

from sklearn import metrics

import warnings

pd.options.mode.chained_assignment = None

warnings.filterwarnings("ignore", category=DeprecationWarning)



# Initialize logistic regression model

lModel = LinearRegression()



# Train the model

yLabelsLog = np.log1p(yLabels)

lModel.fit(X = dataTrain,y = yLabelsLog)



# Make predictions

preds = lModel.predict(X= dataTrain)

print ("RMSLE Value For Linear Regression: ",rmsle(np.exp(yLabelsLog),np.exp(preds),False))
ridge_m_ = Ridge()

ridge_params_ = { 'max_iter':[3000],'alpha':[0.1, 1, 2, 3, 4, 10, 30,100,200,300,400,800,900,1000]}

rmsle_scorer = metrics.make_scorer(rmsle, greater_is_better=False)

grid_ridge_m = GridSearchCV( ridge_m_,

                          ridge_params_,

                          scoring = rmsle_scorer,

                          cv=5)

yLabelsLog = np.log1p(yLabels)

grid_ridge_m.fit( dataTrain, yLabelsLog )

preds = grid_ridge_m.predict(X= dataTrain)

print (grid_ridge_m.best_params_)

print ("RMSLE Value For Ridge Regression: ",rmsle(np.exp(yLabelsLog),np.exp(preds),False))



fig,ax= plt.subplots()

fig.set_size_inches(12,5)

df = pd.DataFrame(grid_ridge_m.grid_scores_)

df["alpha"] = df["parameters"].apply(lambda x:x["alpha"])

df["rmsle"] = df["mean_validation_score"].apply(lambda x:-x)

sn.pointplot(data=df,x="alpha",y="rmsle",ax=ax)
lasso_m_ = Lasso()



alpha  = 1/np.array([0.1, 1, 2, 3, 4, 10, 30,100,200,300,400,800,900,1000])

lasso_params_ = { 'max_iter':[3000],'alpha':alpha}



grid_lasso_m = GridSearchCV( lasso_m_,lasso_params_,scoring = rmsle_scorer,cv=5)

yLabelsLog = np.log1p(yLabels)

grid_lasso_m.fit( dataTrain, yLabelsLog )

preds = grid_lasso_m.predict(X= dataTrain)

print (grid_lasso_m.best_params_)

print ("RMSLE Value For Lasso Regression: ",rmsle(np.exp(yLabelsLog),np.exp(preds),False))



fig,ax= plt.subplots()

fig.set_size_inches(12,5)

df = pd.DataFrame(grid_lasso_m.grid_scores_)

df["alpha"] = df["parameters"].apply(lambda x:x["alpha"])

df["rmsle"] = df["mean_validation_score"].apply(lambda x:-x)

sn.pointplot(data=df,x="alpha",y="rmsle",ax=ax)
from sklearn.ensemble import RandomForestRegressor

rfModel = RandomForestRegressor(n_estimators=100)

yLabelsLog = np.log1p(yLabels)

rfModel.fit(dataTrain,yLabelsLog)

preds = rfModel.predict(X= dataTrain)

print ("RMSLE Value For Random Forest: ",rmsle(np.exp(yLabelsLog),np.exp(preds),False))
from sklearn.ensemble import GradientBoostingRegressor

gbm = GradientBoostingRegressor(n_estimators=4000,alpha=0.01); ### Test 0.41

yLabelsLog = np.log1p(yLabels)

gbm.fit(dataTrain,yLabelsLog)

preds = gbm.predict(X= dataTrain)

print ("RMSLE Value For Gradient Boost: ",rmsle(np.exp(yLabelsLog),np.exp(preds),False))
predsTest = gbm.predict(X= dataTest)

fig,(ax1,ax2)= plt.subplots(ncols=2)

fig.set_size_inches(12,5)

sn.distplot(yLabels,ax=ax1,bins=50)

sn.distplot(np.exp(predsTest),ax=ax2,bins=50)
submission = pd.DataFrame({

        "datetime": datetimecol,

        "count": [max(0, x) for x in np.exp(predsTest)]

    })

submission.to_csv('bike_predictions_gbm_separate_without_fe.csv', index=False)
