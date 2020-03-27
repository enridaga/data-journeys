
import pandas as pd
auto = pd.read_csv("../input/autompg-dataset/auto-mpg.csv")

auto.head(5)
#Check for datatypes and null values
auto.info()
#Observation - horsepower datatype is object need to dive deeper
import numpy as np
auto.horsepower.unique()
#Here hp has ? in the fields, we need to replace them
#Lets conver the unwanted symbol to nan
auto = auto.replace('?',np.nan)
auto.isna().sum()
#6 hp has missing values
#Need to conver hp to integer
auto['horsepower'] = pd.to_numeric(auto['horsepower'])
#Lets fill the na's with mean value
auto['horsepower'] = auto['horsepower'].fillna(auto['horsepower'].mean())
#Lets check the model year, Here for predicting milage model age will be needed than model year.
#Lets create a new feature
auto['model year'].unique()
import datetime
auto['model year'] = auto['model year']+1900 # To conver this to YYYY
auto.head(1)
auto['Age'] = (datetime.datetime.now().year)-auto['model year']
auto.head(1)
#Lets drop model 
auto.drop(['model year'],axis=1,inplace=True)
auto['car name'].unique()
#Car name also wont determine the milage we can drop it as well
auto.drop(['car name'],axis=1, inplace=True)
#At this point datacleaning is completed lets visualize the data now with MATPLOTLIB
#Datanow looks clean, lets perform some visualization
import matplotlib.pyplot as plt
plt.figure(figsize=(12,5))

plt.bar(auto['Age'],auto['mpg'])

plt.xlabel("Age")
plt.ylabel("Miles per galon")
plt.show()
# As age increase the MPG decreases
plt.scatter(auto['acceleration'],auto['mpg'])
plt.xlabel("acceleration")
plt.ylabel("Miles per galon")
#More the acceleration more the MPG
plt.scatter(auto['weight'],auto['mpg'])
plt.xlabel("weight")
plt.ylabel("Miles per galon")
# As weight increases milage decreases

auto.plot.scatter(x='weight',y='mpg')
plt.xlabel("weight")
plt.ylabel("Miles per galon")
plt.bar(auto['cylinders'],auto['mpg'])
plt.xlabel("cylinders")
plt.ylabel("Miles per galon")
#Not relation, can be droped
auto.drop(['cylinders','origin'],inplace=True, axis=1)
auto.head()
#Lets check the corelation
autocorr = auto.corr()
autocorr
# Lets plot this with heat map
import seaborn as sns
sns.heatmap(autocorr,annot=True)
#Lets apply the ML model
auto.shape
X = auto.drop('mpg',axis=1)
Y = auto['mpg']
#Lets apply simple linear regression to predict the age.
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(X,Y,test_size=0.2)
from sklearn.linear_model import LinearRegression
linear_model = LinearRegression(normalize=True).fit(x_train,y_train)
print("Trainign score: ",linear_model.score(x_train,y_train))
y_pred = linear_model.predict(x_test)
from sklearn.metrics import r2_score
print("test score : ", r2_score(y_test, y_pred))