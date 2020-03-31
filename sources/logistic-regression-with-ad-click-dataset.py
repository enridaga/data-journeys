
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
### matplotlib inline
ad_data = pd.read_csv('../input/advertising/advertising.csv')
ad_data.head()
ad_data.columns # displays column names

ad_data.info()
ad_data.describe()
sns.set_style('whitegrid')
ad_data['Age'].hist(bins=30)
plt.xlabel('Age')
pd.crosstab(ad_data['Country'], ad_data['Clicked on Ad']).sort_values( 1,ascending = False).tail(10)

ad_data[ad_data['Clicked on Ad']==1]['Country'].value_counts().head(10)
ad_data['Country'].value_counts().head(10)
pd.crosstab(index=ad_data['Country'],columns='count').sort_values(['count'], ascending=False).head(10)
ad_data.isnull().sum()

type(ad_data['Timestamp'][1])
# Extract datetime variables using timestamp column
ad_data['Timestamp'] = pd.to_datetime(ad_data['Timestamp']) 

# Converting timestamp column into datatime object in order to extract new features
ad_data['Month'] = ad_data['Timestamp'].dt.month 

# Creates a new column called Month
ad_data['Day'] = ad_data['Timestamp'].dt.day     

# Creates a new column called Day
ad_data['Hour'] = ad_data['Timestamp'].dt.hour   

# Creates a new column called Hour
ad_data["Weekday"] = ad_data['Timestamp'].dt.dayofweek 

# Dropping timestamp column to avoid redundancy
ad_data = ad_data.drop(['Timestamp'], axis=1) # deleting timestamp
# ad_data['Hour'] = ad_data['timeStamp'].apply(lambda time: time.hour)
# ad_data['Month'] = ad_data['timeStamp'].apply(lambda time: time.month)
# ad_data['Day of Week'] = ad_data['timeStamp'].apply(lambda time: time.dayofweek)
ad_data.head()
sns.countplot(x = 'Clicked on Ad', data = ad_data)
# Jointplot of daily time spent on site and age 
sns.jointplot(x = "Age", y= "Daily Time Spent on Site", data = ad_data) 
# scatterplot of daily time spent on site and age with clicking ads as hue
sns.scatterplot(x = "Age", y= "Daily Time Spent on Site",hue='Clicked on Ad', data = ad_data) 
# Jointplot of daily time spent on site and age clicking ads as hue
sns.lmplot(x = "Age", y= "Daily Time Spent on Site",hue='Clicked on Ad', data = ad_data) 
# Creating a pairplot with hue defined by Clicked on Ad column
sns.pairplot(ad_data, hue = 'Clicked on Ad', vars = ['Daily Time Spent on Site', 'Age', 'Area Income', 'Daily Internet Usage'],palette = 'rocket')
plots = ['Daily Time Spent on Site', 'Age', 'Area Income','Daily Internet Usage']
for i in plots:
    plt.figure(figsize = (12, 6))
    
    plt.subplot(2,3,1)
    sns.boxplot(data= ad_data, y=ad_data[i],x='Clicked on Ad')
    plt.subplot(2,3,2)
    sns.boxplot(data= ad_data, y=ad_data[i])
    plt.subplot(2,3,3)
    sns.distplot(ad_data[i],bins= 20,)       
    plt.tight_layout()
    plt.title(i)    
    plt.show()
    
print('oldest person didn\'t clicked on the ad was of was of:', ad_data['Age'].max(), 'Years')
print('oldest person who clicked on the ad was of:', ad_data[ad_data['Clicked on Ad']==0]['Age'].max(), 'Years')
print('Youngest person was of:', ad_data['Age'].min(), 'Years')
print('Youngest person who clicked on the ad was of:', ad_data[ad_data['Clicked on Ad']==0]['Age'].min(), 'Years')

print('Average age was of:', ad_data['Age'].mean(), 'Years')
fig = plt.figure(figsize = (12,10))
sns.heatmap(ad_data.corr(), cmap='viridis', annot = True) 
# Degree of relationship i.e correlation using heatmap
f,ax=plt.subplots(1,2,figsize=(14,5))
ad_data['Month'][ad_data['Clicked on Ad']==1].value_counts().sort_index().plot(ax=ax[0])
ax[0].set_ylabel('Count of Clicks')
pd.crosstab(ad_data["Clicked on Ad"], ad_data["Month"]).T.plot(kind = 'Bar',ax=ax[1])
#ad_data.groupby(['Month'])['Clicked on Ad'].sum() 
plt.tight_layout()
plt.suptitle('Months Vs Clicks',y=0,size=20)
plt.show()
f,ax=plt.subplots(1,2,figsize=(14,5))
pd.crosstab(ad_data["Clicked on Ad"], ad_data["Hour"]).T.plot(style = [], ax = ax[0])
pd.pivot_table(ad_data, index = ['Weekday'], values = ['Clicked on Ad'],aggfunc= np.sum).plot(kind = 'Bar', ax=ax[1]) # 0 - Monday
plt.tight_layout()
plt.show()
from sklearn.model_selection import train_test_split
X = ad_data[['Daily Time Spent on Site', 'Age', 'Area Income','Daily Internet Usage', 'Male']]
y = ad_data['Clicked on Ad']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=101)
print(X_train.shape, y_train.shape)
print(X_test.shape, y_test.shape)
from sklearn.linear_model import LogisticRegression
logmodel = LogisticRegression(solver='lbfgs')
logmodel.fit(X_train,y_train)
predictions = logmodel.predict(X_test)
from sklearn.metrics import classification_report
print(classification_report(y_test,predictions))
# Importing a pure confusion matrix from sklearn.metrics family
from sklearn.metrics import confusion_matrix

# Printing the confusion_matrix
print(confusion_matrix(y_test, predictions))
logmodel.coef_
