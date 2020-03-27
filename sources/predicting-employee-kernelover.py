
# Import the neccessary modules for data manipulation and visual representation

import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import matplotlib as matplot

import seaborn as sns

#%matplotlib inline
#Read the analytics csv file and store our dataset into a dataframe called "df"

df = pd.DataFrame.from_csv('../input/HR_comma_sep.csv', index_col=None)
# Check to see if there are any missing values in our data set

df.isnull().any()
# Get a quick overview of what we are dealing with in our dataset

df.head()
# Renaming certain columns for better readability

df = df.rename(columns={'satisfaction_level': 'satisfaction', 

                        'last_evaluation': 'evaluation',

                        'number_project': 'projectCount',

                        'average_montly_hours': 'averageMonthlyHours',

                        'time_spend_company': 'yearsAtCompany',

                        'Work_accident': 'workAccident',

                        'promotion_last_5years': 'promotion',

                        'sales' : 'department',

                        'left' : 'turnover'

                        })
# Move the reponse variable "turnover" to the front of the table

front = df['turnover']

df.drop(labels=['turnover'], axis=1,inplace = True)

df.insert(0, 'turnover', front)

df.head()
# The dataset contains 10 columns and 14999 observations

df.shape
# Check the type of our features. 

df.dtypes
# Looks like about 76% of employees stayed and 24% of employees left. 

# NOTE: When performing cross validation, its important to maintain this turnover ratio

turnover_rate = df.turnover.value_counts() / len(df)

turnover_rate
# Display the statistical overview of the employees

df.describe()
# Overview of summary (Turnover V.S. Non-turnover)

turnover_Summary = df.groupby('turnover')

turnover_Summary.mean()
#Correlation Matrix

corr = df.corr()

corr = (corr)

sns.heatmap(corr, 

            xticklabels=corr.columns.values,

            yticklabels=corr.columns.values)



corr
# Let's compare the means of our employee turnover satisfaction against the employee population satisfaction

#emp_population = df['satisfaction'].mean()

emp_population = df['satisfaction'][df['turnover'] == 0].mean()

emp_turnover_satisfaction = df[df['turnover']==1]['satisfaction'].mean()



print( 'The mean satisfaction for the employee population with no turnover is: ' + str(emp_population))

print( 'The mean satisfaction for employees that had a turnover is: ' + str(emp_turnover_satisfaction) )
import scipy.stats as stats

stats.ttest_1samp(a=  df[df['turnover']==1]['satisfaction'], # Sample of Employee satisfaction who had a Turnover

                  popmean = emp_population)  # Employee Who Had No Turnover satisfaction mean
degree_freedom = len(df[df['turnover']==1])



LQ = stats.t.ppf(0.025,degree_freedom)  # Left Quartile



RQ = stats.t.ppf(0.975,degree_freedom)  # Right Quartile



print ('The t-distribution left quartile range is: ' + str(LQ))

print ('The t-distribution right quartile range is: ' + str(RQ))

# Set up the matplotlib figure

f, axes = plt.subplots(ncols=3, figsize=(15, 6))



# Graph Employee Satisfaction

sns.distplot(df.satisfaction, kde=False, color="g", ax=axes[0]).set_title('Employee Satisfaction Distribution')

axes[0].set_ylabel('Employee Count')



# Graph Employee Evaluation

sns.distplot(df.evaluation, kde=False, color="r", ax=axes[1]).set_title('Employee Evaluation Distribution')

axes[1].set_ylabel('Employee Count')



# Graph Employee Average Monthly Hours

sns.distplot(df.averageMonthlyHours, kde=False, color="b", ax=axes[2]).set_title('Employee Average Monthly Hours Distribution')

axes[2].set_ylabel('Employee Count')
f, ax = plt.subplots(figsize=(15, 4))

sns.countplot(y="salary", hue='turnover', data=df).set_title('Employee Salary Turnover Distribution');
# Employee distri

# Types of colors

color_types = ['#78C850','#F08030','#6890F0','#A8B820','#A8A878','#A040A0','#F8D030',  

                '#E0C068','#EE99AC','#C03028','#F85888','#B8A038','#705898','#98D8D8','#7038F8']



# Count Plot (a.k.a. Bar Plot)

sns.countplot(x='department', data=df, palette=color_types).set_title('Employee Department Distribution');

 

# Rotate x-labels

plt.xticks(rotation=-45)
f, ax = plt.subplots(figsize=(15, 5))

sns.countplot(y="department", hue='turnover', data=df).set_title('Employee Department Turnover Distribution');
ax = sns.barplot(x="projectCount", y="projectCount", hue="turnover", data=df, estimator=lambda x: len(x) / len(df) * 100)

ax.set(ylabel="Percent")
# Kernel Density Plot

fig = plt.figure(figsize=(15,4),)

ax=sns.kdeplot(df.loc[(df['turnover'] == 0),'evaluation'] , color='b',shade=True,label='no turnover')

ax=sns.kdeplot(df.loc[(df['turnover'] == 1),'evaluation'] , color='r',shade=True, label='turnover')

ax.set(xlabel='Employee Evaluation', ylabel='Frequency')

plt.title('Employee Evaluation Distribution - Turnover V.S. No Turnover')
#KDEPlot: Kernel Density Estimate Plot

fig = plt.figure(figsize=(15,4))

ax=sns.kdeplot(df.loc[(df['turnover'] == 0),'averageMonthlyHours'] , color='b',shade=True, label='no turnover')

ax=sns.kdeplot(df.loc[(df['turnover'] == 1),'averageMonthlyHours'] , color='r',shade=True, label='turnover')

ax.set(xlabel='Employee Average Monthly Hours', ylabel='Frequency')

plt.title('Employee AverageMonthly Hours Distribution - Turnover V.S. No Turnover')
#KDEPlot: Kernel Density Estimate Plot

fig = plt.figure(figsize=(15,4))

ax=sns.kdeplot(df.loc[(df['turnover'] == 0),'satisfaction'] , color='b',shade=True, label='no turnover')

ax=sns.kdeplot(df.loc[(df['turnover'] == 1),'satisfaction'] , color='r',shade=True, label='turnover')

plt.title('Employee Satisfaction Distribution - Turnover V.S. No Turnover')
#ProjectCount VS AverageMonthlyHours [BOXPLOT]

#Looks like the average employees who stayed worked about 200hours/month. Those that had a turnover worked about 250hours/month and 150hours/month



import seaborn as sns

sns.boxplot(x="projectCount", y="averageMonthlyHours", hue="turnover", data=df)
#ProjectCount VS Evaluation

#Looks like employees who did not leave the company had an average evaluation of around 70% even with different projectCounts

#There is a huge skew in employees who had a turnover though. It drastically changes after 3 projectCounts. 

#Employees that had two projects and a horrible evaluation left. Employees with more than 3 projects and super high evaluations left

import seaborn as sns

sns.boxplot(x="projectCount", y="evaluation", hue="turnover", data=df)
sns.lmplot(x='satisfaction', y='evaluation', data=df,

           fit_reg=False, # No regression line

           hue='turnover')   # Color by evolution stage
ax = sns.barplot(x="yearsAtCompany", y="yearsAtCompany", hue="turnover", data=df, estimator=lambda x: len(x) / len(df) * 100)

ax.set(ylabel="Percent")
# Import KMeans Model

from sklearn.cluster import KMeans



# Graph and create 3 clusters of Employee Turnover

kmeans = KMeans(n_clusters=3,random_state=2)

kmeans.fit(df[df.turnover==1][["satisfaction","evaluation"]])



kmeans_colors = ['green' if c == 0 else 'blue' if c == 2 else 'red' for c in kmeans.labels_]



fig = plt.figure(figsize=(10, 6))

plt.scatter(x="satisfaction",y="evaluation", data=df[df.turnover==1],

            alpha=0.25,color = kmeans_colors)

plt.xlabel("Satisfaction")

plt.ylabel("Evaluation")

plt.scatter(x=kmeans.cluster_centers_[:,0],y=kmeans.cluster_centers_[:,1],color="black",marker="X",s=100)

plt.title("Clusters of Employee Turnover")

plt.show()
from sklearn import tree

from sklearn.tree import DecisionTreeClassifier

from sklearn.model_selection import train_test_split

plt.style.use('fivethirtyeight')

plt.rcParams['figure.figsize'] = (12,6)



# Renaming certain columns for better readability

df = df.rename(columns={'satisfaction_level': 'satisfaction', 

                        'last_evaluation': 'evaluation',

                        'number_project': 'projectCount',

                        'average_montly_hours': 'averageMonthlyHours',

                        'time_spend_company': 'yearsAtCompany',

                        'Work_accident': 'workAccident',

                        'promotion_last_5years': 'promotion',

                        'sales' : 'department',

                        'left' : 'turnover'

                        })



# Convert these variables into categorical variables

df["department"] = df["department"].astype('category').cat.codes

df["salary"] = df["salary"].astype('category').cat.codes



# Create train and test splits

target_name = 'turnover'

X = df.drop('turnover', axis=1)





y=df[target_name]



X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.15, random_state=123, stratify=y)



dtree = tree.DecisionTreeClassifier(

    #max_depth=3,

    class_weight="balanced",

    min_weight_fraction_leaf=0.01

    )

dtree = dtree.fit(X_train,y_train)



## plot the importances ##

importances = dtree.feature_importances_

feat_names = df.drop(['turnover'],axis=1).columns





indices = np.argsort(importances)[::-1]

plt.figure(figsize=(12,6))

plt.title("Feature importances by DecisionTreeClassifier")

plt.bar(range(len(indices)), importances[indices], color='lightblue',  align="center")

plt.step(range(len(indices)), np.cumsum(importances[indices]), where='mid', label='Cumulative')

plt.xticks(range(len(indices)), feat_names[indices], rotation='vertical',fontsize=14)

plt.xlim([-1, len(indices)])

plt.show()
# Import the neccessary modules for data manipulation and visual representation

import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import matplotlib as matplot

import seaborn as sns

#%matplotlib inline

#Read the analytics csv file and store our dataset into a dataframe called "df"

from sklearn.linear_model import LogisticRegression

from sklearn.preprocessing import LabelEncoder

from sklearn.model_selection import train_test_split

from sklearn.metrics import accuracy_score, classification_report, precision_score, recall_score, confusion_matrix, precision_recall_curve

from sklearn.preprocessing import RobustScaler

df = pd.DataFrame.from_csv('../input/HR_comma_sep.csv', index_col=None)



# Renaming certain columns for better readability

df = df.rename(columns={'satisfaction_level': 'satisfaction', 

                        'last_evaluation': 'evaluation',

                        'number_project': 'projectCount',

                        'average_montly_hours': 'averageMonthlyHours',

                        'time_spend_company': 'yearsAtCompany',

                        'Work_accident': 'workAccident',

                        'promotion_last_5years': 'promotion',

                        'sales' : 'department',

                        'left' : 'turnover'

                        })



# Convert these variables into categorical variables

df["department"] = df["department"].astype('category').cat.codes

df["salary"] = df["salary"].astype('category').cat.codes





# Move the reponse variable "turnover" to the front of the table

front = df['turnover']

df.drop(labels=['turnover'], axis=1,inplace = True)

df.insert(0, 'turnover', front)



# Create an intercept term for the logistic regression equation

df['int'] = 1

indep_var = ['satisfaction', 'evaluation', 'yearsAtCompany', 'int', 'turnover']

df = df[indep_var]



# Create train and test splits

target_name = 'turnover'

X = df.drop('turnover', axis=1)



y=df[target_name]

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.15, random_state=123, stratify=y)



X_train.head()

#
import statsmodels.api as sm

iv = ['satisfaction','evaluation','yearsAtCompany', 'int']

logReg = sm.Logit(y_train, X_train[iv])

answer = logReg.fit()



answer.summary

answer.params
# Create function to compute coefficients

coef = answer.params

def y (coef, Satisfaction, Evaluation, YearsAtCompany) : 

    return coef[3] + coef[0]*Satisfaction + coef[1]*Evaluation + coef[2]*YearsAtCompany



import numpy as np



# An Employee with 0.7 Satisfaction and 0.8 Evaluation and worked 3 years has a 14% chance of turnover

y1 = y(coef, 0.7, 0.8, 3)

p = np.exp(y1) / (1+np.exp(y1))

p
from sklearn.linear_model import LogisticRegression

from sklearn.preprocessing import LabelEncoder

from sklearn.model_selection import train_test_split

from sklearn.metrics import accuracy_score, classification_report, precision_score, recall_score, confusion_matrix, precision_recall_curve

from sklearn.preprocessing import RobustScaler

# Create base rate model

def base_rate_model(X) :

    y = np.zeros(X.shape[0])

    return y
# Create train and test splits

target_name = 'turnover'

X = df.drop('turnover', axis=1)

#robust_scaler = RobustScaler()

#X = robust_scaler.fit_transform(X)

y=df[target_name]

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.15, random_state=123, stratify=y)
# Check accuracy of base rate model

y_base_rate = base_rate_model(X_test)

from sklearn.metrics import accuracy_score

print ("Base rate accuracy is %2.2f" % accuracy_score(y_test, y_base_rate))
# Check accuracy of Logistic Model

from sklearn.linear_model import LogisticRegression

model = LogisticRegression(penalty='l2', C=1)



model.fit(X_train, y_train)

print ("Logistic accuracy is %2.2f" % accuracy_score(y_test, model.predict(X_test)))
# Using 10 fold Cross-Validation to train our Logistic Regression Model

from sklearn import model_selection

from sklearn.linear_model import LogisticRegression

kfold = model_selection.KFold(n_splits=10, random_state=7)

modelCV = LogisticRegression(class_weight = "balanced")

scoring = 'roc_auc'

results = model_selection.cross_val_score(modelCV, X_train, y_train, cv=kfold, scoring=scoring)

print("AUC: %.3f (%.3f)" % (results.mean(), results.std()))
# Compare the Logistic Regression Model V.S. Base Rate Model V.S. Random Forest Model

from sklearn.metrics import roc_auc_score

from sklearn.metrics import classification_report

from sklearn.ensemble import RandomForestClassifier



from sklearn import tree

from sklearn.tree import DecisionTreeClassifier

from sklearn.linear_model import LogisticRegression

from sklearn.ensemble import ExtraTreesClassifier

from sklearn.ensemble import BaggingClassifier

from sklearn.ensemble import AdaBoostClassifier

from sklearn.ensemble import GradientBoostingClassifier

from sklearn.ensemble import VotingClassifier





print ("---Base Model---")

base_roc_auc = roc_auc_score(y_test, base_rate_model(X_test))

print ("Base Rate AUC = %2.2f" % base_roc_auc)

print(classification_report(y_test, base_rate_model(X_test)))



# NOTE: By adding in "class_weight = balanced", the Logistic Auc increased by about 10%! This adjusts the threshold value

logis = LogisticRegression(class_weight = "balanced")

logis.fit(X_train, y_train)

print ("\n\n ---Logistic Model---")

logit_roc_auc = roc_auc_score(y_test, logis.predict(X_test))

print ("Logistic AUC = %2.2f" % logit_roc_auc)

print(classification_report(y_test, logis.predict(X_test)))



# Decision Tree Model

dtree = tree.DecisionTreeClassifier(

    #max_depth=3,

    class_weight="balanced",

    min_weight_fraction_leaf=0.01

    )

dtree = dtree.fit(X_train,y_train)

print ("\n\n ---Decision Tree Model---")

dt_roc_auc = roc_auc_score(y_test, dtree.predict(X_test))

print ("Decision Tree AUC = %2.2f" % dt_roc_auc)

print(classification_report(y_test, dtree.predict(X_test)))



# Random Forest Model

rf = RandomForestClassifier(

    n_estimators=1000, 

    max_depth=None, 

    min_samples_split=10, 

    class_weight="balanced"

    #min_weight_fraction_leaf=0.02 

    )

rf.fit(X_train, y_train)

print ("\n\n ---Random Forest Model---")

rf_roc_auc = roc_auc_score(y_test, rf.predict(X_test))

print ("Random Forest AUC = %2.2f" % rf_roc_auc)

print(classification_report(y_test, rf.predict(X_test)))





# Ada Boost

ada = AdaBoostClassifier(n_estimators=400, learning_rate=0.1)

ada.fit(X_train,y_train)

print ("\n\n ---AdaBoost Model---")

ada_roc_auc = roc_auc_score(y_test, ada.predict(X_test))

print ("AdaBoost AUC = %2.2f" % ada_roc_auc)

print(classification_report(y_test, ada.predict(X_test)))
# Create ROC Graph

from sklearn.metrics import roc_curve

fpr, tpr, thresholds = roc_curve(y_test, logis.predict_proba(X_test)[:,1])

rf_fpr, rf_tpr, rf_thresholds = roc_curve(y_test, rf.predict_proba(X_test)[:,1])

dt_fpr, dt_tpr, dt_thresholds = roc_curve(y_test, dtree.predict_proba(X_test)[:,1])

ada_fpr, ada_tpr, ada_thresholds = roc_curve(y_test, ada.predict_proba(X_test)[:,1])



plt.figure()



# Plot Logistic Regression ROC

plt.plot(fpr, tpr, label='Logistic Regression (area = %0.2f)' % logit_roc_auc)



# Plot Random Forest ROC

plt.plot(rf_fpr, rf_tpr, label='Random Forest (area = %0.2f)' % rf_roc_auc)



# Plot Decision Tree ROC

plt.plot(dt_fpr, dt_tpr, label='Decision Tree (area = %0.2f)' % dt_roc_auc)



# Plot AdaBoost ROC

plt.plot(ada_fpr, ada_tpr, label='AdaBoost (area = %0.2f)' % ada_roc_auc)



# Plot Base Rate ROC

plt.plot([0,1], [0,1],label='Base Rate' 'k--')



plt.xlim([0.0, 1.0])

plt.ylim([0.0, 1.05])

plt.xlabel('False Positive Rate')

plt.ylabel('True Positive Rate')

plt.title('ROC Graph')

plt.legend(loc="lower right")

plt.show()