
#%matplotlib inline 
import numpy as np 
import scipy as sp 
import matplotlib as mpl
import matplotlib.cm as cm 
import matplotlib.pyplot as plt
import pandas as pd 
#from pandas.tools.plotting import scatter_matrix
pd.set_option('display.width', 500)
pd.set_option('display.max_columns', 100)
pd.set_option('display.notebook_repr_html', True)
import seaborn as sns
sns.set(style="whitegrid")
import warnings
warnings.filterwarnings('ignore')
import string
import math
import sys
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
import sklearn
from IPython.core.interactiveshell import InteractiveShell

InteractiveShell.ast_node_interactivity = "all"
train = pd.read_csv('../input/titanic/train.csv')
Test = pd.read_csv("../input/titanic/test.csv")

combine = [train, Test]
combined = pd.concat(combine)
train.tail()
train.info()

print('_'*40)

Test.info()
train.count()
print('Train columns with null values:',train.isnull().sum(), sep = '\n')
print("-"*42)


print('Test/Validation columns with null values:', Test.isnull().sum(),sep = '\n')
print("-"*42)
train.describe(include='all')
figure, survive_bar = plt.subplots(figsize=(7, 7))
sns.barplot(x= train["Survived"].value_counts().index, y = train["Survived"].value_counts(), ax = survive_bar)
survive_bar.set_xticklabels(['Not Survived', 'Survived'])
survive_bar.set_ylabel('Frequency Count')
survive_bar.set_title('Count of Survival', fontsize = 16)

for patch in survive_bar.patches:
    label_x = patch.get_x() + patch.get_width()/2  # find midpoint of rectangle
    label_y = patch.get_y() + patch.get_height()/2
    survive_bar.text(label_x, label_y,
                #left - freq below - rel freq wrt population as a percentage
               str(int(patch.get_height())) + '(' +
               '{:.0%}'.format(patch.get_height()/len(train.Survived))+')',
               horizontalalignment='center', verticalalignment='center')
figure, embarked_bar = plt.subplots(figsize=(7, 7))
sns.barplot(x= train["Embarked"].value_counts().index, y = train["Embarked"].value_counts(), ax = embarked_bar)
embarked_bar.set_xticklabels(['Southampton', 'Chernboug', 'Queenstown'])
embarked_bar.set_ylabel('Frequency Count')
embarked_bar.set_title('Where did the passengers board the Titanic?', fontsize = 16)

fig, myplot = plt.subplots(figsize = (15,6), nrows = 2,ncols = 3)

categorical_features = ["Survived","Pclass","Sex","SibSp","Parch","Embarked"]

row, col, num_cols = 0,0,3

for u in categorical_features:
    sns.barplot(x = train[u].value_counts().index,y = train[u].value_counts(), ax  = myplot[row, col])
    myplot[row, col].set_xlabel("")
    myplot[row, col].set_title(u + " Titanic", fontsize = 15)
    myplot[row, col].set_ylabel("Count")
    col = col + 1
    if col == 3:
        col = 0
        row = row + 1


plt.subplots_adjust(hspace = 0.5)
plt.subplots_adjust(wspace = 0.3)
# i put roundbracket around x,y,z to make more sense. just like how x \in [1,2,3] and if x is a tuple or bracket
#we have   u \in [(1,2,3),(2,3,5),...] where u = (x,y,z)

#for each patch in each graph from [0,0] to [1,2], we want to do the following...
for v in range(2):
    for z in range(3):
        for patch in myplot[v,z].patches:
            label_x = patch.get_x() + patch.get_width()/2  # find midpoint of rectangle
            label_y = patch.get_y() + patch.get_height()/2
            myplot[v,z].text(label_x, label_y, 
                             str(int(patch.get_height())) + '('+'{:.0%}'.format(patch.get_height()/len(train.Survived))+')',
                            horizontalalignment='center', verticalalignment='center')

null_ages = pd.isnull(train.Age)
known_ages = pd.notnull(train.Age)
preimputation = train.Age[known_ages]
sns.distplot(preimputation)
#here we show a distribution of ages before imputation.
# Explore Age distibution as a whole
figure, myaxis = plt.subplots(figsize=(10, 4.5))

sns.kdeplot(data=train["Age"], kernel='gau', ax=myaxis, color="Red", shade=True, legend=True)
#Heatmap to observe correlation
f,ax = plt.subplots(figsize=(10, 10))
sns.heatmap(train.corr(), annot=True, linewidths=0.5, fmt='.2f',ax=ax)
ax.set_ylim(7, 0)
def print_percentage(df,col_name,col_values):
    for x in col_values:
        group = df.loc[df[col_name]==x]
        print ('{0} survival rate: {1:.3f}'.format(x, (group['Survived'].sum()/group.shape[0])))
print_percentage(train,'Sex',["male","female"])
figure, myaxis = plt.subplots(figsize=(10, 7.5))


sns.barplot(x = "Sex", 
            y = "Survived", 
            data=train, 
            ax = myaxis,
            estimator = np.mean,
            palette = {'male':"green", 'female':"Pink"},
            linewidth=2)

myaxis.set_title("Survived/Non-Survived Passenger Gender Distribution", fontsize = 20)
myaxis.set_xlabel("Sex",fontsize = 15)
myaxis.set_ylabel("Proportion of passengers survived", fontsize = 15)

for patch in myaxis.patches:
    label_x = patch.get_x() + patch.get_width()/2  # find midpoint of rectangle
    label_y = patch.get_y() + patch.get_height()/2
    myaxis.text(label_x, label_y,
                #left - freq below - rel freq wrt population as a percentage
                '{:.3%}'.format(patch.get_height()),
               horizontalalignment='center', verticalalignment='center')
figure, myaxis = plt.subplots(figsize=(10, 7.5))

sns.countplot(x = "Sex", 
                   hue="Survived",
                   data = train, 
                   linewidth=2, 
                   palette = {1:"seagreen", 0:"gray"}, ax = myaxis)


## Fixing title, xlabel and ylabel
myaxis.set_title("Passenger Gender Distribution - Survived vs Not-survived", fontsize = 20)
myaxis.set_xlabel("Sex", fontsize = 15);
myaxis.set_ylabel("Number of Passenger Survived", fontsize = 15)
myaxis.legend(["Not Survived", "Survived"], loc = 'upper right')
print_percentage(train,'Pclass',[1,2,3])
#Plot 1: We can use a bar plot:

figure, pclass_bar = plt.subplots(figsize = (8,10))
sns.barplot(x = "Pclass", 
            y = "Survived", 
            estimator = np.mean,
            data=train, 
            ax = pclass_bar,
            linewidth=2)
pclass_bar.set_title("Passenger Class Distribution - Survived vs Non-Survived", fontsize = 18)
pclass_bar.set_xlabel("Passenger class (Pclass)", fontsize = 15);
pclass_bar.set_ylabel("% of Passenger Survived", fontsize = 15);
labels = ['Upper (1)', 'Middle (2)', 'Lower (3)']
#val = sorted(train.Pclass.unique())
val = [0,1,2] ## this is just a temporary trick to get the label right. 
pclass_bar.set_xticklabels(labels);



#Plot 2: We can use a line plot:
sns.catplot('Pclass', 'Survived', kind='point', data=train)

sns.set(font_scale=1)
g = sns.catplot(x="Sex", y="Survived", col="Pclass",
                    data=train, saturation=.5,
                    kind="bar", ci=None, aspect=.6)
(g.set_axis_labels("", "Survival Rate")
    .set_xticklabels(["Men", "Women"])
    .set_titles("{col_name} {col_var}")
    .set(ylim=(0, 1))
    .despine(left=True))  
plt.subplots_adjust(top=0.8)
g.fig.suptitle('How many Men and Women Survived by Passenger Class');

for myaxis in g.axes[0]:
    for patch in myaxis.patches:
        label_x = patch.get_x() + patch.get_width()/2  # find midpoint of rectangle
        label_y = patch.get_y() + patch.get_height()/2
        myaxis.text(label_x, label_y,
                    #left - freq below - rel freq wrt population as a percentage
                    '{:.3%}'.format(patch.get_height()),
                   horizontalalignment='center', verticalalignment='center')

#Another plot that gives the exact same result is as follows: It is good to know different variations.


g = sns.factorplot(x="Pclass", y="Survived", hue="Sex", data=train,
                   size=6, kind="bar", palette="muted")
g.despine(left=True)
g = g.set_ylabels("survival probability")

myaxis = g.ax
for patch in myaxis.patches:
    label_x = patch.get_x() + patch.get_width()/2  # find midpoint of rectangle
    label_y = patch.get_y() + patch.get_height()/2
    myaxis.text(label_x, label_y,
                #left - freq below - rel freq wrt population as a percentage
                '{:.3%}'.format(patch.get_height()),
               horizontalalignment='center', verticalalignment='center')
print_percentage(train,'Embarked',['S','C','Q'])
figure, embarked_bar = plt.subplots(figsize = (8,10))
sns.barplot(x = "Embarked", 
            y = "Survived", 
            estimator = np.mean,
            data=train, 
            ax = embarked_bar,
            linewidth=2)
embarked_bar.set_title("Passenger Embarked Distribution - Survived vs Non-Survived", fontsize = 15)
embarked_bar.set_xlabel("Embarked Place", fontsize = 15);
embarked_bar.set_ylabel("% of Passenger Survived", fontsize = 15);


sns.set(font_scale=1)
g = sns.catplot(x="Pclass", col="Embarked",
                    data=train, saturation=.5,
                    kind="count", ci=None)
sns.set(font_scale=1)
g = sns.catplot(x="Sex", col="Embarked",
                    data=train, saturation=.5,
                    kind="count", ci=None)
# Explore Age distibution
figure, myaxis = plt.subplots(figsize=(10, 4.5))

sns.kdeplot(data=train["Age"][(train["Survived"] == 0) & (
    train["Age"].notnull())], kernel='gau', ax=myaxis, color="Red", shade=True, legend=True)


# a faster code using loc function is sns.kdeplot(train.loc[(train['Survived'] == 0),'Age'] , color='gray',shade=True,label='not survived')

sns.kdeplot(data=train["Age"][(train["Survived"] == 1) & (
    train["Age"].notnull())], kernel='gau', ax=myaxis, color="Blue", shade=True, legend=True)

myaxis.set_xlabel("Age")
myaxis.set_ylabel("Probability Density")
myaxis.legend(["Not Survived", "Survived"], loc='upper right')
myaxis.set_title("Superimposed KDE plot for age of Survived and Not Survived",
                 loc='center', fontdict={'fontsize': 16}, color='r')
pal = {1:"seagreen", 0:"gray"}
g = sns.FacetGrid(train,size=5, col="Sex", row="Survived", margin_titles=True, hue = "Survived",
                  palette=pal)
g = g.map(plt.hist, "Age", edgecolor = 'white');
g.fig.suptitle("Survived by Sex and Age", size = 25)
plt.subplots_adjust(top=0.90)
g = sns.FacetGrid(train,size=5, col="Sex", row="Embarked", margin_titles=True, hue = "Survived",
                  palette = pal
                  )
g = g.map(plt.hist, "Age", edgecolor = 'white').add_legend();
g.fig.suptitle("Survived by Sex and Age", size = 25)
plt.subplots_adjust(top=0.90)
null_ages = pd.isnull(train.Age)
known_ages = pd.notnull(train.Age)
preimputation = train.Age[known_ages]
sns.distplot(preimputation)
# Explore Age distibution
figure, fare = plt.subplots(figsize=(10, 4.5))

sns.kdeplot(data=train.loc[(train['Survived'] == 0),'Fare'], kernel='gau', ax=fare, color="Red", shade=True, legend=True)

sns.kdeplot(data=train.loc[(train['Survived'] == 1),'Fare'], kernel='gau', ax=fare, color="Blue", shade=True, legend=True)

fare.set_xlabel("Fare")
fare.set_ylabel("Probability Density")
fare.legend(["Not Survived", "Survived"], loc='upper right')
fare.set_title("Superimposed KDE plot for Fare of Survived and Not Survived",
                 loc='center', fontdict={'fontsize': 16}, color='r')
figure, fare = plt.subplots(figsize=(20, 5))
sns.distplot(train.loc[(train['Survived'] == 0),'Fare'], hist=True, color='red', ax=fare)
sns.distplot(train.loc[(train['Survived'] == 1),'Fare'], hist=True, color='blue', ax=fare)

fare.set_xlabel("Fare")
fare.set_ylabel("Probability Density")
fare.legend(["Not Survived", "Survived"], loc='upper right')
fare.set_title("Superimposed distribution plot for Fare of Survived and Not Survived",
                 loc='center', fontdict={'fontsize': 16}, color='r')
train.Parch.value_counts()
print_percentage(train,'Parch',[0,1,2,3,4,5,6])
figure, parch_bar = plt.subplots(figsize = (8,10))
sns.barplot(x = "Parch", 
            y = "Survived", 
            estimator = np.mean,
            data=train, 
            ax = parch_bar,
            linewidth=2)
parch_bar.set_title("Parch Distribution - Survived vs Non-Survived", fontsize = 18)
parch_bar.set_xlabel("Parch", fontsize = 15);
parch_bar.set_ylabel("% of Passenger Survived", fontsize = 15);
figure, myaxis = plt.subplots(figsize=(10, 7.5))

sns.countplot(x = "Parch", 
                   hue="Survived",
                   data = train, 
                   linewidth=2, 
                   palette = {1:"seagreen", 0:"gray"}, ax = myaxis)


## Fixing title, xlabel and ylabel
myaxis.set_title("Passenger Parch Distribution - Survived vs Not-survived", fontsize = 20)
myaxis.set_xlabel("Parch", fontsize = 15);
myaxis.set_ylabel("Number of Passenger Survived", fontsize = 15)
myaxis.legend(["Not Survived", "Survived"], loc = 'upper right')
# sns.set(font_scale=1)
# g = sns.catplot(x="SibSp", y="Survived", col="Parch",
#                     data=train, saturation=.5,
#                     kind="bar", ci=None, aspect=.6)
# (g.set_axis_labels("", "Survival Rate")
#     .set_xticklabels(["Men", "Women"])
#     .set_titles("{col_name} {col_var}")
#     .set(ylim=(0, 1))
#     .despine(left=True))  
# plt.subplots_adjust(top=0.8)
# g.fig.suptitle('How many Men and Women Survived by Passenger Class');


train.SibSp.value_counts()
print_percentage(train,'SibSp',[0,1,2,3,4,5,8])
figure, parch_bar = plt.subplots(figsize = (8,10))
sns.barplot(x = "SibSp", 
            y = "Survived", 
            estimator = np.mean,
            data=train, 
            ax = parch_bar,
            linewidth=2)
parch_bar.set_title("Siblings Distribution - Survived vs Non-Survived", fontsize = 18)
parch_bar.set_xlabel("SibSp", fontsize = 15);
parch_bar.set_ylabel("% of Passenger Survived", fontsize = 15);
figure, myaxis = plt.subplots(figsize=(10, 7.5))

sns.countplot(x = "SibSp", 
                   hue="Survived",
                   data = train, 
                   linewidth=2, 
                   palette = {1:"seagreen", 0:"gray"}, ax = myaxis)


## Fixing title, xlabel and ylabel
myaxis.set_title("Siblings Distribution - Survived vs Not-survived", fontsize = 20)
myaxis.set_xlabel("Siblings", fontsize = 15);
myaxis.set_ylabel("Number of Passenger Survived", fontsize = 15)
myaxis.legend(["Not Survived", "Survived"], loc = 'upper right')
# Check variable Fare for missing values:
print('Amount of missing data in Fare for training set:', train.Fare.isnull().sum())
print('Amount of missing data in Fare for test set:', Test.Fare.isnull().sum())
Test[Test['Fare'].isnull()]
median_fare = Test.groupby(['Pclass', 'Parch']).Fare.median()[3][0]
# Filling the missing value in Fare with the median Fare of 3rd class passenger who has Parch 0.
Test['Fare'] = Test['Fare'].fillna(median_fare)
figure, fare = plt.subplots(figsize=(20, 6.6))
sns.distplot(train.Fare, hist=False,  color='red', label = "Training Data",ax=fare)
sns.distplot(Test.Fare, hist=False, label = "Test Data", color='blue', ax=fare)

fare.set_xlabel("Fare")
fare.set_ylabel("Probability Density")
fare.legend(["Training Data", "Test Data"], loc='upper right')
fare.set_title("Superimposed distribution plot for Fare of Training set vs Test set",
                 loc='center', fontdict={'fontsize': 16}, color='r')
# Check variable Embarked for missing values:
print('Amount of missing data in Embarked for train:', train.Embarked.isnull().sum())
print('Amount of missing data in Embarked for test:', Test.Embarked.isnull().sum())
train[train['Embarked'].isnull()]
# pclass1_fare = train[train.Pclass == 1].sort_values(['Embarked','Fare'])
# g = sns.FacetGrid(pclass1_fare,col = "Embarked")
# g.map(sns.distplot, "Fare", kde = False)
train['Embarked'] = train['Embarked'].fillna('S')
# Check variable Embarked for missing values:
print('Amount of missing data in Age for train:', train.Age.isnull().sum())
print('Amount of missing data in Age for test:', Test.Age.isnull().sum())
# Explore Age distibution
figure, myaxis = plt.subplots(figsize=(10, 4.5))

preimputation=sns.kdeplot(data=train["Age"][(train["Survived"] == 0) & (
    train["Age"].notnull())], kernel='gau', ax=myaxis, color="Red", shade=True, legend=True)


# a faster code using loc function is sns.kdeplot(train.loc[(train['Survived'] == 0),'Age'] , color='gray',shade=True,label='not survived')

preimputation=sns.kdeplot(data=train["Age"][(train["Survived"] == 1) & (
    train["Age"].notnull())], kernel='gau', ax=myaxis, color="Blue", shade=True, legend=True)

myaxis.set_xlabel("Age")
myaxis.set_ylabel("Probability Density")
myaxis.legend(["Not Survived", "Survived"], loc='upper right')
myaxis.set_title("Superimposed KDE plot for age of Survived and Not Survived",
                 loc='center', fontdict={'fontsize': 16}, color='r')
copy5 = train.copy()
missing_age_rows2 = copy5.Age.isna()
missing_age_rows2
#age_by_pclass_SibSp = copy5.groupby(['Pclass', 'SibSp']).median()['Age']
#age_by_pclass_SibSp[1].index.tolist()
#age_by_pclass_SibSp[3][8] = age_by_pclass_SibSp[3][5] #since no age values for pclass 3 and sibsp 8 we fill it with
#pclass 3 and sibsp5

#for pclass in range(1, 4):
    #for siblings in age_by_pclass_SibSp[pclass].index.tolist():
        #print('Median age of Pclass {} with {} siblings: {}'.format(pclass, siblings, age_by_pclass_SibSp[pclass][siblings]))
#print('Median age of all passengers: {}'.format(copy5['Age'].median()))
age_by_pclass_SibSp = copy5.groupby(['Pclass', 'SibSp']).median()['Age']
age_by_pclass_SibSp
age_by_pclass_SibSp[1].index.tolist()
age_by_pclass_SibSp[3][8] = age_by_pclass_SibSp[3][5] #since no age values for pclass 3 and sibsp 8 we fill it with
#pclass 3 and sibsp5
age_by_pclass_SibSp
for pclass in range(1, 4):
    for siblings in age_by_pclass_SibSp[pclass].index.tolist():
        print('Median age of Pclass {} with {} siblings: {}'.format(pclass, siblings, age_by_pclass_SibSp[pclass][siblings]))
print('Median age of all passengers: {}'.format(copy5['Age'].median()))

copy5['Age'] = copy5.groupby(['Pclass', 'SibSp'])[
    'Age'].apply(lambda x: x.fillna(x.median()))
# this line is the single code that we need to fill up all the 
#missing values: powerful one liner from 
#https://www.kaggle.com/gunesevitan/advanced-feature-engineering-tutorial-with-titanic.

#however do not forget that the above line of code does not take care of the 7 missing NaN values from passengers
#from Pclass 3 & SibSp 8. So we fill in the remaining 7 missing values to be age 11.

copy5['Age'] = copy5.Age.fillna(11) #think this step cause no values for NA value.
copy5.info()
# Explore Age distibution
figure, myaxis = plt.subplots(figsize=(20, 4.5))

preimputation=sns.kdeplot(data=train["Age"][(train["Survived"] == 0) & (
    train["Age"].notnull())], kernel='gau', ax=myaxis, color="Blue", shade=True, legend=True)


# a faster code using loc function is sns.kdeplot(train.loc[(train['Survived'] == 0),'Age'] , color='gray',shade=True,label='not survived')


afterimputation=sns.kdeplot(data=copy5["Age"][(copy5["Survived"] == 0) & (
    copy5["Age"].notnull())], kernel='gau', ax=myaxis, color="maroon", shade=True, legend=True)


# a faster code using loc function is sns.kdeplot(train.loc[(train['Survived'] == 0),'Age'] , color='gray',shade=True,label='not survived')


myaxis.set_xlabel("Age")
myaxis.set_ylabel("Probability Density")
myaxis.legend(["pre", "after"], loc='upper right')
myaxis.set_title("Superimposed KDE plot for age of not survived: pre-imputation vs after-imputation",
                 loc='center', fontdict={'fontsize': 16}, color='r')
# Explore Age distibution
figure, myaxis = plt.subplots(figsize=(20, 4.5))

preimputation=sns.kdeplot(data=train["Age"][(train["Survived"] == 1) & (
    train["Age"].notnull())], kernel='gau', ax=myaxis, color="Blue", shade=True, legend=True)


# a faster code using loc function is sns.kdeplot(train.loc[(train['Survived'] == 0),'Age'] , color='gray',shade=True,label='not survived')

afterimputation=sns.kdeplot(data=copy5["Age"][(copy5["Survived"] == 1) & (
    copy5["Age"].notnull())], kernel='gau', ax=myaxis, color="maroon", shade=True, legend=True)

myaxis.set_xlabel("Age")
myaxis.set_ylabel("Probability Density")
myaxis.legend(["pre", "after"], loc='upper right')
myaxis.set_title("Superimposed KDE plot for age of survived: pre-imputation vs after-imputation",
                 loc='center', fontdict={'fontsize': 16}, color='r')

copy2 = train.copy() #wanna work with copy here so dont mess up the original data values.
copy2['Age'].fillna(copy2['Age'].median(),inplace = True)
# Explore Age distibution
figure, myaxis = plt.subplots(figsize=(20, 4.5))

preimputation=sns.kdeplot(data=train["Age"][(train["Survived"] == 0) & (
    train["Age"].notnull())], kernel='gau', ax=myaxis, color="Blue", shade=True, legend=True)


median1=sns.kdeplot(data=copy2["Age"][(copy2["Survived"] == 0) & (
    copy2["Age"].notnull())], kernel='gau', ax=myaxis, color="maroon", shade=True, legend=True)



myaxis.set_xlabel("Age")
myaxis.set_ylabel("Probability Density")
myaxis.legend(["pre", "after"], loc='upper right')
myaxis.set_title("Superimposed KDE plot for age of not survived: pre-imputation vs after-imputation",
                 loc='center', fontdict={'fontsize': 16}, color='r')
# Explore Age distibution
figure, myaxis = plt.subplots(figsize=(20, 4.5))




preimputation=sns.kdeplot(data=train["Age"][(train["Survived"] == 1) & (
    train["Age"].notnull())], kernel='gau', ax=myaxis, color="Blue", shade=True, legend=True)





median1=sns.kdeplot(data=copy2["Age"][(copy2["Survived"] == 1) & (
    copy2["Age"].notnull())], kernel='gau', ax=myaxis, color="maroon", shade=True, legend=True)

myaxis.set_xlabel("Age")
myaxis.set_ylabel("Probability Density")
myaxis.legend(["pre", "after"], loc='upper right')
myaxis.set_title("Superimposed KDE plot for age of survived: pre-imputation vs after-imputation",
                 loc='center', fontdict={'fontsize': 16}, color='r')
#Finally do not forget to let train = copy5.

train = copy5
test_age_by_pclass_SibSp = Test.groupby(['Pclass', 'SibSp']).median()['Age']
test_age_by_pclass_SibSp
Test['Age'] = Test.groupby(['Pclass', 'SibSp'])['Age'].apply(lambda x: x.fillna(x.median()))
Test.info()
from collections import Counter

def detect_outliers(df, n, features):
    outliers_indices = [] #create a empty list to keep track of the passenger row number.
    for col in features:
        # 1st quartile (25%)
        Q1 = np.nanpercentile(df[col], 25)
        # 3rd quartile (75%)
        Q3 = np.nanpercentile(df[col], 75)
        # Interquartile range (IQR)
        IQR = Q3 - Q1
        
        # outlier step
        outlier_step = 1.5 * IQR
        # Determine a list of indices of outliers for feature col
        outlier_list_col = df[(df[col] < Q1 - outlier_step)
                              | (df[col] > Q3 + outlier_step)].index
        
        #print(df[(df[col] < Q1 - outlier_step)
                              #| (df[col] > Q3 + outlier_step)].index)
        print(col,Q1-outlier_step,Q3+outlier_step)
        # append the found outlier indices for col to the list of outlier indices
        outliers_indices.extend(outlier_list_col)
        
    #print(outliers_indices)
    
    # select observations containing more than 2 outliers
    outliers_indices = Counter(outliers_indices)
    multiple_outliers = list(k for k, v in outliers_indices.items() if v > n)
    #print(outliers_indices)
    
    return multiple_outliers


Outliers_to_drop = detect_outliers(train, 2, ["Age", "SibSp", "Parch", "Fare"])

Outliers_to_drop

train.loc[Outliers_to_drop]
def outlier_treatment(datacolumn):
    sorted(datacolumn) #must sort data first since we dealing with quartile ranges.
    Q1, Q3 = np.nanpercentile(datacolumn,[25,75])
    IQR = Q3-Q1
    lower_range = Q1-(1.5*IQR)
    upper_range = Q3+ (1.5*IQR)
    return lower_range, upper_range



print('Upper and Lower bound for Age:',outlier_treatment(train.Age), sep = '\n')
print('Upper and Lower bound for Fare:',outlier_treatment(train.Fare), sep = '\n')
print('Upper and Lower bound for Parch:',outlier_treatment(train.Parch), sep = '\n')
print('Upper and Lower bound for Siblings:',outlier_treatment(train.SibSp), sep = '\n')




# check how many unique values each feature has:


print('Number of Unique values for Name is', len(train.Name.unique()))
print('Number of Unique values for PassengerID is', len(train.PassengerId.unique()))
print('Number of Unique values for Fare is', len(train.Fare.unique()))
print('Number of Unique values for Survived is', len(train.Survived.unique()))
print('Number of Unique values for Pclass is', len(train.Pclass.unique()))
print('Number of Unique values for Parch is', len(train.Parch.unique()))
print('Number of Unique values for SibSp is', len(train.SibSp.unique()))
print('Number of Unique values for Embarked is', len(train.Embarked.unique()))
print('Number of Unique values for Cabin is', len(train.Cabin.unique()))
print('Number of Unique values for Sex is', len(train.Sex.unique()))
print('Number of Unique values for Ticket is', len(train.Ticket.unique()))
train.drop(["PassengerId"],inplace=True,axis=1)
Test.drop(["PassengerId"],inplace=True,axis=1)
groupby_pclass_ticket= train.groupby(["Pclass","Ticket"])
#for key, item in groupby_pclass_ticket:
    #groupby_pclass_ticket.get_group(key)
    
#groupby_pclass_ticket.describe(include = 'all')    
train.drop(["Ticket"],inplace=True,axis=1)
Test.drop(["Ticket"],inplace=True,axis=1)
train.Name.head(20)
# Create function that take name and separates it into title, family name and deletes all puntuation from name column:
def name_sep(data):
    families=[]
    titles = []
    new_name = []
    #for each row in dataset:
    for i in range(len(data)):
        name = data.iloc[i]
        # extract name inside brakets into name_bracket:
        if '(' in name:
            name_no_bracket = name.split('(')[0] 
        else:
            name_no_bracket = name
            
        family = name_no_bracket.split(",")[0]
        title = name_no_bracket.split(",")[1].strip().split(" ")[0]
        
        #remove punctuations accept brackets:
        for c in string.punctuation:
            name = name.replace(c,"").strip()
            family = family.replace(c,"").strip()
            title = title.replace(c,"").strip()
            
        families.append(family)
        titles.append(title)
        new_name.append(name)
            
    return families, titles, new_name 
train['Surname'], train['Title'], train['Newname']  = name_sep(train.Name)
Test['Surname'], Test['Title'], Test['Newname'] = name_sep(Test.Name)
train.head()
train.Title.value_counts()
train['Title'] = train['Title'].replace(['Ms', 'Mlle'],'Miss')
train['Title'] = train['Title'].replace(['Mme'],'Mrs')
train['Title'] = train['Title'].replace(['Dr','Rev','the','Jonkheer','Lady','Sir', 'Don'],'Nobles')
train['Title'] = train['Title'].replace(['Major','Col', 'Capt'],'Navy')
train.Title.value_counts()
sns.barplot(x = 'Title', y = 'Survived', data = train)
nobles_survival = train[train.Title == "Nobles"].groupby(['Sex']).Survived.value_counts()
nobles_survival
Test.Title.value_counts()

Test['Title'] = Test['Title'].replace(['Ms','Dona'],'Miss')
Test['Title'] = Test['Title'].replace(['Dr','Rev'],'Nobles')
Test['Title'] = Test['Title'].replace(['Col'],'Navy')
Test.Title.value_counts()
print('Missing values in Train set:', train.Cabin.isnull().sum())
print('Missing values in Test set:', Test.Cabin.isnull().sum())
train.Cabin.value_counts()
def cabin_sep(data_cabin):
    cabin_type = []

    for i in range(len(data_cabin)):

            if data_cabin.isnull()[i] == True: 
                cabin_type.append('M') #missing cabin = M 
            else:    
                cabin = data_cabin[i]
                cabin_type.append(cabin[:1]) 
            
    return cabin_type

# apply cabin sep on test and train set:
#train['cabin_type'] = cabin_sep(train.Cabin)
#Test['cabin_type'] = cabin_sep(Test.Cabin)


#train.head()
train['Cabin'] = train['Cabin'].fillna('M').astype(str).apply(lambda cabin: cabin[0])
idx = train[train['Cabin'] == 'T'].index
train.loc[idx, 'Cabin'] = 'A'
train.Cabin.value_counts()
Test['Cabin'] = Test['Cabin'].fillna('M').astype(str).apply(lambda cabin: cabin[0])
Test.Cabin.value_counts()
train_categorical_features = ['Pclass', 'Sex','Title','Cabin', 'Embarked']

# No need to use sklearn's encoders
# pandas has a pandas.get_dummies() function that takes in a series
#     and returns a HOT encoded dataframe of that series
#     use the add_prefix() method of dataframe to add the feature name in front of the category name
#     then join the dataframe sideways (similar to pd.concat([train, dummies], axis=1))
for feature in train_categorical_features:
    dummies = pd.get_dummies(train[feature]).add_prefix(feature + '_')
    train = train.join(dummies)
    
train
test_categorical_features = ['Pclass', 'Sex','Title', 'Cabin', 'Embarked']

# No need to use sklearn's encoders
# pandas has a pandas.get_dummies() function that takes in a series
#     and returns a HOT encoded dataframe of that series
#     use the add_prefix() method of dataframe to add the feature name in front of the category name
#     then join the dataframe sideways (similar to pd.concat([train, dummies], axis=1))
for feature in test_categorical_features:
    dummies = pd.get_dummies(Test[feature]).add_prefix(feature + '_')
    Test = Test.join(dummies)
    
Test
#importing from sklearn
from sklearn.pipeline import make_pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import learning_curve,GridSearchCV
from sklearn.ensemble.gradient_boosting import GradientBoostingClassifier
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LinearRegression
import statsmodels.formula.api as sm
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC, LinearSVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import Perceptron
from sklearn.linear_model import SGDClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.metrics import confusion_matrix
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
#import xgboost as xgb
from sklearn.metrics import roc_curve, auc
import scikitplot as skplt #conda install -c conda-forge scikit-plot
from sklearn.metrics import accuracy_score 
from sklearn.metrics import mean_absolute_error, accuracy_score
drop_column = ['Pclass','Name','Sex','Cabin', 'Embarked','Surname','Title','Newname']
train.drop(drop_column, axis=1, inplace = True)

drop_column = ['Pclass','Name','Sex','Cabin', 'Embarked','Surname','Title','Newname']
Test.drop(drop_column, axis=1, inplace = True)
original_train_set_without_survived = train.drop("Survived", axis=1)
orginal_train_set_with_only_survived = train["Survived"]
#In this part, we will do the train test split.

x_train, x_test, y_train, y_test = train_test_split(
    original_train_set_without_survived, orginal_train_set_with_only_survived, train_size=0.8, test_size=0.2, random_state=0)
print(x_train.shape, y_train.shape)
print(x_test.shape, y_test.shape)
# Feature Scaling
## We will be using standardscaler to transform the data.
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()

## transforming "train_x"
x_train = sc.fit_transform(x_train)
## transforming "test_x"
x_test = sc.transform(x_test)

## transforming "The testset"
Test = sc.transform(Test)
## call on the model object
logreg = LogisticRegression()

## fit the model with "train_x" and "train_y"
logreg.fit(x_train,y_train)
#Alternatively also can use
#print ("So, Our accuracy Score is: {}".format(round(logreg.score(x_test,y_test),8)))

## Once the model is trained we want to find out how well the model is performing, so we test the model. 
## we use "test_x" portion of the data(this data was not used to fit the model) to predict model outcome. 
y_pred = logreg.predict(x_test)

## Once predicted we save that outcome in "y_pred" variable.
## Then we compare the predicted value( "y_pred") and actual value("test_y") to see how well our model is performing. 

print ("So, Our accuracy Score is: {}".format(round(accuracy_score(y_test,y_pred),8)))
def confusion_matrix_model(model_used):
    cm=confusion_matrix(y_test,model_used.predict(x_test))
    col=["Predicted Dead","Predicted Survived"]
    cm=pd.DataFrame(cm)
    cm.columns=["Predicted Dead","Predicted Survived"]
    cm.index=["Actual Dead","Actual Survived"]
    cm[col]=np.around(cm[col].div(cm[col].sum(axis=1),axis=0),decimals=2)
    return cm



confusion_matrix_model(logreg)
from sklearn.metrics import roc_curve, auc
#plt.style.use('seaborn-pastel')
y_score = logreg.decision_function(x_test) #y_pred_proba = logreg.predict_proba(X_test)[:, 1] same as this?

FPR, TPR, THR = roc_curve(y_test, y_score) #[fpr, tpr, thr] = roc_curve(y_test, y_pred_proba) same as this
ROC_AUC = auc(FPR, TPR)
print (logreg.__class__.__name__+" auc is %2.8f" % ROC_AUC) #Gives AUC score.


#Next is draw roc graph.
plt.figure(figsize =[10,9])
plt.plot(FPR, TPR, label= 'ROC curve(area = %0.2f)'%ROC_AUC, linewidth= 4)
plt.plot([0,1],[0,1], 'k--', linewidth = 4)
plt.xlim([0.0,1.0])
plt.ylim([0.0,1.05])
plt.xlabel('False Positive Rate', fontsize = 17)
plt.ylabel('True Positive Rate', fontsize = 17)
plt.title('ROC for Logistic Regression (Titanic)', fontsize= 17)
plt.show()
from sklearn.metrics import precision_recall_curve

y_score = logreg.decision_function(x_test)

precision, recall, _ = precision_recall_curve(y_test, y_score)
PR_AUC = auc(recall, precision)

plt.figure(figsize=[11,9])
plt.plot(recall, precision, label='PR curve (area = %0.2f)' % PR_AUC, linewidth=4)
plt.xlabel('Recall', fontsize=18)
plt.ylabel('Precision', fontsize=18)
plt.title('Precision Recall Curve for Titanic survivors', fontsize=18)
plt.legend(loc="lower right")
plt.show()
k_fold = KFold(n_splits=10, shuffle=True, random_state=0)
x = original_train_set_without_survived
y = orginal_train_set_with_only_survived
#normalize it by using our sc previously defined - where sc = StandardScaler()
x = sc.fit_transform(x)
def Acc_score(model):
    return np.mean(cross_val_score(model,x,y,cv=k_fold,scoring="accuracy"))
logreg2=LogisticRegression()

print("Accuracy Scores: " + format(cross_val_score(logreg2,x,y,cv=k_fold,scoring="accuracy")))
print(" ") #leave empty line
print("Mean Accuracy Score: " + str(Acc_score(logreg2)))
print(" ")
print("Standard Deviation:", cross_val_score(logreg2,x,y,cv=k_fold,scoring="accuracy").std())
scores_auc = cross_val_score(logreg2, x, y, cv=k_fold, scoring='roc_auc')
#Notice scoring = roc_auc now.  https://scikit-learn.org/stable/modules/model_evaluation.html
print("AUC score for 10 fold Cross Validation:", scores_auc)
print(" ")
print("Mean AUC score for 10 fold Cross Validation:", scores_auc.mean())
def plt_roc_curve(name, model, cv_x_test, cv_y_test, has_proba=True):
    if has_proba:
        fpr,tpr,thr=skplt.metrics.roc_curve(cv_y_test,model.predict_proba(cv_x_test)[:,1])
    else:
        fpr,tpr,thr=skplt.metrics.roc_curve(cv_y_test,model.decision_function(cv_x_test))
    auc= skplt.metrics.auc(fpr,tpr) #x axis is fpr, y axis is tpr

    plt.plot(fpr,tpr,label='ROC curve for %s (AUC = %0.8f)' % (name, auc))
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim((0,1))
    plt.ylim((0,1))
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend(loc="lower right")
    #plt.show()
    return fpr, tpr, auc
def run_k_fold(modeller, no_of_folds):
    scores = []
    mean_fpr = np.linspace(0, 1, 100)
    tprs = []
    aucs = []
    plt.figure(figsize =[10,9])
    
    k_fold = KFold(n_splits=no_of_folds, shuffle=True, random_state=0)
    fold = 1
    # the below line's x, y is from outside just now.
    #k_fold.split gives you all 10 rounds in a list of tuples [round1, round2,...]
    #each round consists of the rows to be in the training set (90%) and the rows to be in 
    #the test set (10%) for that round. 
    for rounds in k_fold.split(x, y): #don't use round as it is a in built function
        train_rows = rounds[0]
        test_rows  = rounds[1]
        CV_x_train = x[train_rows]
        CV_y_train = y[train_rows]
        CV_x_test = x[test_rows]
        CV_y_test = y[test_rows]
        model = modeller()
        model.fit(CV_x_train, CV_y_train)
        scores.append(model.score(CV_x_test, CV_y_test))
        curr_fpr, curr_tpr, curr_auc = plt_roc_curve(
            'log reg fold ' + str(fold), model, CV_x_test, CV_y_test)

        tprs.append(np.interp(mean_fpr, curr_fpr, curr_tpr))
        tprs[-1][0] = 0.0
        roc_auc = auc(curr_fpr, curr_tpr)
        aucs.append(roc_auc)
        fold += 1

    mean_tpr = np.mean(tprs, axis=0)
    mean_tpr[-1] = 1.0
    mean_auc = auc(mean_fpr, mean_tpr)
    std_auc = np.std(aucs)  # popn std dev?
    plt.plot(mean_fpr, mean_tpr, color='b',
             label=r'Mean ROC (AUC = %0.8f $\pm$ %0.2f)' % (mean_auc, std_auc),
             lw=2, alpha=.8)

    std_tpr = np.std(tprs, axis=0)
    tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
    tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
    plt.fill_between(mean_fpr, tprs_lower, tprs_upper, color='grey', alpha=.2,
                     label=r'$\pm$ 1 std. dev.')
    
    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic example')
    plt.legend(loc="lower right")
    plt.show()

    return np.mean(scores)
accuracy_score = run_k_fold(LogisticRegression, 10)
print("Accuracy Score:", accuracy_score)
#notice that the mean roc score is different from the previous code. this is because
#we are finding the AUC of the MEAN ROC curves. In  a sense we have a 11th curve called
#the mean roc curve and we find the area under that 11th curve. While the previous code
#just mean the 10 auc of the respective rocs.
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV
max_depth = range(1,30)
#max_feature = list(range(1,x.shape[1]+1)) #dynamic coding
max_feature = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,'auto'] #how one knows start at 21


criterion=["entropy", "gini"]

param = {'max_depth':max_depth, 
         'max_features':max_feature, 
         'criterion': criterion}
grid = GridSearchCV(DecisionTreeClassifier(random_state=42), 
                                param_grid = param, 
                                 verbose=False, 
                                 cv=KFold(n_splits=10, shuffle=True, random_state=42),
                                n_jobs = -1)
grid.fit(x, y) 


print( grid.best_params_)
print(" ")
print (grid.best_score_)
print(" ")
print (grid.best_estimator_)

dectree_grid = grid.best_estimator_
## using the best found hyper paremeters to get the score. 
dectree_grid.score(x,y)
from sklearn.ensemble import RandomForestClassifier

rf_model = RandomForestClassifier(bootstrap=True, class_weight=None, criterion='entropy',
                       max_depth=9, max_features='auto', max_leaf_nodes=None,
                       min_impurity_decrease=0.0, min_impurity_split=None,
                       min_samples_leaf=1, min_samples_split=16,
                       min_weight_fraction_leaf=0.0, n_estimators=140,
                       n_jobs=None, oob_score=False, random_state=42, verbose=0,
                       warm_start=True)

rf_model.fit(x,y)

#print("%.3f" % rf_model.oob_score_)
rf_model.score(x,y)
Test = sc.fit_transform(Test)
Test1 = pd.read_csv("../input/titanic/test.csv")

output3 = pd.DataFrame({"PassengerId": Test1.PassengerId, "Survived":rf_model.predict(Test)})
output3.PassengerId = output3.PassengerId.astype(int)
output3.Survived = output3.Survived.astype(int)

output3.to_csv("output3.csv", index=False)
print("Your submission was successfully saved!")
output3.head(10)