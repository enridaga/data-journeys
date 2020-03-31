
# We can use the pandas library in python to read in the csv file.

import pandas as pd

#for numerical computaions we can use numpy library

import numpy as np
# This creates a pandas dataframe and assigns it to the titanic variable.

titanic = pd.read_csv("../input/train.csv")

# Print the first 5 rows of the dataframe.

titanic.head()
titanic_test = pd.read_csv("../input/test.csv")

#transpose

titanic_test.head().T

#note their is no Survived column here which is our target varible we are trying to predict
#shape command will give number of rows/samples/examples and number of columns/features/predictors in dataset

#(rows,columns)

titanic.shape
#Describe gives statistical information about numerical columns in the dataset

titanic.describe()

#you can check from count if there are missing vales in columns, here age has got missing values
#info method provides information about dataset like 

#total values in each column, null/not null, datatype, memory occupied etc

titanic.info()
#lets see if there are any more columns with missing values 

null_columns=titanic.columns[titanic.isnull().any()]

titanic.isnull().sum()
#how about test set??

titanic_test.isnull().sum()
### matplotlib inline

import matplotlib.pyplot as plt

import seaborn as sns

sns.set(font_scale=1)



pd.options.display.mpl_style = 'default'

labels = []

values = []

for col in null_columns:

    labels.append(col)

    values.append(titanic[col].isnull().sum())

ind = np.arange(len(labels))

width=0.6

fig, ax = plt.subplots(figsize=(6,5))

rects = ax.barh(ind, np.array(values), color='purple')

ax.set_yticks(ind+((width)/2.))

ax.set_yticklabels(labels, rotation='horizontal')

ax.set_xlabel("Count of missing values")

ax.set_ylabel("Column Names")

ax.set_title("Variables with missing values");
titanic.hist(bins=10,figsize=(9,7),grid=False);
g = sns.FacetGrid(titanic, col="Sex", row="Survived", margin_titles=True)

g.map(plt.hist, "Age",color="purple");
g = sns.FacetGrid(titanic, hue="Survived", col="Pclass", margin_titles=True,

                  palette={1:"seagreen", 0:"gray"})

g=g.map(plt.scatter, "Fare", "Age",edgecolor="w").add_legend();
g = sns.FacetGrid(titanic, hue="Survived", col="Sex", margin_titles=True,

                palette="Set1",hue_kws=dict(marker=["^", "v"]))

g.map(plt.scatter, "Fare", "Age",edgecolor="w").add_legend()

plt.subplots_adjust(top=0.8)

g.fig.suptitle('Survival by Gender , Age and Fare');
titanic.Embarked.value_counts().plot(kind='bar', alpha=0.55)

plt.title("Passengers per boarding location");
sns.factorplot(x = 'Embarked',y="Survived", data = titanic,color="r");
sns.set(font_scale=1)

g = sns.factorplot(x="Sex", y="Survived", col="Pclass",

                    data=titanic, saturation=.5,

                    kind="bar", ci=None, aspect=.6)

(g.set_axis_labels("", "Survival Rate")

    .set_xticklabels(["Men", "Women"])

    .set_titles("{col_name} {col_var}")

    .set(ylim=(0, 1))

    .despine(left=True))  

plt.subplots_adjust(top=0.8)

g.fig.suptitle('How many Men and Women Survived by Passenger Class');
ax = sns.boxplot(x="Survived", y="Age", 

                data=titanic)

ax = sns.stripplot(x="Survived", y="Age",

                   data=titanic, jitter=True,

                   edgecolor="gray")

sns.plt.title("Survival by Age",fontsize=12);
titanic.Age[titanic.Pclass == 1].plot(kind='kde')    

titanic.Age[titanic.Pclass == 2].plot(kind='kde')

titanic.Age[titanic.Pclass == 3].plot(kind='kde')

 # plots an axis lable

plt.xlabel("Age")    

plt.title("Age Distribution within classes")

# sets our legend for our graph.

plt.legend(('1st Class', '2nd Class','3rd Class'),loc='best') ;
corr=titanic.corr()#["Survived"]

plt.figure(figsize=(10, 10))



sns.heatmap(corr, vmax=.8, linewidths=0.01,

            square=True,annot=True,cmap='YlGnBu',linecolor="white")

plt.title('Correlation between features');
#correlation of features with target variable

titanic.corr()["Survived"]
g = sns.factorplot(x="Age", y="Embarked",

                    hue="Sex", row="Pclass",

                    data=titanic[titanic.Embarked.notnull()],

                    orient="h", size=2, aspect=3.5, 

                   palette={'male':"purple", 'female':"blue"},

                    kind="violin", split=True, cut=0, bw=.2);
#Lets check which rows have null Embarked column

titanic[titanic['Embarked'].isnull()]
sns.boxplot(x="Embarked", y="Fare", hue="Pclass", data=titanic);
titanic["Embarked"] = titanic["Embarked"].fillna('C')
#there is an empty fare column in test set

titanic_test.describe()
titanic_test[titanic_test['Fare'].isnull()]
#we can replace missing value in fare by taking median of all fares of those passengers 

#who share 3rd Passenger class and Embarked from 'S' 

def fill_missing_fare(df):

    median_fare=df[(df['Pclass'] == 3) & (df['Embarked'] == 'S')]['Fare'].median()

#'S'

       #print(median_fare)

    df["Fare"] = df["Fare"].fillna(median_fare)

    return df



titanic_test=fill_missing_fare(titanic_test)
titanic["Deck"]=titanic.Cabin.str[0]

titanic_test["Deck"]=titanic_test.Cabin.str[0]

titanic["Deck"].unique() # 0 is for null values
g = sns.factorplot("Survived", col="Deck", col_wrap=4,

                    data=titanic[titanic.Deck.notnull()],

                    kind="count", size=2.5, aspect=.8);
titanic = titanic.assign(Deck=titanic.Deck.astype(object)).sort("Deck")

g = sns.FacetGrid(titanic, col="Pclass", sharex=False,

                  gridspec_kws={"width_ratios": [5, 3, 3]})

g.map(sns.boxplot, "Deck", "Age");
titanic.Deck.fillna('Z', inplace=True)

titanic_test.Deck.fillna('Z', inplace=True)

titanic["Deck"].unique() # Z is for null values
# Create a family size variable including the passenger themselves

titanic["FamilySize"] = titanic["SibSp"] + titanic["Parch"]+1

titanic_test["FamilySize"] = titanic_test["SibSp"] + titanic_test["Parch"]+1

print(titanic["FamilySize"].value_counts())
# Discretize family size

titanic.loc[titanic["FamilySize"] == 1, "FsizeD"] = 'singleton'

titanic.loc[(titanic["FamilySize"] > 1)  &  (titanic["FamilySize"] < 5) , "FsizeD"] = 'small'

titanic.loc[titanic["FamilySize"] >4, "FsizeD"] = 'large'



titanic_test.loc[titanic_test["FamilySize"] == 1, "FsizeD"] = 'singleton'

titanic_test.loc[(titanic_test["FamilySize"] >1) & (titanic_test["FamilySize"] <5) , "FsizeD"] = 'small'

titanic_test.loc[titanic_test["FamilySize"] >4, "FsizeD"] = 'large'

print(titanic["FsizeD"].unique())

print(titanic["FsizeD"].value_counts())
sns.factorplot(x="FsizeD", y="Survived", data=titanic);
#Create feture for length of name 

# The .apply method generates a new series

titanic["NameLength"] = titanic["Name"].apply(lambda x: len(x))



titanic_test["NameLength"] = titanic_test["Name"].apply(lambda x: len(x))

#print(titanic["NameLength"].value_counts())



bins = [0, 20, 40, 57, 85]

group_names = ['short', 'okay', 'good', 'long']

titanic['NlengthD'] = pd.cut(titanic['NameLength'], bins, labels=group_names)

titanic_test['NlengthD'] = pd.cut(titanic_test['NameLength'], bins, labels=group_names)



sns.factorplot(x="NlengthD", y="Survived", data=titanic)

print(titanic["NlengthD"].unique())
import re



#A function to get the title from a name.

def get_title(name):

    # Use a regular expression to search for a title.  Titles always consist of capital and lowercase letters, and end with a period.

    title_search = re.search(' ([A-Za-z]+)\.', name)

    #If the title exists, extract and return it.

    if title_search:

        return title_search.group(1)

    return ""



#Get all the titles and print how often each one occurs.

titles = titanic["Name"].apply(get_title)

print(pd.value_counts(titles))





#Add in the title column.

titanic["Title"] = titles



# Titles with very low cell counts to be combined to "rare" level

rare_title = ['Dona', 'Lady', 'Countess','Capt', 'Col', 'Don', 

                'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer']



# Also reassign mlle, ms, and mme accordingly

titanic.loc[titanic["Title"] == "Mlle", "Title"] = 'Miss'

titanic.loc[titanic["Title"] == "Ms", "Title"] = 'Miss'

titanic.loc[titanic["Title"] == "Mme", "Title"] = 'Mrs'

titanic.loc[titanic["Title"] == "Dona", "Title"] = 'Rare Title'

titanic.loc[titanic["Title"] == "Lady", "Title"] = 'Rare Title'

titanic.loc[titanic["Title"] == "Countess", "Title"] = 'Rare Title'

titanic.loc[titanic["Title"] == "Capt", "Title"] = 'Rare Title'

titanic.loc[titanic["Title"] == "Col", "Title"] = 'Rare Title'

titanic.loc[titanic["Title"] == "Don", "Title"] = 'Rare Title'

titanic.loc[titanic["Title"] == "Major", "Title"] = 'Rare Title'

titanic.loc[titanic["Title"] == "Rev", "Title"] = 'Rare Title'

titanic.loc[titanic["Title"] == "Sir", "Title"] = 'Rare Title'

titanic.loc[titanic["Title"] == "Jonkheer", "Title"] = 'Rare Title'

titanic.loc[titanic["Title"] == "Dr", "Title"] = 'Rare Title'



#titanic.loc[titanic["Title"].isin(['Dona', 'Lady', 'Countess','Capt', 'Col', 'Don', 

#                'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer']), "Title"] = 'Rare Title'



#titanic[titanic['Title'].isin(['Dona', 'Lady', 'Countess'])]

#titanic.query("Title in ('Dona', 'Lady', 'Countess')")



titanic["Title"].value_counts()





titles = titanic_test["Name"].apply(get_title)

print(pd.value_counts(titles))



#Add in the title column.

titanic_test["Title"] = titles



# Titles with very low cell counts to be combined to "rare" level

rare_title = ['Dona', 'Lady', 'Countess','Capt', 'Col', 'Don', 

                'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer']



# Also reassign mlle, ms, and mme accordingly

titanic_test.loc[titanic_test["Title"] == "Mlle", "Title"] = 'Miss'

titanic_test.loc[titanic_test["Title"] == "Ms", "Title"] = 'Miss'

titanic_test.loc[titanic_test["Title"] == "Mme", "Title"] = 'Mrs'

titanic_test.loc[titanic_test["Title"] == "Dona", "Title"] = 'Rare Title'

titanic_test.loc[titanic_test["Title"] == "Lady", "Title"] = 'Rare Title'

titanic_test.loc[titanic_test["Title"] == "Countess", "Title"] = 'Rare Title'

titanic_test.loc[titanic_test["Title"] == "Capt", "Title"] = 'Rare Title'

titanic_test.loc[titanic_test["Title"] == "Col", "Title"] = 'Rare Title'

titanic_test.loc[titanic_test["Title"] == "Don", "Title"] = 'Rare Title'

titanic_test.loc[titanic_test["Title"] == "Major", "Title"] = 'Rare Title'

titanic_test.loc[titanic_test["Title"] == "Rev", "Title"] = 'Rare Title'

titanic_test.loc[titanic_test["Title"] == "Sir", "Title"] = 'Rare Title'

titanic_test.loc[titanic_test["Title"] == "Jonkheer", "Title"] = 'Rare Title'

titanic_test.loc[titanic_test["Title"] == "Dr", "Title"] = 'Rare Title'



titanic_test["Title"].value_counts()
titanic["Ticket"].tail()
titanic["TicketNumber"] = titanic["Ticket"].str.extract('(\d{2,})', expand=True)

titanic["TicketNumber"] = titanic["TicketNumber"].apply(pd.to_numeric)





titanic_test["TicketNumber"] = titanic_test["Ticket"].str.extract('(\d{2,})', expand=True)

titanic_test["TicketNumber"] = titanic_test["TicketNumber"].apply(pd.to_numeric)
#some rows in ticket column dont have numeric value so we got NaN there

titanic[titanic["TicketNumber"].isnull()]
titanic.TicketNumber.fillna(titanic["TicketNumber"].median(), inplace=True)

titanic_test.TicketNumber.fillna(titanic_test["TicketNumber"].median(), inplace=True)
from sklearn.preprocessing import LabelEncoder,OneHotEncoder



labelEnc=LabelEncoder()



cat_vars=['Embarked','Sex',"Title","FsizeD","NlengthD",'Deck']

for col in cat_vars:

    titanic[col]=labelEnc.fit_transform(titanic[col])

    titanic_test[col]=labelEnc.fit_transform(titanic_test[col])



titanic.head()
with sns.plotting_context("notebook",font_scale=1.5):

    sns.set_style("whitegrid")

    sns.distplot(titanic["Age"].dropna(),

                 bins=80,

                 kde=False,

                 color="red")

    sns.plt.title("Age Distribution")

    plt.ylabel("Count");
from sklearn.ensemble import RandomForestRegressor

#predicting missing values in age using Random Forest

def fill_missing_age(df):

    

    #Feature set

    age_df = df[['Age','Embarked','Fare', 'Parch', 'SibSp',

                 'TicketNumber', 'Title','Pclass','FamilySize',

                 'FsizeD','NameLength',"NlengthD",'Deck']]

    # Split sets into train and test

    train  = age_df.loc[ (df.Age.notnull()) ]# known Age values

    test = age_df.loc[ (df.Age.isnull()) ]# null Ages

    

    # All age values are stored in a target array

    y = train.values[:, 0]

    

    # All the other values are stored in the feature array

    X = train.values[:, 1::]

    

    # Create and fit a model

    rtr = RandomForestRegressor(n_estimators=2000, n_jobs=-1)

    rtr.fit(X, y)

    

    # Use the fitted model to predict the missing values

    predictedAges = rtr.predict(test.values[:, 1::])

    

    # Assign those predictions to the full data set

    df.loc[ (df.Age.isnull()), 'Age' ] = predictedAges 

    

    return df
titanic=fill_missing_age(titanic)

titanic_test=fill_missing_age(titanic_test)
with sns.plotting_context("notebook",font_scale=1.5):

    sns.set_style("whitegrid")

    sns.distplot(titanic["Age"].dropna(),

                 bins=80,

                 kde=False,

                 color="tomato")

    sns.plt.title("Age Distribution")

    plt.ylabel("Count")

    plt.xlim((15,100));
from sklearn import preprocessing



std_scale = preprocessing.StandardScaler().fit(titanic[['Age', 'Fare']])

titanic[['Age', 'Fare']] = std_scale.transform(titanic[['Age', 'Fare']])





std_scale = preprocessing.StandardScaler().fit(titanic_test[['Age', 'Fare']])

titanic_test[['Age', 'Fare']] = std_scale.transform(titanic_test[['Age', 'Fare']])
titanic.corr()["Survived"]
# Import the linear regression class

from sklearn.linear_model import LinearRegression

# Sklearn also has a helper that makes it easy to do cross validation

from sklearn.cross_validation import KFold



# The columns we'll use to predict the target

predictors = ["Pclass", "Sex", "Age","SibSp", "Parch", "Fare",

              "Embarked","NlengthD", "FsizeD", "Title","Deck"]

target="Survived"

# Initialize our algorithm class

alg = LinearRegression()



# Generate cross validation folds for the titanic dataset.  It return the row indices corresponding to train and test.

# We set random_state to ensure we get the same splits every time we run this.

kf = KFold(titanic.shape[0], n_folds=3, random_state=1)



predictions = []
for train, test in kf:

    # The predictors we're using the train the algorithm.  Note how we only take the rows in the train folds.

    train_predictors = (titanic[predictors].iloc[train,:])

    # The target we're using to train the algorithm.

    train_target = titanic[target].iloc[train]

    # Training the algorithm using the predictors and target.

    alg.fit(train_predictors, train_target)

    # We can now make predictions on the test fold

    test_predictions = alg.predict(titanic[predictors].iloc[test,:])

    predictions.append(test_predictions)
predictions = np.concatenate(predictions, axis=0)

# Map predictions to outcomes (only possible outcomes are 1 and 0)

predictions[predictions > .5] = 1

predictions[predictions <=.5] = 0





accuracy=sum(titanic["Survived"]==predictions)/len(titanic["Survived"])

accuracy
from sklearn import cross_validation

from sklearn.linear_model import LogisticRegression

from sklearn.model_selection import cross_val_score

from sklearn.model_selection import ShuffleSplit



predictors = ["Pclass", "Sex", "Fare", "Embarked","Deck","Age",

              "FsizeD", "NlengthD","Title","Parch"]



# Initialize our algorithm

lr = LogisticRegression(random_state=1)

# Compute the accuracy score for all the cross validation folds.

cv = ShuffleSplit(n_splits=10, test_size=0.3, random_state=50)



scores = cross_val_score(lr, titanic[predictors], 

                                          titanic["Survived"],scoring='f1', cv=cv)

# Take the mean of the scores (because we have one for each fold)

print(scores.mean())
from sklearn import cross_validation

from sklearn.ensemble import RandomForestClassifier

from sklearn.cross_validation import KFold

from sklearn.model_selection import cross_val_predict



import numpy as np

predictors = ["Pclass", "Sex", "Age",

              "Fare","NlengthD","NameLength", "FsizeD", "Title","Deck"]



# Initialize our algorithm with the default paramters

# n_estimators is the number of trees we want to make

# min_samples_split is the minimum number of rows we need to make a split

# min_samples_leaf is the minimum number of samples we can have at the place where a tree branch ends (the bottom points of the tree)

rf = RandomForestClassifier(random_state=1, n_estimators=10, min_samples_split=2, 

                            min_samples_leaf=1)

kf = KFold(titanic.shape[0], n_folds=5, random_state=1)

cv = ShuffleSplit(n_splits=10, test_size=0.3, random_state=50)



predictions = cross_validation.cross_val_predict(rf, titanic[predictors],titanic["Survived"],cv=kf)

predictions = pd.Series(predictions)

scores = cross_val_score(rf, titanic[predictors], titanic["Survived"],

                                          scoring='f1', cv=kf)

# Take the mean of the scores (because we have one for each fold)

print(scores.mean())
predictors = ["Pclass", "Sex", "Age",

              "Fare","NlengthD","NameLength", "FsizeD", "Title","Deck","TicketNumber"]

rf = RandomForestClassifier(random_state=1, n_estimators=50, max_depth=9,min_samples_split=6, min_samples_leaf=4)

rf.fit(titanic[predictors],titanic["Survived"])

kf = KFold(titanic.shape[0], n_folds=5, random_state=1)

predictions = cross_validation.cross_val_predict(rf, titanic[predictors],titanic["Survived"],cv=kf)

predictions = pd.Series(predictions)

scores = cross_val_score(rf, titanic[predictors], titanic["Survived"],scoring='f1', cv=kf)

# Take the mean of the scores (because we have one for each fold)

print(scores.mean())
importances=rf.feature_importances_

std = np.std([rf.feature_importances_ for tree in rf.estimators_],

             axis=0)

indices = np.argsort(importances)[::-1]

sorted_important_features=[]

for i in indices:

    sorted_important_features.append(predictors[i])

#predictors=titanic.columns

plt.figure()

plt.title("Feature Importances By Random Forest Model")

plt.bar(range(np.size(predictors)), importances[indices],

       color="r", yerr=std[indices], align="center")

plt.xticks(range(np.size(predictors)), sorted_important_features, rotation='vertical')



plt.xlim([-1, np.size(predictors)]);
import numpy as np

from sklearn.ensemble import GradientBoostingClassifier



from sklearn.feature_selection import SelectKBest, f_classif

from sklearn.cross_validation import KFold

### matplotlib inline

import matplotlib.pyplot as plt

#predictors = ["Pclass", "Sex", "Age", "Fare",

 #             "FsizeD", "Embarked", "NlengthD","Deck","TicketNumber"]

predictors = ["Pclass", "Sex", "Age",

              "Fare","NlengthD", "FsizeD","NameLength","Deck","Embarked"]

# Perform feature selection

selector = SelectKBest(f_classif, k=5)

selector.fit(titanic[predictors], titanic["Survived"])



# Get the raw p-values for each feature, and transform from p-values into scores

scores = -np.log10(selector.pvalues_)



indices = np.argsort(scores)[::-1]



sorted_important_features=[]

for i in indices:

    sorted_important_features.append(predictors[i])



plt.figure()

plt.title("Feature Importances By SelectKBest")

plt.bar(range(np.size(predictors)), scores[indices],

       color="seagreen", yerr=std[indices], align="center")

plt.xticks(range(np.size(predictors)), sorted_important_features, rotation='vertical')



plt.xlim([-1, np.size(predictors)]);
from sklearn import cross_validation

from sklearn.linear_model import LogisticRegression

predictors = ["Pclass", "Sex", "Age", "Fare", "Embarked","NlengthD",

              "FsizeD", "Title","Deck"]



# Initialize our algorithm

lr = LogisticRegression(random_state=1)

# Compute the accuracy score for all the cross validation folds.  

cv = ShuffleSplit(n_splits=10, test_size=0.3, random_state=50)

scores = cross_val_score(lr, titanic[predictors], titanic["Survived"], scoring='f1',cv=cv)

print(scores.mean())
from sklearn.ensemble import AdaBoostClassifier

predictors = ["Pclass", "Sex", "Age", "Fare", "Embarked","NlengthD",

              "FsizeD", "Title","Deck","TicketNumber"]

adb=AdaBoostClassifier()

adb.fit(titanic[predictors],titanic["Survived"])

cv = ShuffleSplit(n_splits=10, test_size=0.3, random_state=50)

scores = cross_val_score(adb, titanic[predictors], titanic["Survived"], scoring='f1',cv=cv)

print(scores.mean())
predictions=["Pclass", "Sex", "Age", "Fare", "Embarked","NlengthD",

              "FsizeD", "Title","Deck","NameLength","TicketNumber"]

from sklearn.ensemble import VotingClassifier

eclf1 = VotingClassifier(estimators=[

        ('lr', lr), ('rf', rf), ('adb', adb)], voting='soft')

eclf1 = eclf1.fit(titanic[predictors], titanic["Survived"])

predictions=eclf1.predict(titanic[predictors])

predictions



test_predictions=eclf1.predict(titanic_test[predictors])



test_predictions=test_predictions.astype(int)

submission = pd.DataFrame({

        "PassengerId": titanic_test["PassengerId"],

        "Survived": test_predictions

    })



submission.to_csv("titanic_submission.csv", index=False)