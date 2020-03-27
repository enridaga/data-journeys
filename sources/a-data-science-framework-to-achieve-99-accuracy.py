
# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 





import sys #access to system parameters https://docs.python.org/3/library/sys.html

print("Python version: {}". format( sys.version))



import pandas as pd #collection of functions for data processing and analysis modeled after R dataframes with SQL like features

print("pandas version: {}". format( pd.__version__))



import matplotlib #collection of functions for scientific and publication-ready visualization

print("matplotlib version: {}". format( matplotlib.__version__)) 



import numpy as np #foundational package for scientific computing

print("NumPy version: {}". format( np.__version__)) 



import scipy as sp #collection of functions for scientific computing and advance mathematics

print("SciPy version: {}". format( sp.__version__)) 





import IPython 

from IPython import display #pretty printing of dataframes in Jupyter notebook

from IPython.display import Image

print(" IPython version: {}". format( IPython.__version__)) 



import sklearn #collection of machine learning algorithms

print("scikit-learn version: {}". format( sklearn.__version__))



#measure execution of code snippets: https://docs.python.org/3/library/timeit.html

import timeit as t

import random

from time import time



# Ignore warnings

import warnings

warnings.filterwarnings('ignore')



print('-'*25)





# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
#Common Model Algorithms

from sklearn import svm, tree, linear_model, neighbors, naive_bayes, ensemble, discriminant_analysis, gaussian_process



#Common Model Helpers

from sklearn.preprocessing import Imputer , Normalizer , scale, OneHotEncoder, LabelEncoder

from sklearn.model_selection import train_test_split, StratifiedShuffleSplit, GridSearchCV, StratifiedKFold, RandomizedSearchCV

from sklearn.feature_selection import RFECV

from sklearn import metrics



# Visualization

import matplotlib as mpl

import matplotlib.pyplot as plt

import matplotlib.pylab as pylab

import seaborn as sns

from pandas.tools.plotting import scatter_matrix



# Configure visualizations

##%matplotlib inline = show plots in Jupyter Notebook browser

#%matplotlib inline

mpl.style.use( 'ggplot' )

sns.set_style( 'white' )

pylab.rcParams[ 'figure.figsize' ] = 8 , 6
#load as dataframe

data_raw = pd.read_csv('../input/train.csv')



#Note: The test file is really validation data for competition submission, because we do not know the survival status

#We will create real test data in a later section, so we can evaluate our model before competition submission

validation_raw  = pd.read_csv('../input/test.csv')



#preview data

print (data_raw.info())

#data_raw.head() #https://pandas.pydata.org/pandas-docs/stable/generated/pandas.DataFrame.head.html

#data_raw.tail() #https://pandas.pydata.org/pandas-docs/stable/generated/pandas.DataFrame.tail.html

data_raw.sample(10) #https://pandas.pydata.org/pandas-docs/stable/generated/pandas.DataFrame.sample.html
#Quantitative Descriptive Statistics

print(data_raw.isnull().sum())

print("-"*10)



#Qualitative Descriptive Statistics

print(data_raw['Sex'].value_counts())

print("-"*10)

print(data_raw['Embarked'].value_counts())

data_raw.describe(include = 'all')
#create a copy of data

#remember python assignment or equal passes by reference vs values, so we use the copy function: https://stackoverflow.com/questions/46327494/python-pandas-dataframe-copydeep-false-vs-copydeep-true-vs

data1 = data_raw.copy(deep = True)
#cleanup age with median

data1['Age'].fillna(data1['Age'].median(), inplace = True)



#preview data again

print(data1.isnull().sum())

print("-"*10)

#print(data_raw.isnull().sum())
#cleanup embarked with mode

data1['Embarked'].fillna(data1['Embarked'].mode()[0], inplace = True)



#preview data again

print(data1.isnull().sum())

print("-"*10)

#print(data_raw.isnull().sum())
#delete the cabin feature/column and others previously stated to exclude

drop_column = ['Cabin','PassengerId','Name', 'Ticket']

data1.drop(drop_column, axis=1, inplace = True)



#preview data again

print(data1.isnull().sum())

print("-"*10)

#print(data_raw.isnull().sum())
#convert to explicit category data type

#data1['Pclass'] = data1['Pclass'].astype('category')

#data1['Sex'] = data1['Sex'].astype('category')

#data1['Embarked'] = data1['Embarked'].astype('category')



print("Original Features: ", list(data1.columns), '\n')

data1_dummy = pd.get_dummies(data1)

print("Features with Dummies: ", data1_dummy.columns.values, '\n')



print (data1_dummy.dtypes)

data1_dummy.head()
#Quantitative Descriptive Statistics

print (data1.info())

print("-"*10)

print(data1.isnull().sum())

print("-"*10)



#Qualitative Descriptive Statistics

print(data1.Sex.value_counts())

print("-"*10)

print(data1.Embarked.value_counts())

print("-"*10)

data1.describe(include = 'all')
#define x and y variables for original features aka feature selection

data1_x = ['Pclass','Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']

data1_y = ['Survived']

data1_xy = data1_x + data1_y

print('Original X Y: ', data1_xy, '\n')



#define x and y variables for dummy features aka feature selection

data1_dummy_x = data1_dummy.iloc[:,1:].columns.tolist()

data1_dummy_y = data1_dummy.iloc[:,0:1].columns.tolist()

data1_dummy_xy = data1_dummy_x + data1_dummy_y

print('Dummy Coding X Y: ', data1_dummy_xy, '\n')



#split train and test data with function defaults

train1_x, test1_x, train1_y, test1_y = train_test_split(data1[data1_x], data1[data1_y])

train1_dummy_x, test1_dummy_x, train1_dummy_y, test1_dummy_y = train_test_split(data1_dummy[data1_dummy_x], data1_dummy[data1_dummy_y])



print("Data Shape: {}".format(data1.shape))

print("Train1 Shape: {}".format(train1_x.shape))

print("Test1 Shape: {}".format(test1_x.shape), '\n')



print("Data1_Dummy Shape: {}".format(data1_dummy.shape))

print("Train1_Dummy Shape: {}".format(train1_dummy_x.shape))

print("Test1_Dummy Shape: {}".format(test1_dummy_x.shape), '\n')
#Correlation by Survival; excluding continuous variables of age and fare

#using group by https://pandas.pydata.org/pandas-docs/stable/generated/pandas.DataFrame.groupby.html

for x in data1_x:

    if data1[x].dtype != 'float64' :

        print('Survival Correlation by:', x)

        print(data1[[x, data1_y[0]]].groupby(x, as_index=False).mean())

        print('-'*10, '\n')



###Feature Engineering###



#Family Size

data1['FamilySize'] = data1 ['SibSp'] + data1['Parch'] + 1

print ('Survival Correlation by: Family Size \n',

       data1[['FamilySize', 'Survived']].groupby(['FamilySize'], as_index=False).mean(),

      '\n','-'*10, '\n')



#IsAlone

data1['IsAlone'] = 1 #create a new feature and initialize to yes/1 is alone

data1['IsAlone'].loc[data1['FamilySize'] > 1] = 0 # now update to no/0 if family size is greater than 1

print ('Survival Correlation by: IsAlone \n',

       data1[['IsAlone', 'Survived']].groupby(['IsAlone'], as_index=False).mean(),

      '\n','-'*10, '\n')



##Handling continuous data##

#qcut or frequency bins: https://pandas.pydata.org/pandas-docs/stable/generated/pandas.qcut.html

#cut or value bins: https://pandas.pydata.org/pandas-docs/stable/generated/pandas.cut.html

#qcut vs cut: https://stackoverflow.com/questions/30211923/what-is-the-difference-between-pandas-qcut-and-pandas-cut



#Fare Bins/Buckets using qcut

data1['FareBin'] = pd.qcut(data1['Fare'], 4)

print ('Survival Correlation by: FareBin \n',

       data1[['FareBin', 'Survived']].groupby(['FareBin'], as_index=False).mean(),

      '\n','-'*10, '\n')



#Age Bins/Buckets using cut

data1['AgeBin'] = pd.cut(data1['Age'].astype(int), 5)

print ('Survival Correlation by: AgeBin \n',

       data1[['AgeBin', 'Survived']].groupby(['AgeBin'], as_index=False).mean(),

      '\n','-'*10, '\n')





#simple frequency table of class and sex

#using crosstabs: https://pandas.pydata.org/pandas-docs/stable/generated/pandas.crosstab.html

print(pd.crosstab(data1['Pclass'], data1['Sex']))

#optional plotting w/pandas: https://pandas.pydata.org/pandas-docs/stable/visualization.html



#we will use matplotlib.pyplot: https://matplotlib.org/api/pyplot_api.html

#to organize our graphics will use figure: https://matplotlib.org/api/_as_gen/matplotlib.pyplot.figure.html#matplotlib.pyplot.figure

#subplot: https://matplotlib.org/api/_as_gen/matplotlib.pyplot.subplot.html#matplotlib.pyplot.subplot

#and subplotS: https://matplotlib.org/api/_as_gen/matplotlib.pyplot.subplots.html?highlight=matplotlib%20pyplot%20subplots#matplotlib.pyplot.subplots



#graph distribution of quantitative data

plt.figure(figsize=[16,12])



plt.subplot(231)

plt.boxplot(x=data1['Fare'], showmeans = True, meanline = True)

plt.title('Fare Boxplot')

plt.ylabel('Fare ($)')



plt.subplot(232)

plt.boxplot(data1['Age'], showmeans = True, meanline = True)

plt.title('Age Boxplot')

plt.ylabel('Age (Years)')



plt.subplot(233)

plt.boxplot(data1['FamilySize'], showmeans = True, meanline = True)

plt.title('Family Size Boxplot')

plt.ylabel('Family Size (#)')



plt.subplot(234)

plt.hist(x = [data1[data1['Survived']==1]['Fare'], data1[data1['Survived']==0]['Fare']], 

         stacked=True, color = ['g','r'],label = ['Survived','Dead'])

plt.title('Fare Histogram by Survival')

plt.xlabel('Fare ($)')

plt.ylabel('# of Passengers')

plt.legend()



plt.subplot(235)

plt.hist(x = [data1[data1['Survived']==1]['Age'], data1[data1['Survived']==0]['Age']], 

         stacked=True, color = ['g','r'],label = ['Survived','Dead'])

plt.title('Age Histogram by Survival')

plt.xlabel('Age (Years)')

plt.ylabel('# of Passengers')

plt.legend()



plt.subplot(236)

plt.hist(x = [data1[data1['Survived']==1]['FamilySize'], data1[data1['Survived']==0]['FamilySize']], 

         stacked=True, color = ['g','r'],label = ['Survived','Dead'])

plt.title('Family Size Histogram by Survival')

plt.xlabel('Family Size (#)')

plt.ylabel('# of Passengers')

plt.legend()





#we will use seaborn graphics for multi-variable comparison: https://seaborn.pydata.org/api.html



#graph distribution of qualitative data: Pclass

#we know class mattered in survival, now let's compare class and a 2nd feature

fig, (axis1,axis2,axis3) = plt.subplots(1,3,figsize=(16,12))



sns.boxplot(x = 'Pclass', y = 'Fare', hue = 'Survived', data = data1, ax = axis1)

axis1.set_title('Pclass vs Fare Survival Comparison')



sns.violinplot(x = 'Pclass', y = 'Age', hue = 'Survived', data = data1, split = True, ax = axis2)

axis2.set_title('Pclass vs Age Survival Comparison')



sns.boxplot(x = 'Pclass', y ='FamilySize', hue = 'Survived', data = data1, ax = axis3)

axis3.set_title('Pclass vs Family Size Survival Comparison')





#graph distribution of qualitative data: Sex

#we know sex mattered in survival, now let's compare sex and a 2nd feature

fig, qaxis = plt.subplots(1,3,figsize=(16,12))



sns.barplot(x = 'Sex', y = 'Survived', hue = 'Embarked', data=data1, ax = qaxis[0])

axis1.set_title('Sex vs Embarked Survival Comparison')



sns.barplot(x = 'Sex', y = 'Survived', hue = 'Pclass', data=data1, ax  = qaxis[1])

axis1.set_title('Sex vs Pclass Survival Comparison')



sns.barplot(x = 'Sex', y = 'Survived', hue = 'IsAlone', data=data1, ax  = qaxis[2])

axis1.set_title('Sex vs IsAlone Survival Comparison')





#graph individual features by survival

fig, saxis = plt.subplots(2, 3,figsize=(16,12))



sns.barplot(x = 'Embarked', y = 'Survived', data=data1, ax = saxis[0,0])

sns.barplot(x = 'Pclass', y = 'Survived', order=[1,2,3], data=data1, ax = saxis[0,1])

sns.barplot(x = 'IsAlone', y = 'Survived', order=[1,0], data=data1, ax = saxis[0,2])



g = sns.factorplot(x = 'FareBin', y = 'Survived',  data=data1, ax = saxis[1,0])

h = sns.factorplot(x = 'AgeBin', y = 'Survived',  data=data1, ax = saxis[1,1])

j = sns.factorplot(x = 'FamilySize', y = 'Survived', data=data1, ax = saxis[1,2])

#close factor plot facetgrid we don't need: https://stackoverflow.com/questions/33925494/seaborn-produces-separate-figures-in-subplots

plt.close(g.fig)

plt.close(h.fig)

plt.close(j.fig)





#more side-by-side comparisons

fig, (maxis1, maxis2) = plt.subplots(1, 2,figsize=(16,12))



#how does family size factor with sex & survival

sns.pointplot(x="FamilySize", y="Survived", hue="Sex", data=data1,

              palette={"male": "blue", "female": "pink"},

              markers=["*", "o"], linestyles=["-", "--"], ax = maxis1)



#how does class factor with sex & survival

sns.pointplot(x="Pclass", y="Survived", hue="Sex", data=data1,

              palette={"male": "blue", "female": "pink"},

              markers=["*", "o"], linestyles=["-", "--"], ax = maxis2)



#how does embark port factor with class, sex, and survival

plt.figure()

sns.factorplot(x="Pclass", y="Survived", hue="Sex", col="Embarked",

                   data=data1, aspect=0.9, size=3.5, ci=95.0)





#plot distributions of Age of passengers who survived or did not survive

def plot_distribution( df , feature , target , **kwargs ):

    plt.figure()

    row = kwargs.get( 'row', None )

    col = kwargs.get( 'col', None )

    facet = sns.FacetGrid( df, hue=target, aspect=4, row=row ,col=col )

    facet.map( sns.kdeplot, feature, shade= True )

    facet.set( xlim=( 0 , df[feature].max() ) )

    facet.add_legend()



plot_distribution(data1 , feature='Age' , target='Survived' , row='Sex')





#pair plots

plt.figure()

f = sns.pairplot(data1, hue='Survived', palette = 'seismic',size=1.2,diag_kind = 'kde',diag_kws=dict(shade=True),plot_kws=dict(s=10) )

f.set(xticklabels=[])





#correlation heatmap

def correlation_heatmap(df):

    plt.figure()

    _ , ax = plt.subplots(figsize =(14, 12))

    #colormap = plt.cm.viridis

    colormap = sns.diverging_palette(220, 10, as_cmap = True)

    

    _ = sns.heatmap(

        df.corr(), 

        cmap = colormap,

        square=True, 

        cbar_kws={'shrink':.9 }, 

        ax=ax,

        annot=True, 

        linewidths=0.1,vmax=1.0, linecolor='white',

        annot_kws={'fontsize':12 }

    )

    

    plt.title('Pearson Correlation of Features', y=1.05, size=15)



correlation_heatmap(data1)









#Machine Learning Algorithm (MLA) Selection and initialization

MLA = [

    #Ensemble Methods

    ensemble.AdaBoostClassifier(),

    ensemble.BaggingClassifier(),

    ensemble.ExtraTreesClassifier(),

    ensemble.GradientBoostingClassifier(),

    ensemble.RandomForestClassifier(n_estimators = 100),

    #ensemble.VotingClassifier(),



    #Gaussian Processes

    gaussian_process.GaussianProcessClassifier(),

    

    #GLM

    linear_model.LogisticRegressionCV(),

    linear_model.PassiveAggressiveClassifier(),

    linear_model. RidgeClassifierCV(),

    linear_model.SGDClassifier(),

    linear_model.Perceptron(),

    

    #Navies Bayes

    naive_bayes.GaussianNB(),

    

    #Nearest Neighbor

    neighbors.KNeighborsClassifier(n_neighbors = 3),

    

    #SVM

    svm.SVC(probability=True),

    svm.LinearSVC(),

    

    #Trees    

    tree.DecisionTreeClassifier(),

    tree.ExtraTreeClassifier(),

    

    #Discriminant Analysis

    discriminant_analysis.LinearDiscriminantAnalysis(),

    discriminant_analysis.QuadraticDiscriminantAnalysis()

    ]



#create table to compare MLA

MLA_columns = ['MLA Name', 'MLA Parameters','MLA Train Accuracy', 'MLA Test Accuracy', 'MLA Best Accuracy', 'MLA Time']

MLA_compare = pd.DataFrame(columns = MLA_columns)





#index through MLA and save to table

row_index = 0

for alg in MLA:

    #set name column

    MLA_compare.loc[row_index, 'MLA Name'] = alg.__class__.__name__

    

    #get and set algorithm, execution time, and accuracy

    start_time = t.default_timer()

    alg.fit(train1_dummy_x,train1_dummy_y)

    run_time = t.default_timer() - start_time     

    

    MLA_compare.loc[row_index, 'MLA Parameters'] = str(alg.get_params())

    MLA_compare.loc[row_index, 'MLA Time'] = run_time

    MLA_compare.loc[row_index, 'MLA Train Accuracy'] = alg.score(train1_dummy_x,train1_dummy_y)*100

    MLA_compare.loc[row_index, 'MLA Test Accuracy'] = alg.score(test1_dummy_x,test1_dummy_y)*100

    

    row_index+=1



#print and sort table: https://pandas.pydata.org/pandas-docs/stable/generated/pandas.DataFrame.sort_values.html

MLA_compare.sort_values(by = ['MLA Test Accuracy'], ascending = False, inplace = True)

print(MLA_compare)

MLA_compare





#logreg = LogisticRegression()

#%timeit logreg.fit(train1_dummy_x,train1_dummy_y) 

#print("LogReg Training w/dummy set score: {:.2f}". format(logreg.score(train1_dummy_x,train1_dummy_y)*100)) 

#print("LogReg Test w/dummy set score: {:.2f}". format(logreg.score(test1_dummy_x,test1_dummy_y)*100))

print('-'*10,)



#barplot using https://seaborn.pydata.org/generated/seaborn.barplot.html

sns.barplot(x='MLA Test Accuracy', y = 'MLA Name', data = MLA_compare, color = 'm')



#prettify using pyplot: https://matplotlib.org/api/pyplot_api.html

plt.title('Machine Learning Algorithm Accuracy Score \n')

plt.xlabel('Accuracy Score (%)')

plt.ylabel('Algorithm')
#create a 2nd copy of our data

data2 = data_raw.copy(deep = True)



#Feature Engineering

#Note: we will not do any imputing missing data at this time

data2['FareBin'] = pd.qcut(data1['Fare'], 4)

data2['AgeBin'] = pd.cut(data1['Age'].astype(int), 5)

data2['Title'] = data2['Name'].str.split(", ", expand=True)[1].str.split(".", expand=True)[0]

#data2.sample(10)





#coin flip model with random 1/survived 0/died



#Iterate over DataFrame rows as (index, Series) pairs: https://pandas.pydata.org/pandas-docs/stable/generated/pandas.DataFrame.iterrows.html

for index, row in data2.iterrows(): 

    #random number generator: https://docs.python.org/2/library/random.html

    if random.random() > .5:     # Random float x, 0.0 <= x < 1.0    

        data2.set_value(index, 'Random_Predict', 1) #predict survived/1

    else: 

        data2.set_value(index, 'Random_Predict', 0) #predict died/0

    



#score random guess of survival. Use shortcut 1 = Right Guess and 0 = Wrong Guess

#the mean of the column will then equal the accuracy

data2['Random_Score'] = 0 #assume prediction wrong

data2.loc[(data2['Survived'] == data2['Random_Predict']), 'Random_Score'] = 1 #set to 1 for correct prediction

print('Coin Flip Model Accuracy: {:.2f}%'.format(data2['Random_Score'].mean()*100))



#we can also use scikit's accuracy_score function to save us a few lines of code

#http://scikit-learn.org/stable/modules/generated/sklearn.metrics.accuracy_score.html#sklearn.metrics.accuracy_score

print('Coin Flip Model Accuracy w/SciKit: {:.2f}%'.format(metrics.accuracy_score(data2['Survived'], data2['Random_Predict'])*100))







#group by or pivot table: https://pandas.pydata.org/pandas-docs/stable/generated/pandas.DataFrame.groupby.html

pivot_female = data2[data2.Sex=='female'].groupby(['Sex','Pclass', 'Embarked','FareBin'])['Survived'].mean()

print('\n\nSurvival Decision Tree w/Female Node: \n',pivot_female)



pivot_male = data2[data2.Sex=='male'].groupby(['Sex','Title'])['Survived'].mean()

print('\n\nSurvival Decision Tree w/Male Node: \n',pivot_male)



#Question 1: Were you on the Titanic; majority died

data2['Tree_Predict'] = 0



#Question 2: Are you female; majority survived

data2.loc[(data2['Sex'] == 'female'), 'Tree_Predict'] = 1



#Question 3A Female - Class and Question 4 Embarked gain minimum information



#Question 5B Female - FareBin; set anything less than .5 in female node decision tree back to 0

data2.loc[(data2['Sex'] == 'female') & (data2['Pclass'] == 3) & 

          (data2['Embarked'] == 'C') & (data2['Fare'] > 8) & (data2['Fare'] <15),

          'Tree_Predict'] = 0



data2.loc[(data2['Sex'] == 'female') & (data2['Pclass'] == 3) & 

          (data2['Embarked'] == 'S') & (data2['Fare'] > 8),

          'Tree_Predict'] = 0



#Question 3B Male: Title; set anything greater than .5 to 1 for majority survived

male_title = ['Master', 'Sir']

data2.loc[(data2['Sex'] == 'male') &

          (data2['Title'].isin(male_title)),

          'Tree_Predict'] = 1



#Score Decision Tree Model

print('\n\nDecision Tree Model Accuracy: {:.2f}%\n'.format(metrics.accuracy_score(data2['Survived'], data2['Tree_Predict'])*100))



#Accuracy Summary Report with http://scikit-learn.org/stable/modules/generated/sklearn.metrics.classification_report.html#sklearn.metrics.classification_report

#Where recall score = (true positives)/(true positive + false negative) w/1 being best:http://scikit-learn.org/stable/modules/generated/sklearn.metrics.recall_score.html#sklearn.metrics.recall_score

#And F1 score = weighted average of precision and recall w/1 being best: http://scikit-learn.org/stable/modules/generated/sklearn.metrics.f1_score.html#sklearn.metrics.f1_score

print(metrics.classification_report(data2['Survived'], data2['Tree_Predict']))



#Credit: http://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html

import itertools

def plot_confusion_matrix(cm, classes,

                          normalize=False,

                          title='Confusion matrix',

                          cmap=plt.cm.Blues):

    """

    This function prints and plots the confusion matrix.

    Normalization can be applied by setting `normalize=True`.

    """

    if normalize:

        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

        print("Normalized confusion matrix")

    else:

        print('Confusion matrix, without normalization')



    print(cm)



    plt.imshow(cm, interpolation='nearest', cmap=cmap)

    plt.title(title)

    plt.colorbar()

    tick_marks = np.arange(len(classes))

    plt.xticks(tick_marks, classes, rotation=45)

    plt.yticks(tick_marks, classes)



    fmt = '.2f' if normalize else 'd'

    thresh = cm.max() / 2.

    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):

        plt.text(j, i, format(cm[i, j], fmt),

                 horizontalalignment="center",

                 color="white" if cm[i, j] > thresh else "black")



    plt.tight_layout()

    plt.ylabel('True label')

    plt.xlabel('Predicted label')



# Compute confusion matrix

cnf_matrix = metrics.confusion_matrix(data2['Survived'], data2['Tree_Predict'])

np.set_printoptions(precision=2)



class_names = ['Dead', 'Survived']

# Plot non-normalized confusion matrix

plt.figure()

plot_confusion_matrix(cnf_matrix, classes=class_names,

                      title='Confusion matrix, without normalization')



# Plot normalized confusion matrix

plt.figure()

plot_confusion_matrix(cnf_matrix, classes=class_names, normalize=True, 

                      title='Normalized confusion matrix')
