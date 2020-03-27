

# here we will import the libraries used for machine learning

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv), data manipulation as in SQL

import matplotlib.pyplot as plt # this is used for the plot the graph 

import seaborn as sns # used for plot interactive graph. I like it most for plot

#%matplotlib inline

from sklearn.linear_model import LogisticRegression # to apply the Logistic regression

from sklearn.model_selection import train_test_split # to split the data into two parts

from sklearn.cross_validation import KFold # use for cross validation

from sklearn.model_selection import GridSearchCV# for tuning parameter

from sklearn.ensemble import RandomForestClassifier # for random forest classifier

from sklearn.naive_bayes import GaussianNB

from sklearn.neighbors import KNeighborsClassifier

from sklearn.tree import DecisionTreeClassifier

from sklearn import svm # for Support Vector Machine

from sklearn import metrics # for the check the error and accuracy of the model

# Any results you write to the current directory are saved as output.

# dont worry about the error if its not working then insteda of model_selection we can use cross_validation
data = pd.read_csv("../input/data.csv",header=0)# here header 0 means the 0 th row is our coloumn 

                                                # header in data
# have a look at the data

print(data.head(2))# as u can see our data have imported and having 33 columns

# head is used for to see top 5 by default I used 2 so it will print 2 rows

# If we will use print(data.tail(2))# it will print last 2 rows in data
# now lets look at the type of data we have. We can use 

data.info()
# now we can drop this column Unnamed: 32

data.drop("Unnamed: 32",axis=1,inplace=True) # in this process this will change in our data itself 

# if you want to save your old data then you can use below code

# data1=data.drop("Unnamed:32",axis=1)

# here axis 1 means we are droping the column
# here you can check the column has been droped

data.columns # this gives the column name which are persent in our data no Unnamed: 32 is not now there
# like this we also don't want the Id column for our analysis

data.drop("id",axis=1,inplace=True)
# As I said above the data can be divided into three parts.lets divied the features according to their category

features_mean= list(data.columns[1:11])

features_se= list(data.columns[11:20])

features_worst=list(data.columns[21:31])

print(features_mean)

print("-----------------------------------")

print(features_se)

print("------------------------------------")

print(features_worst)
# lets now start with features_mean 

# now as ou know our diagnosis column is a object type so we can map it to integer value

data['diagnosis']=data['diagnosis'].map({'M':1,'B':0})
data.describe() # this will describe the all statistical function of our data
# lets get the frequency of cancer stages

sns.countplot(data['diagnosis'],label="Count")
# from this graph we can see that there is a more number of bengin stage of cancer which can be cure
# now lets draw a correlation graph so that we can remove multi colinearity it means the columns are

# dependenig on each other so we should avoid it because what is the use of using same column twice

# lets check the correlation between features

# now we will do this analysis only for features_mean then we will do for others and will see who is doing best

corr = data[features_mean].corr() # .corr is used for find corelation

plt.figure(figsize=(14,14))

sns.heatmap(corr, cbar = True,  square = True, annot=True, fmt= '.2f',annot_kws={'size': 15},

           xticklabels= features_mean, yticklabels= features_mean,

           cmap= 'coolwarm') # for more on heatmap you can visit Link(http://seaborn.pydata.org/generated/seaborn.heatmap.html)

prediction_var = ['texture_mean','perimeter_mean','smoothness_mean','compactness_mean','symmetry_mean']

# now these are the variables which will use for prediction
#now split our data into train and test

train, test = train_test_split(data, test_size = 0.3)# in this our main data is splitted into train and test

# we can check their dimension

print(train.shape)

print(test.shape)
train_X = train[prediction_var]# taking the training data input 

train_y=train.diagnosis# This is output of our training data

# same we have to do for test

test_X= test[prediction_var] # taking test data inputs

test_y =test.diagnosis   #output value of test dat
model=RandomForestClassifier(n_estimators=100)# a simple random forest model
model.fit(train_X,train_y)# now fit our model for traiing data
prediction=model.predict(test_X)# predict for the test data

# prediction will contain the predicted value by our model predicted values of dignosis column for test inputs
metrics.accuracy_score(prediction,test_y) # to check the accuracy

# here we will use accuracy measurement between our predicted value and our test output values
# lets now try with SVM
model = svm.SVC()

model.fit(train_X,train_y)

prediction=model.predict(test_X)

metrics.accuracy_score(prediction,test_y)
prediction_var = features_mean # taking all features
train_X= train[prediction_var]

train_y= train.diagnosis

test_X = test[prediction_var]

test_y = test.diagnosis
model=RandomForestClassifier(n_estimators=100)
model.fit(train_X,train_y)

prediction = model.predict(test_X)

metrics.accuracy_score(prediction,test_y)
featimp = pd.Series(model.feature_importances_, index=prediction_var).sort_values(ascending=False)

print(featimp) # this is the property of Random Forest classifier that it provide us the importance 

# of the features used
# first lets do with SVM also using all features
model = svm.SVC()

model.fit(train_X,train_y)

prediction=model.predict(test_X)

metrics.accuracy_score(prediction,test_y)
# as you can see the accuracy of SVM decrease very much

# now lets take only top 5 important features given by RandomForest classifier
prediction_var=['concave points_mean','perimeter_mean' , 'concavity_mean' , 'radius_mean','area_mean']      
train_X= train[prediction_var]

train_y= train.diagnosis

test_X = test[prediction_var]

test_y = test.diagnosis
model=RandomForestClassifier(n_estimators=100)

model.fit(train_X,train_y)

prediction = model.predict(test_X)

metrics.accuracy_score(prediction,test_y)
model = svm.SVC()

model.fit(train_X,train_y)

prediction=model.predict(test_X)

metrics.accuracy_score(prediction,test_y)
# so from this discussion we got multi colinearty effecting our SVM part a lot 

# but its not affecting so much randomforest because for random forest we dont need to make so much effort for our analysis part

# now lets do with the 3rd part of data which is worst

# first start with all features_worst
prediction_var = features_worst
train_X= train[prediction_var]

train_y= train.diagnosis

test_X = test[prediction_var]

test_y = test.diagnosis
model = svm.SVC()

model.fit(train_X,train_y)

prediction=model.predict(test_X)

metrics.accuracy_score(prediction,test_y)
# but same problem With SVM, very much less accuray I think we have to tune its parameter

# that i will do later in intermidate part

#now we can get the important features from random forest now run Random Forest for it 
model=RandomForestClassifier(n_estimators=100)

model.fit(train_X,train_y)

prediction = model.predict(test_X)

metrics.accuracy_score(prediction,test_y)
# the accuracy for RandomForest invcrease it means the value are more catogrical in Worst part

#lets get the important features

featimp = pd.Series(model.feature_importances_, index=prediction_var).sort_values(ascending=False)

print(featimp) # this is the property of Random Forest classifier that it provide us the importance 

# of the features used
# same parameter but with great importance and here it seamed the only conacve points_worst is making 

# very important so it may be bias lets check only for top 5 important features
prediction_var = ['concave points_worst','radius_worst','area_worst','perimeter_worst','concavity_worst'] 
train_X= train[prediction_var]

train_y= train.diagnosis

test_X = test[prediction_var]

test_y = test.diagnosis
model=RandomForestClassifier(n_estimators=100)

model.fit(train_X,train_y)

prediction = model.predict(test_X)

metrics.accuracy_score(prediction,test_y)
#check for SVM

model = svm.SVC()

model.fit(train_X,train_y)

prediction=model.predict(test_X)

metrics.accuracy_score(prediction,test_y)
# now I think for simplicity the Randomforest will be better for prediction
# Now explore a little bit more

# now from features_mean i will try to find the variable which can be use for classify

# so lets plot a scatter plot for identify those variable who have a separable boundary between two class

#of cancer
# Lets start with the data analysis for features_mean

# Just try to understand which features can be used for prediction

# I will plot scatter plot for the all features_mean for both of diagnosis Category

# and from it we will find which are easily can used for differenciate between two category
color_function = {0: "blue", 1: "red"} # Here Red color will be 1 which means M and blue foo 0 means B

colors = data["diagnosis"].map(lambda x: color_function.get(x))# mapping the color fuction with diagnosis column

pd.scatter_matrix(data[features_mean], c=colors, alpha = 0.5, figsize = (15, 15)); # plotting scatter plot matrix
# So predicton features will be 

features_mean
# So predicton features will be 

predictor_var = ['radius_mean','perimeter_mean','area_mean','compactness_mean','concave points_mean']
# Now with these variable we will try to explore a liitle bit we will move to how to use cross validiation

# for a detail on cross validation use this link https://www.analyticsvidhya.com/blog/2015/11/improve-model-performance-cross-validation-in-python-r/
def model(model,data,prediction,outcome):

    # This function will be used for to check accuracy of different model

    # model is the m

    kf = KFold(data.shape[0], n_folds=10) # if you have refer the link then you must understand what is n_folds

    
prediction_var = ['radius_mean','perimeter_mean','area_mean','compactness_mean','concave points_mean']
# so those features who are capable of classify classe will be more useful
# so in this part i am going to explain about only some concept of machine learnig 

# here I will also compare the accuracy of different models

# I will First use cross validation with different model

# then I will explain about how to to tune the parameter of models using gridSearchCV 
# As we are going to use many models lets make a function

# Which we can use with different models

def classification_model(model,data,prediction_input,output):

    # here the model means the model 

    # data is used for the data 

    #prediction_input means the inputs used for prediction

    # output mean the value which are to be predicted

    # here we will try to find out the Accuarcy of model by using same data for fiiting and 

    #comparison for same data

    #Fit the model:

    model.fit(data[prediction_input],data[output]) #Here we fit the model using training set

  

    #Make predictions on training set:

    predictions = model.predict(data[prediction_input])

  

    #Print accuracy

    # now checkin accuracy for same data

    accuracy = metrics.accuracy_score(predictions,data[output])

    print("Accuracy : %s" % "{0:.3%}".format(accuracy))

 

    

    kf = KFold(data.shape[0], n_folds=5)

    # About cross validitaion please follow this link

    #https://www.analyticsvidhya.com/blog/2015/11/improve-model-performance-cross-validation-in-python-r/

    #let me explain a little bit data.shape[0] means number of rows in data

    #n_folds is for number of folds

    error = []

    for train, test in kf:

        # as the data is divided into train and test using KFold

        # now as explained above we have fit many models 

        # so here also we are going to fit model

        #in the cross validation the data in train and test will change for evry iteration

        train_X = (data[prediction_input].iloc[train,:])# in this iloc is used for index of trainig data

        # here iloc[train,:] means all row in train in kf amd the all columns

        train_y = data[output].iloc[train]# here is only column so it repersenting only row in train

        # Training the algorithm using the predictors and target.

        model.fit(train_X, train_y)

    

        # now do this for test data also

        test_X=data[prediction_input].iloc[test,:]

        test_y=data[output].iloc[test]

        error.append(model.score(test_X,test_y))

        # printing the score 

        print("Cross-Validation Score : %s" % "{0:.3%}".format(np.mean(error)))

    

    

    
# Now from Here start using different model
model = DecisionTreeClassifier()

prediction_var = ['radius_mean','perimeter_mean','area_mean','compactness_mean','concave points_mean']

outcome_var= "diagnosis"

classification_model(model,data,prediction_var,outcome_var)
# now move to svm
model = svm.SVC()



classification_model(model,data,prediction_var,outcome_var)
# I am facing problem with SVM dont know why?

#lets leave that we will try to do it later 
model = KNeighborsClassifier()

classification_model(model,data,prediction_var,outcome_var)
# same here cross validation scores are not good

# now move to RandomForestclassifier

model = RandomForestClassifier(n_estimators=100)

classification_model(model,data,prediction_var,outcome_var)
# cross validation score are also not bed

# so Random forest is good

# lets try with logistic regression

model=LogisticRegression()

classification_model(model,data,prediction_var,outcome_var)
data_X= data[prediction_var]

data_y= data["diagnosis"]
# lets Make a function for Grid Search CV

def Classification_model_gridsearchCV(model,param_grid,data_X,data_y):

    clf = GridSearchCV(model,param_grid,cv=10,scoring="accuracy")

    # this is how we use grid serch CV we are giving our model

    # the we gave parameters those we want to tune

    # Cv is for cross validation

    # scoring means to score the classifier

    

    clf.fit(train_X,train_y)

    print("The best parameter found on development set is :")

    # this will gie us our best parameter to use

    print(clf.best_params_)

    print("the bset estimator is ")

    print(clf.best_estimator_)

    print("The best score is ")

    # this is the best score that we can achieve using these parameters#

    print(clf.best_score_)

    

    

    

    
# Here we have to take parameters that are used for Decison tree Classifier

# you will understand these terms once you follow the link above

param_grid = {'max_features': ['auto', 'sqrt', 'log2'],

              'min_samples_split': [2,3,4,5,6,7,8,9,10], 

              'min_samples_leaf':[2,3,4,5,6,7,8,9,10] }

# here our gridasearchCV will take all combinations of these parameter and apply it to model 

# and then it will find the best parameter for model

model= DecisionTreeClassifier()

Classification_model_gridsearchCV(model,param_grid,data_X,data_y)

# call our function
model = KNeighborsClassifier()



k_range = list(range(1, 30))

leaf_size = list(range(1,30))

weight_options = ['uniform', 'distance']

param_grid = {'n_neighbors': k_range, 'leaf_size': leaf_size, 'weights': weight_options}

Classification_model_gridsearchCV(model,param_grid,data_X,data_y)
model=svm.SVC()

param_grid = [

              {'C': [1, 10, 100, 1000], 

               'kernel': ['linear']

              },

              {'C': [1, 10, 100, 1000], 

               'gamma': [0.001, 0.0001], 

               'kernel': ['rbf']

              },

 ]

Classification_model_gridsearchCV(model,param_grid,data_X,data_y)