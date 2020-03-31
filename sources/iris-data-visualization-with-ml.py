
##Importing basic Python libraries necessary for data manipulation and visualization

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

##Enabling matplotlib in jupyter notebook

### matplotlib inline
##Step 1: Loading and examining the Iris dataset file into a Pandas DataFrame for analysis.
#Creating a DataFrame with variable name 'iris'.

iris = pd.read_csv ('../input/iris/Iris.csv')
iris.head()
#Assigning Id column as the index for slightly easier data manipulation later on.

iris.set_index('Id', inplace = True)
iris.head()
#Changing all column names to lowercase for easier selection (personal preference).

iris.columns = map(str.lower, iris)
iris.head()
#Step 2: Performing statistical analysis on the dataset, as well as checking for possible errors (missing values).

iris.info()
iris.describe()
iris[['sepallengthcm', 'sepalwidthcm', 'petallengthcm', 'petalwidthcm']].isna().describe()

#Fortunately, there are no missing values in this dataset.
#Step 3: Plotting a few graphs to gain a sensing of the relation between the features.
#Using Seaborn's pairplot to gain a broad overview of the dataset.

sns.pairplot(iris, hue = 'species', diag_kind = 'hist', palette = 'Set1')
#Using Seaborn's jointplot for a slightly more indepth look.

sns.jointplot(x = 'sepallengthcm', y = 'sepalwidthcm', data = iris)
#Using Seaborn's scatterplot to obtain a similar plot, but color-coded by their species

sns.scatterplot(x = 'sepallengthcm', y = 'sepalwidthcm', data = iris, hue = 'species', palette = 'Set1')
plt.title('Sepal Width vs Sepal Length')
plt.legend(loc = 'center left', bbox_to_anchor = (1.0, 0.5))
##We shall now continue with the plots, with a deeper look into the distribution and correlation between the features.
#Using matplotlib to create a figure for a more compressed view of the four features.
#Using histograms for an overview of the varying values of each feature.

iris.hist(figsize = (12, 12), ec = 'black')
#Using matplotlib to create a figure consisting of four subplots.
#Using Seaborn's boxplot to gain insight into the distribution of the four features.

plt.figure(figsize = (12, 12))
sns.set_style('whitegrid')
sns.set_palette('Set1')
plt.subplot(2, 2, 1)
sns.boxplot(x = 'species', y = 'petallengthcm', data = iris)
plt.subplot(2, 2, 2)
sns.boxplot(x = 'species', y = 'petalwidthcm', data = iris)
plt.subplot(2, 2, 3)
sns.boxplot(x = 'species', y = 'sepallengthcm', data = iris)
plt.subplot(2, 2, 4)
sns.boxplot(x = 'species', y= 'sepalwidthcm', data = iris,)
#Now, I wish to show the correlation between the features(if any), using Seaborn's heatmap feature.
#Firstly, doing some basic transformation of the pandas DataFrame.

iris.head()
iris_corr = iris.corr()
iris_corr.head()
#Using Seaborn's heatmap to visualize the correlation.

sns.heatmap(iris_corr, cbar = True, annot = True, cmap = 'RdBu_r')
##I will be going through my process of implementing logistic regression in the following rows using a step-by-step
##approach once again.
##Step 1: Creating a training and test set (look up Google for reasons on why we need to split a training and test set)

#Brief review of our DataFrame:
iris.head()
#Creating our matrix of flower features and their respective values for each data point, X.

#We drop the 'species' column since it is our dependent variable in this case.
X = iris.drop('species', axis = 1) 
X.head()
#Assigning our dependent variable, the species of the flower, to y.

y = iris['species']
y.value_counts()
#From here, we will be importing relevant modules from libraries to allow us to create our training and test data sets.

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 101)
#with this, our training(X_train, y_train) and test(X_test, y_test) sets have been created.
##Step 2: Training our model by fitting it to the training data sets.
#Importing our LogisticRegression module and creating an instance to carry out the training of the model

from sklearn.linear_model import LogisticRegression

logm = LogisticRegression()
#Fitting the model to our training data.

logm.fit(X_train, y_train)
##Step 3: Obtaining our predictions by using the model with the test set.
lg_predictions = logm.predict(X_test)
##Step 4: Evaluating our results by comparing the predicted results (lg_predictions) to the actual results (y_test).
#To do this, I will import a few modules which allows us to visualize this comparison.

from sklearn.metrics import classification_report, confusion_matrix
print(confusion_matrix(y_test, lg_predictions))
print(classification_report(y_test, lg_predictions))
##Step 1: Creating a training and test data set.
X = iris.drop('species', axis = 1)
y = iris['species']
X_train, x_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 101)
##Step 2: Implementing kNN, starting with an arbitrary value of k=5.
#Now, we will again import the kNeighborsClassifier to carry out our implementation and learning.

from sklearn.neighbors import KNeighborsClassifier
#Creating an instance and fitting it to our training data.

knn = KNeighborsClassifier(n_neighbors = 5, metric = 'euclidean') 
knn.fit(X_train, y_train)
#Creating our predictions.

knn_predictions = knn.predict(X_test)
##Step 3: Evaluating our predicted results against the actual results.
print(confusion_matrix(y_test, knn_predictions))
print(classification_report(y_test, knn_predictions))
##Step 4: Choosing the 'best' value of 'k'
#In order to do this, we can plot a graph of the error-rate of the model against the k-value that is being selected.
#This allows us to get a better gauge of the domain of k values which may allow us to obtain a better accuracy on our
#model.
#We will use a for loop to repeat steps 3-4 using different values of k.
error_rate = []

for i in range (1, 30):
    knn_i = KNeighborsClassifier(n_neighbors = i)
    knn_i.fit(X_train, y_train)
    knn_i_pred = knn_i.predict(X_test)
    error_rate.append(np.mean (knn_i_pred != y_test))
plt.figure(figsize = (12, 8))
plt.plot(np.arange(1, 30), error_rate, 'o-')
#Seems like our initial value of k=5 was a pretty good estimate. Hence, we shall stick with it.
##Step 1: Creating a training/test data set.
#X and y have already been defined from above examples.

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 101)
##Step 2: Importing our module to fit and train the model for predictions.

from sklearn.tree import DecisionTreeClassifier
tree = DecisionTreeClassifier()
tree.fit(X_train, y_train)
tree_predictions = tree.predict(X_test)
##Step 3: Evaluating our metrics given the trained model.

print(confusion_matrix(y_test, tree_predictions))
print(classification_report(y_test, tree_predictions))
##We shall now carry on with Random Forests implementation. For more
##details regarding this model, please look up details from other
##sources for more detailed explanations.
##Step 1: Importing our random forests module and training it.

from sklearn.ensemble import RandomForestClassifier
rfc = RandomForestClassifier()
rfc.fit(X_train, y_train)
rfc_predictions = rfc.predict(X_test)
##Step 2: Evaluating our metrics given the predicted values.

print(confusion_matrix(y_test, rfc_predictions))
print(classification_report(y_test, rfc_predictions))
##Step 1: Creating a training and test data set.
##Since the steps have been repeated multiple times in the above few
##examples, I shall be skipping this step from here on.
##Step 2: Importing our SVM classifier and training it.

from sklearn.svm import SVC
svc = SVC()
svc.fit(X_train, y_train)
svc_predictions = svc.predict(X_test)
##Step 3: Evaluating our metrics given predicted values.

print(confusion_matrix(y_test, svc_predictions))
print(classification_report(y_test, svc_predictions))