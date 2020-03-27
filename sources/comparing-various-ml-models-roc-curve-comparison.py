
# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
import pandas as pd

import numpy as np

from matplotlib import pyplot as plt

import seaborn as sns
data = pd.read_csv("../input/mushrooms.csv")

data.head(6)
data.isnull().sum()
data['class'].unique()
data.shape
from sklearn.preprocessing import LabelEncoder

labelencoder=LabelEncoder()

for col in data.columns:

    data[col] = labelencoder.fit_transform(data[col])

 

data.head()
data['stalk-color-above-ring'].unique()
print(data.groupby('class').size())
'''

# Create a figure instance

fig, axes = plt.subplots(nrows=2 ,ncols=2 ,figsize=(9, 9))



# Create an axes instance and the boxplot

bp1 = axes[0,0].boxplot(data['stalk-color-above-ring'],patch_artist=True)



bp2 = axes[0,1].boxplot(data['stalk-color-below-ring'],patch_artist=True)



bp3 = axes[1,0].boxplot(data['stalk-surface-below-ring'],patch_artist=True)



bp4 = axes[1,1].boxplot(data['stalk-surface-above-ring'],patch_artist=True)

'''

ax = sns.boxplot(x='class', y='stalk-color-above-ring', 

                data=data)

ax = sns.stripplot(x="class", y='stalk-color-above-ring',

                   data=data, jitter=True,

                   edgecolor="gray")

sns.plt.title("Class w.r.t stalkcolor above ring",fontsize=12)
X = data.iloc[:,1:23]  # all rows, all the features and no labels

y = data.iloc[:, 0]  # all rows, label only

X.head()

y.head()
X.describe()
y.head()
data.corr()
# Scale the data to be between -1 and 1

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

X=scaler.fit_transform(X)

X
from sklearn.decomposition import PCA

pca = PCA()

pca.fit_transform(X)
covariance=pca.get_covariance()

#covariance
explained_variance=pca.explained_variance_

explained_variance
with plt.style.context('dark_background'):

    plt.figure(figsize=(6, 4))

    

    plt.bar(range(22), explained_variance, alpha=0.5, align='center',

            label='individual explained variance')

    plt.ylabel('Explained variance ratio')

    plt.xlabel('Principal components')

    plt.legend(loc='best')

    plt.tight_layout()
N=data.values

pca = PCA(n_components=2)

x = pca.fit_transform(N)

plt.figure(figsize = (5,5))

plt.scatter(x[:,0],x[:,1])

plt.show()
from sklearn.cluster import KMeans

kmeans = KMeans(n_clusters=2, random_state=5)

X_clustered = kmeans.fit_predict(N)



LABEL_COLOR_MAP = {0 : 'g',

                   1 : 'y'

                  }



label_color = [LABEL_COLOR_MAP[l] for l in X_clustered]

plt.figure(figsize = (5,5))

plt.scatter(x[:,0],x[:,1], c= label_color)

plt.show()
pca_modified=PCA(n_components=17)

pca_modified.fit_transform(X)
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=4)
from sklearn.linear_model import LogisticRegression

from sklearn.model_selection import cross_val_score

from sklearn import metrics



model_LR= LogisticRegression()
model_LR.fit(X_train,y_train)
y_prob = model_LR.predict_proba(X_test)[:,1] # This will give you positive class prediction probabilities  

y_pred = np.where(y_prob > 0.5, 1, 0) # This will threshold the probabilities to give class predictions.

model_LR.score(X_test, y_pred)
confusion_matrix=metrics.confusion_matrix(y_test,y_pred)

confusion_matrix
auc_roc=metrics.roc_auc_score(y_test,y_pred)

auc_roc
from sklearn.metrics import roc_curve, auc

false_positive_rate, true_positive_rate, thresholds = roc_curve(y_test, y_prob)

roc_auc = auc(false_positive_rate, true_positive_rate)

roc_auc
import matplotlib.pyplot as plt

plt.figure(figsize=(10,10))

plt.title('Receiver Operating Characteristic')

plt.plot(false_positive_rate,true_positive_rate, color='red',label = 'AUC = %0.2f' % roc_auc)

plt.legend(loc = 'lower right')

plt.plot([0, 1], [0, 1],linestyle='--')

plt.axis('tight')

plt.ylabel('True Positive Rate')

plt.xlabel('False Positive Rate')
from sklearn.linear_model import LogisticRegression

from sklearn.model_selection import cross_val_score

from sklearn import metrics



LR_model= LogisticRegression()



tuned_parameters = {'C': [0.001, 0.01, 0.1, 1, 10, 100, 1000] ,

              'penalty':['l1','l2']

                   }
data.corr()
from sklearn.model_selection import GridSearchCV



LR= GridSearchCV(LR_model, tuned_parameters,cv=10)
LR.fit(X_train,y_train)
print(LR.best_params_)
y_prob = LR.predict_proba(X_test)[:,1] # This will give you positive class prediction probabilities  

y_pred = np.where(y_prob > 0.5, 1, 0) # This will threshold the probabilities to give class predictions.

LR.score(X_test, y_pred)
confusion_matrix=metrics.confusion_matrix(y_test,y_pred)

confusion_matrix
auc_roc=metrics.classification_report(y_test,y_pred)

auc_roc
auc_roc=metrics.roc_auc_score(y_test,y_pred)

auc_roc
from sklearn.metrics import roc_curve, auc

false_positive_rate, true_positive_rate, thresholds = roc_curve(y_test, y_prob)

roc_auc = auc(false_positive_rate, true_positive_rate)

roc_auc


import matplotlib.pyplot as plt

plt.figure(figsize=(10,10))

plt.title('Receiver Operating Characteristic')

plt.plot(false_positive_rate,true_positive_rate, color='red',label = 'AUC = %0.2f' % roc_auc)

plt.legend(loc = 'lower right')

plt.plot([0, 1], [0, 1],linestyle='--')

plt.axis('tight')

plt.ylabel('True Positive Rate')

plt.xlabel('False Positive Rate')
LR_ridge= LogisticRegression(penalty='l2')

LR_ridge.fit(X_train,y_train)
y_prob = LR_ridge.predict_proba(X_test)[:,1] # This will give you positive class prediction probabilities  

y_pred = np.where(y_prob > 0.5, 1, 0) # This will threshold the probabilities to give class predictions.

LR_ridge.score(X_test, y_pred)
confusion_matrix=metrics.confusion_matrix(y_test,y_pred)

confusion_matrix
auc_roc=metrics.classification_report(y_test,y_pred)

auc_roc
auc_roc=metrics.roc_auc_score(y_test,y_pred)

auc_roc


from sklearn.metrics import roc_curve, auc

false_positive_rate, true_positive_rate, thresholds = roc_curve(y_test, y_prob)

roc_auc = auc(false_positive_rate, true_positive_rate)

roc_auc


import matplotlib.pyplot as plt

plt.figure(figsize=(10,10))

plt.title('Receiver Operating Characteristic')

plt.plot(false_positive_rate,true_positive_rate, color='red',label = 'AUC = %0.2f' % roc_auc)

plt.legend(loc = 'lower right')

plt.plot([0, 1], [0, 1],linestyle='--')

plt.axis('tight')

plt.ylabel('True Positive Rate')

plt.xlabel('False Positive Rate')
from sklearn.naive_bayes import GaussianNB

model_naive = GaussianNB()

model_naive.fit(X_train, y_train)
y_prob = model_naive.predict_proba(X_test)[:,1] # This will give you positive class prediction probabilities  

y_pred = np.where(y_prob > 0.5, 1, 0) # This will threshold the probabilities to give class predictions.

model_naive.score(X_test, y_pred)
print("Number of mislabeled points from %d points : %d"

      % (X_test.shape[0],(y_test!= y_pred).sum()))
scores = cross_val_score(model_naive, X, y, cv=10, scoring='accuracy')

print(scores)
scores.mean()
confusion_matrix=metrics.confusion_matrix(y_test,y_pred)

confusion_matrix
auc_roc=metrics.classification_report(y_test,y_pred)

auc_roc
auc_roc=metrics.roc_auc_score(y_test,y_pred)

auc_roc
from sklearn.metrics import roc_curve, auc

false_positive_rate, true_positive_rate, thresholds = roc_curve(y_test, y_prob)

roc_auc = auc(false_positive_rate, true_positive_rate)

roc_auc
import matplotlib.pyplot as plt

plt.figure(figsize=(10,10))

plt.title('Receiver Operating Characteristic')

plt.plot(false_positive_rate,true_positive_rate, color='red',label = 'AUC = %0.2f' % roc_auc)

plt.legend(loc = 'lower right')

plt.plot([0, 1], [0, 1],linestyle='--')

plt.axis('tight')

plt.ylabel('True Positive Rate')

plt.xlabel('False Positive Rate')
from sklearn.svm import SVC

svm_model= SVC()
tuned_parameters = {

 'C': [1, 10, 100,500, 1000], 'kernel': ['linear','rbf'],

 'C': [1, 10, 100,500, 1000], 'gamma': [1,0.1,0.01,0.001, 0.0001], 'kernel': ['rbf'],

 #'degree': [2,3,4,5,6] , 'C':[1,10,100,500,1000] , 'kernel':['poly']

    }
from sklearn.grid_search import RandomizedSearchCV



model_svm = RandomizedSearchCV(svm_model, tuned_parameters,cv=10,scoring='accuracy',n_iter=20)
model_svm.fit(X_train, y_train)

print(model_svm.best_score_)
print(model_svm.grid_scores_)
print(model_svm.best_params_)


y_pred= model_svm.predict(X_test)

print(metrics.accuracy_score(y_pred,y_test))
confusion_matrix=metrics.confusion_matrix(y_test,y_pred)

confusion_matrix
auc_roc=metrics.classification_report(y_test,y_pred)

auc_roc
auc_roc=metrics.roc_auc_score(y_test,y_pred)

auc_roc
from sklearn.metrics import roc_curve, auc

false_positive_rate, true_positive_rate, thresholds = roc_curve(y_test, y_pred)

roc_auc = auc(false_positive_rate, true_positive_rate)

roc_auc
import matplotlib.pyplot as plt

plt.figure(figsize=(10,10))

plt.title('Receiver Operating Characteristic')

plt.plot(false_positive_rate,true_positive_rate, color='red',label = 'AUC = %0.2f' % roc_auc)

plt.legend(loc = 'lower right')

plt.plot([0, 1], [0, 1],linestyle='--')

plt.axis('tight')

plt.ylabel('True Positive Rate')

plt.xlabel('False Positive Rate')
tuned_parameters = {

 'C': [1, 10, 100,500, 1000], 'kernel': ['linear','rbf'],

 'C': [1, 10, 100,500, 1000], 'gamma': [1,0.1,0.01,0.001, 0.0001], 'kernel': ['rbf'],

 'degree': [2,3,4,5,6] , 'C':[1,10,100,500,1000] , 'kernel':['poly']

    }
from sklearn.grid_search import RandomizedSearchCV



model_svm = RandomizedSearchCV(svm_model, tuned_parameters,cv=10,scoring='accuracy',n_iter=20)
model_svm.fit(X_train, y_train)

print(model_svm.best_score_)
print(model_svm.grid_scores_)
print(model_svm.best_params_)
y_pred= model_svm.predict(X_test)

print(metrics.accuracy_score(y_pred,y_test))
confusion_matrix=metrics.confusion_matrix(y_test,y_pred)

confusion_matrix
auc_roc=metrics.classification_report(y_test,y_pred)

auc_roc
auc_roc=metrics.roc_auc_score(y_test,y_pred)

auc_roc
from sklearn.metrics import roc_curve, auc

false_positive_rate, true_positive_rate, thresholds = roc_curve(y_test, y_pred)

roc_auc = auc(false_positive_rate, true_positive_rate)

roc_auc
import matplotlib.pyplot as plt

plt.figure(figsize=(10,10))

plt.title('Receiver Operating Characteristic')

plt.plot(false_positive_rate,true_positive_rate, color='red',label = 'AUC = %0.2f' % roc_auc)

plt.legend(loc = 'lower right')

plt.plot([0, 1], [0, 1],linestyle='--')

plt.axis('tight')

plt.ylabel('True Positive Rate')

plt.xlabel('False Positive Rate')
from sklearn.ensemble import RandomForestClassifier



model_RR=RandomForestClassifier()



#tuned_parameters = {'min_samples_leaf': range(5,10,5), 'n_estimators' : range(50,200,50),

                    #'max_depth': range(5,15,5), 'max_features':range(5,20,5)

                    #}

               
model_RR.fit(X_train,y_train)
y_prob = model_RR.predict_proba(X_test)[:,1] # This will give you positive class prediction probabilities  

y_pred = np.where(y_prob > 0.5, 1, 0) # This will threshold the probabilities to give class predictions.

model_RR.score(X_test, y_pred)
confusion_matrix=metrics.confusion_matrix(y_test,y_pred)

confusion_matrix
auc_roc=metrics.classification_report(y_test,y_pred)

auc_roc
auc_roc=metrics.roc_auc_score(y_test,y_pred)

auc_roc
from sklearn.metrics import roc_curve, auc

false_positive_rate, true_positive_rate, thresholds = roc_curve(y_test, y_prob)

roc_auc = auc(false_positive_rate, true_positive_rate)
import matplotlib.pyplot as plt

plt.figure(figsize=(10,10))

plt.title('Receiver Operating Characteristic')

plt.plot(false_positive_rate,true_positive_rate, color='red',label = 'AUC = %0.2f' % roc_auc)

plt.legend(loc = 'lower right')

plt.plot([0, 1], [0, 1],linestyle='--')

plt.axis('tight')

plt.ylabel('True Positive Rate')

plt.xlabel('False Positive Rate')
from sklearn.ensemble import RandomForestClassifier



model_RR=RandomForestClassifier()



tuned_parameters = {'min_samples_leaf': range(10,100,10), 'n_estimators' : range(10,100,10),

                    'max_features':['auto','sqrt','log2']

                    }

    
from sklearn.grid_search import RandomizedSearchCV



RR_model= RandomizedSearchCV(model_RR, tuned_parameters,cv=10,scoring='accuracy',n_iter=20,n_jobs= -1)
RR_model.fit(X_train,y_train)
print(RR_model.grid_scores_)
print(RR_model.best_score_)
print(RR_model.best_params_)
y_prob = RR_model.predict_proba(X_test)[:,1] # This will give you positive class prediction probabilities  

y_pred = np.where(y_prob > 0.5, 1, 0) # This will threshold the probabilities to give class predictions.

RR_model.score(X_test, y_pred)
confusion_matrix=metrics.confusion_matrix(y_test,y_pred)

confusion_matrix
auc_roc=metrics.classification_report(y_test,y_pred)

auc_roc
auc_roc=metrics.roc_auc_score(y_test,y_pred)

auc_roc


from sklearn.metrics import roc_curve, auc

false_positive_rate, true_positive_rate, thresholds = roc_curve(y_test, y_prob)

roc_auc = auc(false_positive_rate, true_positive_rate)

roc_auc
import matplotlib.pyplot as plt

plt.figure(figsize=(10,10))

plt.title('Receiver Operating Characteristic')

plt.plot(false_positive_rate,true_positive_rate, color='red',label = 'AUC = %0.2f' % roc_auc)

plt.legend(loc = 'lower right')

plt.plot([0, 1], [0, 1],linestyle='--')

plt.axis('tight')

plt.ylabel('True Positive Rate')

plt.xlabel('False Positive Rate')
from sklearn.tree import DecisionTreeClassifier



model_tree = DecisionTreeClassifier()
model_tree.fit(X_train, y_train)
y_prob = model_tree.predict_proba(X_test)[:,1] # This will give you positive class prediction probabilities  

y_pred = np.where(y_prob > 0.5, 1, 0) # This will threshold the probabilities to give class predictions.

model_tree.score(X_test, y_pred)
confusion_matrix=metrics.confusion_matrix(y_test,y_pred)

confusion_matrix
auc_roc=metrics.classification_report(y_test,y_pred)

auc_roc
auc_roc=metrics.roc_auc_score(y_test,y_pred)

auc_roc
from sklearn.metrics import roc_curve, auc

false_positive_rate, true_positive_rate, thresholds = roc_curve(y_test, y_prob)

roc_auc = auc(false_positive_rate, true_positive_rate)
import matplotlib.pyplot as plt

plt.figure(figsize=(10,10))

plt.title('Receiver Operating Characteristic')

plt.plot(false_positive_rate,true_positive_rate, color='red',label = 'AUC = %0.2f' % roc_auc)

plt.legend(loc = 'lower right')

plt.plot([0, 1], [0, 1],linestyle='--')

plt.axis('tight')

plt.ylabel('True Positive Rate')

plt.xlabel('False Positive Rate')
from sklearn.tree import DecisionTreeClassifier



model_DD = DecisionTreeClassifier()





tuned_parameters= {'criterion': ['gini','entropy'], 'max_features': ["auto","sqrt","log2"],

                   'min_samples_leaf': range(1,100,1) , 'max_depth': range(1,50,1)

                  }

           
from sklearn.grid_search import RandomizedSearchCV

DD_model= RandomizedSearchCV(model_DD, tuned_parameters,cv=10,scoring='accuracy',n_iter=20,n_jobs= -1,random_state=5)
DD_model.fit(X_train, y_train)
print(DD_model.grid_scores_)
print(DD_model.best_score_)
print(DD_model.best_params_)
y_prob = DD_model.predict_proba(X_test)[:,1] # This will give you positive class prediction probabilities  

y_pred = np.where(y_prob > 0.5, 1, 0) # This will threshold the probabilities to give class predictions.

DD_model.score(X_test, y_pred)
confusion_matrix=metrics.confusion_matrix(y_test,y_pred)

confusion_matrix
auc_roc=metrics.classification_report(y_test,y_pred)

auc_roc
auc_roc=metrics.roc_auc_score(y_test,y_pred)

auc_roc
from sklearn.metrics import roc_curve, auc

false_positive_rate, true_positive_rate, thresholds = roc_curve(y_test, y_prob)

roc_auc = auc(false_positive_rate, true_positive_rate)

roc_auc
import matplotlib.pyplot as plt

plt.figure(figsize=(10,10))

plt.title('Receiver Operating Characteristic')

plt.plot(false_positive_rate,true_positive_rate, color='red',label = 'AUC = %0.2f' % roc_auc)

plt.legend(loc = 'lower right')

plt.plot([0, 1], [0, 1],linestyle='--')

plt.axis('tight')

plt.ylabel('True Positive Rate')

plt.xlabel('False Positive Rate')
from sklearn.neural_network import MLPClassifier
mlp = MLPClassifier()

mlp.fit(X_train,y_train)
y_prob = mlp.predict_proba(X_test)[:,1] # This will give you positive class prediction probabilities  

y_pred = np.where(y_prob > 0.5, 1, 0) # This will threshold the probabilities to give class predictions.

mlp.score(X_test, y_pred)
confusion_matrix=metrics.confusion_matrix(y_test,y_pred)

confusion_matrix
auc_roc=metrics.classification_report(y_test,y_pred)

auc_roc
auc_roc=metrics.roc_auc_score(y_test,y_pred)

auc_roc
from sklearn.metrics import roc_curve, auc

false_positive_rate, true_positive_rate, thresholds = roc_curve(y_test, y_prob)

roc_auc = auc(false_positive_rate, true_positive_rate)

roc_auc
import matplotlib.pyplot as plt

plt.figure(figsize=(10,10))

plt.title('Receiver Operating Characteristic')

plt.plot(false_positive_rate,true_positive_rate, color='red',label = 'AUC = %0.2f' % roc_auc)

plt.legend(loc = 'lower right')

plt.plot([0, 1], [0, 1],linestyle='--')

plt.axis('tight')

plt.ylabel('True Positive Rate')

plt.xlabel('False Positive Rate')
'''

from sklearn.neural_network import MLPClassifier



mlp = MLPClassifier()



tuned_parameters={'hidden_layer_sizes': range(1,200,10) , 'activation': ['tanh','logistic','relu'],

                  'alpha':[0.0001,0.001,0.01,0.1,1,10], 'max_iter': range(50,200,50)

    

}

'''
#from sklearn.grid_search import RandomizedSearchCV

#model_mlp= RandomizedSearchCV(mlp_model, tuned_parameters,cv=10,scoring='accuracy',n_iter=5,n_jobs= -1,random_state=5)
#model_mlp.fit(X_train, y_train)
#print(model_mlp.grid_scores_)
#print(model_mlp.best_score_)
#print(model_mlp.best_params_)
'''

y_prob = model_LR.predict_proba(X_test)[:,1] # This will give you positive class prediction probabilities  

y_pred = np.where(y_prob > 0.5, 1, 0) # This will threshold the probabilities to give class predictions.

model_LR.score(X_test, y_pred)

'''
#confusion_matrix=metrics.confusion_matrix(y_test,y_pred)

#confusion_matrix
#auc_roc=metrics.classification_report(y_test,y_pred)

#auc_roc
#auc_roc=metrics.roc_auc_score(y_test,y_pred)

#auc_roc
'''

from sklearn.metrics import roc_curve, auc

false_positive_rate, true_positive_rate, thresholds = roc_curve(y_test, y_prob)

roc_auc = auc(false_positive_rate, true_positive_rate)

'''
'''

import matplotlib.pyplot as plt

plt.figure(figsize=(10,10))

plt.title('Receiver Operating Characteristic')

plt.plot(false_positive_rate,true_positive_rate, color='red',label = 'AUC = %0.2f' % roc_auc)

plt.legend(loc = 'lower right')

plt.plot([0, 1], [0, 1],linestyle='--')

plt.axis('tight')

plt.ylabel('True Positive Rate')

plt.xlabel('False Positive Rate')

'''