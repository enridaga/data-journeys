
# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.
#import dataset

data=pd.read_csv('../input/biomechanical-features-of-orthopedic-patients/column_2C_weka.csv')
data.info()
data.head()
data['class'].unique()
data.describe()
color_list = ['cyan' if i=='Abnormal' else 'orange' for i in data.loc[:,'class']]
pd.plotting.scatter_matrix(data.loc[:, data.columns != 'class'],
                           c=color_list,
                           figsize= [17,17],
                           diagonal='hist',
                           alpha=0.5,
                           s = 200,
                           marker = '*',
                           edgecolor= "black")
                                        
plt.show()
plt.figure(figsize=(15,15))
sns.heatmap(data.corr(),annot=True,cmap='RdBu_r')
sns.countplot(x="class", data=data)
plt.show()
# create data1 and data2 that includes pelvic_incidence that is feature and sacral_slope that is target variable
data1 = data[data['class'] =='Abnormal']
data2 = data[data['class'] =='Normal']
x = np.array(data1.loc[:,'pelvic_incidence']).reshape(-1,1)
y = np.array(data1.loc[:,'sacral_slope']).reshape(-1,1)
x2=np.array(data2.loc[:,'pelvic_incidence']).reshape(-1,1)
y2=np.array(data2.loc[:,'sacral_slope']).reshape(-1,1)
# Scatter
plt.figure(figsize=[5,5])
plt.scatter(x=x,y=y,color='cyan',marker="*",label='Abnormal')
plt.scatter(x=x2,y=y2,color='orange',marker="*",label="Normal")
plt.xlabel('pelvic_incidence')
plt.ylabel('sacral_slope')
plt.legend()
plt.show()

data.loc[:,'class'].value_counts()
data.head()
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors = 3)
x,y = data.loc[:,data.columns != 'class'], data.loc[:,'class']
knn.fit(x,y)
prediction = knn.predict(x)
print('Prediction: {}'.format(prediction))
# train test split 
#train %70  and test %30 
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size = 0.3,random_state = 1)

knn = KNeighborsClassifier(n_neighbors = 5)
x,y = data.loc[:,data.columns != 'class'], data.loc[:,'class']

knn.fit(x_train,y_train)
prediction = knn.predict(x_test)

#print('Prediction: {}'.format(prediction))
print('With KNN (K=5) accuracy is: ',knn.score(x_test,y_test)) # accuracy
prediction
#find k value
score_list=[]
train_accuracy = []
for each in range (1,25):
    knn=KNeighborsClassifier(n_neighbors=each)
    knn.fit(x_train,y_train)
    score_list.append(knn.score(x_test,y_test))
    train_accuracy.append(knn.score(x_train, y_train))
plt.plot(range(1,25),score_list,label="Testing Accuracy",color="red")
plt.plot(range(1,25), train_accuracy, label = 'Training Accuracy',color="green")
plt.xlabel("Number of Neighbors")
plt.ylabel("accuracy")
plt.title('Value VS Accuracy')
plt.savefig('graph.png')
plt.legend()
plt.show()
print("Best accuracy is {} with K = {}".format(np.max(score_list),1+score_list.index(np.max(score_list))))
data.head()
plt.scatter(data['pelvic_radius'],data['degree_spondylolisthesis'])
plt.xlabel('pelvic_radius')
plt.ylabel('degree_spondylolisthesis')
plt.show()
data2=data.loc[:,['degree_spondylolisthesis','pelvic_radius']]
from sklearn.cluster import KMeans
kmeans=KMeans(n_clusters=2)
kmeans.fit(data2)
labels=kmeans.predict(data2)
plt.scatter(data['pelvic_radius'],data['degree_spondylolisthesis'],c=labels)
plt.xlabel('pelvic_radius')
plt.ylabel('degree_spondylolisthesis')
plt.show()
wcss = []
for i in range(1,15):
    kmeans = KMeans(n_clusters=i)
    kmeans.fit(data2)
    wcss.append(kmeans.inertia_)
plt.plot(range(1,15),wcss,'-*')
plt.xlabel('Number of k (cluster value)')
plt.ylabel('Wcss')
plt.show()
data3 = data.drop('class',axis = 1)
data3.head()
from scipy.cluster.hierarchy import linkage, dendrogram 

merg=linkage(data2,method="ward")
dendrogram(merg) #hiartical clustring algoritmasını kullanıyorum.
dendrogram(merg,leaf_rotation=90)
plt.xlabel("data paoints")
plt.ylabel("euclidean distance")
plt.show()
from scipy.cluster.hierarchy import linkage,dendrogram

merg = linkage(data3.iloc[200:220,:],method = 'single')
dendrogram(merg, leaf_rotation = 90, leaf_font_size = 6)
plt.show()