
import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
# Following are several helpful packages to load in 
# Imported Libraries are as fallows
import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# Classification tree libraries
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import BaggingClassifier

# Other Libraries
from sklearn.metrics import confusion_matrix,accuracy_score

# Read the credit card dataset
base_dataset=pd.read_csv("/kaggle/input/creditcardfraud/creditcard.csv")
df=base_dataset
base_dataset.head()
base_dataset.shape
sns.countplot(base_dataset['Class'])
""" iterate through all the columns of a dataframe and modify the data type
    to reduce memory usage.        
"""
start_mem = base_dataset.memory_usage().sum() / 1024**2
print('Memory usage of dataframe is {:.2f} MB'.format(start_mem))

for col in base_dataset.describe().columns:
    col_type = base_dataset[col].dtype

    if col_type != object:
        c_min = base_dataset[col].min()
        c_max = base_dataset[col].max()
        if str(col_type)[:3] == 'int':
            if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                base_dataset[col] = base_dataset[col].astype(np.int8)
            elif c_min > base_dataset.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                base_dataset[col] = base_dataset[col].astype(np.int16)
            elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                base_dataset[col] = base_dataset[col].astype(np.int32)
            elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                base_dataset[col] = base_dataset[col].astype(np.int64)  
        else:
            if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                base_dataset[col] = base_dataset[col].astype(np.float16)
            elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                base_dataset[col] = base_dataset[col].astype(np.float32)
            else:
                base_dataset[col] = base_dataset[col].astype(np.float64)
    else:
        base_dataset[col] = base_dataset[col].astype('category')

end_mem = base_dataset.memory_usage().sum() / 1024**2
print('Memory usage after optimization is: {:.2f} MB'.format(end_mem))
print('Decreased by {:.1f}%'.format(100 * (start_mem - end_mem) / start_mem))    
base_dataset.isna().sum()
for i in base_dataset.columns:
    if i != 'Class' and i !='Time':
        sns.boxplot(x=base_dataset[i])
        plt.show()
from IPython.display import Image
Image("/kaggle/input/outlierimage/outlier.png")
def outliers_transform(base_dataset):
    for i in base_dataset.var().sort_values(ascending=False).index[0:10]:
        x=np.array(base_dataset[i])
        qr1=np.quantile(x,0.25)
        qr3=np.quantile(x,0.75)
        iqr=qr3-qr1
        utv=qr3+(1.5*(iqr))
        ltv=qr1-(1.5*(iqr))
        y=[]
        #"""Based on clients input(ltv,utv) run the below code """
        for p in x:
            if p <ltv or p>utv:
                y.append(np.median(x))
            else:
                y.append(p)
        base_dataset[i]=y
outliers_transform(base_dataset)
sns.boxplot(x=base_dataset['V1'])
sns.boxplot(x=base_dataset['V2'])
base_dataset.columns
sd = StandardScaler()
sd.fit_transform(pd.DataFrame(base_dataset['Amount']))
z1=sd.transform(pd.DataFrame(base_dataset['Amount']))
base_dataset['Amount']=z1

z2 =sd.fit_transform(pd.DataFrame(base_dataset['Time']))
base_dataset['Time']=z2


base_dataset.head()
for i in base_dataset.var().index:
    sns.distplot(base_dataset[i],kde=False)
    plt.show()
plt.figure(figsize=(20,10))
sns.heatmap(base_dataset.corr())
y = base_dataset['Class']
x = base_dataset.drop('Class',axis=1)
y.head(10)
x.head()
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(x,y,test_size=0.20,random_state=43)
print(X_train.shape,X_test.shape,y_train.shape,y_test.shape)
final_accuracy_scores_decisionTree_gini=[]
dt=DecisionTreeClassifier(criterion='gini')
dt.fit(X_train,y_train)
dt.predict(X_train)
dt.predict(X_test)
print("")
print("---------------------------------------------------------------------------------------------------------")
print("For the machine learning model : {}".format(i))
print("Confusion matrix for test samples")
print(confusion_matrix(y_test,dt.predict(X_test)))
print("Accuracy score for test samples",accuracy_score(y_test,dt.predict(X_test)))
print("Confusion matrix for training samples")
print(confusion_matrix(y_train,dt.predict(X_train)))
print("Accuracy score for training samples",accuracy_score(y_train,dt.predict(X_train)))
final_accuracy_scores_decisionTree_gini.append([dt,confusion_matrix(y_test,dt.predict(X_test)),accuracy_score(y_test,dt.predict(X_test)),confusion_matrix(y_train,dt.predict(X_train)),accuracy_score(y_train,dt.predict(X_train))])
from sklearn.model_selection import cross_val_score
print("K-Fold results for machine learning model : {} ".format(dt))
print(cross_val_score(dt,X_train,y_train,cv=10))
predicted_decisionTree_gini = dt.predict(X_test)
predicted_decisionTree_gini
final_accuracy_scores_randomForest_gini=[]
dt=RandomForestClassifier(criterion='gini')
dt.fit(X_train,y_train)
dt.predict(X_train)
dt.predict(X_test)
print("")
print("---------------------------------------------------------------------------------------------------------")
print("For the machine learning model : {}".format(i))
print("Confusion matrix for test samples")
print(confusion_matrix(y_test,dt.predict(X_test)))
print("Accuracy score for test samples",accuracy_score(y_test,dt.predict(X_test)))
print("Confusion matrix for training samples")
print(confusion_matrix(y_train,dt.predict(X_train)))
print("Accuracy score for training samples",accuracy_score(y_train,dt.predict(X_train)))
final_accuracy_scores_randomForest_gini.append([dt,confusion_matrix(y_test,dt.predict(X_test)),accuracy_score(y_test,dt.predict(X_test)),confusion_matrix(y_train,dt.predict(X_train)),accuracy_score(y_train,dt.predict(X_train))])
from sklearn.model_selection import cross_val_score
print("K-Fold results for machine learning model : {} ".format(dt))
print(cross_val_score(dt,X_train,y_train,cv=10))
predicted_randomForest_gini = dt.predict(X_test)
predicted_randomForest_gini
final_accuracy_scores_Bagging=[]
dt=BaggingClassifier()
dt.fit(X_train,y_train)
dt.predict(X_train)
dt.predict(X_test)
print("")
print("---------------------------------------------------------------------------------------------------------")
print("For the machine learning model : {}".format(i))
print("Confusion matrix for test samples")
print(confusion_matrix(y_test,dt.predict(X_test)))
print("Accuracy score for test samples",accuracy_score(y_test,dt.predict(X_test)))
print("Confusion matrix for training samples")
print(confusion_matrix(y_train,dt.predict(X_train)))
print("Accuracy score for training samples",accuracy_score(y_train,dt.predict(X_train)))
final_accuracy_scores_Bagging.append([dt,confusion_matrix(y_test,dt.predict(X_test)),accuracy_score(y_test,dt.predict(X_test)),confusion_matrix(y_train,dt.predict(X_train)),accuracy_score(y_train,dt.predict(X_train))])
from sklearn.model_selection import cross_val_score
print("K-Fold results for machine learning model : {} ".format(dt))
print(cross_val_score(dt,X_train,y_train,cv=10))
predicted_bagging = dt.predict(X_test)
predicted_bagging
from sklearn.metrics import accuracy_score
accuracy_score(y_test,dt.predict(X_test))
final_accuracy_scores_DecisionTree_entropy=[]
dt=DecisionTreeClassifier(criterion='entropy')
dt.fit(X_train,y_train)
dt.predict(X_train)
dt.predict(X_test)
print("")
print("---------------------------------------------------------------------------------------------------------")
print("For the machine learning model : {}".format(i))
print("Confusion matrix for test samples")
print(confusion_matrix(y_test,dt.predict(X_test)))
print("Accuracy score for test samples",accuracy_score(y_test,dt.predict(X_test)))
print("Confusion matrix for training samples")
print(confusion_matrix(y_train,dt.predict(X_train)))
print("Accuracy score for training samples",accuracy_score(y_train,dt.predict(X_train)))
final_accuracy_scores_DecisionTree_entropy.append([dt,confusion_matrix(y_test,dt.predict(X_test)),accuracy_score(y_test,dt.predict(X_test)),confusion_matrix(y_train,dt.predict(X_train)),accuracy_score(y_train,dt.predict(X_train))])
from sklearn.model_selection import cross_val_score
print("K-Fold results for machine learning model : {} ".format(dt))
print(cross_val_score(dt,X_train,y_train,cv=10))
predicted_decisionTree_entropy = dt.predict(X_test)
predicted_decisionTree_entropy
from sklearn.metrics import accuracy_score
accuracy_score(y_test,dt.predict(X_test))
final_accuracy_scores_RandomForest_entropy=[]
dt=RandomForestClassifier(criterion='entropy')
dt.fit(X_train,y_train)
dt.predict(X_train)
dt.predict(X_test)
print("")
print("---------------------------------------------------------------------------------------------------------")
print("For the machine learning model : {}".format(i))
print("Confusion matrix for test samples")
print(confusion_matrix(y_test,dt.predict(X_test)))
print("Accuracy score for test samples",accuracy_score(y_test,dt.predict(X_test)))
print("Confusion matrix for training samples")
print(confusion_matrix(y_train,dt.predict(X_train)))
print("Accuracy score for training samples",accuracy_score(y_train,dt.predict(X_train)))
final_accuracy_scores_RandomForest_entropy.append([dt,confusion_matrix(y_test,dt.predict(X_test)),accuracy_score(y_test,dt.predict(X_test)),confusion_matrix(y_train,dt.predict(X_train)),accuracy_score(y_train,dt.predict(X_train))])
from sklearn.model_selection import cross_val_score
print("K-Fold results for machine learning model : {} ".format(dt))
print(cross_val_score(dt,X_train,y_train,cv=10))
predicted_randomForest_entropy = dt.predict(X_test)
predicted_randomForest_entropy
from sklearn.metrics import accuracy_score
accuracy_score(y_test,dt.predict(X_test))