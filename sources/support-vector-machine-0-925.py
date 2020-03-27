
import numpy as np, pandas as pd, os
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score

train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')

train.head()
# LOAD LIBRARIES
from sklearn.svm import SVC
from sklearn.feature_selection import VarianceThreshold
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
# INITIALIZE VARIABLES
oof = np.zeros(len(train))
preds = np.zeros(len(test))
cols = [c for c in train.columns if c not in ['id', 'target', 'wheezy-copper-turtle-magic']]

# BUILD 512 SEPARATE NON-LINEAR MODELS
for i in range(512):
    
    # EXTRACT SUBSET OF DATASET WHERE WHEEZY-MAGIC EQUALS I
    train2 = train[train['wheezy-copper-turtle-magic']==i]
    test2 = test[test['wheezy-copper-turtle-magic']==i]
    idx1 = train2.index; idx2 = test2.index
    train2.reset_index(drop=True,inplace=True)
    
    # FEATURE SELECTION (USE APPROX 40 OF 255 FEATURES)
    sel = VarianceThreshold(threshold=1.5).fit(train2[cols])
    train3 = sel.transform(train2[cols])
    test3 = sel.transform(test2[cols])
        
    # STRATIFIED K FOLD (Using splits=25 scores 0.002 better but is slower)
    skf = StratifiedKFold(n_splits=11, random_state=42)
    for train_index, test_index in skf.split(train3, train2['target']):
        
        # MODEL WITH SUPPORT VECTOR MACHINE
        clf = SVC(probability=True,kernel='poly',degree=4,gamma='auto')
        clf.fit(train3[train_index,:],train2.loc[train_index]['target'])
        oof[idx1[test_index]] = clf.predict_proba(train3[test_index,:])[:,1]
        preds[idx2] += clf.predict_proba(test3)[:,1] / skf.n_splits
        
    #if i%10==0: print(i)
        
# PRINT VALIDATION CV AUC
auc = roc_auc_score(train['target'],oof)
print('CV score =',round(auc,5))
sub = pd.read_csv('../input/sample_submission.csv')
sub['target'] = preds
sub.to_csv('submission.csv',index=False)

import matplotlib.pyplot as plt
plt.hist(preds,bins=100)
plt.title('Test.csv predictions')
plt.show()
# LOAD LIBRARIES
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import warnings
warnings.filterwarnings("ignore")

# PLOT FIRST 8 VARIABLES
plt.figure(figsize=(15,15))
for i in range(8):
    plt.subplot(3,3,i+1)
    #plt.hist(train.iloc[:,i+1],bins=100)
    sns.distplot(train.iloc[:,i+1],bins=100)
    plt.title( train.columns[i+1] )
    plt.xlabel('')
    
# PLOT GAUSSIAN FOR COMPARISON
plt.subplot(3,3,9)
std = round(np.std(train.iloc[:,8]),2)
data = np.random.normal(0,std,len(train))
sns.distplot(data,bins=100)
plt.xlim((-17,17))
plt.ylim((0,0.37))
plt.title("Gaussian with m=0, std="+str(std))

plt.subplots_adjust(hspace=0.3)
plt.show()
# NORMALITY PLOTS FOR FIRST 8 VARIABLES
plt.figure(figsize=(15,15))
for i in range(8):
    plt.subplot(3,3,i+1)
    stats.probplot(train.iloc[:,1], plot=plt)
    plt.title( train.columns[i+1] )
    
# NORMALITY PLOT FOR GAUSSIAN
plt.subplot(3,3,9)
stats.probplot(data, plot=plt)   
plt.title("Gaussian with m=0, std="+str(std))

plt.subplots_adjust(hspace=0.4)
plt.show()
train0 = train[ train['wheezy-copper-turtle-magic']==0 ]
# PLOT FIRST 8 VARIABLES
plt.figure(figsize=(15,15))
for i in range(8):
    plt.subplot(3,3,i+1)
    #plt.hist(train0.iloc[:,i+1],bins=10)
    sns.distplot(train0.iloc[:,i+1],bins=10)
    plt.title( train.columns[i+1] )
    plt.xlabel('')
    
# PLOT GAUSSIAN FOR COMPARISON
plt.subplot(3,3,9)
std0 = round(np.std(train0.iloc[:,8]),2)
data0 = np.random.normal(0,std0,2*len(train0))
sns.distplot(data0,bins=10)
plt.xlim((-17,17))
plt.ylim((0,0.1))
plt.title("Gaussian with m=0, std="+str(std0))
    
plt.subplots_adjust(hspace=0.3)
plt.show()
# NORMALITY PLOTS FOR FIRST 8 VARIABLES
plt.figure(figsize=(15,15))
for i in range(8):
    plt.subplot(3,3,i+1)
    stats.probplot(train0.iloc[:,1], plot=plt)
    plt.title( train.columns[i+1] )
    
# NORMALITY PLOT FOR GAUSSIAN
plt.subplot(3,3,9)
stats.probplot(data0, plot=plt)   
plt.title("Gaussian with m=0, std="+str(std0))

plt.subplots_adjust(hspace=0.4)
plt.show()