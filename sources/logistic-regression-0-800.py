
import numpy as np, pandas as pd, os
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score

train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')

train.head()
cols = [c for c in train.columns if c not in ['id', 'target']]
oof = np.zeros(len(train))
skf = StratifiedKFold(n_splits=5, random_state=42)
   
for train_index, test_index in skf.split(train.iloc[:,1:-1], train['target']):
    clf = LogisticRegression(solver='liblinear',penalty='l2',C=1.0)
    clf.fit(train.loc[train_index][cols],train.loc[train_index]['target'])
    oof[test_index] = clf.predict_proba(train.loc[test_index][cols])[:,1]
    
auc = roc_auc_score(train['target'],oof)
print('LR without interactions scores CV =',round(auc,5))
# INITIALIZE VARIABLES
cols.remove('wheezy-copper-turtle-magic')
interactions = np.zeros((512,255))
oof = np.zeros(len(train))
preds = np.zeros(len(test))

# BUILD 512 SEPARATE MODELS
for i in range(512):
    # ONLY TRAIN WITH DATA WHERE WHEEZY EQUALS I
    train2 = train[train['wheezy-copper-turtle-magic']==i]
    test2 = test[test['wheezy-copper-turtle-magic']==i]
    idx1 = train2.index; idx2 = test2.index
    train2.reset_index(drop=True,inplace=True)
    test2.reset_index(drop=True,inplace=True)
    
    skf = StratifiedKFold(n_splits=25, random_state=42)
    for train_index, test_index in skf.split(train2.iloc[:,1:-1], train2['target']):
        # LOGISTIC REGRESSION MODEL
        clf = LogisticRegression(solver='liblinear',penalty='l1',C=0.05)
        clf.fit(train2.loc[train_index][cols],train2.loc[train_index]['target'])
        oof[idx1[test_index]] = clf.predict_proba(train2.loc[test_index][cols])[:,1]
        preds[idx2] += clf.predict_proba(test2[cols])[:,1] / 25.0
        # RECORD INTERACTIONS
        for j in range(255):
            if clf.coef_[0][j]>0: interactions[i,j] = 1
            elif clf.coef_[0][j]<0: interactions[i,j] = -1
    #if i%25==0: print(i)
        
# PRINT CV AUC
auc = roc_auc_score(train['target'],oof)
print('LR with interactions scores CV =',round(auc,5))
sub = pd.read_csv('../input/sample_submission.csv')
sub['target'] = preds
sub.to_csv('submission.csv',index=False)
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

plt.figure(figsize=(15,5))

# PLOT ALL ZIPPY
plt.subplot(1,2,1)
sns.distplot(train[ (train['target']==0) ]['zippy-harlequin-otter-grandmaster'], label = 't=0')
sns.distplot(train[ (train['target']==1) ]['zippy-harlequin-otter-grandmaster'], label = 't=1')
plt.title("Without interaction, zippy has no correlation \n (showing all rows)")
plt.xlim((-5,5))
plt.legend()

# PLOT ZIPPY WHERE WHEEZY-MAGIC=0
plt.subplot(1,2,2)
sns.distplot(train[ (train['wheezy-copper-turtle-magic']==0) & (train['target']==0) ]
             ['zippy-harlequin-otter-grandmaster'], label = 't=0')
sns.distplot(train[ (train['wheezy-copper-turtle-magic']==0) & (train['target']==1) ]
             ['zippy-harlequin-otter-grandmaster'], label = 't=1')
plt.title("With interaction, zippy has postive correlation \n (only showing rows where wheezy-copper-turtle-magic=0)")
plt.legend()

plt.show()
plt.figure(figsize=(15,5))

# PLOT ALL HASTY
plt.subplot(1,2,1)
sns.distplot(train[ (train['target']==0) ]['hasty-puce-fowl-fepid'], label = 't=0')
sns.distplot(train[ (train['target']==1) ]['hasty-puce-fowl-fepid'], label = 't=1')
plt.title("Without interaction, hasty has no correlation \n (showing all rows)")
plt.xlim((-5,5))
plt.legend()

# PLOT HASTY WHERE WHEEZY-MAGIC=0
plt.subplot(1,2,2)
sns.distplot(train[ (train['wheezy-copper-turtle-magic']==0) & (train['target']==0) ]
             ['hasty-puce-fowl-fepid'], label = 't=0')
sns.distplot(train[ (train['wheezy-copper-turtle-magic']==0) & (train['target']==1) ]
             ['hasty-puce-fowl-fepid'], label = 't=1')
plt.title("With interaction, hasty has negative correlation \n (only showing rows where wheezy-copper-turtle-magic=0)")
plt.legend()

plt.show()
# PLOT INTERACTIONS WITH WHEEZY-MAGIC
plt.figure(figsize=(15,8))
plt.matshow(interactions.transpose(),fignum=1)
plt.title("Variable Interactions with wheezy-copper-turtle-magic \n \
    Yellow = combines to create positive correlation with target. Blue = negative correlation",fontsize=16)
plt.xlabel('values of wheezy-copper-turtle-magic')
plt.ylabel('columns of train')
plt.show()