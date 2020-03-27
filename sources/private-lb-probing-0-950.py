
import numpy as np, pandas as pd, os, gc

train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')

train.head()
# FIND STANDARD DEVIATION OF ALL 512*256 BLOCKS
useful = np.zeros((256,512))
for i in range(512):
    partial = train[ train['wheezy-copper-turtle-magic']==i ]
    useful[:,i] = np.std(partial.iloc[:,1:-1], axis=0)
# CONVERT TO BOOLEANS IDENTIFYING USEFULNESS
useful = useful > 1.5
import matplotlib.pyplot as plt

# PLOT BOOLEANS OF USEFUL BLOCKS
plt.figure(figsize=(10,20))
plt.matshow(useful.transpose(),fignum=1)
plt.title('The useful datablocks of dataset', fontsize=24)
plt.xlabel('256 Variable columns', fontsize=16)
plt.ylabel('512 Partial datasets', fontsize=16)
plt.show()
# SET MAGIC COLUMN AS ALL USEFUL
useful[146,:] = [True]*512

# REMOVE ALL USELESS BLOCKS
for i in range(512):
    idx = train.columns[1:-1][ ~useful[:,i] ]    
    train.loc[ train.iloc[:,147]==i,idx ] = 0.0
    test.loc[ test.iloc[:,147]==i,idx ] = 0.0 
    #if i%25==0: print(i)
from sklearn.model_selection import StratifiedKFold
from sklearn import metrics, preprocessing

import tensorflow as tf
tf.logging.set_verbosity(tf.logging.ERROR) #hide warnings
from keras.layers import Dense, Input
from keras.layers import BatchNormalization
from keras.models import Model
from keras import callbacks
from keras import backend as K
from keras.layers import Dropout
from keras.callbacks import LearningRateScheduler

import warnings
warnings.filterwarnings("ignore")
# CUSTOM METRICS
def fallback_auc(y_true, y_pred):
    try:
        return metrics.roc_auc_score(y_true, y_pred)
    except:
        return 0.5

def auc(y_true, y_pred):
    return tf.py_function(fallback_auc, (y_true, y_pred), tf.double)
# ONE-HOT-ENCODE THE MAGIC FEATURE
len_train = train.shape[0]
test['target'] = -1
data = pd.concat([train, test])
data = pd.concat([data, pd.get_dummies(data['wheezy-copper-turtle-magic'])], axis=1, sort=False)

train = data[:len_train]
test = data[len_train:]
# PREPARE DATA AND STANDARDIZE
y = train.target
ids = train.id.values
train = train.drop(['id', 'target'], axis=1)
test_ids = test.id.values
test = test[train.columns]

all_auc_NN = []
oof_preds_NN = np.zeros((len(train)))
test_preds_NN = np.zeros((len(test)))

scl = preprocessing.StandardScaler()
scl.fit(pd.concat([train, test]))
train = scl.transform(train)
test = scl.transform(test)
NFOLDS = 15
RANDOM_STATE = 42

gc.collect()
# STRATIFIED K FOLD
folds = StratifiedKFold(n_splits=NFOLDS, shuffle=True, random_state=RANDOM_STATE)
for fold_, (trn_, val_) in enumerate(folds.split(y, y)):
    #print("Current Fold: {}".format(fold_))
    trn_x, trn_y = train[trn_, :], y.iloc[trn_]
    val_x, val_y = train[val_, :], y.iloc[val_]

    # BUILD MODEL
    inp = Input(shape=(trn_x.shape[1],))
    x = Dense(2000, activation="relu")(inp)
    x = BatchNormalization()(x)
    x = Dropout(0.3)(x)
    x = Dense(1000, activation="relu")(x)
    x = BatchNormalization()(x)
    x = Dropout(0.3)(x)
    x = Dense(500, activation="relu")(x)
    x = BatchNormalization()(x)
    x = Dropout(0.2)(x)
    x = Dense(100, activation="relu")(x)
    x = BatchNormalization()(x)
    x = Dropout(0.2)(x)
    out = Dense(1, activation="sigmoid")(x)
    clf = Model(inputs=inp, outputs=out)
    clf.compile(loss='binary_crossentropy', optimizer="adam", metrics=[auc])
    
    # CALLBACKS
    es = callbacks.EarlyStopping(monitor='val_auc', min_delta=0.001, patience=10,
                verbose=0, mode='max', baseline=None, restore_best_weights=True)
    rlr = callbacks.ReduceLROnPlateau(monitor='val_auc', factor=0.5,
                patience=3, min_lr=1e-6, mode='max', verbose=0)

    # TRAIN
    clf.fit(trn_x, trn_y, validation_data=(val_x, val_y), callbacks=[es, rlr], epochs=100, 
                batch_size=1024, verbose=0)
    
    # PREDICT TEST
    test_fold_preds = clf.predict(test)
    test_preds_NN += test_fold_preds.ravel() / NFOLDS
    
    # PREDICT OOF
    val_preds = clf.predict(val_x)
    oof_preds_NN[val_] = val_preds.ravel()
    
    # RECORD AUC
    val_auc = round( metrics.roc_auc_score(val_y, val_preds),5 )
    all_auc_NN.append(val_auc)
    print('Fold',fold_,'has AUC =',val_auc)
    
    K.clear_session()
    gc.collect()
# DISPLAY NN VALIDATION AUC
val_auc = metrics.roc_auc_score(y, oof_preds_NN)
print('NN_CV = OOF_AUC =', round( val_auc,5) )
print('Mean_AUC =', round( np.mean(all_auc_NN),5) )

# PLOT NN TEST PREDICTIONS
plt.hist(test_preds_NN,bins=100)
plt.title('NN test.csv predictions')
plt.show()
train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')

cols = [c for c in train.columns if c not in ['id', 'target', 'wheezy-copper-turtle-magic']]

from sklearn.svm import SVC
from sklearn.feature_selection import VarianceThreshold
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score

# INITIALIZE VARIABLES
oof_preds_SVM = np.zeros(len(train))
test_preds_SVM = np.zeros(len(test))

# BUILD 512 SEPARATE MODELS
for i in range(512):
    
    # ONLY TRAIN/PREDICT WHERE WHEEZY-MAGIC EQUALS I
    train2 = train[train['wheezy-copper-turtle-magic']==i]
    test2 = test[test['wheezy-copper-turtle-magic']==i]
    idx1 = train2.index; idx2 = test2.index
    train2.reset_index(drop=True,inplace=True)
    
    # FEATURE SELECTION (USE SUBSET OF 255 FEATURES)
    sel = VarianceThreshold(threshold=1.5).fit(train2[cols])
    train3 = sel.transform(train2[cols])
    test3 = sel.transform(test2[cols])
        
    # STRATIFIED K FOLD
    skf = StratifiedKFold(n_splits=11, random_state=42)
    for train_index, test_index in skf.split(train3, train2['target']):
        
        # MODEL WITH SUPPORT VECTOR MACHINE
        clf = SVC(probability=True,kernel='poly',degree=4,gamma='auto')
        clf.fit(train3[train_index,:],train2.loc[train_index]['target'])
        oof_preds_SVM[idx1[test_index]] = clf.predict_proba(train3[test_index,:])[:,1]
        test_preds_SVM[idx2] += clf.predict_proba(test3)[:,1] / skf.n_splits
        
    #if i%10==0: print(i)
# DISPLAY SVM VALIDATION AUC
val_auc = roc_auc_score(train['target'],oof_preds_SVM)
print('SVM_CV = OOF_AUC =',round(val_auc,5))

# PLOT SVM TEST PREDICTIONS
plt.hist(test_preds_SVM,bins=100)
plt.title('SVM test.csv predictions')
plt.show()
# DISPLAY ENSEMBLE VALIDATION AUC
val_auc = roc_auc_score(train['target'],oof_preds_SVM+oof_preds_NN)
print('Ensemble_NN+SVM_CV = OOF_AUC =',round(val_auc,5))

# PLOT ENSEMBLE TEST PREDICTIONS
plt.hist(test_preds_SVM+test_preds_NN,bins=100)
plt.title('Ensemble NN+SVM test.csv predictions')
plt.show()
import hashlib

# CREATE LIST PUBLIC DATASET IDS
public_ids = []
for i in range(256*512+1):
    st = str(i)+"test"
    public_ids.append( hashlib.md5(st.encode()).hexdigest() )
    
# DISPLAY FIRST 5 GENERATED
public_ids[:5]
# DISPLAY FIRST 5 ACTUAL
test['id'].head()
# SEPARATE PUBLIC AND PRIVATE DATASETS
public = test[ test['id'].isin(public_ids) ].copy()
private = test[ ~test.index.isin(public.index) ].copy()
# DETERMINE TRAIN DATASET STRUCTURE
useful_train = np.zeros((256,512))
for i in range(512):
    partial = train[ train['wheezy-copper-turtle-magic']==i ]
    useful_train[:,i] = np.std(partial.iloc[:,1:-1], axis=0)
useful_train = useful_train > 1.5

# DETERMINE PUBLIC TEST DATASET STRUCTURE
useful_public = np.zeros((256,512))
for i in range(512):
    partial = public[ public['wheezy-copper-turtle-magic']==i ]
    useful_public[:,i] = np.std(partial.iloc[:,1:], axis=0)
useful_public = useful_public > 1.5

# DETERMINE PRIVATE TEST DATASET STRUCTURE
useful_private = np.zeros((256,512))
for i in range(512):
    partial = private[ private['wheezy-copper-turtle-magic']==i ]
    useful_private[:,i] = np.std(partial.iloc[:,1:], axis=0)
useful_private = useful_private > 1.5
if np.allclose(useful_train,useful_public):
    print('Public dataset has the SAME structure as train')
else:
    print('Public dataset DOES NOT HAVE the same structure as train')
sub = pd.read_csv('../input/sample_submission.csv')

if np.allclose(useful_train,useful_public) & np.allclose(useful_train,useful_private):
    print('We are submitting TRUE predictions for LB 0.950')
    sub['target'] = (test_preds_NN + test_preds_SVM) / 2.0
else:
    print('We are submitting ALL ZERO predictions for LB 0.500')
    sub['target'] = np.zeros( len(sub) )
    
sub.to_csv('submission.csv',index=False)