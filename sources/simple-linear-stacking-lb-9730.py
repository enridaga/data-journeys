
# File containing validation data
# (These are selected from the last day of the original training set
#  to correspond to the times of day used in the test set.)
VAL_FILE = '../input/training-and-validation-data-pickle/validation.pkl.gz'
import numpy as np
import pandas as pd
import os
from sklearn.linear_model import LogisticRegression
from scipy.special import expit, logit
from sklearn.metrics import roc_auc_score

print(os.listdir("../input"))
almost_zero = 1e-10
almost_one = 1 - almost_zero
# Just names to identify the models
base_models = {
    'lgb1 ': "Python LGBM based on Pranav Pandya's R version",
    'wbftl': "anttip's Wordbatch FM-FTRL",
    'nngpu': "Downampled Neural Network run on GPU"
    }
# Files with validation set predictions from each of the base models
# (These were fit on a subset of the training data that ends a day before
#  the end of the full training set.)
cvfiles = {
    'lgb1 ': '../input/validate-pranav-lgb-model/pranav_lgb_val_nostop.csv',
    'wbftl': '../input/validate-anttip-s-wordbatch-fm-ftrl-9711-version/wordbatch_fm_ftrl_val.csv',
    'nngpu': '../input/gpu-validation/gpu_val1.csv'
    }
# Files with test set predictions
# (These were fit on the full training set
#  or on a subset at the end, to accommodate memory limits.)
subfiles = {
    'lgb1 ': '../input/try-pranav-s-r-lgbm-in-python/sub_lgbm_r_to_python_nocv.csv',
    'wbftl': '../input/anttip-s-wordbatch-fm-ftrl-9711-version/wordbatch_fm_ftrl.csv',
    'nngpu': '../input/talkingdata-gpu-example-with-multiple-runs/gpu_test2.csv'
    }
# Public leaderbaord scores, for comparison
lbscores = {
    'lgb1 ': .9694,
    'wbftl': .9711,
    'nngpu': .9678
}
model_order = [m for m in base_models]  # To make sure order is consistent when converting to array
cvdata = pd.DataFrame( { 
    m:pd.read_csv(cvfiles[m])['is_attributed'].clip(almost_zero,almost_one).apply(logit) 
    for m in base_models
    } )
X_train = np.array(cvdata[model_order])
y_train = pd.read_pickle(VAL_FILE)['is_attributed']  # relies on validation cases being in same order
cvdata.head()
cvdata.corr()
stack_model = LogisticRegression()
stack_model.fit(X_train, y_train)
stack_model.coef_
weights = stack_model.coef_/stack_model.coef_.sum()
columns = cvdata[model_order].columns
scores = [ roc_auc_score( y_train, expit(cvdata[c]) )  for c in columns ]
names = [ base_models[c] for c in columns ]
lb = [ lbscores[c] for c in columns ]
pd.DataFrame( data={'LB score': lb, 'CV score':scores, 'weight':weights.reshape(-1)}, index=names )
print(  'Stacker score: ', roc_auc_score( y_train, stack_model.predict_proba(X_train)[:,1] )  )
final_sub = pd.DataFrame()
subs = {m:pd.read_csv(subfiles[m]).rename({'is_attributed':m},axis=1) for m in base_models}
first_model = list(base_models.keys())[0]
final_sub['click_id'] = subs[first_model]['click_id']
df = subs[first_model]
for m in base_models:
    if m != first_model:
        df = df.merge(subs[m], on='click_id')  # being careful in case clicks are in different order
df.head()
X_test = np.array( df.drop(['click_id'],axis=1)[model_order].clip(almost_zero,almost_one).apply(logit) )
final_sub['is_attributed'] = stack_model.predict_proba(X_test)[:,1]
final_sub.head(10)
final_sub.to_csv("sub_stacked.csv", index=False, float_format='%.9f')
