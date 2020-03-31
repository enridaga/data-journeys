
from sklearn.model_selection import StratifiedKFold, KFold, RepeatedKFold
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error
from eli5.sklearn import PermutationImportance
from sklearn.tree import DecisionTreeRegressor
from catboost import CatBoostRegressor,Pool
import matplotlib.patches as patch
from scipy.stats import kurtosis
import matplotlib.pyplot as plt
from sklearn.svm import NuSVR
from scipy.stats import skew
from scipy.stats import norm
from scipy import linalg
from sklearn import tree
from sklearn import svm
import lightgbm as lgb
import xgboost as xgb
from tqdm import tqdm
import seaborn as sns
import pandas as pd
import numpy as np
import graphviz
import warnings
import random
import eli5
import shap  # package used to calculate Shap values
import time
import glob
import sys
import os
### matplotlib inline
### precision 4
warnings.filterwarnings('ignore')
plt.style.use('ggplot')
np.set_printoptions(suppress=True)
pd.set_option("display.precision", 15)
print('pandas: {}'.format(pd.__version__))
print('numpy: {}'.format(np.__version__))
print('Python: {}'.format(sys.version))
print(os.listdir("../input/"))
### %time
train = pd.read_csv('../input/train.csv' , dtype={'acoustic_data': np.int16, 'time_to_failure': np.float32})
print("Train has: rows:{} cols:{}".format(train.shape[0], train.shape[1]))
submission = pd.read_csv('../input/sample_submission.csv', index_col='seg_id')
submission.head()
print("submission has: rows:{} cols:{}".format(submission.shape[0], submission.shape[1]))
len(os.listdir(os.path.join('../input/', 'test')))
train.info()
train.shape
train.head()
# we change the size of Dataset due to play with it (just loaded %0001)
mini_train= pd.read_csv("../input/train.csv",nrows=150000)
mini_train.describe()
mini_train.shape
mini_train.isna().sum()
type(mini_train)
#acoustic_data means signal
mini_train["acoustic_data"].hist();
plt.plot(mini_train["time_to_failure"], mini_train["acoustic_data"])
plt.title("time_to_failure histogram")
sns.distplot(mini_train.acoustic_data.values, color="Red", bins=100, kde=False)
sns.kdeplot(mini_train["acoustic_data"] )
# based on : https://www.kaggle.com/inversion/basic-feature-benchmark
rows = 150_000
segments = int(np.floor(train.shape[0] / rows))
segments
X_train = pd.DataFrame(index=range(segments), dtype=np.float64,
                       columns=['ave', 'std', 'max', 'min','sum','skew','kurt'])
y_train = pd.DataFrame(index=range(segments), dtype=np.float64,
                       columns=['time_to_failure'])
y_train.head()
X_train.head()
### %time
for segment in tqdm(range(segments)):
    seg = train.iloc[segment*rows:segment*rows+rows]
    x = seg['acoustic_data'].values
    y = seg['time_to_failure'].values[-1]
    
    y_train.loc[segment, 'time_to_failure'] = y
    X_train.loc[segment, 'ave'] = x.mean()
    X_train.loc[segment, 'std'] = x.std()
    X_train.loc[segment, 'max'] = x.max()
    X_train.loc[segment, 'min'] = x.min()
    X_train.loc[segment, 'sum'] = x.sum()
    X_train.loc[segment, 'skew'] =skew(x)
    X_train.loc[segment, 'kurt'] = kurtosis(x)
X_train.head()
y_train.head()
def check_missing_data(df):
    flag=df.isna().sum().any()
    if flag==True:
        total = df.isnull().sum()
        percent = (df.isnull().sum())/(df.isnull().count()*100)
        output = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
        data_type = []
        # written by MJ Bahmani
        for col in df.columns:
            dtype = str(df[col].dtype)
            data_type.append(dtype)
        output['Types'] = data_type
        return(np.transpose(output))
    else:
        return(False)
check_missing_data(X_train)
X_test = pd.DataFrame(columns=X_train.columns, dtype=np.float64, index=submission.index)
X_test.head()
### %time
for seg_id in  tqdm(X_test.index):
    seg = pd.read_csv('../input/test/' + seg_id + '.csv')
    
    x = seg['acoustic_data'].values
    
    X_test.loc[seg_id, 'ave'] = x.mean()
    X_test.loc[seg_id, 'std'] = x.std()
    X_test.loc[seg_id, 'max'] = x.max()
    X_test.loc[seg_id, 'min'] = x.min()
    X_test.loc[seg_id, 'sum'] = x.sum()
    X_test.loc[seg_id, 'skew'] =skew(x)
    X_test.loc[seg_id, 'kurt'] = kurtosis(x)
X_test.shape
scaler = StandardScaler()
scaler.fit(X_train)
X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)
X=X_train.copy()
y=y_train.copy()
train_X, val_X, train_y, val_y = train_test_split(X, y, random_state=1)
rfc_model = RandomForestRegressor(random_state=0).fit(train_X, train_y)
perm = PermutationImportance(rfc_model, random_state=1).fit(val_X, val_y)
eli5.show_weights(perm, feature_names = val_X.columns.tolist(), top=7)
train_X, val_X, train_y, val_y = train_test_split(X, y, random_state=1)
tree_model = DecisionTreeRegressor(random_state=0, max_depth=5, min_samples_split=5).fit(train_X, train_y)
tree_graph = tree.export_graphviz(tree_model, out_file=None, feature_names=X.columns)
graphviz.Source(tree_graph)
row_to_show = 5
data_for_prediction = val_X.iloc[row_to_show]  # use 1 row of data here. Could use multiple rows if desired
data_for_prediction_array = data_for_prediction.values.reshape(1, -1)


tree_model.predict(data_for_prediction_array);
# Create object that can calculate shap values
explainer = shap.TreeExplainer(tree_model)

# Calculate Shap values
shap_values = explainer.shap_values(data_for_prediction)
shap.initjs()
shap.force_plot(explainer.expected_value, shap_values, data_for_prediction)
from matplotlib import pyplot as plt
from pdpbox import pdp, get_dataset, info_plots

# Create the data that we will plot
pdp_goals = pdp.pdp_isolate(model=tree_model, dataset=val_X, model_features=X.columns, feature='std')

# plot it
pdp.pdp_plot(pdp_goals, 'std')
plt.show()
# Create the data that we will plot
pdp_goals = pdp.pdp_isolate(model=rfc_model, dataset=val_X, model_features=X.columns, feature='kurt')

# plot it
pdp.pdp_plot(pdp_goals, 'kurt')
plt.show()
# Create the data that we will plot
pdp_goals = pdp.pdp_isolate(model=rfc_model, dataset=val_X, model_features=X.columns, feature='max')

# plot it
pdp.pdp_plot(pdp_goals, 'max')
plt.show()
# Create the data that we will plot
pdp_goals = pdp.pdp_isolate(model=rfc_model, dataset=val_X, model_features=X.columns, feature='min')

# plot it
pdp.pdp_plot(pdp_goals, 'min')
plt.show()
svm = NuSVR()
svm.fit(X_train_scaled, y_train.values.flatten())
y_pred_svm = svm.predict(X_train_scaled)
score = mean_absolute_error(y_train.values.flatten(), y_pred_svm)
print(f'Score: {score:0.3f}')
folds = KFold(n_splits=5, shuffle=True, random_state=42)
params = {'objective' : "regression", 
               'boosting':"gbdt",
               'metric':"mae",
               'boost_from_average':"false",
               'num_threads':8,
               'learning_rate' : 0.001,
               'num_leaves' : 52,
               'max_depth':-1,
               'tree_learner' : "serial",
               'feature_fraction' : 0.85,
               'bagging_freq' : 1,
               'bagging_fraction' : 0.85,
               'min_data_in_leaf' : 10,
               'min_sum_hessian_in_leaf' : 10.0,
               'verbosity' : -1}
### %time
y_pred_lgb = np.zeros(len(X_test_scaled))
for fold_n, (train_index, valid_index) in tqdm(enumerate(folds.split(X))):
    print('Fold', fold_n, 'started at', time.ctime())
    X_train, X_valid = X.iloc[train_index], X.iloc[valid_index]
    y_train, y_valid = y.iloc[train_index], y.iloc[valid_index]
        
    model = lgb.LGBMRegressor(**params, n_estimators = 22000, n_jobs = -1)
    model.fit(X_train, y_train, 
                    eval_set=[(X_train, y_train), (X_valid, y_valid)], eval_metric='mae',
                    verbose=1000, early_stopping_rounds=200)
            
    y_pred_valid = model.predict(X_valid)
    y_pred_lgb += model.predict(X_test_scaled, num_iteration=model.best_iteration_) / folds.n_splits
train_pool = Pool(X,y)
cat_model = CatBoostRegressor(
                               iterations=3000,# change 25 to 3000 to get best performance 
                               learning_rate=0.03,
                               eval_metric='MAE',
                              )
cat_model.fit(X,y,silent=True)
y_pred_cat = cat_model.predict(X_test)
y_pred_svm= svm.predict(X_test_scaled)
submission['time_to_failure'] = y_pred_cat
submission.to_csv('submission_svm.csv')
submission['time_to_failure'] = y_pred_lgb
submission.to_csv('submission_lgb.csv')
submission['time_to_failure'] = y_pred_cat
submission.to_csv('submission_cat.csv')
y_pred_rf=rfc_model.predict(X_test_scaled)

submission['time_to_failure'] = y_pred_rf
submission.to_csv('submission_rf.csv')
blending = y_pred_svm*0.5 + y_pred_lgb*0.5 
submission['time_to_failure']=blending
submission.to_csv('submission_lgb_svm.csv')
blending = y_pred_svm*0.5 + y_pred_cat*0.5 
submission['time_to_failure']=blending
submission.to_csv('submission_cat_svm.csv')