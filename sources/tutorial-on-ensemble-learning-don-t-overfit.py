
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix
from catboost import CatBoostClassifier,Pool
from sklearn.metrics import accuracy_score
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from pandas import get_dummies
import plotly.graph_objs as go
from sklearn import datasets
import plotly.plotly as py
import seaborn as sns
import xgboost as xgb
import lightgbm as lgb
import pandas as pd
import numpy as np
import matplotlib
import warnings
import sklearn
import scipy
import numpy
import time
import json
import sys
import csv
import os
print('matplotlib: {}'.format(matplotlib.__version__))
print('sklearn: {}'.format(sklearn.__version__))
print('scipy: {}'.format(scipy.__version__))
print('seaborn: {}'.format(sns.__version__))
print('pandas: {}'.format(pd.__version__))
print('numpy: {}'.format(np.__version__))
print('Python: {}'.format(sys.version))
# for get better result chage fold_n to 5
fold_n=5
folds = StratifiedKFold(n_splits=fold_n, shuffle=True, random_state=10)
warnings.filterwarnings('ignore')
sns.set(color_codes=True)
plt.style.available
### matplotlib inline
### precision 2
import os
print([filename for filename in os.listdir('../input') if '.csv' in filename])
# import Dataset to play with it
train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')
sample_submission = pd.read_csv('../input/sample_submission.csv')
sample_submission.head()
train.shape, test.shape, sample_submission.shape
train.head()
test.head()
train.tail()
train.columns
print(train.info())
train['target'].value_counts().plot.bar();
f,ax=plt.subplots(1,2,figsize=(18,8))
train['target'].value_counts().plot.pie(explode=[0,0.1],autopct='%1.1f%%',ax=ax[0],shadow=True)
ax[0].set_title('target')
ax[0].set_ylabel('')
sns.countplot('target',data=train,ax=ax[1])
ax[1].set_title('target')
plt.show()
train["1"].hist();
train["2"].hist();
train["3"].hist();
train[train.columns[2:]].mean().plot('hist');plt.title('Mean Frequency');
sns.set(rc={'figure.figsize':(9,7)})
sns.distplot(train['target']);
sns.violinplot(data=train,x="target", y="1");
sns.violinplot(data=train,x="target", y="20");
train['target'].value_counts()
cols=["target","id"]
X = train.drop(cols,axis=1)
y = train["target"]
X_test  = test.drop("id",axis=1)
# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size = 0.2, random_state = 0)
from sklearn.ensemble import RandomForestClassifier
Model=RandomForestClassifier(max_depth=2)
Model.fit(X_train,y_train)
y_pred=Model.predict(X_val)
print(classification_report(y_pred,y_val))
print(confusion_matrix(y_pred,y_val))
#Accuracy Score
print('accuracy is ',accuracy_score(y_pred,y_val))
from sklearn.ensemble import BaggingClassifier
bag_Model=BaggingClassifier()
Model.fit(X_train,y_train)
y_pred=Model.predict(X_val)
print(classification_report(y_pred,y_val))
print(confusion_matrix(y_pred,y_val))
#Accuracy Score
print('accuracy is ',accuracy_score(y_pred,y_val))
from sklearn.ensemble import AdaBoostClassifier
Model=AdaBoostClassifier()
Model.fit(X_train,y_train)
y_pred=Model.predict(X_val)
print(classification_report(y_pred,y_val))
print(confusion_matrix(y_pred,y_val))
#Accuracy Score
print('accuracy is ',accuracy_score(y_pred,y_val))
from sklearn.ensemble import GradientBoostingClassifier
Model=GradientBoostingClassifier()
Model.fit(X_train,y_train)
y_pred=Model.predict(X_val)
print(classification_report(y_pred,y_val))
print(confusion_matrix(y_pred,y_val))
#Accuracy Score
print('accuracy is ',accuracy_score(y_pred,y_val))
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
Model=LinearDiscriminantAnalysis()
Model.fit(X_train,y_train)
y_pred=Model.predict(X_val)
print(classification_report(y_pred,y_val))
print(confusion_matrix(y_pred,y_val))
#Accuracy Score
print('accuracy is ',accuracy_score(y_pred,y_val))
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
Model=QuadraticDiscriminantAnalysis()
Model.fit(X_train,y_train)
y_pred=Model.predict(X_val)
print(classification_report(y_pred,y_val))
print(confusion_matrix(y_pred,y_val))
#Accuracy Score
print('accuracy is ',accuracy_score(y_pred,y_val))
train_X, val_X, train_y, val_y = train_test_split(X, y, random_state=1)
rfc_model = RandomForestClassifier(random_state=0).fit(train_X, train_y)
import eli5
from eli5.sklearn import PermutationImportance

perm = PermutationImportance(rfc_model, random_state=1).fit(val_X, val_y)
eli5.show_weights(perm, feature_names = val_X.columns.tolist(),top=300)
train_X, val_X, train_y, val_y = train_test_split(X, y, random_state=1)
tree_model = DecisionTreeClassifier(random_state=0, max_depth=5, min_samples_split=5).fit(train_X, train_y)
features = [c for c in train.columns if c not in ['id', 'target']]
from sklearn import tree
import graphviz
tree_graph = tree.export_graphviz(tree_model, out_file=None, feature_names=features)
display(graphviz.Source(tree_graph))
from matplotlib import pyplot as plt
from pdpbox import pdp, get_dataset, info_plots

# Create the data that we will plot
pdp_goals = pdp.pdp_isolate(model=tree_model, dataset=val_X, model_features=features, feature='26')

# plot it
pdp.pdp_plot(pdp_goals, '26')
plt.show()
# Create the data that we will plot
pdp_goals = pdp.pdp_isolate(model=rfc_model, dataset=val_X, model_features=features, feature='264')

# plot it
pdp.pdp_plot(pdp_goals, '264')
plt.show()
import shap  # package used to calculate Shap values

# Create object that can calculate shap values
explainer = shap.TreeExplainer(rfc_model)

# Calculate Shap values
shap_values = explainer.shap_values(X_train)
shap.summary_plot(shap_values, X_train)
shap.initjs()
shap.force_plot(explainer.expected_value[1], shap_values[1], X_train.iloc[:,1:10])
# based on following kernel https://www.kaggle.com/dromosys/sctp-working-lgb
params = {'num_leaves': 9,
         'min_data_in_leaf': 42,
         'objective': 'binary',
         'max_depth': 16,
         'learning_rate': 0.0123,
         'boosting': 'gbdt',
         'bagging_freq': 5,
         'bagging_fraction': 0.8,
         'feature_fraction': 0.8201,
         'bagging_seed': 11,
         'reg_alpha': 1.728910519108444,
         'reg_lambda': 4.9847051755586085,
         'random_state': 42,
         'metric': 'auc',
         'verbosity': -1,
         'subsample': 0.81,
         'min_gain_to_split': 0.01077313523861969,
         'min_child_weight': 19.428902804238373,
         'num_threads': 4}
## based on following kernel https://www.kaggle.com/dromosys/sctp-working-lgb
y_pred_lgb = np.zeros(len(X_test))
for fold_n, (train_index, valid_index) in enumerate(folds.split(X,y)):
    print('Fold', fold_n, 'started at', time.ctime())
    X_train, X_valid = X.iloc[train_index], X.iloc[valid_index]
    y_train, y_valid = y.iloc[train_index], y.iloc[valid_index]
    
    train_data = lgb.Dataset(X_train, label=y_train)
    valid_data = lgb.Dataset(X_valid, label=y_valid)
        
    lgb_model = lgb.train(params,train_data,num_boost_round=2000,#change 20 to 2000
                    valid_sets = [train_data, valid_data],verbose_eval=300,early_stopping_rounds = 200)##change 10 to 200
            
    y_pred_lgb += lgb_model.predict(X_test, num_iteration=lgb_model.best_iteration)/5
y_pred_rfc = rfc_model.predict(X_test)
y_pred_tree = tree_model.predict(X_test)
train_pool = Pool(train_X,train_y)
cat_model = CatBoostClassifier(
                               iterations=3000,# change 25 to 3000 to get best performance 
                               learning_rate=0.03,
                               objective="Logloss",
                               eval_metric='AUC',
                              )
cat_model.fit(train_X,train_y,silent=True)
y_pred_cat = cat_model.predict(X_test)
from sklearn.ensemble import GradientBoostingClassifier
Model=GradientBoostingClassifier()
Model.fit(X_train,y_train)
y_pred=Model.predict(X_test)
submission_rfc = pd.DataFrame({
        "id": test["id"],
        "target": y_pred_rfc
    })
submission_rfc.to_csv('submission_rfc.csv', index=False)
submission_tree = pd.DataFrame({
        "id": test["id"],
        "target": y_pred_tree
    })
submission_tree.to_csv('submission_tree.csv', index=False)
submission_cat = pd.DataFrame({
        "id": test["id"],
        "target": y_pred_cat
    })
submission_cat.to_csv('submission_cat.csv', index=False)
submission_lgb = pd.DataFrame({
        "id": test["id"],
        "target": y_pred_lgb
    })
submission_lgb.to_csv('submission_lgb.csv', index=False)