
fit_gaussians = False
use_plotly=True
# data manipulation
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# visualization
import seaborn as sns
import matplotlib.pyplot as plt
#%matplotlib inline
sns.set()

import plotly.offline as py
py.init_notebook_mode(connected=True)
import plotly.graph_objs as go

# sklearn models & tools
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.metrics import roc_auc_score
from sklearn.metrics import make_scorer
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import RobustScaler
from sklearn.decomposition import PCA

# ignore warnings
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)

import os
print(os.listdir("../input"))
train = pd.read_csv("../input/train.csv")
test = pd.read_csv("../input/test.csv")
submission = pd.read_csv("../input/sample_submission.csv")
train.shape
train.head(10)
train.target.dtype
org_vars = train.drop(["target", "ID_code"], axis=1).columns.values
len(org_vars)
train["Id"] = train.index.values
original_trainid = train.ID_code.values

train.drop("ID_code", axis=1, inplace=True)
train.isnull().sum().sum()
test.head(10)
test.isnull().sum().sum()
test.shape
test["Id"] = test.index.values
original_testid = test.ID_code.values

test.drop("ID_code", axis=1, inplace=True)
submission.head()
fig, ax = plt.subplots(1,2,figsize=(20,5))
sns.countplot(train.target.values, ax=ax[0], palette="husl")
sns.violinplot(x=train.target.values, y=train.index.values, ax=ax[1], palette="husl")
sns.stripplot(x=train.target.values, y=train.index.values,
              jitter=True, ax=ax[1], color="black", size=0.5, alpha=0.5)
ax[1].set_xlabel("Target")
ax[1].set_ylabel("Index");
ax[0].set_xlabel("Target")
ax[0].set_ylabel("Counts");
train.loc[train.target==1].shape[0] / train.loc[train.target==0].shape[0]
train_correlations = train.drop(["target"], axis=1).corr()
train_correlations = train_correlations.values.flatten()
train_correlations = train_correlations[train_correlations != 1]

test_correlations = test.corr()
test_correlations = test_correlations.values.flatten()
test_correlations = test_correlations[test_correlations != 1]

plt.figure(figsize=(20,5))
sns.distplot(train_correlations, color="Red", label="train")
sns.distplot(test_correlations, color="Green", label="test")
plt.xlabel("Correlation values found in train (except 1)")
plt.ylabel("Density")
plt.title("Are there correlations between features?"); 
plt.legend();
parameters = {'min_samples_leaf': [20, 25]}
forest = RandomForestClassifier(max_depth=15, n_estimators=15)
grid = GridSearchCV(forest, parameters, cv=3, n_jobs=-1, verbose=2, scoring=make_scorer(roc_auc_score))
grid.fit(train.drop("target", axis=1).values, train.target.values)
grid.best_score_
grid.best_params_
n_top = 5 
importances = grid.best_estimator_.feature_importances_
idx = np.argsort(importances)[::-1][0:n_top]
feature_names = train.drop("target", axis=1).columns.values

plt.figure(figsize=(20,5))
sns.barplot(x=feature_names[idx], y=importances[idx]);
plt.title("What are the top important features to start with?");
fig, ax = plt.subplots(n_top,2,figsize=(20,5*n_top))

for n in range(n_top):
    sns.distplot(train.loc[train.target==0, feature_names[idx][n]], ax=ax[n,0], color="Orange", norm_hist=True)
    sns.distplot(train.loc[train.target==1, feature_names[idx][n]], ax=ax[n,0], color="Red", norm_hist=True)
    sns.distplot(test.loc[:, feature_names[idx][n]], ax=ax[n,1], color="Mediumseagreen", norm_hist=True)
    ax[n,0].set_title("Train {}".format(feature_names[idx][n]))
    ax[n,1].set_title("Test {}".format(feature_names[idx][n]))
    ax[n,0].set_xlabel("")
    ax[n,1].set_xlabel("")
top = train.loc[:, feature_names[idx]]
top.describe()
top = top.join(train.target)
sns.pairplot(top, hue="target")
y_proba = grid.predict_proba(test.values)
y_proba_train = grid.predict_proba(train.drop("target", axis=1).values)
fig, ax = plt.subplots(2,1,figsize=(20,8))
sns.distplot(y_proba_train[train.target==1,1], norm_hist=True, color="mediumseagreen",
             ax=ax[0], label="1")
sns.distplot(y_proba_train[train.target==0,1], norm_hist=True, color="coral",
             ax=ax[0], label="0")
sns.distplot(y_proba[:,1], norm_hist=True,
             ax=ax[1], color="purple")
ax[1].set_xlabel("Predicted probability for test data");
ax[1].set_ylabel("Density");
ax[0].set_xlabel("Predicted probability for train data");
ax[0].set_ylabel("Density");
ax[0].legend();
submission["target"] = y_proba
submission.to_csv("submission_baseline_forest.csv", index=False)
original_features = train.drop(["target", "Id"], axis=1).columns.values
original_features
encoder = LabelEncoder()
for your_feature in top.drop("target", axis=1).columns.values:
    train[your_feature + "_qbinned"] = pd.qcut(
        train.loc[:, your_feature].values,
        q=10,
        labels=False
    )
    train[your_feature + "_qbinned"] = encoder.fit_transform(
        train[your_feature + "_qbinned"].values.reshape(-1, 1)
    )
    
    
    train[your_feature + "_rounded"] = np.round(train.loc[:, your_feature].values)
    train[your_feature + "_rounded_10"] = np.round(10*train.loc[:, your_feature].values)
    train[your_feature + "_rounded_100"] = np.round(100*train.loc[:, your_feature].values)
cv = StratifiedKFold(n_splits=3, random_state=0)
forest = RandomForestClassifier(max_depth=15, n_estimators=15, min_samples_leaf=20,
                                n_jobs=-1)

scores = []
X = train.drop("target", axis=1).values
y = train.target.values

for train_idx, test_idx in cv.split(X, y):
    x_train = X[train_idx]
    x_test = X[test_idx]
    y_train = y[train_idx]
    y_test = y[test_idx]
    
    forest.fit(x_train, y_train)
    y_proba = forest.predict_proba(x_test)
    y_pred = np.zeros(y_proba.shape[0])
    y_pred[y_proba[:,1] >= 0.166] = 1
    
    score = roc_auc_score(y_test, y_pred)
    print(score)
    scores.append(score)

print(np.round(np.mean(scores),4))
print(np.round(np.std(scores), 4))
importances = forest.feature_importances_
feature_names = train.drop("target", axis=1).columns.values
idx = np.argsort(importances)[::-1][0:30]

plt.figure(figsize=(20,5))
sns.barplot(x=feature_names[idx], y=importances[idx]);
plt.xticks(rotation=90);
col1 = "var_81"
col2 = "var_12"
N=70000
fig, ax = plt.subplots(1,1, figsize=(20,10))
sns.kdeplot(train[col1].values[0:N], train[col2].values[0:N])
ax.scatter(train[col1].values[0:N], train[col2].values[0:N],
           s=2, c=train.target.values[0:N], cmap="coolwarm", alpha=0.5)
ax.set_xlabel(col1)
ax.set_xlabel(col2);
combined = train.drop(["target", "Id"], axis=1).append(test.drop("Id", axis=1))
combined.shape
max_components = 10
start_components = 3
n_splits = 3
K = train.shape[0]

X = train.loc[:, original_features].values[0:K]
y = train.target.values[0:K]
seeds = np.random.RandomState(0).randint(0,100, size=(max_components-start_components))
seeds
scaler = RobustScaler()
X_scaled = scaler.fit_transform(X)
if fit_gaussians:
    components = np.arange(start_components, max_components, 1)
    kf = StratifiedKFold(random_state=0, n_splits=n_splits)
    
    scores = np.zeros(shape=(max_components-start_components, n_splits))

    for m in components:
        split=0
        print("Components " + str(m))
        for train_index, test_index in kf.split(X_scaled, y):
            print("Split " + str(split))
            x_train, x_test = X_scaled[train_index], X_scaled[test_index]
            gm = GaussianMixture(n_components=m, random_state=seeds[m-start_components])
            gm.fit(x_train)
            score = gm.score(x_test)
            scores[m-start_components,split] = score
            split +=1
    
    print(np.round(np.mean(scores, axis=1), 2))
    print(np.round(np.std(scores, axis=1), 2))
    best_idx = np.argmax(np.mean(scores, axis=1))
    best_component = components[best_idx]
    best_seed = seeds[best_idx]
    print("Best component found " + str(best_component))
    
else:
    best_seed = seeds[0]
    best_component = 3
X = train.loc[:, original_features].values

gm = GaussianMixture(n_components=best_component, random_state=best_seed)
X_scaled = scaler.transform(X)
gm.fit(X_scaled)
train["cluster"] = gm.predict(X_scaled)
train["logL"] = gm.score_samples(X_scaled)
test["cluster"] = gm.predict(test.loc[:, original_features].values)
test["logL"] = gm.score_samples(test.loc[:, original_features].values)
fig, ax = plt.subplots(1,2,figsize=(20,5))
sns.countplot(train.cluster, palette="Set2", ax=ax[0])
sns.distplot(train.logL, color="Dodgerblue", ax=ax[1]);
cluster_occupation = train.groupby("cluster").target.value_counts() / train.groupby("cluster").size() * 100
cluster_occupation = cluster_occupation.loc[:, 1]

target_occupation = train.groupby("target").cluster.value_counts() / train.groupby("target").size() * 100
target_occupation = target_occupation.loc[1, :]
target_occupation.index = target_occupation.index.droplevel("target")

fig, ax = plt.subplots(1,2,figsize=(20,5))
ax[0].set_title("How many % of the data per cluster has hot targets?")
sns.barplot(cluster_occupation.index, cluster_occupation.values, ax=ax[0], color="cornflowerblue")
ax[0].set_ylabel("% of cluster data")
ax[0].set_ylim([0,100])

ax[1].set_title("How many % of total hot targets are in one cluster?")
sns.barplot(target_occupation.index, target_occupation.values, ax=ax[1], color="tomato")
ax[1].set_ylabel("% of hot targets")
ax[1].set_ylim([0,100]);
plt.figure(figsize=(20,5))
for n in range(gm.means_.shape[0]):
    plt.plot(gm.means_[n,:], 'o')
plt.title("How do the gaussian means look like?")
plt.ylabel("Cluster mean value")
plt.xlabel("Feature")