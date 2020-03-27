
## Import the required python utilities
from plotly.offline import init_notebook_mode, iplot
import matplotlib.pyplot as plt
import plotly.graph_objs as go
import pandas as pd
import numpy as np

## Import sklearn important modules
from sklearn.decomposition import PCA, SparsePCA, MiniBatchSparsePCA, KernelPCA, IncrementalPCA
from sklearn.random_projection import GaussianRandomProjection, SparseRandomProjection
from sklearn.decomposition import TruncatedSVD, FastICA, NMF, FactorAnalysis
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE
import lightgbm as lgb

init_notebook_mode(connected=True)
path = "../input/"
train = pd.read_csv(path+'train.csv')

target = train['target']
train = train.drop(["target", "ID"], axis=1)

print ("Rows: " + str(train.shape[0]) + ", Columns: " + str(train.shape[1]))
train.head()
standardized_train = StandardScaler().fit_transform(train.values)
feature_df = train.describe().T
feature_df = feature_df.reset_index().rename(columns = {'index' : 'columns'})
feature_df['distinct_vals'] = feature_df['columns'].apply(lambda x : len(train[x].value_counts()))
feature_df['column_var'] = feature_df['columns'].apply(lambda x : np.var(train[x]))
feature_df['column_std'] = feature_df['columns'].apply(lambda x : np.std(train[x]))
feature_df['column_mean'] = feature_df['columns'].apply(lambda x : np.mean(train[x]))
feature_df['target_corr'] = feature_df['columns'].apply(lambda x : np.corrcoef(target, train[x])[0][1])
feature_df.head()
len(feature_df[feature_df['column_var'].astype(float) == 0.0])
feature_df = feature_df.sort_values('column_var', ascending = True)
feature_df['column_var'] = (feature_df['column_var'] - feature_df['column_var'].min()) / (feature_df['column_var'].max() - feature_df['column_var'].min())
trace1 = go.Scatter(x=feature_df['columns'], y=feature_df['column_var'], opacity=0.75, marker=dict(color="red"))
layout = dict(height=400, title='Feature Variance', legend=dict(orientation="h"));
fig = go.Figure(data=[trace1], layout=layout);
iplot(fig);
trace1 = go.Histogram(x=feature_df[feature_df['column_var'] <= 0.01]['column_var'], opacity=0.45, marker=dict(color="red"))
layout = dict(height=400, title='Distribution of Variable Variance <= 0.01', legend=dict(orientation="h"));
fig = go.Figure(data=[trace1], layout=layout);
iplot(fig);

trace1 = go.Histogram(x=feature_df[feature_df['column_var'] > 0.01]['column_var'], opacity=0.45, marker=dict(color="red"))
layout = dict(height=400, title='Distribution of Variable Variance > 0.01', legend=dict(orientation="h"));
fig = go.Figure(data=[trace1], layout=layout);
iplot(fig);
trace1 = go.Histogram(x=feature_df['target_corr'], opacity=0.45, marker=dict(color="green"))
layout = dict(height=400, title='Distribution of correlation with target', legend=dict(orientation="h"));
fig = go.Figure(data=[trace1], layout=layout);
iplot(fig);
# Calculating Eigenvectors and eigenvalues of Cov matirx
mean_vec = np.mean(standardized_train, axis=0)
cov_matrix = np.cov(standardized_train.T)
eig_vals, eig_vecs = np.linalg.eig(cov_matrix)

# Create a list of (eigenvalue, eigenvector) tuples
eig_pairs = [ (np.abs(eig_vals[i]),eig_vecs[:,i]) for i in range(len(eig_vals))]

# Sort the eigenvalue, eigenvector pair from high to low
eig_pairs.sort(key = lambda x: x[0], reverse= True)

# Calculation of Explained Variance from the eigenvalues
tot = sum(eig_vals)

# Individual explained variance
var_exp = [(i/tot)*100 for i in sorted(eig_vals, reverse=True)] 
var_exp_real = [v.real for v in var_exp]

# Cumulative explained variance
cum_var_exp = np.cumsum(var_exp) 
cum_exp_real = [v.real for v in cum_var_exp]

## plot the variance and cumulative variance 
trace1 = go.Scatter(x=train.columns, y=var_exp_real, name="Individual Variance", opacity=0.75, marker=dict(color="red"))
trace2 = go.Scatter(x=train.columns, y=cum_exp_real, name="Cumulative Variance", opacity=0.75, marker=dict(color="blue"))
layout = dict(height=400, title='Variance Explained by Variables', legend=dict(orientation="h", x=0, y=1.2));
fig = go.Figure(data=[trace1, trace2], layout=layout);
iplot(fig);
def _get_number_components(model, threshold):
    component_variance = model.explained_variance_ratio_
    explained_variance = 0.0
    components = 0

    for var in component_variance:
        explained_variance += var
        components += 1
        if(explained_variance >= threshold):
            break
    return components

### Get the optimal number of components
pca = PCA()
train_pca = pca.fit_transform(standardized_train)
components = _get_number_components(pca, threshold=0.85)
components
def plot_3_components(x_trans, title):
    trace = go.Scatter3d(x=x_trans[:,0], y=x_trans[:,1], z = x_trans[:,2],
                          name = target, mode = 'markers', text = target, showlegend = False,
                          marker = dict(size = 8, color=x_trans[:,1], 
                          line = dict(width = 1, color = '#f7f4f4'), opacity = 0.5))
    layout = go.Layout(title = title, showlegend= True)
    fig = dict(data=[trace], layout=layout)
    iplot(fig)

def plot_2_components(x_trans, title):
    trace = go.Scatter(x=x_trans[:,0], y=x_trans[:,1], name=target, mode='markers',
        text = target, showlegend = False,
        marker = dict(size = 8, color=x_trans[:,1], line = dict(width = 1, color = '#fefefe'), opacity = 0.7))
    layout = go.Layout(title = title, hovermode= 'closest',
        xaxis= dict(title= 'First Component',
            ticklen = 5, zeroline= False, gridwidth= 2),
        yaxis=dict(title= 'Second Component',
            ticklen = 5, gridwidth = 2), showlegend= True)
    fig = dict(data=[trace], layout=layout)
    iplot(fig)
### Implement PCA 
obj_pca = model = PCA(n_components = components)
X_pca = obj_pca.fit_transform(standardized_train)

## Visualize the Components 
plot_3_components(X_pca, 'PCA - First Three Component')
plot_2_components(X_pca, 'PCA - First Two Components')
### Implement Truncated SVD 
obj_svd = TruncatedSVD(n_components = components)
X_svd = obj_svd.fit_transform(standardized_train)

## Visualize the Components 
plot_3_components(X_svd, 'Truncated SVD - First three components')
plot_2_components(X_svd, 'Truncated SVD - First two components')
### Implement ICA 
obj_ica = FastICA(n_components = 30)
X_ica = obj_ica.fit_transform(standardized_train)

## Visualize the Components 
plot_3_components(X_ica, 'ICA - First three components')
plot_2_components(X_ica, 'ICA - First two components')
### Implement Factor Analysis 
obj_fa = FactorAnalysis(n_components = 30)
X_fa = obj_fa.fit_transform(standardized_train)

## Visualize the Components 
plot_3_components(X_fa, 'Factor Analysis - First three components')
# plot_2_components(X, 'Factor Analysis - First two components')
### Implement NonNegative Matrix Factorization
obj = NMF(n_components = 2)
X_nmf = obj.fit_transform(train)

## Visualize the Components 
# plot_3_components(X, 'NNMF - First three components')
plot_2_components(X_nmf, 'NNMF - First two components')
### Implement Gaussian Random Projection
obj_grp = GaussianRandomProjection(n_components = 30, eps=0.1)
X_grp = obj_grp.fit_transform(standardized_train)

## Visualize the Components 
plot_3_components(X_grp, 'Gaussian Random Projection - First three components')
plot_2_components(X_grp, 'Gaussian Random Projection - First two components')
### Implement Sparse Random Projection
obj_srp = SparseRandomProjection(n_components = 30, eps=0.1)
X_srp = obj_srp.fit_transform(standardized_train)

## Visualize the Components 
plot_3_components(X_srp, 'Sparse Random Projection - First three components')
plot_2_components(X_srp, 'Sparse Random Projection - First two components')
tsne_model = TSNE(n_components=2, verbose=1, random_state=42, n_iter=500)
tsne_results = tsne_model.fit_transform(X_svd)

traceTSNE = go.Scatter(
    x = tsne_results[:,0],
    y = tsne_results[:,1],
    name = target,
     hoveron = target,
    mode = 'markers',
    text = target,
    showlegend = True,
    marker = dict(
        size = 8,
        color = '#c94ff2',
        showscale = False,
        line = dict(
            width = 2,
            color = 'rgb(255, 255, 255)'
        ),
        opacity = 0.8
    )
)
data = [traceTSNE]

layout = dict(title = 'TSNE (T-Distributed Stochastic Neighbour Embedding)',
              hovermode= 'closest',
              yaxis = dict(zeroline = False),
              xaxis = dict(zeroline = False),
              showlegend= False,

             )

fig = dict(data=data, layout=layout)
iplot(fig)
## add the decomposed features in the train dataset
def _add_decomposition(df, decomp, ncomp, flag):
    for i in range(1, ncomp+1):
        df[flag+"_"+str(i)] = decomp[:, i - 1]

_add_decomposition(train, X_pca, 30, 'pca')
_add_decomposition(train, X_svd, 30, 'svd')
_add_decomposition(train, X_ica, 30, 'ica')
_add_decomposition(train, X_fa, 30, 'fa')
_add_decomposition(train, X_grp, 30, 'grp')
_add_decomposition(train, X_srp, 30, 'srp')
## create the lists of decomposed and non decomposed features 
all_features = [x for x in train.columns if x not in ["ID", "target"]]
all_features = [x for x in all_features if "_" not in x]
decomposed_features = [x for x in train.columns if "_" in x]

## split the dataset into train test validation
target_log = np.log1p(target.values)
train_x, val_x, train_y, val_y = train_test_split(train, target_log, test_size=0.20, random_state=2018)
## create a baseline model with all features 
params = {'learning_rate': 0.01, 
          'max_depth': 16, 
          'boosting': 'gbdt', 
          'objective': 'regression', 
          'metric': 'rmse', 
          'is_training_metric': True, 
          'num_leaves': 144, 
          'feature_fraction': 0.9, 
          'bagging_fraction': 0.7, 
          'bagging_freq': 5, 
          'seed':2018}

## model without decomposed features 
train_X = lgb.Dataset(train_x[all_features], label=train_y)
val_X = lgb.Dataset(val_x[all_features], label=val_y)
model1 = lgb.train(params, train_X, 1000, val_X, verbose_eval=100, early_stopping_rounds=100)
## create a model with decomposed features 
train_X = lgb.Dataset(train_x[decomposed_features], label=train_y)
val_X = lgb.Dataset(val_x[decomposed_features], label=val_y)
model2 = lgb.train(params, train_X, 3000, val_X, verbose_eval=100, early_stopping_rounds=100)
## Find important features using Random Forests 
complete_features = all_features + decomposed_features
model = RandomForestRegressor(n_jobs=-1, random_state=2018)
model.fit(train[complete_features], target)
importances = model.feature_importances_

## get list of important features 
importances_df = pd.DataFrame({'importance': importances, 'feature': complete_features})
importances_df = importances_df.sort_values(by=['importance'], ascending=[False])
important_features = importances_df[:750]['feature'].values
## create a model with important features   
train_X = lgb.Dataset(train_x[important_features], label=train_y)  
val_X = lgb.Dataset(val_x[important_features], label=val_y)  
model3 = lgb.train(params, train_X, 3000, val_X, verbose_eval=100, early_stopping_rounds=100)  
test = pd.read_csv(path+"test.csv")
testid = test.ID.values
test = test.drop('ID', axis = 1)
## obtain the components from test data
standardized_test = StandardScaler().fit_transform(test[all_features].values)
tsX_pca = obj_pca.transform(standardized_test)
tsX_svd = obj_svd.transform(standardized_test)
tsX_ica = obj_ica.transform(standardized_test)
tsX_fa  = obj_fa.transform(standardized_test)
tsX_grp = obj_grp.transform(standardized_test)
tsX_srp = obj_srp.transform(standardized_test)
## add the components in test data
_add_decomposition(test, tsX_pca, 30, 'pca')
_add_decomposition(test, tsX_svd, 30, 'svd')
_add_decomposition(test, tsX_ica, 30, 'ica')
_add_decomposition(test, tsX_fa, 30, 'fa')
_add_decomposition(test, tsX_grp, 30, 'grp')
_add_decomposition(test, tsX_srp, 30, 'srp')
## create submission file 
pred = np.expm1(model3.predict(test[important_features], num_iteration=model3.best_iteration))
sub = pd.DataFrame()
sub['ID'] = testid
sub['target'] = pred
sub.to_csv('submission.csv', index=False)
sub.head()
