
import gc 
import warnings

import scipy
import numpy as np
import pandas as pd
from tqdm import tqdm
from copy import deepcopy

from scipy.stats import skew, kurtosis, gmean, ks_2samp

import hdbscan
from sklearn.cluster import KMeans, MeanShift

from sklearn.ensemble import ExtraTreesRegressor
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import ElasticNetCV
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import LeaveOneOut
from sklearn.decomposition import TruncatedSVD, FastICA, NMF, FactorAnalysis
from sklearn.decomposition import PCA, SparsePCA, MiniBatchSparsePCA, KernelPCA, IncrementalPCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.manifold import TSNE
from sklearn.random_projection import GaussianRandomProjection
from sklearn.random_projection import SparseRandomProjection
from sklearn.manifold import TSNE
from sklearn.preprocessing import scale

from mlxtend.feature_selection import SequentialFeatureSelector as SFS

from sklearn.base import BaseEstimator, TransformerMixin, RegressorMixin
from sklearn.externals.joblib import Parallel, delayed
from sklearn.base import clone, is_classifier
from sklearn.model_selection._split import check_cv

import seaborn as sns
import matplotlib.pyplot as plt
import scikitplot as skplt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.ticker import NullFormatter
### matplotlib inline
# How many rows to include in this kernel (to make iteration speedier)
SAMPLES = 1000

# Read train and test files
train_df = pd.read_csv('../input/train.csv').sample(SAMPLES)
test_df = pd.read_csv('../input/test.csv').sample(SAMPLES)

# Get the combined data
total_df = pd.concat([train_df.drop('target', axis=1), test_df], axis=0).drop('ID', axis=1)

# Get the target
y = np.log1p(train_df.target)

# Log-transform all column
total_df.loc[:, :] = np.log1p(total_df.values)

# Scale values. Since it's time-series we shouldn't scale column-wise
mean = np.mean(total_df.values)
std = np.std(total_df.values)
total_df.loc[:, :] = ((total_df.values - mean) / std)
    
# Train and test
train_idx = range(0, len(train_df))
test_idx = range(len(train_df), len(total_df))

# Longest list of sorted columns I could find, from:
# https://www.kaggle.com/johnfarrell/giba-s-property-extended-extended-result
ordered_cols = [
    'f190486d6', '58e2e02e6', 'eeb9cd3aa', '9fd594eec', '6eef030c1', '15ace8c9f', 
    'fb0f5dbfe', '58e056e12', '20aa07010', '024c577b9', 'd6bb78916', 'b43a7cfd5', 
    '58232a6fb', '1702b5bf0', '324921c7b', '62e59a501', '2ec5b290f', '241f0f867', 
    'fb49e4212', '66ace2992', 'f74e8f13d', '5c6487af1', '963a49cdc', '26fc93eb7', 
    '1931ccfdd', '703885424', '70feb1494', '491b9ee45', '23310aa6f', 'e176a204a', 
    '6619d81fc', '1db387535', 'fc99f9426', '91f701ba2', '0572565c2', '190db8488', 
    'adb64ff71', 'c47340d97', 'c5a231d81', '0ff32eb98'
][::-1]
aggregate_df = pd.DataFrame()

# Wrapper function
def diff2(x):
    return np.diff(x, n=2)

# Different pre-processing to be used before each primary function
preprocess_steps = [
    [],
    [np.diff], [diff2],
    [np.unique], [np.unique, np.diff], [np.unique, diff2]    
]

# Different statistics to calculate on each preprocessed step
stats = [len, np.min, np.max, np.median, np.std, skew, kurtosis] + 19 * [np.percentile]
stats_kwargs = [{} for i in range(7)] + [{'q': np.round(i, 2)} for i in np.linspace(0.05, 0.95, 19)]

# Only operate on non-nulls
for funs in preprocess_steps:
    
    # Apply pre-processing steps
    x = total_df[total_df != 0]
    for f in funs:
        x = f(x)
        
    # Go through our set of stat functions
    for stat, stat_kwargs in zip(stats, stats_kwargs):
        
        # Construct feature name
        name_components = [
            stat.__name__,
            "_".join([f.__name__ for f in funs]),
            "_".join(["{}={}".format(k, v) for k,v in stat_kwargs.items()])
        ]
        feature_name = "-".join([e for e in name_components if e])

        # Calc and save new feature in our dataframe
        aggregate_df[feature_name] = total_df.apply(lambda x: stat(x, **stat_kwargs), axis=1)
        
# Extra features
aggregate_df['number_of_different'] = total_df.nunique(axis=1)
aggregate_df['non_zero_count'] = total_df.astype(bool).sum(axis=1) 
aggregate_df['sum_zeros'] = (total_df == 0).astype(int).sum(axis=1)
aggregate_df['non_zero_fraction'] = total_df.shape[1] / total_df.astype(bool).sum(axis=1) 
aggregate_df['geometric_mean'] = total_df.apply(
    lambda x: np.exp(np.log(x[x>0]).mean()), axis=1
)
aggregate_df.reset_index(drop=True, inplace=True)
aggregate_df['geometric_mean'] = aggregate_df['geometric_mean'].replace(np.nan, 0)
aggregate_df['non_zero_fraction'] = aggregate_df['non_zero_fraction'].replace(np.inf, 0)

# Show user which aggregates were created
print(f">> Created {len(aggregate_df.columns)} features for; {aggregate_df.columns.tolist()}")
COMPONENTS = 10

# Convert to sparse matrix
sparse_matrix = scipy.sparse.csr_matrix(total_df.values)

# Data to be passed to t-SNE
tsvd = TruncatedSVD(n_components=1000).fit_transform(sparse_matrix)

# V1 List of decomposition methods
methods = [
    {'method': KernelPCA(n_components=2, kernel="rbf"), 'data': 'total'},
    {'method': FactorAnalysis(n_components=COMPONENTS), 'data': 'total'},
    {'method': TSNE(n_components=3, init='pca'), 'data': 'tsvd'},
    {'method': TruncatedSVD(n_components=COMPONENTS), 'data': 'sparse'},
    {'method': PCA(n_components=COMPONENTS), 'data': 'total'},
    {'method': FastICA(n_components=COMPONENTS), 'data': 'total'},
    {'method': GaussianRandomProjection(n_components=COMPONENTS, eps=0.1), 'data': 'total'},
    {'method': SparseRandomProjection(n_components=COMPONENTS, dense_output=True), 'data': 'total'}
]

# Run all the methods
embeddings = []
for run in methods:
    name = run['method'].__class__.__name__
    
    # Run method on appropriate data
    if run['data'] == 'sparse':
        embedding = run['method'].fit_transform(sparse_matrix)
    elif run['data'] == 'tsvd':
        embedding = run['method'].fit_transform(tsvd)
    else:
        embedding = run['method'].fit_transform(total_df)
        
    # Save in list of all embeddings
    embeddings.append(
        pd.DataFrame(embedding, columns=[f"{name}_{i}" for i in range(embedding.shape[1])])
    )
    print(f">> Ran {name}")
    gc.collect()    
    
# Put all components into one dataframe
components_df = pd.concat(embeddings, axis=1).reset_index(drop=True)
from keras.layers import *
from keras.optimizers import *
from keras.callbacks import *
from keras.models import Model, Sequential

enc_input = Input((total_df.shape[1], ))
enc_output = Dense(512, activation='relu')(enc_input)
enc_output = Dropout(0.5)(enc_output)
enc_output = Dense(5, activation='relu')(enc_output)

dec_input = Dense(512, activation='relu')(enc_output)
dec_output = Dropout(0.5)(dec_input)
dec_output = Dense(total_df.shape[1], activation='relu')(dec_output)

# This model maps an input to its reconstruction
vanilla_encoder = Model(enc_input, enc_output)
vanilla_autoencoder = Model(enc_input, dec_output)
vanilla_autoencoder.compile(optimizer=Adam(0.0001), loss='mean_squared_error')
vanilla_autoencoder.summary()

# Fit the autoencoder
vanilla_autoencoder.fit(
    total_df.values, total_df.values,
    epochs=6, # INCREASE THIS ONE
    batch_size=256,
    shuffle=True,
    callbacks=[
        ReduceLROnPlateau(monitor='loss', patience=5, verbose=1),
        EarlyStopping(monitor='loss', patience=10, mode='min', min_delta=1e-5)
    ]
)

# Put into dataframe
dense_ae_df = pd.DataFrame(
    vanilla_encoder.predict(total_df.values, batch_size=256), 
    columns=['dense_AE_{}'.format(i) for i in range(5)]
).reset_index(drop=True)
# Define regressors and class-levels to go through
classes = range(2, 7)
regressors = [
    ExtraTreesClassifier(
        n_estimators=100, max_features=0.5,
        max_depth=None, max_leaf_nodes=270,
        min_impurity_decrease=0.0001,
        n_jobs=-1, class_weight='balanced'
    ),
    LogisticRegression(
        class_weight='balanced'
    )
]

class ClassifierTransformer(BaseEstimator, TransformerMixin):
    """https://www.kaggle.com/sggpls/pipeline-kernel-xgb-fe-lb1-39"""
    
    def __init__(self, estimator=None, n_classes=2, cv=3):
        self.estimator = estimator
        self.n_classes = n_classes
        self.cv = cv
        
    @staticmethod
    def _get_labels(y, n_classes):
        y_labels = np.zeros(len(y))
        y_us = np.sort(np.unique(y))
        step = int(len(y_us) / n_classes)
        
        for i_class in range(n_classes):
            if i_class + 1 == n_classes:
                y_labels[y >= y_us[i_class * step]] = i_class
            else:
                y_labels[
                    np.logical_and(
                        y >= y_us[i_class * step],
                        y < y_us[(i_class + 1) * step]
                    )
                ] = i_class
        return y_labels
        
    def fit(self, X, y):
        y_labels = self._get_labels(y, self.n_classes)
        cv = check_cv(self.cv, y_labels, classifier=is_classifier(self.estimator))
        self.estimators_ = []
        
        for train, _ in cv.split(X, y_labels):
            self.estimators_.append(
                clone(self.estimator).fit(X[train], y_labels[train])
            )
        return self
    
    def transform(self, X, y=None):
        cv = check_cv(self.cv, y, classifier=is_classifier(self.estimator))
        
        X_prob = np.zeros((X.shape[0], self.n_classes))
        X_pred = np.zeros(X.shape[0])
        
        for estimator, (_, test) in zip(self.estimators_, cv.split(X)):
            X_prob[test] = estimator.predict_proba(X[test])
            X_pred[test] = estimator.predict(X[test])
        return np.hstack([X_prob, np.array([X_pred]).T])

# Put all features into one dataframe (i.e. aggregate, timeseries, components)
feature_df = pd.concat([components_df, aggregate_df, dense_ae_df], axis=1).fillna(0)    
    
# Collect predictions
clf_features = []
clf_columns = []
for n in tqdm(classes):
    for regr in regressors:
        clf = ClassifierTransformer(regr, n_classes=n, cv=5)
        clf.fit(tsvd[train_idx], y)
        clf_features.append(
            clf.transform(tsvd)
        )
        clf_columns += [f"{n}-{regr.__class__.__name__}_pred{i}" for i in range(n+1)]

# Save into dataframe
clf_features = np.concatenate(clf_features, axis=1)
classifier_df = pd.DataFrame(clf_features, columns=clf_columns)
# Final results
lda_df = []

# Don't show warnings about collinearity issues
with warnings.catch_warnings():    
    warnings.simplefilter("ignore") 
    
    # Try for different splits of classes
    n_classes = [2, 3, 4, 5, 6]
    for classes in tqdm(n_classes):

        # Create labels / target categories
        labels = ClassifierTransformer._get_labels(y, classes)

        # Run LDA
        lda = LinearDiscriminantAnalysis()
        lda.fit(total_df.values[train_idx], labels)
        lda_trafo = lda.transform(total_df)

        # The transformed data is in a numpy matrix. This may be inconvenient and we therefore
        # put transformed/projected data into new dataframe, where we specify column names and index
        lda_df.append(pd.DataFrame(
            lda_trafo,
            columns=["LDA{}_Comp{}".format(classes, i+1) for i in range(lda_trafo.shape[1])]
        ))

# Put into one dataframe
lda_df = pd.concat(lda_df, axis=1)

# Plot the LDAs with unique labels 2, 3, 4 in the component space
# _, axes = plt.subplots(1, 3, figsize=(20, 5))
fig = plt.figure(figsize=(20, 5))

# Plot 1D plot with label colors
ax = fig.add_subplot(131)
ax.scatter(lda_df.loc[train_idx, 'LDA2_Comp1'], np.zeros(len(train_idx)), c=labels)
ax.xaxis.set_major_formatter(NullFormatter())
ax.yaxis.set_major_formatter(NullFormatter())
ax.set_title("LDA with 2 labels")
ax.set_xlabel("Component 1")

# Plot 2D plot with label colors
ax = fig.add_subplot(132)
ax.scatter(lda_df.loc[train_idx, 'LDA3_Comp1'], lda_df.loc[train_idx, 'LDA3_Comp2'], c=labels)
ax.xaxis.set_major_formatter(NullFormatter())
ax.yaxis.set_major_formatter(NullFormatter())
ax.set_title("LDA with 3 labels")
ax.set_xlabel("Component 1")
ax.set_ylabel("Component 2")

# Plot 3D plot with label colors
ax = fig.add_subplot(133, projection='3d')
ax.scatter(lda_df.loc[train_idx, 'LDA4_Comp1'], lda_df.loc[train_idx, 'LDA4_Comp2'], lda_df.loc[train_idx, 'LDA4_Comp3'], c=labels)
ax.xaxis.set_major_formatter(NullFormatter())
ax.yaxis.set_major_formatter(NullFormatter())
ax.zaxis.set_major_formatter(NullFormatter())
ax.set_title("LDA with 4 labels")
ax.set_xlabel("Component 1")
ax.set_ylabel("Component 2")
ax.set_zlabel("Component 3")

# Show plot
plt.show()
from sklearn.model_selection import train_test_split
from keras.callbacks import *

# Split into train & test
X_train, X_val, y_train, y_val = train_test_split(
    total_df.iloc[train_idx][ordered_cols],
    y.values, 
    test_size=0.1,
    random_state=42,
    shuffle=True
)

# Loss function
def rmse(y_true, y_pred):
    return K.sqrt(K.mean(K.square(y_pred - y_true), axis=-1)) 

# Simple LSTM model
model = Sequential()
model.add(LSTM(64, return_sequences=True, input_shape=(len(ordered_cols), 1)))
model.add(LSTM(64, return_sequences=True))
model.add(LSTM(32))
model.add(Dense(1, activation='linear'))
model.compile(loss=rmse, optimizer='adam')

# Fit model to train data, validate on validation data
model.fit(
    np.expand_dims(X_train, 2), y_train,
    epochs=50,
    validation_data=(np.expand_dims(X_val, 2), y_val),
    callbacks=[
        ReduceLROnPlateau(patience=2, verbose=1),
        EarlyStopping(patience=3, min_delta=1e-5)
    ]
)

# Save regression result in dataframe
lstm_regr_df = pd.DataFrame(
    model.predict(np.expand_dims(total_df[ordered_cols].values, 2)),
    columns=['LSTM_regression']
)
lstm_regr_df.head()
n_clusters = [4, 8, 16, 32, 64]

skplt.cluster.plot_elbow_curve(
    KMeans(random_state=42),
    sparse_matrix[train_idx],
    cluster_ranges=n_clusters,
    figsize=(20, 5)
)
plt.show()
# Clustering algos
cluster_methods = {'KMeans-{}'.format(n): KMeans(n_clusters=n) for n in n_clusters}

# Target statistics to derive for each cluster
stats = [len, np.min, np.max, np.mean, np.median, np.std, skew, kurtosis]

# Put results into this dataframe
cluster_df = pd.DataFrame()

# Create plot
_, axes = plt.subplots(2, len(cluster_methods), figsize=(20, 7))

# Convenience function
def get_cluster_colors(clusterer, palette='hls'):
    """Create cluster colors based on labels and probability assignments"""
    n_clusters = len(np.unique(clusterer.labels_))
    color_palette = sns.color_palette(palette, n_clusters)
    cluster_colors = [color_palette[x] if x >= 0 else (0.5, 0.5, 0.5) for x in clusterer.labels_]
    if hasattr(clusterer, 'probabilities_'):
        cluster_colors = [sns.desaturate(x, p) for x, p in zip(cluster_colors, clusterer.probabilities_)]
    return cluster_colors
    
# Loop through all the clustering methods
for i, (name, method) in tqdm(enumerate(cluster_methods.items())):
    
    # Perform clustering on SVD data
    method.fit(tsvd[train_idx])
    
    # Plot on t-SNE for visual inspectino
    axes[0][i].scatter(
        components_df['TSNE_0'], components_df['TSNE_1'], 
        s=50, linewidth=0, 
        c=get_cluster_colors(method), 
        alpha=0.25
    )
    axes[0][i].set_title(name+" on t-SNE")
    
    # Plot barplots to see average target in clusters
    unique_clusters = np.unique(method.labels_)
    
    # Predict on entire datasest
    predicted_clusters = method.predict(tsvd)
    cluster_df[name] = predicted_clusters
    
    # Cluster targets
    cluster_targets = {cluster: y.iloc[train_idx][method.labels_ == cluster] for cluster in unique_clusters}
    
    # Go through each function we want to run on the clusters
    for fun in stats:
        cluster2fun = {}
        
        # Go through each cluster
        for cluster in unique_clusters:
            cluster2fun[cluster] = fun(cluster_targets[cluster])
        
        # Map result to final dataframe
        cluster_df[name+" "+fun.__name__] = cluster_df[name].map(cluster2fun)
        
    # Show target values for cluster algo
    means = [np.mean(cluster_targets[c]) for c in unique_clusters]
    stds = [np.std(cluster_targets[c]) for c in unique_clusters]
    axes[1][i].bar(unique_clusters, means, 0.5, yerr=stds)
    axes[1][i].set_xlabel('Cluster')
    axes[1][i].set_ylabel('Target Value')

# Show plot
plt.tight_layout()
plt.show()
from gplearn.genetic import SymbolicTransformer

function_set = ['add', 'sub', 'mul', 'div',
                'inv', 'log', 'abs', 'neg', 
                'sqrt', 'max', 'min']

gp = SymbolicTransformer(
    generations=10, population_size=50000,
    hall_of_fame=100, n_components=10,
    function_set=function_set,
    parsimony_coefficient=0.0005,
    max_samples=0.9, verbose=1,
    random_state=42, n_jobs=4
)

# Fit & save to dataframe
gp.fit(total_df.iloc[train_idx], y)
gp_features = gp.transform(total_df)
genetic_df = pd.DataFrame(gp_features, columns=[f'Genetic_{i}' for i in range(gp_features.shape[1])])
def series_to_supervised(data, n_in=1, n_out=1, dropnan=True):
    """
    Frame a time series as a supervised learning dataset.
    Taken from: https://machinelearningmastery.com/convert-time-series-supervised-learning-problem-python/
    Arguments:
        data: Sequence of observations as a list or NumPy array.
        n_in: Number of lag observations as input (X).
        n_out: Number of observations as output (y).
        dropnan: Boolean whether or not to drop rows with NaN values.
    Returns:
        Pandas DataFrame of series framed for supervised learning.
    """
    n_vars = 1 if type(data) is list else data.shape[1]
    df = pd.DataFrame(data)
    cols, names = list(), list()
    # input sequence (t-n, ... t-1)
    for i in range(n_in, 0, -1):
        cols.append(df.shift(i))
        names += [('var%d(t-%d)' % (j+1, i)) for j in range(n_vars)]
    # forecast sequence (t, t+1, ... t+n)
    for i in range(0, n_out):
        cols.append(df.shift(-i))
        if i == 0:
            names += [('var%d(t)' % (j+1)) for j in range(n_vars)]
        else:
            names += [('var%d(t+%d)' % (j+1, i)) for j in range(n_vars)]
    # put it all together
    agg = pd.concat(cols, axis=1)
    agg.columns = names
    # drop rows with NaN values
    if dropnan:
        agg.dropna(inplace=True)
    return agg

# Forecast storage
forecasts = {k: [] for k in ['ElasticNet_idx', 'RandomForest_lags3']}

# Run various forecasts
for i, row in tqdm(total_df[ordered_cols].iterrows(), total=len(total_df)):
    
    # Linear regression 2 steps into the future based on index
    regr = ElasticNetCV(cv=LeaveOneOut())
    regr.fit(
        np.arange(0, len(row)).reshape(-1, 1), 
        row.values.reshape(-1, 1).ravel()
    )
    forecasts['ElasticNet_idx'].append(regr.predict([[len(row)+2]])[0])
    
    # Random forest based on 3 lagged features
    sdf = series_to_supervised(row.values.tolist(), 3, 2)
    regr = ExtraTreesRegressor(n_estimators=10)
    regr.fit(sdf[['var1(t-3)', 'var1(t-2)', 'var1(t-1)']], sdf['var1(t+1)'])
    forecasts['RandomForest_lags3'].append(
        regr.predict([row.values[-3:]])[0]
    )
    
# put into dataframe
forecasts_df = pd.DataFrame(forecasts)
latent_dim = 5
timesteps = len(ordered_cols)

# Define model (based on https://blog.keras.io/building-autoencoders-in-keras.html)
inputs = Input(shape=(timesteps, 1))
encoded = LSTM(5)(inputs)
decoded = RepeatVector(timesteps)(encoded)
decoded = LSTM(1, return_sequences=True)(decoded)
sequence_autoencoder = Model(inputs, decoded)
sequence_encoder = Model(inputs, encoded)

# Compile with MSE
sequence_autoencoder.compile(
    optimizer=Adam(0.001),
    loss='mean_squared_error'
)

# Fitting only on sorted columns
lstm_X = np.expand_dims(total_df[ordered_cols].values, 2)

# Fit to all data (test + train)
sequence_autoencoder.fit(
    lstm_X, lstm_X,
    epochs=100,
    batch_size=256,
    shuffle=True,
    verbose=1,
    callbacks=[
        ReduceLROnPlateau(monitor='loss', patience=5, verbose=1),
        EarlyStopping(monitor='loss', patience=10, mode='min', min_delta=1e-5)
    ]
)

# Put encoded result into dataframe
lstm_ae_df = pd.DataFrame(
    sequence_encoder.predict(lstm_X, batch_size=16), 
    columns=['lstm_AE_{}'.format(i) for i in range(5)]
).reset_index(drop=True)
# Put all features into one dataframe (i.e. aggregate, timeseries, components)
feature_df = pd.concat([
    components_df,
    aggregate_df,
    dense_ae_df,
    classifier_df,
    lda_df,
    cluster_df,
    genetic_df,
    forecasts_df,
    lstm_ae_df,
    lstm_regr_df
], axis=1).fillna(0)

# Go through each feature
results = []
for col in tqdm(feature_df.columns):
    
    # Get the column values in training
    X = feature_df.iloc[train_idx][col].values.reshape(-1, 1)
    
    # Get CV scores
    scores = cross_val_score(
        ExtraTreesRegressor(n_estimators=30),
        X, y,
        scoring='neg_mean_squared_error',
        cv=10
    )
    scores = np.sqrt(-scores)
    for score in scores:
        results.append({'feature': col, 'score': score, 'mean_score': np.mean(scores)})
        
# Put results in dataframe
results = pd.DataFrame(results).sort_values('mean_score')

# Function for plotting feature scores. Will be used again later
def plot_feature_scores(results, max_cols=100, feature_list=None, title=""):
    
    # Save copy
    results_subset = results.copy()
    
    # Only get subset of features for plotting
    if len(np.unique(results_subset.feature)) > max_cols:
        results_subset = results_subset[results_subset.mean_score < np.sort(np.unique(results_subset.mean_score))[100]]
        
    # Only select certain features
    if feature_list is not None:
        results_subset = results_subset[results_subset.feature.isin(feature_list)]

    # Create plot of scores
    _, axes = plt.subplots(1, 1, figsize=(20, 5))
    sns.barplot(x='feature', y='score', data=results_subset, ax=axes)
    plt.xticks(rotation=90)
    plt.title(title)
    plt.show()

# Plot the feature scores
plot_feature_scores(results)
# Train / test bool target
y_clf = np.zeros(len(total_df))
y_clf[test_idx] = 1

def get_safe_KS(df, thr=0.1):
    """Use KS to determine columns with KS statistic above threshold between train & test"""

    # Book-keeping
    drop_features = []

    # Go through all columns
    with tqdm() as pbar:
        for col in feature_df.columns:

            # Columns left
            cols_left = [c for c in feature_df.columns if c not in drop_features]
            pbar.update(1)

            # Look at distribution in feature
            statistic, pvalue = ks_2samp(
                feature_df.loc[train_idx, col].values, 
                feature_df.loc[test_idx, col].values
            )
            if pvalue < 0.05 and statistic > 0.1:
                pbar.set_description(f"Dropping: {col}. KS: {statistic}. p-value: {pvalue}. {len(cols_left)} features left.")
                drop_features.append(col)
            
    # Return columns to keep
    return cols_left

def get_safe_adversarial(df, thr=0.7):
    """Recursively eliminate features from adversarial validation with highest feature importance,
    Continues untill the accuracy of the oob-score for a random forest decreases below given threshold
    """
    
    # Book-keeping
    current_score = np.inf
    drop_features = []
    
    # Start eliminating features
    with tqdm() as pbar:
        while current_score > thr:

            # Columns left
            cols_left = [c for c in df.columns if c not in drop_features]

            # Fit random forest model
            regr = ExtraTreesClassifier(n_estimators=100, oob_score=True, bootstrap=True)
            regr.fit(df[cols_left], y_clf)
            current_score = regr.oob_score_
            pbar.update(1)

            # Get most important feature for classification
            best_feature = cols_left[np.argmax(regr.feature_importances_)]

            # Add to drop and inform user
            if current_score > thr:
                pbar.set_description(f"Acc: {regr.oob_score_}. Dropping: {best_feature}.")
                drop_features.append(best_feature)
            else:
                pbar.set_description(f"Adversarial Elimination reached threshold acc of {thr}. {len(cols_left)} features left.")
    return cols_left

# Create plot for KS elimination
cols_left = get_safe_KS(feature_df, 0.1)
plot_feature_scores(results, feature_list=cols_left, title="After Kolmogorov–Smirnov feature elimination")

# Create plot for adversarial elimination
cols_left = get_safe_adversarial(feature_df, 0.7)
plot_feature_scores(results, feature_list=cols_left, title="After adversarial feature elimination")
# Create forward feature selector
selector = SFS(
    ExtraTreesRegressor(n_estimators=30),
    k_features=(1,15),
    forward=True,
    scoring='neg_mean_squared_error',
    cv=10,
    n_jobs=-1, 
    verbose=0
)

# Fit model and get best features
selector.fit(feature_df[cols_left].values[train_idx], y)

# Plot results
results = []
current_features = []
for step, info in selector.subsets_.items():

    # What was added / removed on this iteration
    added_feature = [i for i in info['feature_idx'] if i not in current_features][0]
    removed_feature = [i for i in current_features if i not in info['feature_idx']]    
    
    # Update book-keeping
    current_features.append(added_feature)
    
    # Save for plotting
    label = f"Added {feature_df.columns[added_feature]}"
    if removed_feature:
        label += f". Removed {feature_df.columns[removed_feature[0]]}"
        current_features.remove(removed_feature[0])
    scores = np.sqrt(-info['cv_scores'])
    for score in scores:
        results.append({'label': label, 'score': score, 'mean_score': np.mean(scores)})
        
# Put results in dataframe
results = pd.DataFrame(results)

# Create plot of scores
_, axes = plt.subplots(1, 1, figsize=(20, 5))
sns.barplot(x='label', y='score', data=results, ax=axes)
axes.set_ylim((results.score.min(), results.score.max()))
plt.xticks(rotation=90)
plt.show()
