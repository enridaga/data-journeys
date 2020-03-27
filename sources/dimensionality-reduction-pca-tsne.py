
__author__ = 'Tilii: https://kaggle.com/tilii7' 



import warnings

with warnings.catch_warnings():

    warnings.filterwarnings('ignore')

    import numpy as np

    import pandas as pd

    import matplotlib.pyplot as plt

    #%matplotlib inline

    import matplotlib.cm as cm

    from sklearn.decomposition import PCA

    from sklearn.manifold import TSNE

    
print('\nLoading files ...')

train = pd.read_csv('../input/train.csv')

test = pd.read_csv('../input/test.csv')

X = train.drop(['id', 'target'], axis=1).values

y = train['target'].values.astype(np.int8)

target_names = np.unique(y)

print('\nThere are %d unique target valuess in this dataset:' % (len(target_names)), target_names)
n_comp = 20

# PCA

print('\nRunning PCA ...')

pca = PCA(n_components=n_comp, svd_solver='full', random_state=1001)

X_pca = pca.fit_transform(X)

print('Explained variance: %.4f' % pca.explained_variance_ratio_.sum())



print('Individual variance contributions:')

for j in range(n_comp):

    print(pca.explained_variance_ratio_[j])

colors = ['blue', 'red']

plt.figure(1, figsize=(10, 10))



for color, i, target_name in zip(colors, [0, 1], target_names):

    plt.scatter(X_pca[y == i, 0], X_pca[y == i, 1], color=color, s=1,

                alpha=.8, label=target_name, marker='.')

plt.legend(loc='best', shadow=False, scatterpoints=3)

plt.title(

        "Scatter plot of the training data projected on the 1st "

        "and 2nd principal components")

plt.xlabel("Principal axis 1 - Explains %.1f %% of the variance" % (

        pca.explained_variance_ratio_[0] * 100.0))

plt.ylabel("Principal axis 2 - Explains %.1f %% of the variance" % (

        pca.explained_variance_ratio_[1] * 100.0))



plt.savefig('pca-porto-01.png', dpi=150)

plt.show()

# tsne = TSNE(n_components=2, init='pca', random_state=1001, perplexity=30, method='barnes_hut', n_iter=1000, verbose=1)

# X_tsne = tsne.fit_transform(X) # this will either fail or take a while (most likely overnight)



# plt.figure(2, figsize=(10, 10))



# for color, i, target_name in zip(colors, [0, 1], target_names):

#     plt.scatter(X_tsne[y == i, 0], X_tsne[y == i, 1], color=color, s=1,

#                 alpha=.8, label=target_name, marker='.')

# plt.legend(loc='best', shadow=False, scatterpoints=3)

# plt.title('Scatter plot of t-SNE embedding')

# plt.xlabel('X')

# plt.ylabel('Y')



# plt.savefig('t-SNE-porto-01.png', dpi=150)

# plt.show()
from sklearn.preprocessing import MinMaxScaler



def scale_data(X, scaler=None):

    if not scaler:

        scaler = MinMaxScaler(feature_range=(-1, 1))

        scaler.fit(X)

    X = scaler.transform(X)

    return X, scaler



X = train.drop(['id', 'target'], axis=1)

test.drop(['id'], axis=1, inplace=True)

n_train = X.shape[0]

train_test = pd.concat((X, test)).reset_index(drop=True)

col_to_drop = X.columns[X.columns.str.endswith('_cat')]

col_to_dummify = X.columns[X.columns.str.endswith('_cat')].astype(str).tolist()



for col in col_to_dummify:

    dummy = pd.get_dummies(train_test[col].astype('category'))

    columns = dummy.columns.astype(str).tolist()

    columns = [col + '_' + w for w in columns]

    dummy.columns = columns

    train_test = pd.concat((train_test, dummy), axis=1)



train_test.drop(col_to_dummify, axis=1, inplace=True)

train_test_scaled, scaler = scale_data(train_test)

X = np.array(train_test_scaled[:n_train, :])

test = np.array(train_test_scaled[n_train:, :])

print('\n Shape of processed train data:', X.shape)

print(' Shape of processed test data:', test.shape)
print('\nRunning PCA again ...')

pca = PCA(n_components=n_comp, svd_solver='full', random_state=1001)

X_pca = pca.fit_transform(X)

print('Explained variance: %.4f' % pca.explained_variance_ratio_.sum())



print('Individual variance contributions:')

for j in range(n_comp):

    print(pca.explained_variance_ratio_[j])



plt.figure(1, figsize=(10, 10))



for color, i, target_name in zip(colors, [0, 1], target_names):

    plt.scatter(X_pca[y == i, 0], X_pca[y == i, 1], color=color, s=1,

                alpha=.8, label=target_name, marker='.')

plt.legend(loc='best', shadow=False, scatterpoints=3)

plt.title(

        "Scatter plot of the training data projected on the 1st "

        "and 2nd principal components")

plt.xlabel("Principal axis 1 - Explains %.1f %% of the variance" % (

        pca.explained_variance_ratio_[0] * 100.0))

plt.ylabel("Principal axis 2 - Explains %.1f %% of the variance" % (

        pca.explained_variance_ratio_[1] * 100.0))



plt.savefig('pca-porto-02.png', dpi=150)

plt.show()