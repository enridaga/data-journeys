
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
#%matplotlib inline
plt.style.use('bmh')
plt.rcParams['figure.figsize'] = (10, 10)
title_config = {'fontsize': 20, 'y': 1.05}
train = pd.read_csv('../input/train.csv')
train.head()
X_train = train.iloc[:, 2:].values.astype('float64')
y_train = train['target'].values
pd.DataFrame(X_train[y_train == 0]).plot.kde(ind=100, legend=False)
plt.title('Likelihood KDE Plots for the Negative Class', **title_config);
pd.DataFrame(X_train[y_train == 1]).plot.kde(ind=100, legend=False)
plt.title('Likelihood KDE Plots for the Positive Class', **title_config);
from sklearn.preprocessing import StandardScaler

scaled = pd.DataFrame(StandardScaler().fit_transform(X_train))
scaled[y_train == 0].plot.kde(ind=100, legend=False)
plt.title('Likelihood KDE Plots for the Negative Class after Standardization', **title_config);
scaled[y_train == 1].plot.kde(ind=100, legend=False)
plt.title('Likelihood KDE Plots for the Positive Class after Standardization', **title_config);
from sklearn.preprocessing import QuantileTransformer

transformed = pd.DataFrame(QuantileTransformer(output_distribution='normal').fit_transform(X_train))
transformed[y_train == 0].plot.kde(ind=100, legend=False)
plt.title('Likelihood KDE Plots for the Negative Class after Quantile Transformation', **title_config);
transformed[y_train == 1].plot.kde(ind=100, legend=False)
plt.title('Likelihood KDE Plots for the Positive Class after Quantile Transformation', **title_config);
plt.imshow(transformed.corr())
plt.colorbar()
plt.title('Correlation Matrix Plot of the Features', **title_config);
plt.hist(transformed[y_train == 0].mean() - transformed[y_train == 1].mean())
plt.title('Histogram of Sample Mean Differences between Two Classes', **title_config);
plt.hist(transformed[y_train == 0].var() - transformed[y_train == 1].var())
plt.title('Histogram of Sample Variance Differences between Two Classes', **title_config);
select = (transformed[y_train == 0].var() - transformed[y_train == 1].var()).nsmallest(2).index
plt.scatter(transformed.loc[y_train == 0, select[0]], transformed.loc[y_train == 0, select[1]], alpha=0.5, label='Negative')
plt.scatter(transformed.loc[y_train == 1, select[0]], transformed.loc[y_train == 1, select[1]], alpha=0.5, label='Positive')
plt.xlabel(f'Transformed var_{select[0]}')
plt.ylabel(f'Transformed var_{select[1]}')
plt.title('Positive Class Looks More Concentrated Despite Higher Sample Variance', **title_config)
plt.legend();
transformed.loc[y_train == 0, select[0]].mean() - transformed.loc[y_train == 1, select[0]].mean()
transformed.loc[y_train == 0, select[1]].mean() - transformed.loc[y_train == 1, select[1]].mean()
plt.scatter(X_train[y_train == 0, select[0]], X_train[y_train == 0, select[1]], alpha=0.5, label='Negative')
plt.scatter(X_train[y_train == 1, select[0]], X_train[y_train == 1, select[1]], alpha=0.5, label='Positive')
plt.xlabel(f'var_{select[0]}')
plt.ylabel(f'var_{select[1]}')
plt.title('Bounds in Data', **title_config)
plt.legend();
size0 = (y_train == 0).sum()
size1 = y_train.size - size0
x0 = np.random.normal(transformed.loc[y_train == 0, select[0]].mean(),
                      transformed.loc[y_train == 0, select[0]].std(), size=size0)
y0 = np.random.normal(transformed.loc[y_train == 0, select[1]].mean(),
                      transformed.loc[y_train == 0, select[1]].std(), size=size0)
x1 = np.random.normal(transformed.loc[y_train == 1, select[0]].mean(),
                      transformed.loc[y_train == 1, select[0]].std(), size=size1)
y1 = np.random.normal(transformed.loc[y_train == 1, select[1]].mean(),
                      transformed.loc[y_train == 1, select[1]].std(), size=size1)
plt.scatter(x0, y0, alpha=0.5, label='Negative')
plt.scatter(x1, y1, alpha=0.5, label='Positive')
plt.xlabel(f'Simulated var_{select[0]}')
plt.ylabel(f'Simulated var_{select[1]}')
plt.title('Simulated Data for the Puzzle', **title_config)
plt.legend();
from sklearn.pipeline import make_pipeline
from sklearn.naive_bayes import GaussianNB

pipeline = make_pipeline(QuantileTransformer(output_distribution='normal'), GaussianNB())
pipeline.fit(X_train, y_train)
from sklearn.metrics import roc_curve, auc

fpr, tpr, thr = roc_curve(y_train, pipeline.predict_proba(X_train)[:,1])
plt.plot(fpr, tpr)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic Plot', **title_config)
auc(fpr, tpr)
from sklearn.model_selection import cross_val_score

cross_val_score(pipeline, X_train, y_train, scoring='roc_auc', cv=10).mean()
from sklearn.metrics import roc_auc_score

pipeline.fit(X_train, y_train)
model = pipeline.named_steps['gaussiannb']
size = 1000000
size0 = int(size * model.class_prior_[0])
size1 = size - size0
sample0 = np.concatenate([[np.random.normal(i, j, size=size0)]
                          for i, j in zip(model.theta_[0], np.sqrt(model.sigma_[0]))]).T
sample1 = np.concatenate([[np.random.normal(i, j, size=size1)]
                          for i, j in zip(model.theta_[1], np.sqrt(model.sigma_[1]))]).T
X_sample = np.concatenate([sample0, sample1])
y_sample = np.concatenate([np.zeros(size0), np.ones(size1)])
roc_auc_score(y_sample, model.predict_proba(X_sample)[:,1])
test = pd.read_csv('../input/test.csv')
test.head()
X_test = test.iloc[:, 1:].values.astype('float64')
submission = pd.read_csv('../input/sample_submission.csv')
submission['target'] = pipeline.predict_proba(X_test)[:,1]
submission.to_csv('submission.csv', index=False)