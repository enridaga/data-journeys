
import numpy as np
import pandas as pd
import missingno as msno

train = pd.read_csv('../input/application_train.csv')
msno.matrix(train.sample(500), inline=True, sparkline=True, figsize=(20,10), sort=None, color=(0.25, 0.45, 0.6))
msno.matrix(train.iloc[0:100, 40:94], inline=True, sparkline=True, figsize=(20,10), sort='ascending', fontsize=12, labels=True, color=(0.25, 0.45, 0.6))
msno.dendrogram(train, inline=True, fontsize=12, figsize=(20,30))
train['incomplete'] = 1
train.loc[train.isnull().sum(axis=1) < 35, 'incomplete'] = 0

mean_c = np.mean(train.loc[train['incomplete'] == 0, 'TARGET'].values)
mean_i = np.mean(train.loc[train['incomplete'] == 1, 'TARGET'].values)
print('default ratio for more complete: {:.1%} \ndefault ratio for less complete: {:.1%}'.format(mean_c, mean_i))
from scipy.stats import chi2_contingency

props = pd.crosstab(train.incomplete, train.TARGET)
c = chi2_contingency(props, lambda_="log-likelihood")
print(props, "\n p-value= ", c[1])
