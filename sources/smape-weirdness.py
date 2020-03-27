
# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from matplotlib import pyplot as plt
#%matplotlib inline
import seaborn as sns

def smape(y_true, y_pred):
    denominator = (np.abs(y_true) + np.abs(y_pred)) / 200.0
    diff = np.abs(y_true - y_pred) / denominator
    diff[denominator == 0] = 0.0
    return np.nanmean(diff)
y_true = np.array(3)
y_pred = np.ones(1)
x = np.linspace(0,10,1000)
res = [smape(y_true, i * y_pred) for i in x]
plt.plot(x, res)
y_true = np.array([1,9])
y_pred = np.ones(len(y_true))
x = np.linspace(0,10,1000)
res = [smape(y_true, i * y_pred) for i in x]
plt.plot(x, res)
print('SMAPE min:%0.2f' % np.min(res), ' at %0.2f' % x[np.argmin(res)])
print('SMAPE is :%0.2f' % smape(y_true, y_pred*np.nanmedian(y_true)), 
      ' at median %0.2f' % np.nanmedian(y_true))
np.random.seed(0)
y_true = np.random.uniform(1, 9, 100)
y_pred = np.ones(len(y_true))
x = np.linspace(0,10,1000)
res = [smape(y_true, i * y_pred) for i in x]
plt.plot(x, res)
print('SMAPE min:%0.2f' % np.min(res), ' at %0.2f' % x[np.argmin(res)])
print('SMAPE is :%0.2f' % smape(y_true, y_pred*np.nanmedian(y_true)), 
      ' at median %0.2f' % np.nanmedian(y_true))
np.random.seed(0)
y_true = np.random.lognormal(1, 1, 100)
y_pred = np.ones(len(y_true))
x = np.linspace(0,10,1000)
res = [smape(y_true, i * y_pred) for i in x]
plt.plot(x, res)
print('SMAPE min:%0.2f' % np.min(res), ' at %0.2f' % x[np.argmin(res)])
print('SMAPE is :%0.2f' % smape(y_true, y_pred*np.nanmedian(y_true)), 
      ' at median %0.2f' % np.nanmedian(y_true))
y_true = np.array([0])
y_pred = np.ones(len(y_true))
x = np.linspace(0,10,1000)
res = [smape(y_true, i * y_pred) for i in x]
plt.plot(x, res)
np.random.seed(0)
y_true = np.array([0,9])
y_pred = np.ones(len(y_true))
x = np.linspace(0,10,1000)
res = [smape(y_true, i * y_pred) for i in x]
plt.plot(x, res)
print('SMAPE min:%0.2f' % np.min(res), ' at %0.2f' % x[np.argmin(res)])
print('SMAPE is :%0.2f' % smape(y_true, y_pred*np.nanmedian(y_true)), 
      ' at median %0.2f' % np.nanmedian(y_true))
np.random.seed(0)
y_true = np.random.lognormal(1, 1, 100)
y_true[y_true < 3] = 0
print('There are %d zeros in the series' % np.sum(y_true == 0))
y_pred = np.ones(len(y_true))
x = np.linspace(0,10,1000)
res = [smape(y_true, i * y_pred) for i in x]
plt.plot(x, res)
print('SMAPE min:%0.2f' % np.min(res), ' at %0.2f' % x[np.argmin(res)])
print('SMAPE is :%0.2f' % smape(y_true, y_pred*np.nanmedian(y_true)), 
      ' at median %0.2f' % np.nanmedian(y_true))
