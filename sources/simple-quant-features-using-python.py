
import pandas as pd
import numpy as np
import time
import lightgbm as lgb
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt
import seaborn as sns

plt.style.use('seaborn')
sns.set(font_scale=2)

import warnings 
warnings.filterwarnings('ignore')
import os
from kaggle.competitions import twosigmanews
# You can only call make_env() once, so don't lose it!
env = twosigmanews.make_env()
print('Done!')
(market_train_df, news_train_df) = env.get_training_data()
for i, df in market_train_df.groupby('assetCode'):
    break
df['MA_7MA'] = df['close'].rolling(window=7).mean()
df['MA_15MA'] = df['close'].rolling(window=15).mean()
df['MA_30MA'] = df['close'].rolling(window=30).mean()
df['MA_60MA'] = df['close'].rolling(window=60).mean()
ewma = pd.Series.ewm
df['close_30EMA'] = ewma(df["close"], span=30).mean()
plt.figure(figsize=(10, 10))
plt.plot(df['close'].values)
plt.plot(df['MA_7MA'].values)
plt.plot(df['MA_60MA'].values)
plt.plot(df['close_30EMA'].values)
plt.legend(['Close', 'MA_7MA', 'MA_60MA', 'close_30EMA'])
plt.show()
df['close_26EMA'] = ewma(df["close"], span=26).mean()
df['close_12EMA'] = ewma(df["close"], span=12).mean()

df['MACD'] = df['close_12EMA'] - df['close_26EMA']
no_of_std = 2

df['MA_7MA'] = df['close'].rolling(window=7).mean()
df['MA_7MA_std'] = df['close'].rolling(window=7).std() 
df['MA_7MA_BB_high'] = df['MA_7MA'] + no_of_std * df['MA_7MA_std']
df['MA_7MA_BB_low'] = df['MA_7MA'] - no_of_std * df['MA_7MA_std']

plt.figure(figsize=(10, 10))
plt.plot(df['close'].values)
plt.plot(df['MA_7MA'].values)
plt.plot(df['MA_7MA_BB_high'].values)
plt.plot(df['MA_7MA_BB_low'].values)
plt.legend(['Close', 'MA_7MA', 'MA_7MA_BB_high', 'MA_7MA_BB_low'])
plt.xlim(2200, 2500)
plt.ylim(30, 50)
plt.show()
no_of_std = 2

df['MA_15MA'] = df['close'].rolling(window=15).mean()
df['MA_15MA_std'] = df['close'].rolling(window=15).std() 
df['MA_15MA_BB_high'] = df['MA_15MA'] + no_of_std * df['MA_15MA_std']
df['MA_15MA_BB_low'] = df['MA_15MA'] - no_of_std * df['MA_15MA_std']

plt.figure(figsize=(10, 10))
plt.plot(df['close'].values)
plt.plot(df['MA_15MA'].values)
plt.plot(df['MA_15MA_BB_high'].values)
plt.plot(df['MA_15MA_BB_low'].values)
plt.legend(['Close', 'MA_15MA', 'MA_15MA_BB_high', 'MA_15MA_BB_low'])
plt.xlim(2000, 2500)
plt.show()
no_of_std = 2

df['MA_30MA'] = df['close'].rolling(window=30).mean()
df['MA_30MA_std'] = df['close'].rolling(window=30).std() 
df['MA_30MA_BB_high'] = df['MA_30MA'] + no_of_std * df['MA_30MA_std']
df['MA_30MA_BB_low'] = df['MA_30MA'] - no_of_std * df['MA_30MA_std']

plt.figure(figsize=(10, 10))
plt.plot(df['close'].values)
plt.plot(df['MA_30MA'].values)
plt.plot(df['MA_30MA_BB_high'].values)
plt.plot(df['MA_30MA_BB_low'].values)
plt.legend(['Close', 'MA_30MA', 'MA_30MA_BB_high', 'MA_30MA_BB_low'])
plt.xlim(2000, 2500)
plt.show()
def rsiFunc(prices, n=14):
    deltas = np.diff(prices)
    seed = deltas[:n+1]
    up = seed[seed>=0].sum()/n
    down = -seed[seed<0].sum()/n
    rs = up/down
    rsi = np.zeros_like(prices)
    rsi[:n] = 100. - 100./(1.+rs)

    for i in range(n, len(prices)):
        delta = deltas[i-1] # cause the diff is 1 shorter

        if delta>0:
            upval = delta
            downval = 0.
        else:
            upval = 0.
            downval = -delta

        up = (up*(n-1) + upval)/n
        down = (down*(n-1) + downval)/n

        rs = up/down
        rsi[i] = 100. - 100./(1.+rs)

    return rsi
rsi_6 = rsiFunc(df['close'].values, 6)
rsi_14 = rsiFunc(df['close'].values, 14)
rsi_20 = rsiFunc(df['close'].values, 20)
fig, ax = plt.subplots(2, 1, figsize=(10, 10))

ax[0].plot(df['MA_15MA'].values)
ax[0].plot(df['MA_15MA_BB_high'].values)
ax[0].plot(df['MA_15MA_BB_low'].values)
ax[1].plot(rsi_6)
ax[1].plot(rsi_14)
ax[1].plot(rsi_20)

ax[0].set_xlim([1500, 2000])
ax[0].legend(['MA_15MA', 'MA_15MA_BB_high', 'MA_15MA_BB_low'])
ax[1].set_xlim([1500, 2000])
ax[1].legend(['rsi_6', 'rsi_14', 'rsi_20'])
plt.show()
df['VMA_7MA'] = df['volume'].rolling(window=7).mean()
df['VMA_15MA'] = df['volume'].rolling(window=15).mean()
df['VMA_30MA'] = df['volume'].rolling(window=30).mean()
df['VMA_60MA'] = df['volume'].rolling(window=60).mean()
plt.figure(figsize=(10, 5))
# plt.plot(df['close'].values)
plt.plot(df['VMA_7MA'].values)
plt.plot(df['VMA_15MA'].values)
plt.plot(df['VMA_30MA'].values)
plt.plot(df['VMA_60MA'].values)
plt.legend(['Close', 'VMA_7MA', 'VMA_15MA', 'VMA_30MA', 'VMA_60MA'])
plt.xlim([1500, 2000])
plt.show()
fig, ax = plt.subplots(3, 1, figsize=(10, 20))

ax[0].plot(df['MA_15MA'].values)
ax[0].plot(df['MA_15MA_BB_high'].values)
ax[0].plot(df['MA_15MA_BB_low'].values)
ax[1].plot(rsi_6)
ax[1].plot(rsi_14)
ax[1].plot(rsi_20)
ax[2].plot(df['VMA_15MA'].values)
ax[2].plot(df['VMA_30MA'].values)


ax[0].set_xlim([1500, 2000])
ax[0].legend(['MA_15MA', 'MA_15MA_BB_high', 'MA_15MA_BB_low'])
ax[1].set_xlim([1500, 2000])
ax[1].legend(['rsi_6', 'rsi_14', 'rsi_20'])
ax[2].set_xlim([1500, 2000])
ax[2].legend(['VMA_15MA', 'VMA_30MA'])
plt.show()
