
import numpy as np
import pandas as pd
import os

import matplotlib.pyplot as plt
### matplotlib inline
from tqdm import tqdm_notebook
from sklearn.preprocessing import StandardScaler
from sklearn.svm import NuSVR, SVR
from sklearn.metrics import mean_absolute_error
pd.options.display.precision = 15
from sklearn.svm import NuSVR, SVR
import lightgbm as lgb
import xgboost as xgb
import time
import datetime


from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold, KFold, RepeatedKFold
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.linear_model import Ridge, RidgeCV
import gc
from catboost import CatBoostRegressor
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")
import gc
train = pd.read_csv('../input/train.csv', dtype={'acoustic_data': np.int16, 'time_to_failure': np.float64})
#min_1 = train.acoustic_data.mean() - 3 * train.acoustic_data.std()
#max_1 = train.acoustic_data.mean() + 3 * train.acoustic_data.std() 
#train["sharp_rise1"] = np.where((train.acoustic_data >= min_1) & (train.acoustic_data <= max_1), 0, 100)
#del min_1,max_1
#gc.collect()
#min_2 = train.acoustic_data.mean() - 2 * train.acoustic_data.std()
#max_2 = train.acoustic_data.mean() + 2 * train.acoustic_data.std() 
#train["sharp_rise2"] = np.where((train.acoustic_data >= min_2) & (train.acoustic_data <= max_2), 0, 50)
#del min_2,max_2
#gc.collect()
#differences = np.diff(train.time_to_failure)
#train = train.drop(train.index[len(train)-1])
#train["differences"] = differences

#train.differences.unique()
#train["differences"] = np.around(train["differences"],10)
#train.differences.unique()
#train = train.convert_objects(convert_numeric=True)
#train["change"]=train.differences * 1e9 + 1

#train.change.unique()
#train["change"] = np.around(train["change"],3)
#train["change"]=np.floor(train["change"])

#train.head()
#columns = ['sharp_rise1', 'sharp_rise2',"differences","change"]
#train.drop(columns, inplace=True, axis=1)
#del differences
#gc.collect()
from scipy.signal import hilbert
from scipy.signal import hann
from scipy.signal import convolve

def classic_sta_lta(x, length_sta, length_lta):
    
    sta = np.cumsum(x ** 2)

    # Convert to float
    sta = np.require(sta, dtype=np.float)

    # Copy for LTA
    lta = sta.copy()

    # Compute the STA and the LTA
    sta[length_sta:] = sta[length_sta:] - sta[:-length_sta]
    sta /= length_sta
    lta[length_lta:] = lta[length_lta:] - lta[:-length_lta]
    lta /= length_lta

    # Pad zeros
    sta[:length_lta - 1] = 0

    # Avoid division by zero by setting zero values to tiny float
    dtiny = np.finfo(0.0).tiny
    idx = lta < dtiny
    lta[idx] = dtiny

    return sta / lta
# Create a training file with simple derived features
rows = 150_000
segments = int(np.floor(train.shape[0] / rows))

X_tr = pd.DataFrame(index=range(segments), dtype=np.float64,
                       columns=['mean', 'std', 'max', 'min',
                               'mean_change_abs', 'mean_change_rate', 'abs_max', 'abs_min',
                               'std_first_50000', 'std_last_50000', 'std_first_10000', 'std_last_10000',
                               'avg_first_50000', 'avg_last_50000', 'avg_first_10000', 'avg_last_10000',
                               'min_first_50000', 'min_last_50000', 'min_first_10000', 'min_last_10000',
                               'max_first_50000', 'max_last_50000', 'max_first_10000', 'max_last_10000',
                               'max_to_min', 'max_to_min_diff', 'count_big', 'sum',
                               'mean_change_rate_first_50000', 'mean_change_rate_last_50000', 'mean_change_rate_first_10000', 'mean_change_rate_last_10000','q70','q75','q60','q65','q85',"q90",'q80','q95','q99','Hilbert_mean','Hann_window_mean','classic_sta_lta1_mean','classic_sta_lta2_mean','classic_sta_lta3_mean','classic_sta_lta4_mean','Moving_average_700_mean','Moving_average_1500_mean','Moving_average_3000_mean','Moving_average_6000_mean','exp_Moving_average_300_mean','exp_Moving_average_3000_mean','exp_Moving_average_30000_mean','MA_700MA_std_mean','MA_700MA_BB_high_mean','MA_700MA_BB_low_mean','MA_400MA_std_mean','MA_400MA_BB_high_mean','MA_400MA_BB_low_mean','MA_1000MA_std_mean'])
y_tr = pd.DataFrame(index=range(segments), dtype=np.float64,
                       columns=['time_to_failure'])

total_mean = train['acoustic_data'].mean()
total_std = train['acoustic_data'].std()
total_max = train['acoustic_data'].max()
total_min = train['acoustic_data'].min()
total_sum = train['acoustic_data'].sum()
total_abs_max = np.abs(train['acoustic_data']).sum()

for segment in tqdm_notebook(range(segments)):
    seg = train.iloc[segment*rows:segment*rows+rows]
    x = pd.Series(seg['acoustic_data'].values)
    y = seg['time_to_failure'].values[-1]
    
    y_tr.loc[segment, 'time_to_failure'] = y
    X_tr.loc[segment, 'mean'] = x.mean()
    X_tr.loc[segment, 'std'] = x.std()
    X_tr.loc[segment, 'max'] = x.max()
    X_tr.loc[segment, 'min'] = x.min()
    
    
    X_tr.loc[segment, 'mean_change_abs'] = np.mean(np.diff(x))
    X_tr.loc[segment, 'mean_change_rate'] = np.mean(np.nonzero((np.diff(x) / x[:-1]))[0])
    X_tr.loc[segment, 'abs_max'] = np.abs(x).max()
    X_tr.loc[segment, 'abs_min'] = np.abs(x).min()
    
    X_tr.loc[segment, 'std_first_50000'] = x[:50000].std()
    X_tr.loc[segment, 'std_last_50000'] = x[-50000:].std()
    X_tr.loc[segment, 'std_first_10000'] = x[:10000].std()
    X_tr.loc[segment, 'std_last_10000'] = x[-10000:].std()
    
    X_tr.loc[segment, 'avg_first_50000'] = x[:50000].mean()
    X_tr.loc[segment, 'avg_last_50000'] = x[-50000:].mean()
    X_tr.loc[segment, 'avg_first_10000'] = x[:10000].mean()
    X_tr.loc[segment, 'avg_last_10000'] = x[-10000:].mean()
    
    X_tr.loc[segment, 'min_first_50000'] = x[:50000].min()
    X_tr.loc[segment, 'min_last_50000'] = x[-50000:].min()
    X_tr.loc[segment, 'min_first_10000'] = x[:10000].min()
    X_tr.loc[segment, 'min_last_10000'] = x[-10000:].min()
    
    X_tr.loc[segment, 'max_first_50000'] = x[:50000].max()
    X_tr.loc[segment, 'max_last_50000'] = x[-50000:].max()
    X_tr.loc[segment, 'max_first_10000'] = x[:10000].max()
    X_tr.loc[segment, 'max_last_10000'] = x[-10000:].max()
    
    X_tr.loc[segment, 'max_to_min'] = x.max() / np.abs(x.min())
    X_tr.loc[segment, 'max_to_min_diff'] = x.max() - np.abs(x.min())
    X_tr.loc[segment, 'count_big'] = len(x[np.abs(x) > 500])
    X_tr.loc[segment, 'sum'] = x.sum()
    
    X_tr.loc[segment, 'mean_change_rate_first_50000'] = np.mean(np.nonzero((np.diff(x[:50000]) / x[:50000][:-1]))[0])
    X_tr.loc[segment, 'mean_change_rate_last_50000'] = np.mean(np.nonzero((np.diff(x[-50000:]) / x[-50000:][:-1]))[0])
    X_tr.loc[segment, 'mean_change_rate_first_10000'] = np.mean(np.nonzero((np.diff(x[:10000]) / x[:10000][:-1]))[0])
    X_tr.loc[segment, 'mean_change_rate_last_10000'] = np.mean(np.nonzero((np.diff(x[-10000:]) / x[-10000:][:-1]))[0])
    
     #new:'q70','q75','q60','q65'
    X_tr.loc[segment, 'q70'] = np.quantile(x, 0.70)    
    X_tr.loc[segment, 'q75'] = np.quantile(x, 0.75)   
    X_tr.loc[segment, 'q60'] = np.quantile(x, 0.60)    
    X_tr.loc[segment, 'q65'] = np.quantile(x, 0.65)    
    X_tr.loc[segment, 'q85'] = np.quantile(x, 0.85)
    X_tr.loc[segment, 'q90'] = np.quantile(x, 0.90)
    X_tr.loc[segment, 'q80'] = np.quantile(x, 0.80)
    X_tr.loc[segment, 'q95'] = np.quantile(x, 0.95)
    X_tr.loc[segment, 'q99'] = np.quantile(x, 0.99)


    X_tr.loc[segment, 'Hilbert_mean'] = np.abs(hilbert(x)).mean()
    X_tr.loc[segment, 'Hann_window_mean'] = (convolve(x, hann(150), mode='same') / sum(hann(150))).mean()
    X_tr.loc[segment, 'classic_sta_lta1_mean'] = classic_sta_lta(x, 500, 10000).mean()
    X_tr.loc[segment, 'classic_sta_lta2_mean'] = classic_sta_lta(x, 5000, 100000).mean()
    X_tr.loc[segment, 'classic_sta_lta3_mean'] = classic_sta_lta(x, 3333, 6666).mean()
    X_tr.loc[segment, 'classic_sta_lta4_mean'] = classic_sta_lta(x, 10000, 25000).mean()
    X_tr.loc[segment, 'Moving_average_700_mean'] = x.rolling(window=700).mean().mean(skipna=True)
    X_tr.loc[segment, 'Moving_average_1500_mean'] = x.rolling(window=1500).mean().mean(skipna=True)
    X_tr.loc[segment, 'Moving_average_3000_mean'] = x.rolling(window=3000).mean().mean(skipna=True)
    X_tr.loc[segment, 'Moving_average_6000_mean'] = x.rolling(window=6000).mean().mean(skipna=True)
    ewma = pd.Series.ewm
    X_tr.loc[segment, 'exp_Moving_average_300_mean'] = (ewma(x, span=300).mean()).mean(skipna=True)
    X_tr.loc[segment, 'exp_Moving_average_3000_mean'] = ewma(x, span=3000).mean().mean(skipna=True)
    X_tr.loc[segment, 'exp_Moving_average_30000_mean'] = ewma(x, span=6000).mean().mean(skipna=True)
    no_of_std = 2
    X_tr.loc[segment, 'MA_700MA_std_mean'] = x.rolling(window=700).std().mean()
    X_tr.loc[segment,'MA_700MA_BB_high_mean'] = (X_tr.loc[segment, 'Moving_average_700_mean'] + no_of_std * X_tr.loc[segment, 'MA_700MA_std_mean']).mean()
    X_tr.loc[segment,'MA_700MA_BB_low_mean'] = (X_tr.loc[segment, 'Moving_average_700_mean'] - no_of_std * X_tr.loc[segment, 'MA_700MA_std_mean']).mean()
    X_tr.loc[segment, 'MA_400MA_std_mean'] = x.rolling(window=400).std().mean()
    X_tr.loc[segment,'MA_400MA_BB_high_mean'] = (X_tr.loc[segment, 'Moving_average_700_mean'] + no_of_std * X_tr.loc[segment, 'MA_400MA_std_mean']).mean()
    X_tr.loc[segment,'MA_400MA_BB_low_mean'] = (X_tr.loc[segment, 'Moving_average_700_mean'] - no_of_std * X_tr.loc[segment, 'MA_400MA_std_mean']).mean()
    X_tr.loc[segment, 'MA_1000MA_std_mean'] = x.rolling(window=1000).std().mean()
    
fsegments = 10000

X_tr1 = pd.DataFrame(index=range(segments), dtype=np.float64,
                       columns=['mean', 'std', 'max', 'min',
                               'mean_change_abs', 'mean_change_rate', 'abs_max', 'abs_min',
                               'std_first_50000', 'std_last_50000', 'std_first_10000', 'std_last_10000',
                               'avg_first_50000', 'avg_last_50000', 'avg_first_10000', 'avg_last_10000',
                               'min_first_50000', 'min_last_50000', 'min_first_10000', 'min_last_10000',
                               'max_first_50000', 'max_last_50000', 'max_first_10000', 'max_last_10000',
                               'max_to_min', 'max_to_min_diff', 'count_big', 'sum',
                               'mean_change_rate_first_50000', 'mean_change_rate_last_50000', 'mean_change_rate_first_10000', 'mean_change_rate_last_10000','q70','q75','q60','q65','q85',"q90",'q80',
                               'q95','q99','Hilbert_mean','Hann_window_mean','classic_sta_lta1_mean','classic_sta_lta2_mean','classic_sta_lta3_mean','classic_sta_lta4_mean','Moving_average_700_mean','Moving_average_1500_mean','Moving_average_3000_mean','Moving_average_6000_mean','exp_Moving_average_300_mean','exp_Moving_average_3000_mean','exp_Moving_average_30000_mean','MA_700MA_std_mean','MA_700MA_BB_high_mean','MA_700MA_BB_low_mean','MA_400MA_std_mean','MA_400MA_BB_high_mean','MA_400MA_BB_low_mean','MA_1000MA_std_mean'])
y_tr1 = pd.DataFrame(index=range(segments), dtype=np.float64,
                       columns=['time_to_failure'])

total_mean = train['acoustic_data'].mean()
total_std = train['acoustic_data'].std()
total_max = train['acoustic_data'].max()
total_min = train['acoustic_data'].min()
total_sum = train['acoustic_data'].sum()
total_abs_max = np.abs(train['acoustic_data']).sum()

for segment in tqdm_notebook(range(segments)):
    ind = np.random.randint(0, train.shape[0]-150001)
    seg = train.iloc[ind:ind+rows]
    x = pd.Series(seg['acoustic_data'].values)
    y = seg['time_to_failure'].values[-1]

   
    y_tr1.loc[segment, 'time_to_failure'] = y

    X_tr1.loc[segment, 'mean'] = x.mean()
    X_tr1.loc[segment, 'std'] = x.std()
    X_tr1.loc[segment, 'max'] = x.max()
    X_tr1.loc[segment, 'min'] = x.min()
    
    
    X_tr1.loc[segment, 'mean_change_abs'] = np.mean(np.diff(x))
    X_tr1.loc[segment, 'mean_change_rate'] = np.mean(np.nonzero((np.diff(x) / x[:-1]))[0])
    X_tr1.loc[segment, 'abs_max'] = np.abs(x).max()
    X_tr1.loc[segment, 'abs_min'] = np.abs(x).min()
    
    X_tr1.loc[segment, 'std_first_50000'] = x[:50000].std()
    X_tr1.loc[segment, 'std_last_50000'] = x[-50000:].std()
    X_tr1.loc[segment, 'std_first_10000'] = x[:10000].std()
    X_tr1.loc[segment, 'std_last_10000'] = x[-10000:].std()
    
    X_tr1.loc[segment, 'avg_first_50000'] = x[:50000].mean()
    X_tr1.loc[segment, 'avg_last_50000'] = x[-50000:].mean()
    X_tr1.loc[segment, 'avg_first_10000'] = x[:10000].mean()
    X_tr1.loc[segment, 'avg_last_10000'] = x[-10000:].mean()
    
    X_tr1.loc[segment, 'min_first_50000'] = x[:50000].min()
    X_tr1.loc[segment, 'min_last_50000'] = x[-50000:].min()
    X_tr1.loc[segment, 'min_first_10000'] = x[:10000].min()
    X_tr1.loc[segment, 'min_last_10000'] = x[-10000:].min()
    
    X_tr1.loc[segment, 'max_first_50000'] = x[:50000].max()
    X_tr1.loc[segment, 'max_last_50000'] = x[-50000:].max()
    X_tr1.loc[segment, 'max_first_10000'] = x[:10000].max()
    X_tr1.loc[segment, 'max_last_10000'] = x[-10000:].max()
    
    X_tr1.loc[segment, 'max_to_min'] = x.max() / np.abs(x.min())
    X_tr1.loc[segment, 'max_to_min_diff'] = x.max() - np.abs(x.min())
    X_tr1.loc[segment, 'count_big'] = len(x[np.abs(x) > 500])
    X_tr1.loc[segment, 'sum'] = x.sum()
    
    X_tr1.loc[segment, 'mean_change_rate_first_50000'] = np.mean(np.nonzero((np.diff(x[:50000]) / x[:50000][:-1]))[0])
    X_tr1.loc[segment, 'mean_change_rate_last_50000'] = np.mean(np.nonzero((np.diff(x[-50000:]) / x[-50000:][:-1]))[0])
    X_tr1.loc[segment, 'mean_change_rate_first_10000'] = np.mean(np.nonzero((np.diff(x[:10000]) / x[:10000][:-1]))[0])
    X_tr1.loc[segment, 'mean_change_rate_last_10000'] = np.mean(np.nonzero((np.diff(x[-10000:]) / x[-10000:][:-1]))[0])
   
    
    #new:
    
    X_tr1.loc[segment, 'q70'] = np.quantile(x, 0.70)    
    X_tr1.loc[segment, 'q75'] = np.quantile(x, 0.75)   
    X_tr1.loc[segment, 'q60'] = np.quantile(x, 0.60)    
    X_tr1.loc[segment, 'q65'] = np.quantile(x, 0.65) 
    X_tr1.loc[segment, 'q85'] = np.quantile(x, 0.85)
    X_tr1.loc[segment, 'q90'] = np.quantile(x, 0.90)
    X_tr1.loc[segment, 'q80'] = np.quantile(x, 0.80)
    X_tr1.loc[segment, 'q95'] = np.quantile(x, 0.95)
    X_tr1.loc[segment, 'q99'] = np.quantile(x, 0.99)


    #new:
    

    X_tr1.loc[segment, 'Hilbert_mean'] = np.abs(hilbert(x)).mean()
    X_tr1.loc[segment, 'Hann_window_mean'] = (convolve(x, hann(150), mode='same') / sum(hann(150))).mean()
    X_tr1.loc[segment, 'classic_sta_lta1_mean'] = classic_sta_lta(x, 500, 10000).mean()
    X_tr1.loc[segment, 'classic_sta_lta2_mean'] = classic_sta_lta(x, 5000, 100000).mean()
    X_tr1.loc[segment, 'classic_sta_lta3_mean'] = classic_sta_lta(x, 3333, 6666).mean()
    X_tr1.loc[segment, 'classic_sta_lta4_mean'] = classic_sta_lta(x, 10000, 25000).mean()
    X_tr1.loc[segment, 'Moving_average_700_mean'] = x.rolling(window=700).mean().mean(skipna=True)
    X_tr1.loc[segment, 'Moving_average_1500_mean'] = x.rolling(window=1500).mean().mean(skipna=True)
    X_tr1.loc[segment, 'Moving_average_3000_mean'] = x.rolling(window=3000).mean().mean(skipna=True)
    X_tr1.loc[segment, 'Moving_average_6000_mean'] = x.rolling(window=6000).mean().mean(skipna=True)
    ewma = pd.Series.ewm
    X_tr1.loc[segment, 'exp_Moving_average_300_mean'] = (ewma(x, span=300).mean()).mean(skipna=True)
    X_tr1.loc[segment, 'exp_Moving_average_3000_mean'] = ewma(x, span=3000).mean().mean(skipna=True)
    X_tr1.loc[segment, 'exp_Moving_average_30000_mean'] = ewma(x, span=6000).mean().mean(skipna=True)
    no_of_std = 2 
    X_tr1.loc[segment, 'MA_700MA_std_mean'] = x.rolling(window=700).std().mean()
    X_tr1.loc[segment,'MA_700MA_BB_high_mean'] = (X_tr1.loc[segment, 'Moving_average_700_mean'] + no_of_std * X_tr1.loc[segment, 'MA_700MA_std_mean']).mean()
    X_tr1.loc[segment,'MA_700MA_BB_low_mean'] = (X_tr1.loc[segment, 'Moving_average_700_mean'] - no_of_std * X_tr1.loc[segment, 'MA_700MA_std_mean']).mean()
    X_tr1.loc[segment, 'MA_400MA_std_mean'] = x.rolling(window=400).std().mean()
    X_tr1.loc[segment,'MA_400MA_BB_high_mean'] = (X_tr1.loc[segment, 'Moving_average_700_mean'] + no_of_std * X_tr1.loc[segment, 'MA_400MA_std_mean']).mean()
    X_tr1.loc[segment,'MA_400MA_BB_low_mean'] = (X_tr1.loc[segment, 'Moving_average_700_mean'] - no_of_std * X_tr1.loc[segment, 'MA_400MA_std_mean']).mean()
    X_tr1.loc[segment, 'MA_1000MA_std_mean'] = x.rolling(window=1000).std().mean()
    
X_tr.shape
X_tr = X_tr.append(X_tr1)
y_tr = y_tr.append(y_tr1)
print(f'{X_tr.shape[0]} samples in new train data now.')
del train
gc.collect()
scaler = StandardScaler()
scaler.fit(X_tr)
X_train_scaled = pd.DataFrame(scaler.transform(X_tr), columns=X_tr.columns)
submission = pd.read_csv('../input/sample_submission.csv', index_col='seg_id')
X_test = pd.DataFrame(columns=X_tr.columns, dtype=np.float64, index=submission.index)
plt.figure(figsize=(22, 16))

for i, seg_id in enumerate(tqdm_notebook(X_test.index)):
    seg = pd.read_csv('../input/test/' + seg_id + '.csv')
    
    x = pd.Series(seg['acoustic_data'].values)
    X_test.loc[seg_id, 'mean'] = x.mean()
    X_test.loc[seg_id, 'std'] = x.std()
    X_test.loc[seg_id, 'max'] = x.max()
    X_test.loc[seg_id, 'min'] = x.min()
        
    X_test.loc[seg_id, 'mean_change_abs'] = np.mean(np.diff(x))
    X_test.loc[seg_id, 'mean_change_rate'] = np.mean(np.nonzero((np.diff(x) / x[:-1]))[0])
    X_test.loc[seg_id, 'abs_max'] = np.abs(x).max()
    X_test.loc[seg_id, 'abs_min'] = np.abs(x).min()
    
    X_test.loc[seg_id, 'std_first_50000'] = x[:50000].std()
    X_test.loc[seg_id, 'std_last_50000'] = x[-50000:].std()
    X_test.loc[seg_id, 'std_first_10000'] = x[:10000].std()
    X_test.loc[seg_id, 'std_last_10000'] = x[-10000:].std()
    
    X_test.loc[seg_id, 'avg_first_50000'] = x[:50000].mean()
    X_test.loc[seg_id, 'avg_last_50000'] = x[-50000:].mean()
    X_test.loc[seg_id, 'avg_first_10000'] = x[:10000].mean()
    X_test.loc[seg_id, 'avg_last_10000'] = x[-10000:].mean()
    
    X_test.loc[seg_id, 'min_first_50000'] = x[:50000].min()
    X_test.loc[seg_id, 'min_last_50000'] = x[-50000:].min()
    X_test.loc[seg_id, 'min_first_10000'] = x[:10000].min()
    X_test.loc[seg_id, 'min_last_10000'] = x[-10000:].min()
    
    X_test.loc[seg_id, 'max_first_50000'] = x[:50000].max()
    X_test.loc[seg_id, 'max_last_50000'] = x[-50000:].max()
    X_test.loc[seg_id, 'max_first_10000'] = x[:10000].max()
    X_test.loc[seg_id, 'max_last_10000'] = x[-10000:].max()
    
    X_test.loc[seg_id, 'max_to_min'] = x.max() / np.abs(x.min())
    X_test.loc[seg_id, 'max_to_min_diff'] = x.max() - np.abs(x.min())
    X_test.loc[seg_id, 'count_big'] = len(x[np.abs(x) > 500])
    X_test.loc[seg_id, 'sum'] = x.sum()
    
    X_test.loc[seg_id, 'mean_change_rate_first_50000'] = np.mean(np.nonzero((np.diff(x[:50000]) / x[:50000][:-1]))[0])
    X_test.loc[seg_id, 'mean_change_rate_last_50000'] = np.mean(np.nonzero((np.diff(x[-50000:]) / x[-50000:][:-1]))[0])
    X_test.loc[seg_id, 'mean_change_rate_first_10000'] = np.mean(np.nonzero((np.diff(x[:10000]) / x[:10000][:-1]))[0])
    X_test.loc[seg_id, 'mean_change_rate_last_10000'] = np.mean(np.nonzero((np.diff(x[-10000:]) / x[-10000:][:-1]))[0])
   


    #new
    
    
    X_test.loc[seg_id, 'q70'] = np.quantile(x, 0.70)    
    X_test.loc[seg_id, 'q75'] = np.quantile(x, 0.75)   
    X_test.loc[seg_id, 'q60'] = np.quantile(x, 0.60)    
    X_test.loc[seg_id, 'q65'] = np.quantile(x, 0.65) 
    X_test.loc[seg_id, 'q85'] = np.quantile(x, 0.85)
    X_test.loc[seg_id, 'q90'] = np.quantile(x, 0.90)
    X_test.loc[seg_id, 'q80'] = np.quantile(x, 0.80)
    X_test.loc[seg_id, 'q95'] = np.quantile(x,0.95)
    X_test.loc[seg_id, 'q99'] = np.quantile(x,0.99)
    

    
    X_test.loc[seg_id, 'Hilbert_mean'] = np.abs(hilbert(x)).mean()
    X_test.loc[seg_id, 'Hann_window_mean'] = (convolve(x, hann(150), mode='same') / sum(hann(150))).mean()
    X_test.loc[seg_id, 'classic_sta_lta1_mean'] = classic_sta_lta(x, 500, 10000).mean()
    X_test.loc[seg_id, 'classic_sta_lta2_mean'] = classic_sta_lta(x, 5000, 100000).mean()
    X_test.loc[seg_id, 'classic_sta_lta3_mean'] = classic_sta_lta(x, 3333, 6666).mean()
    X_test.loc[seg_id, 'classic_sta_lta4_mean'] = classic_sta_lta(x, 10000, 25000).mean()
    X_test.loc[seg_id, 'Moving_average_700_mean'] = x.rolling(window=700).mean().mean(skipna=True)
    X_test.loc[seg_id, 'Moving_average_1500_mean'] = x.rolling(window=1500).mean().mean(skipna=True)
    X_test.loc[seg_id, 'Moving_average_3000_mean'] = x.rolling(window=3000).mean().mean(skipna=True)
    X_test.loc[seg_id, 'Moving_average_6000_mean'] = x.rolling(window=6000).mean().mean(skipna=True)
    ewma = pd.Series.ewm
    X_test.loc[seg_id, 'exp_Moving_average_300_mean'] = (ewma(x, span=300).mean()).mean(skipna=True)
    X_test.loc[seg_id, 'exp_Moving_average_3000_mean'] = ewma(x, span=3000).mean().mean(skipna=True)
    X_test.loc[seg_id, 'exp_Moving_average_30000_mean'] = ewma(x, span=6000).mean().mean(skipna=True)
    no_of_std = 2
    X_test.loc[seg_id, 'MA_700MA_std_mean'] = x.rolling(window=700).std().mean()
    X_test.loc[seg_id,'MA_700MA_BB_high_mean'] = (X_test.loc[seg_id, 'Moving_average_700_mean'] + no_of_std * X_test.loc[seg_id, 'MA_700MA_std_mean']).mean()
    X_test.loc[seg_id,'MA_700MA_BB_low_mean'] = (X_test.loc[seg_id, 'Moving_average_700_mean'] - no_of_std * X_test.loc[seg_id, 'MA_700MA_std_mean']).mean()
    X_test.loc[seg_id, 'MA_400MA_std_mean'] = x.rolling(window=400).std().mean()
    X_test.loc[seg_id,'MA_400MA_BB_high_mean'] = (X_test.loc[seg_id, 'Moving_average_700_mean'] + no_of_std * X_test.loc[seg_id, 'MA_400MA_std_mean']).mean()
    X_test.loc[seg_id,'MA_400MA_BB_low_mean'] = (X_test.loc[seg_id, 'Moving_average_700_mean'] - no_of_std * X_test.loc[seg_id, 'MA_400MA_std_mean']).mean()
    X_test.loc[seg_id, 'MA_1000MA_std_mean'] = x.rolling(window=1000).std().mean()
   
    X_test_scaled = pd.DataFrame(scaler.transform(X_test), columns=X_tr1.columns)
n_fold = 5
folds = KFold(n_splits=n_fold, shuffle=True, random_state=11)
def train_model(X=X_train_scaled, X_test=X_test_scaled, y=y_tr, params=None, folds=folds, model_type='lgb', plot_feature_importance=False, model=None):

    oof = np.zeros(len(X))
    prediction = np.zeros(len(X_test))
    scores = []
    feature_importance = pd.DataFrame()
    for fold_n, (train_index, valid_index) in enumerate(folds.split(X)):
        print('Fold', fold_n, 'started at', time.ctime())
        X_train, X_valid = X.iloc[train_index], X.iloc[valid_index]
        y_train, y_valid = y.iloc[train_index], y.iloc[valid_index]
        
        if model_type == 'lgb':
            model = lgb.LGBMRegressor(**params, n_estimators = 50000, n_jobs = -1)
            model.fit(X_train, y_train, 
                    eval_set=[(X_train, y_train), (X_valid, y_valid)], eval_metric='mae',
                    verbose=10000, early_stopping_rounds=200)
            
            y_pred_valid = model.predict(X_valid)
            y_pred = model.predict(X_test, num_iteration=model.best_iteration_)
            
        if model_type == 'xgb':
            train_data = xgb.DMatrix(data=X_train, label=y_train, feature_names=X_tr.columns)
            valid_data = xgb.DMatrix(data=X_valid, label=y_valid, feature_names=X_tr.columns)

            watchlist = [(train_data, 'train'), (valid_data, 'valid_data')]
            model = xgb.train(dtrain=train_data, num_boost_round=20000, evals=watchlist, early_stopping_rounds=200, verbose_eval=500, params=params)
            y_pred_valid = model.predict(xgb.DMatrix(X_valid, feature_names=X_tr.columns), ntree_limit=model.best_ntree_limit)
            y_pred = model.predict(xgb.DMatrix(X_test, feature_names=X_tr.columns), ntree_limit=model.best_ntree_limit)
            
        if model_type == 'rcv':
            model = RidgeCV(alphas=(0.01, 0.1, 1.0, 10.0, 100.0, 1000.0), scoring='neg_mean_absolute_error', cv=5)
            model.fit(X_train, y_train)
            print(model.alpha_)

            y_pred_valid = model.predict(X_valid).reshape(-1,)
            score = mean_absolute_error(y_valid, y_pred_valid)
            print(f'Fold {fold_n}. MAE: {score:.4f}.')
            print('')
            
            y_pred = model.predict(X_test).reshape(-1,)
        
        if model_type == 'sklearn':
            model = model
            model.fit(X_train, y_train)
            
            y_pred_valid = model.predict(X_valid).reshape(-1,)
            score = mean_absolute_error(y_valid, y_pred_valid)
            print(f'Fold {fold_n}. MAE: {score:.4f}.')
            print('')
            
            y_pred = model.predict(X_test).reshape(-1,)
        
        if model_type == 'cat':
            model = CatBoostRegressor(iterations=20000,  eval_metric='MAE', **params)
            model.fit(X_train, y_train, eval_set=(X_valid, y_valid), cat_features=[], use_best_model=True, verbose=False)

            y_pred_valid = model.predict(X_valid)
            y_pred = model.predict(X_test)
        
        oof[valid_index] = y_pred_valid.reshape(-1,)
        scores.append(mean_absolute_error(y_valid, y_pred_valid))

        prediction += y_pred    
        
        if model_type == 'lgb':
            # feature importance
            fold_importance = pd.DataFrame()
            fold_importance["feature"] = X.columns
            fold_importance["importance"] = model.feature_importances_
            fold_importance["fold"] = fold_n + 1
            feature_importance = pd.concat([feature_importance, fold_importance], axis=0)

    prediction /= n_fold
    
    print('CV mean score: {0:.4f}, std: {1:.4f}.'.format(np.mean(scores), np.std(scores)))
    
    if model_type == 'lgb':
        feature_importance["importance"] /= n_fold
        if plot_feature_importance:
            cols = feature_importance[["feature", "importance"]].groupby("feature").mean().sort_values(
                by="importance", ascending=False)[:50].index

            best_features = feature_importance.loc[feature_importance.feature.isin(cols)]

            plt.figure(figsize=(16, 12));
            sns.barplot(x="importance", y="feature", data=best_features.sort_values(by="importance", ascending=False));
            plt.title('LGB Features (avg over folds)');
        
            return oof, prediction, feature_importance
        return oof, prediction
    
    else:
        return oof, prediction
from bayes_opt import BayesianOptimization
X = X_train_scaled
y = y_tr
train_data = lgb.Dataset(data=X, label=y, free_raw_data=False)
def lgb_eval(num_leaves, feature_fraction, max_depth , min_split_gain, min_child_weight,bagging_freq,reg_alpha,reg_lambda):
        params = {
            "objective" : "regression", "bagging_fraction" : 0.8,
            "min_child_samples": 20, "boosting": "gbdt",
            "learning_rate" : 0.01, "subsample" : 0.8, "colsample_bytree" : 0.8, "verbosity": -1, "metric" : 'mae'
        }
        params["bagging_freq"] = int(round(bagging_freq))
        params["reg_alpha"] = reg_alpha
        params["reg_lambda"] = reg_lambda
        params['feature_fraction'] = max(min(feature_fraction, 1), 0)
        params['max_depth'] = int(round(max_depth))
        params['num_leaves'] = int(round(num_leaves))
        params['min_split_gain'] = min_split_gain
        params['min_child_weight'] = min_child_weight
        cv_result = lgb.cv(params, train_data, nfold=5, seed=123, verbose_eval =200,stratified=False)
        return (-1.0 * np.array(cv_result['l1-mean'])).max()
lgbBO = BayesianOptimization(lgb_eval, {'num_leaves': (24, 80),
                                        'feature_fraction': (0.1, 1),
                                        'max_depth': (2, 30),
                                        'min_split_gain': (0.001, 1),
                                        'min_child_weight': (1, 30),
                                        "reg_alpha": (0,3),
                                        "reg_lambda":(0,3),
                                        "bagging_freq": (1,10)}
                            )
lgbBO.maximize(init_points=5, n_iter=15,acq='ei')
# Use the expected improvement acquisition function to handle negative numbers
lgb_params = {'num_leaves': 80,
              'min_child_weight': 28,
              'min_split_gain': 0.745,
          'min_data_in_leaf': 79,
          'objective': 'huber',
          'max_depth': 25,
          'learning_rate': 0.01,
          "boosting": "gbdt",
          "bagging_freq": 4,
          "bagging_fraction": 0.8126672064208567,
          "bagging_seed": 11,
          "metric": 'mae',
          "verbosity": -1,
          'reg_alpha': 0.1058,
          'reg_lambda': 0.2209,
          'feature_fraction': 0.9201
         }
oof_lgb, prediction_lgb, feature_importance = train_model(params=lgb_params, model_type='lgb', plot_feature_importance=True)
dtrain = xgb.DMatrix(X, label=y)
def xgb_evaluate(max_depth, gamma, colsample_bytree,learning_rate,reg_alpha,reg_lambda,min_child_weight):
    params = {'eval_metric': 'mae',
              'max_depth': int(round(max_depth)),
              'subsample': 0.8,
              'eta': 0.1,
              'gamma': gamma,
              'colsample_bytree': colsample_bytree,
              "silent":1,
              "learning_rate":learning_rate,
              "reg_alpha":reg_alpha,
              "reg_lambda":reg_lambda,
              "min_child_weight":min_child_weight
              
             }

    cv_result = xgb.cv(params, dtrain, num_boost_round=1000, nfold=3)    

    return (-1.0 * np.array(cv_result['test-mae-mean'])).max()
xgb_bo = BayesianOptimization(xgb_evaluate, {'max_depth': (3, 30), 
                                             'gamma': (0, 1),
                                             'colsample_bytree': (0.3, 1),
                                             "learning_rate": (0.0, 1.0),
                                             "reg_alpha": (1.0, 10.0),
                                             "reg_lambda":(1.0, 10.0),
                                             "min_child_weight":(0, 10)
                                            })
# Use the expected improvement acquisition function to handle negative numbers
xgb_bo.maximize(init_points=5, n_iter=15, acq='ei')
xgb_params = {'eta': 0.05,
              'gamma': 0.5913,
              'colsample_bytree': 0.9692,
              "learning_rate": 0.04425,
              "reg_alpha":  1.226,
              "reg_lambda": 4.834,
              "min_child_weight": 5,
              'max_depth': 27,
              'subsample': 0.9,
              'objective': 'reg:linear',
              'eval_metric': 'mae',
              'silent': True,
              'nthread': 4}
oof_xgb, prediction_xgb = train_model(params=xgb_params, model_type='xgb')
submission['time_to_failure'] = (prediction_lgb + prediction_xgb) / 2
print(submission.head())
submission.to_csv('submission.csv')





