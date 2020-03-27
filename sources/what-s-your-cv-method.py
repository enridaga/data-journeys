
## Function to reduce the DF size
def reduce_mem_usage(df, verbose=True):
    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
    start_mem = df.memory_usage().sum() / 1024**2    
    for col in df.columns:
        col_type = df[col].dtypes
        if col_type in numerics:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)  
            else:
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)    
    end_mem = df.memory_usage().sum() / 1024**2
    if verbose: print('Mem. usage decreased to {:5.2f} Mb ({:.1f}% reduction)'.format(end_mem, 100 * (start_mem - end_mem) / start_mem))
    return df
import pandas as pd 
import numpy as np
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt
import os
import gc

directory = '/kaggle/input/ashrae-energy-prediction'
os.listdir(directory)
df_train = pd.read_csv(directory+'/train.csv')
df_train  = reduce_mem_usage(df_train)
df_train["timestamp"] = pd.to_datetime(df_train["timestamp"])
folds = [4, 2, 12]
for folds in folds:
    fold = KFold(n_splits=folds, shuffle=False, random_state=42)
    print('-'*50)
    print(folds,'-fold')
    for fold_, (trn_idx, val_idx)  in enumerate(fold.split(df_train, df_train['meter_reading'])):
        tr_x, tr_y = df_train.iloc[trn_idx], df_train['meter_reading'][trn_idx]
        vl_x, vl_y = df_train.iloc[val_idx], df_train['meter_reading'][val_idx]
        print('fold :', fold_)
        print('val_set_range : ',vl_x['timestamp'].min(), vl_x['timestamp'].max())    
del tr_x, tr_y, vl_x, vl_y
# Reference - https://www.kaggle.com/nroman/eda-for-ashrae
fig, axes = plt.subplots(1, 1, figsize=(14, 6), dpi=100)
df_train[['timestamp', 'meter_reading']].set_index('timestamp').resample('M').mean()['meter_reading'].plot(ax=axes, label='By month', alpha=1).set_ylabel('Meter reading', fontsize=14);
df_train[['timestamp', 'meter_reading']].set_index('timestamp').resample('H').mean()['meter_reading'].plot(ax=axes, label='By hour', alpha=0.7).set_ylabel('Meter reading', fontsize=14);
df_train[['timestamp', 'meter_reading']].set_index('timestamp').resample('D').mean()['meter_reading'].plot(ax=axes, label='By day', alpha=0.7).set_ylabel('Meter reading', fontsize=14);
axes.set_title('Mean Meter reading by hour and day, month', fontsize=16);
axes.legend();
df_train['meter'] = df_train['meter'].map({0: 'electricity', 1: 'chilledwater', 2: 'steam', 3: 'hotwater'})
col = ['electricity', 'chilledwater', 'steam', 'hotwater']
fig, axes = plt.subplots(4, 1, figsize=(14, 30), dpi=100)
for i, col in enumerate(col):
    train = df_train[df_train['meter']==col].copy()
    train[['timestamp', 'meter_reading']].set_index('timestamp').resample('M').mean()['meter_reading'].plot(ax=axes[i], label='By month', alpha=1).set_ylabel('Meter reading', fontsize=14);
    train[['timestamp', 'meter_reading']].set_index('timestamp').resample('H').mean()['meter_reading'].plot(ax=axes[i], label='By hour', alpha=0.7).set_ylabel('Meter reading', fontsize=14);
    train[['timestamp', 'meter_reading']].set_index('timestamp').resample('D').mean()['meter_reading'].plot(ax=axes[i], label='By day', alpha=0.7).set_ylabel('Meter reading', fontsize=14);
    axes[i].set_title(col +' Mean Meter reading by hour and day, month', fontsize=16);
    axes[i].legend();
del df_train
del train

df_test = pd.read_csv(directory+'/test.csv')
df_test = df_test['meter']

gc.collect()

fold2 = pd.read_csv('/kaggle/input/2fold/submission.csv')
fold2  = reduce_mem_usage(fold2)
fold2["timestamp"] = pd.to_datetime(fold2["timestamp"])
fold2 = pd.concat([fold2,df_test],axis=1)
fold2['meter'] = fold2['meter'].map({0: 'electricity', 1: 'chilledwater', 2: 'steam', 3: 'hotwater'})
fold2.head()
gc.collect()
column = ['fold0_meter_reading', 'fold1_meter_reading','fold_mean_meter_reading']
fig, axes = plt.subplots(3, 1, figsize=(14, 18), dpi=100)
for i, col in enumerate(column):
    fold2[['timestamp', col]].set_index('timestamp').resample('M').mean()[col].plot(ax=axes[i], label='By month', alpha=1).set_ylabel('Meter reading', fontsize=14);
    fold2[['timestamp', col]].set_index('timestamp').resample('H').mean()[col].plot(ax=axes[i], label='By hour', alpha=0.7).set_ylabel('Meter reading', fontsize=14);
    fold2[['timestamp', col]].set_index('timestamp').resample('D').mean()[col].plot(ax=axes[i], label='By day', alpha=0.7).set_ylabel('Meter reading', fontsize=14);
    axes[i].set_title(col + 'by hour and day, month', fontsize=16);
    axes[i].legend();
column = ['electricity', 'chilledwater', 'steam', 'hotwater']
fig, axes = plt.subplots(2, 2, figsize=(20, 12), dpi=100)
for i, col in enumerate(column):
    train = fold2[fold2['meter']==col].copy()
    train[['timestamp', 'fold0_meter_reading']].set_index('timestamp').resample('M').mean()['fold0_meter_reading'].plot(ax=axes[i%2][i//2], label='By month', alpha=1).set_ylabel('Meter reading', fontsize=14);
    train[['timestamp', 'fold0_meter_reading']].set_index('timestamp').resample('H').mean()['fold0_meter_reading'].plot(ax=axes[i%2][i//2], label='By hour', alpha=0.7).set_ylabel('Meter reading', fontsize=14);
    train[['timestamp', 'fold0_meter_reading']].set_index('timestamp').resample('D').mean()['fold0_meter_reading'].plot(ax=axes[i%2][i//2], label='By day', alpha=0.7).set_ylabel('Meter reading', fontsize=14);
    axes[i%2][i//2].set_title(col + 'by hour and day, month', fontsize=16);
    axes[i%2][i//2].legend();
fig, axes = plt.subplots(2, 2, figsize=(20, 12), dpi=100)
for i, col in enumerate(column):
    train = fold2[fold2['meter']==col].copy()
    train[['timestamp', 'fold1_meter_reading']].set_index('timestamp').resample('M').mean()['fold1_meter_reading'].plot(ax=axes[i%2][i//2], label='By month', alpha=1).set_ylabel('Meter reading', fontsize=14);
    train[['timestamp', 'fold1_meter_reading']].set_index('timestamp').resample('H').mean()['fold1_meter_reading'].plot(ax=axes[i%2][i//2], label='By hour', alpha=0.7).set_ylabel('Meter reading', fontsize=14);
    train[['timestamp', 'fold1_meter_reading']].set_index('timestamp').resample('D').mean()['fold1_meter_reading'].plot(ax=axes[i%2][i//2], label='By day', alpha=0.7).set_ylabel('Meter reading', fontsize=14);
    axes[i%2][i//2].set_title(col + 'by hour and day, month', fontsize=16);
    axes[i%2][i//2].legend();
fig, axes = plt.subplots(2, 2, figsize=(20, 12), dpi=100)
for i, col in enumerate(column):
    train = fold2[fold2['meter']==col].copy()
    train[['timestamp', 'fold_mean_meter_reading']].set_index('timestamp').resample('M').mean()['fold_mean_meter_reading'].plot(ax=axes[i%2][i//2], label='By month', alpha=1).set_ylabel('Meter reading', fontsize=14);
    train[['timestamp', 'fold_mean_meter_reading']].set_index('timestamp').resample('H').mean()['fold_mean_meter_reading'].plot(ax=axes[i%2][i//2], label='By hour', alpha=0.7).set_ylabel('Meter reading', fontsize=14);
    train[['timestamp', 'fold_mean_meter_reading']].set_index('timestamp').resample('D').mean()['fold_mean_meter_reading'].plot(ax=axes[i%2][i//2], label='By day', alpha=0.7).set_ylabel('Meter reading', fontsize=14);
    axes[i%2][i//2].set_title(col + 'by hour and day, month', fontsize=16);
    axes[i%2][i//2].legend();
del fold2
del train
gc.collect()

fold4 = pd.read_csv('/kaggle/input/4fold/submission.csv')
fold4  = reduce_mem_usage(fold4)
fold4["timestamp"] = pd.to_datetime(fold4["timestamp"])
fold4.head()
column = ['fold0_meter_reading', 'fold1_meter_reading','fold2_meter_reading', 'fold3_meter_reading','fold_mean_meter_reading']
fig, axes = plt.subplots(5, 1, figsize=(14, 30), dpi=100)
for i, col in enumerate(column):
    fold4[['timestamp', col]].set_index('timestamp').resample('M').mean()[col].plot(ax=axes[i], label='By month', alpha=1).set_ylabel('Meter reading', fontsize=14);
    fold4[['timestamp', col]].set_index('timestamp').resample('H').mean()[col].plot(ax=axes[i], label='By hour', alpha=0.7).set_ylabel('Meter reading', fontsize=14);
    fold4[['timestamp', col]].set_index('timestamp').resample('D').mean()[col].plot(ax=axes[i], label='By day', alpha=0.7).set_ylabel('Meter reading', fontsize=14);
    axes[i].set_title(col + 'by hour and day, month', fontsize=16);
    axes[i].legend();
gc.collect()
fold4 = pd.concat([fold4,df_test],axis=1)
fold4['meter'] = fold4['meter'].map({0: 'electricity', 1: 'chilledwater', 2: 'steam', 3: 'hotwater'})
column = ['electricity', 'chilledwater', 'steam', 'hotwater']
fig, axes = plt.subplots(2, 2, figsize=(20, 12), dpi=100)
for i, col in enumerate(column):
    train = fold4[fold4['meter']==col].copy()
    train[['timestamp', 'fold0_meter_reading']].set_index('timestamp').resample('M').mean()['fold0_meter_reading'].plot(ax=axes[i%2][i//2], label='By month', alpha=1).set_ylabel('Meter reading', fontsize=14);
    train[['timestamp', 'fold0_meter_reading']].set_index('timestamp').resample('H').mean()['fold0_meter_reading'].plot(ax=axes[i%2][i//2], label='By hour', alpha=0.7).set_ylabel('Meter reading', fontsize=14);
    train[['timestamp', 'fold0_meter_reading']].set_index('timestamp').resample('D').mean()['fold0_meter_reading'].plot(ax=axes[i%2][i//2], label='By day', alpha=0.7).set_ylabel('Meter reading', fontsize=14);
    axes[i%2][i//2].set_title(col + 'by hour and day, month', fontsize=16);
    axes[i%2][i//2].legend();
fig, axes = plt.subplots(2, 2, figsize=(20, 12), dpi=100)
for i, col in enumerate(column):
    train = fold4[fold4['meter']==col].copy()
    train[['timestamp', 'fold1_meter_reading']].set_index('timestamp').resample('M').mean()['fold1_meter_reading'].plot(ax=axes[i%2][i//2], label='By month', alpha=1).set_ylabel('Meter reading', fontsize=14);
    train[['timestamp', 'fold1_meter_reading']].set_index('timestamp').resample('H').mean()['fold1_meter_reading'].plot(ax=axes[i%2][i//2], label='By hour', alpha=0.7).set_ylabel('Meter reading', fontsize=14);
    train[['timestamp', 'fold1_meter_reading']].set_index('timestamp').resample('D').mean()['fold1_meter_reading'].plot(ax=axes[i%2][i//2], label='By day', alpha=0.7).set_ylabel('Meter reading', fontsize=14);
    axes[i%2][i//2].set_title(col + 'by hour and day, month', fontsize=16);
    axes[i%2][i//2].legend();
fig, axes = plt.subplots(2, 2, figsize=(20, 12), dpi=100)
for i, col in enumerate(column):
    train = fold4[fold4['meter']==col].copy()
    train[['timestamp', 'fold2_meter_reading']].set_index('timestamp').resample('M').mean()['fold2_meter_reading'].plot(ax=axes[i%2][i//2], label='By month', alpha=1).set_ylabel('Meter reading', fontsize=14);
    train[['timestamp', 'fold2_meter_reading']].set_index('timestamp').resample('H').mean()['fold2_meter_reading'].plot(ax=axes[i%2][i//2], label='By hour', alpha=0.7).set_ylabel('Meter reading', fontsize=14);
    train[['timestamp', 'fold2_meter_reading']].set_index('timestamp').resample('D').mean()['fold2_meter_reading'].plot(ax=axes[i%2][i//2], label='By day', alpha=0.7).set_ylabel('Meter reading', fontsize=14);
    axes[i%2][i//2].set_title(col + 'by hour and day, month', fontsize=16);
    axes[i%2][i//2].legend();
fig, axes = plt.subplots(2, 2, figsize=(20, 12), dpi=100)
for i, col in enumerate(column):
    train = fold4[fold4['meter']==col].copy()
    train[['timestamp', 'fold3_meter_reading']].set_index('timestamp').resample('M').mean()['fold3_meter_reading'].plot(ax=axes[i%2][i//2], label='By month', alpha=1).set_ylabel('Meter reading', fontsize=14);
    train[['timestamp', 'fold3_meter_reading']].set_index('timestamp').resample('H').mean()['fold3_meter_reading'].plot(ax=axes[i%2][i//2], label='By hour', alpha=0.7).set_ylabel('Meter reading', fontsize=14);
    train[['timestamp', 'fold3_meter_reading']].set_index('timestamp').resample('D').mean()['fold3_meter_reading'].plot(ax=axes[i%2][i//2], label='By day', alpha=0.7).set_ylabel('Meter reading', fontsize=14);
    axes[i%2][i//2].set_title(col + 'by hour and day, month', fontsize=16);
    axes[i%2][i//2].legend();
fig, axes = plt.subplots(2, 2, figsize=(20, 12), dpi=100)
for i, col in enumerate(column):
    train = fold4[fold4['meter']==col].copy()
    train[['timestamp', 'fold_mean_meter_reading']].set_index('timestamp').resample('M').mean()['fold_mean_meter_reading'].plot(ax=axes[i%2][i//2], label='By month', alpha=1).set_ylabel('Meter reading', fontsize=14);
    train[['timestamp', 'fold_mean_meter_reading']].set_index('timestamp').resample('H').mean()['fold_mean_meter_reading'].plot(ax=axes[i%2][i//2], label='By hour', alpha=0.7).set_ylabel('Meter reading', fontsize=14);
    train[['timestamp', 'fold_mean_meter_reading']].set_index('timestamp').resample('D').mean()['fold_mean_meter_reading'].plot(ax=axes[i%2][i//2], label='By day', alpha=0.7).set_ylabel('Meter reading', fontsize=14);
    axes[i%2][i//2].set_title(col + 'by hour and day, month', fontsize=16);
    axes[i%2][i//2].legend();
