
#%matplotlib inline
import os
import gc
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm import tqdm
DATA_DIR = '../input/'

hex2dec = lambda x: int(x, 16)
train = pd.read_csv(DATA_DIR+'train.csv')
cols = [
    "f190486d6","58e2e02e6","eeb9cd3aa","9fd594eec","6eef030c1","15ace8c9f",
    "fb0f5dbfe","58e056e12","20aa07010","024c577b9","d6bb78916",
    "b43a7cfd5","58232a6fb"
]
rows = np.array([2072,3493,379,2972,2367,4415,2791,3980,194,1190,3517,811,4444])-1
tmp = train.loc[rows, ["ID","target"]+cols]
print('original shape', tmp.shape)
tmp
df_cand_col = train.loc[rows, :]
df_cand_col = df_cand_col.iloc[:, 2:]
df_cand_col
df_new = train.loc[rows, cols]
def bf_search(df_new, df_cand):
    cnt = 0
    head_curr = df_new.values[1:, 0]
    tail_curr = df_new.values[:-1, -1]
    while True:
        for c in df_cand.columns:
            if c in df_new:
                continue
            elif np.all(
                df_cand[c].iloc[:-1].values==head_curr
            ) and len(df_cand[c].unique())>1:
                df_new.insert(0, c, df_cand[c].values)
                head_curr = df_new.values[1:, 0]
                print(c, 'found head!', 'new shape', df_new.shape)
                cnt += 1
                break
            elif np.all(
                df_cand[c].iloc[1:].values==tail_curr
            ) and len(df_cand[c].unique())>1:
                df_new[c] = df_cand[c].values
                tail_curr = df_new.values[:-1, -1]
                print(c, 'found tail!', 'new shape', df_new.shape)
                cnt += 1
                break
            else:
                continue
        if cnt==0:
            break
        else:
            cnt = 0
            continue
    return df_new
print('Column searching ...')
df_new = bf_search(df_new, df_cand_col)
df_new
df_new = df_new.T.copy()
df_new
df_cand_row = train[df_new.index].T.copy()
df_cand_row.head()
print('Row searching ...')
df_new = bf_search(df_new, df_cand_row)
df_new = df_new.T.copy()
df_new
df_cand_col = train.loc[df_new.index, :]
df_cand_col = df_cand_col.iloc[:, 2:]
df_cand_col
print('Column searching (second time) ...')
df_new = bf_search(df_new, df_cand_col)
print('new shape', df_new.shape)
train.loc[df_new.index, ["ID","target"]+df_new.columns.tolist()]
print(f'Row indexes({df_new.shape[0]})\n', df_new.index.values.tolist())
print(f'Column indexes({df_new.shape[1]})\n', df_new.columns.values.tolist())
for i, c in enumerate(df_new.columns):
    print(
        'No.', i, 'Column Name', c, 
        'subset count',
        (df_new[c].values==1563411.76).sum(), 
        'train count',
        (train[c].values==1563411.76).sum()
    )
res_cnt = dict((c, (train[c].values==1563411.76).sum()) for c in train.columns[2:])
res_cnt = pd.DataFrame.from_dict(res_cnt, orient='index', columns=['strange_number_cnt'])
res_cnt = res_cnt.sort_values('strange_number_cnt', 0, False)
res_cnt.head(50).T
res_cnt.head(10)
for i, c in enumerate(df_new.T.columns):
    print(
        'No.', i, 'Row Name', c, 
        'subset count',
        (df_new.T[c].values==1563411.76).sum(), 
        'train count',
        (train.T[c].values==1563411.76).sum()
    )
tmp = train.iloc[:, 2:].values
res_t_cnt = dict((idx, (tmp[i, :]==1563411.76).sum()) for i,idx in enumerate(train.index))
res_t_cnt = pd.DataFrame.from_dict(res_t_cnt, orient='index', columns=['strange_number_cnt'])
res_t_cnt = res_t_cnt.sort_values('strange_number_cnt', 0, False)
res_t_cnt.head(50).T
head_row_indexes = res_t_cnt[res_t_cnt['strange_number_cnt']>24].index.tolist()
head_row_indexes
mask = res_t_cnt['strange_number_cnt']>0 
mask&=res_t_cnt['strange_number_cnt']<8
tail_row_indexes = res_t_cnt.loc[mask].index.tolist()
tail_row_indexes
pd.concat([
    train.loc[head_row_indexes, ['target']+df_new.columns.tolist()], 
    train.loc[df_new.index, ['target']+df_new.columns.tolist()],
    train.loc[tail_row_indexes, ['target']+df_new.columns.tolist()], 
])
df_new = pd.concat([
    train.loc[head_row_indexes, df_new.columns.tolist()], 
    train.loc[df_new.index, df_new.columns.tolist()],
    train.loc[tail_row_indexes, df_new.columns.tolist()], 
])
def row_bf_search(df_new):
    df_new = df_new.T.copy()
    df_cand_row = train[df_new.index].T.copy()
    print('Row searching ...')
    df_new = bf_search(df_new, df_cand_row)
    df_new = df_new.T.copy()
    return df_new
def column_bf_search(df_new):
    df_cand_col = train.loc[df_new.index, :]
    df_cand_col = df_cand_col.iloc[:, 2:]
    print('Column searching ...')
    df_new = bf_search(df_new, df_cand_col)
    return df_new
df_new = column_bf_search(df_new)
df_new = row_bf_search(df_new)
res_cnt[:10]
train.loc[df_new.index, ['target']+df_new.columns.tolist()+['fc99f9426','91f701ba2'] + res_cnt.index.values[:6].tolist()[::-1]]
train.loc[df_new.index[-10:], df_new.columns.tolist()[-2:]+['fc99f9426', '91f701ba2'] + res_cnt.index.values[:6].tolist()[::-1]]
train.loc[df_new.index[:8], ['f190486d6', '58e2e02e6']].T
train.loc[df_new.index[-2:], df_new.columns.tolist()[-2:]+['fc99f9426', '91f701ba2']+res_cnt.index.values[:6].tolist()[::-1]]
df_new_new = train.loc[df_new.index, df_new.columns.tolist()+['fc99f9426','91f701ba2'] + res_cnt.index.values[:6].tolist()[::-1]]
df_new_new.shape
print(f'Row indexes({df_new_new.shape[0]})\n', df_new_new.index.values.tolist())
print(f'Column indexes({df_new_new.shape[1]})\n', df_new_new.columns.values.tolist())
train.loc[df_new_new.index, ['ID', 'target']+df_new_new.columns.tolist()]

