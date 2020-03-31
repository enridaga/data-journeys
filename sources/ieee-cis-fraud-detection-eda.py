
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import gc,os,sys
import re

from sklearn import metrics, preprocessing
from sklearn.model_selection import StratifiedKFold
from sklearn.decomposition import PCA, KernelPCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.cluster import KMeans
from tqdm import tqdm

import warnings
warnings.filterwarnings('ignore')

sns.set_style('darkgrid')

pd.options.display.float_format = '{:,.3f}'.format

print(os.listdir("../input"))
# Memory saving function credit to https://www.kaggle.com/gemartin/load-data-reduce-memory-usage
def reduce_mem_usage(df):
    """ iterate through all the columns of a dataframe and modify the data type
        to reduce memory usage.
    """
    start_mem = df.memory_usage().sum() / 1024**2
    
    for col in df.columns:
        col_type = df[col].dtype
        
        if col_type != object:
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
        #else:
            #df[col] = df[col].astype('category')

    end_mem = df.memory_usage().sum() / 1024**2
    print('Memory usage of dataframe is {:.2f} MB --> {:.2f} MB (Decreased by {:.1f}%)'.format(
        start_mem, end_mem, 100 * (start_mem - end_mem) / start_mem))
    return df
### %time
train_id = pd.read_csv('../input/train_identity.csv')
train_trn = pd.read_csv('../input/train_transaction.csv')
test_id = pd.read_csv('../input/test_identity.csv')
test_trn = pd.read_csv('../input/test_transaction.csv')
train_id = reduce_mem_usage(train_id)
train_trn = reduce_mem_usage(train_trn)
test_id = reduce_mem_usage(test_id)
test_trn = reduce_mem_usage(test_trn)
print(train_id.shape, test_id.shape)
print(train_trn.shape, test_trn.shape)
[c for c in train_trn.columns if c not in test_trn.columns]
fc = train_trn['isFraud'].value_counts(normalize=True).to_frame()
fc.plot.bar()
fc.T
fig,ax = plt.subplots(2, 1, figsize=(16,8))

train_trn['_seq_day'] = train_trn['TransactionDT'] // (24*60*60)
train_trn['_seq_week'] = train_trn['_seq_day'] // 7
train_trn.groupby('_seq_day')['isFraud'].mean().to_frame().plot.line(ax=ax[0])
train_trn.groupby('_seq_week')['isFraud'].mean().to_frame().plot.line(ax=ax[1])
import datetime

START_DATE = '2017-11-30'
startdate = datetime.datetime.strptime(START_DATE, "%Y-%m-%d")
train_trn['Date'] = train_trn['TransactionDT'].apply(lambda x: (startdate + datetime.timedelta(seconds=x)))
train_trn['_ymd'] = train_trn['Date'].dt.year.astype(str) + '-' + train_trn['Date'].dt.month.astype(str) + '-' + train_trn['Date'].dt.day.astype(str)
train_trn['_year_month'] = train_trn['Date'].dt.year.astype(str) + '-' + train_trn['Date'].dt.month.astype(str)
train_trn['_weekday'] = train_trn['Date'].dt.dayofweek
train_trn['_hour'] = train_trn['Date'].dt.hour
train_trn['_day'] = train_trn['Date'].dt.day

fig,ax = plt.subplots(4, 1, figsize=(16,12))

train_trn.groupby('_weekday')['isFraud'].mean().to_frame().plot.bar(ax=ax[0])
train_trn.groupby('_hour')['isFraud'].mean().to_frame().plot.bar(ax=ax[1])
train_trn.groupby('_day')['isFraud'].mean().to_frame().plot.bar(ax=ax[2])
train_trn.groupby('_year_month')['isFraud'].mean().to_frame().plot.bar(ax=ax[3])
df = train_trn.groupby(['_ymd'])['isFraud'].agg(['count','mean','sum'])
df.sort_values(by='mean',ascending=False)[:10].T
df.sort_values(by='count',ascending=False)[:10].T
# transaction-count X fraud-rate
plt.scatter(df['count'], df['mean'], s=10)
# transaction-count X fraud-count
plt.scatter(df['count'], df['sum'], s=10)
train_trn['_weekday_hour'] = train_trn['_weekday'].astype(str) + '_' + train_trn['_hour'].astype(str)
train_trn.groupby('_weekday_hour')['isFraud'].mean().to_frame().plot.line(figsize=(16,3))
df = train_trn.groupby('_weekday')['isFraud'].mean().to_frame()
df.sort_values(by='isFraud', ascending=False)
df = train_trn.groupby('_hour')['isFraud'].mean().to_frame()
df.sort_values(by='isFraud', ascending=False).head(10)
df = train_trn.groupby('_weekday_hour')['isFraud'].mean().to_frame()
df.sort_values(by='isFraud', ascending=False).head(10)
train_trn['_amount_qcut10'] = pd.qcut(train_trn['TransactionAmt'],10)
df = train_trn.groupby('_amount_qcut10')['isFraud'].mean().to_frame()
df.sort_values(by='isFraud', ascending=False)
# Not all transactions have corresponding identity information.
#len([c for c in train_trn['TransactionID'] if c not in train_id['TransactionID'].values]) #446307

# Not all fraud transactions have corresponding identity information.
fraud_id = train_trn[train_trn['isFraud'] == 1]['TransactionID']
fraud_id_in_trn = [i for i in fraud_id if i in train_id['TransactionID'].values]
print(f'fraud data count:{len(fraud_id)}, and in trn:{len(fraud_id_in_trn)}')
train_id_trn = pd.merge(train_id, train_trn[['isFraud','TransactionAmt','TransactionID']])
train_id_f0 = train_id_trn[train_id_trn['isFraud'] == 0]
train_id_f1 = train_id_trn[train_id_trn['isFraud'] == 1]
print(train_id_f0.shape, train_id_f1.shape)

def plotHistByFraud(col, bins=20, figsize=(8,3)):
    with np.errstate(invalid='ignore'):
        plt.figure(figsize=figsize)
        plt.hist([train_id_f0[col], train_id_f1[col]], bins=bins, density=True, color=['royalblue', 'orange'])
        
def plotCategoryRateBar(col, topN=np.nan, figsize=(8,3)):
    a, b = train_id_f0, train_id_f1
    if topN == topN: # isNotNan
        vals = b[col].value_counts(normalize=True).to_frame().iloc[:topN,0]
        subA = a.loc[a[col].isin(vals.index.values), col]
        df = pd.DataFrame({'normal':subA.value_counts(normalize=True), 'fraud':vals})
    else:
        df = pd.DataFrame({'normal':a[col].value_counts(normalize=True), 'fraud':b[col].value_counts(normalize=True)})
    df.sort_values('fraud', ascending=False).plot.bar(figsize=figsize)
plotHistByFraud('id_01')
plotHistByFraud('id_02')
plotHistByFraud('id_07')
numid_cols = [f'id_{str(i).zfill(2)}' for i in range(1,12)]
train_id_trn[numid_cols].isna().sum().to_frame().T / len(train_id)
plt.figure(figsize=(10, 5))
sns.heatmap(train_id_trn[['isFraud','TransactionAmt']+numid_cols].corr(), annot=True, fmt='.2f')
train_id_f1[['isFraud'] + numid_cols].head(10)
train_id_f0[['isFraud'] + numid_cols].head(10)
plotCategoryRateBar('id_15')
plotCategoryRateBar('id_16')
plotCategoryRateBar('id_17',10)
plotCategoryRateBar('id_19', 20)
plotHistByFraud('id_19')
print('unique count:', train_id['id_19'].nunique())
plotCategoryRateBar('id_20', 20)
plotHistByFraud('id_20')
print('unique count:', train_id['id_20'].nunique())
plotCategoryRateBar('id_23')
plotCategoryRateBar('id_26', 15)
plotCategoryRateBar('id_28')
plotCategoryRateBar('id_29')
plotCategoryRateBar('id_31', 20)

train_id_f0['_id_31_ua'] = train_id_f0['id_31'].apply(lambda x: x.split()[0] if x == x else 'unknown')
train_id_f1['_id_31_ua'] = train_id_f1['id_31'].apply(lambda x: x.split()[0] if x == x else 'unknown')
plotCategoryRateBar('_id_31_ua', 10)
plotCategoryRateBar('id_32')
plotCategoryRateBar('id_33',15)
plotCategoryRateBar('id_34')
plotCategoryRateBar('id_35')
plotCategoryRateBar('id_38')
plotCategoryRateBar('DeviceType')
plotCategoryRateBar('DeviceInfo',10)
ccols = [f'C{i}' for i in range(1,15)]
dcols = [f'D{i}' for i in range(1,16)]
mcols = [f'M{i}' for i in range(1,10)]
vcols = [f'V{i}' for i in range(1,340)]
train_trn_f0 = train_trn[train_trn['isFraud'] == 0]
train_trn_f1 = train_trn[train_trn['isFraud'] == 1]
print(train_trn_f0.shape, train_trn_f1.shape)

def plotTrnHistByFraud(col, bins=20, figsize=(8,3)):
    with np.errstate(invalid='ignore'):
        plt.figure(figsize=figsize)
        plt.hist([train_trn_f0[col], train_trn_f1[col]], bins=bins, density=True, color=['royalblue', 'orange'])

def plotTrnLogHistByFraud(col, bins=20, figsize=(8,3)):
    with np.errstate(invalid='ignore'):
        plt.figure(figsize=figsize)
        plt.hist([np.log1p(train_trn_f0[col]), np.log1p(train_trn_f1[col])], bins=bins, density=True, color=['royalblue', 'orange'])
        
def plotTrnCategoryRateBar(col, topN=np.nan, figsize=(8,3)):
    a, b = train_trn_f0, train_trn_f1
    if topN == topN: # isNotNan
        vals = b[col].value_counts(normalize=True).to_frame().iloc[:topN,0]
        subA = a.loc[a[col].isin(vals.index.values), col]
        df = pd.DataFrame({'normal':subA.value_counts(normalize=True), 'fraud':vals})
    else:
        df = pd.DataFrame({'normal':a[col].value_counts(normalize=True), 'fraud':b[col].value_counts(normalize=True)})
    df.sort_values('fraud', ascending=False).plot.bar(figsize=figsize)
#START_DATE = '2017-11-30'
START_DATE = '2017-12-01'
startdate = datetime.datetime.strptime(START_DATE, "%Y-%m-%d")

train_date = train_trn['TransactionDT'].apply(lambda x: (startdate + datetime.timedelta(seconds=x)))
test_date = test_trn['TransactionDT'].apply(lambda x: (startdate + datetime.timedelta(seconds=x)))

print('train date:', train_date.min(), '-', train_date.max())
print('test  date:', test_date.min(), '-', test_date.max())
plt.figure(figsize=(12,4))
train_trn['TransactionDT'].hist(bins=20)
test_trn['TransactionDT'].hist(bins=20)
def appendLagDT(df):
    df = df.assign(_date_lag = df['TransactionDT'] - df.groupby(['card1','card2'])['TransactionDT'].shift(1))
    return df

train_trn = appendLagDT(train_trn)
train_trn_f0 = train_trn[train_trn['isFraud'] == 0]
train_trn_f1 = train_trn[train_trn['isFraud'] == 1]
pd.concat([train_trn_f0['_date_lag'].describe(), 
           train_trn_f1['_date_lag'].describe()], axis=1)
plotTrnLogHistByFraud('_date_lag')
plotTrnHistByFraud('TransactionAmt')
plotTrnLogHistByFraud('TransactionAmt')
amt_desc = pd.concat([train_trn_f0['TransactionAmt'].describe(), train_trn_f1['TransactionAmt'].describe()], axis=1)
amt_desc.columns = ['normal','fraud']
amt_desc
def appendLagAmt(df):
    df = df.assign(_amt_lag = df['TransactionAmt'] - df.groupby(['card1','card2'])['TransactionAmt'].shift(1))
    df['_amt_lag_sig'] = df['_amt_lag'].apply(lambda x: '0' if np.isnan(x) else '+' if x >=0 else '-')
    return df

train_trn = appendLagAmt(train_trn)
train_trn_f0 = train_trn[train_trn['isFraud'] == 0]
train_trn_f1 = train_trn[train_trn['isFraud'] == 1]
plotTrnHistByFraud('_amt_lag')
plotTrnCategoryRateBar('_amt_lag_sig')
plotTrnCategoryRateBar('ProductCD')
cols = ['ProductCD','addr1','addr2','dist1','dist2']
train_trn[cols].head(20)
cols = ['addr1','addr2','dist1','dist2']
for f in cols:
    train_trn[f + '_isna'] = train_trn[f].isna()
pd.crosstab(train_trn['ProductCD'], train_trn['addr1_isna'])
pd.crosstab(train_trn['ProductCD'], train_trn['dist1_isna'])
pd.crosstab(train_trn['ProductCD'], train_trn['dist2_isna'])
train_trn[(train_trn['dist1_isna'] == False) & (train_trn['dist2_isna'] == False)][cols]
train_trn = pd.concat([train_trn, pd.get_dummies(train_trn[['ProductCD']])], axis=1)
train_trn.head(5)
cols = ['ProductCD_W','ProductCD_C','ProductCD_H','ProductCD_R','ProductCD_S','dist1_isna','dist2_isna','addr1_isna','addr2_isna']
train_trn[cols].corr()
train_trn['_amount_max_ProductCD'] = train_trn.groupby(['ProductCD'])['TransactionAmt'].transform('max')
train_trn[['ProductCD','_amount_max_ProductCD']].drop_duplicates().sort_values(by='_amount_max_ProductCD', ascending=False)
cols = [f'card{n}' for n in range(1,7)]
train_trn[cols].isnull().sum()
train_trn[cols].nunique()
train_trn[cols].head(10)
train_trn[train_trn['card4']=='visa']['card1'].hist(bins=50)
train_trn[train_trn['card4']=='mastercard']['card1'].hist(bins=50)
train_trn[train_trn['card4']=='visa']['card2'].hist(bins=50)
train_trn[train_trn['card4']=='mastercard']['card2'].hist(bins=50)
plotTrnCategoryRateBar('card1', 15)
plotTrnHistByFraud('card1', bins=30)
plotTrnCategoryRateBar('card2', 15)
plotTrnHistByFraud('card2', bins=30)
train_trn_f0['_card1_card2'] = train_trn_f0['card1'].astype(str) + '_' + train_trn_f0['card2'].astype(str)
train_trn_f1['_card1_card2'] = train_trn_f1['card1'].astype(str) + '_' + train_trn_f1['card2'].astype(str)

plotTrnCategoryRateBar('_card1_card2', 50, figsize=(15,3))
plotTrnCategoryRateBar('card3', 10)
plotTrnCategoryRateBar('card4')
plotTrnCategoryRateBar('card5', 10)
plotTrnCategoryRateBar('card6')
print(len(train_trn))
print(train_trn['card1'].nunique(), train_trn['card2'].nunique(), train_trn['card3'].nunique(), train_trn['card5'].nunique())

train_trn['card_n'] = (train_trn['card1'].astype(str) + '_' + train_trn['card2'].astype(str) \
       + '_' + train_trn['card3'].astype(str) + '_' + train_trn['card5'].astype(str))
print('unique cards:', train_trn['card_n'].nunique())
vc = train_trn['card_n'].value_counts()
vc[vc > 3000].plot.bar()
train_trn.groupby(['card_n'])['isFraud'].mean().sort_values(ascending=False)
train_trn['addr1'].nunique(), train_trn['addr2'].nunique()
plotTrnCategoryRateBar('addr1', 20)
plotTrnHistByFraud('addr1', bins=30)
train_trn['addr1'].value_counts(dropna=False).to_frame().iloc[:10]
plotTrnCategoryRateBar('addr2', 10)
print('addr2 nunique:', train_trn['addr2'].nunique())
train_trn['addr2'].value_counts(dropna=False).to_frame().iloc[:10]
plotTrnCategoryRateBar('dist1', 20)
plotTrnCategoryRateBar('dist2', 20)
train_trn_f0['dist3'] = np.where(train_trn_f0['dist1'].isna(), train_trn_f0['dist2'], train_trn_f0['dist1'])
train_trn_f1['dist3'] = np.where(train_trn_f1['dist1'].isna(), train_trn_f1['dist2'], train_trn_f1['dist1'])

plotTrnCategoryRateBar('dist3', 20)
plotTrnLogHistByFraud('dist3')
plotTrnCategoryRateBar('P_emaildomain',10)
plotTrnCategoryRateBar('R_emaildomain',10)
train_trn['P_emaildomain'].fillna('unknown',inplace=True)
train_trn['R_emaildomain'].fillna('unknown',inplace=True)

inf = pd.DataFrame([], columns=['P_emaildomain','R_emaildomain','Count','isFraud'])
for n in (train_trn['P_emaildomain'] + ' ' + train_trn['R_emaildomain']).unique():
    p, r = n.split()[0], n.split()[1]
    df = train_trn[(train_trn['P_emaildomain'] == p) & (train_trn['R_emaildomain'] == r)]
    inf = inf.append(pd.DataFrame([p, r, len(df), df['isFraud'].mean()], index=inf.columns).T)

inf.sort_values(by='isFraud', ascending=False).head(10)
train_trn_f1['P_emaildomain_prefix'] = train_trn_f1['P_emaildomain'].fillna('unknown').apply(lambda x: x.split('.')[0])
pd.crosstab(train_trn_f1['P_emaildomain_prefix'], train_trn_f1['ProductCD']).T
train_trn['P_emaildomain_prefix'] = train_trn['P_emaildomain'].apply(lambda x: x.split('.')[0])
ct = pd.crosstab(train_trn['P_emaildomain_prefix'], train_trn['ProductCD'])
ct = ct.sort_values(by='W')[-15:]
ct.plot.barh(stacked=True, figsize=(12,4))
for i in range(1,15):
    plotTrnCategoryRateBar(f'C{i}',10)
train_trn[ccols].describe().loc[['count','mean','std','min','max']]
plt.figure(figsize=(10,5))

corr = train_trn[['isFraud'] + ccols].corr()
sns.heatmap(corr, annot=True, fmt='.2f')
cols = ['TransactionDT','TransactionAmt','isFraud'] + ccols
train_trn[train_trn['card1'] == 9500][cols].head(20)
cols = ['TransactionDT','TransactionAmt','isFraud'] + ccols
train_trn[train_trn['card1'] == 4774][cols].head(20)
train_trn[train_trn['card1'] == 14770][cols].head(20)
for i in range(1,16):
    plotTrnCategoryRateBar(f'D{i}',10)
train_trn[dcols].describe().loc[['count','mean','std','min','max']]
plt.figure(figsize=(12,4))

plt.scatter(train_trn_f0['TransactionDT'], train_trn_f0['D1'], s=2)
plt.scatter(train_trn_f1['TransactionDT'], train_trn_f1['D1'], s=2, c='r')
plt.scatter(test_trn['TransactionDT'], test_trn['D1'], s=2, c='g')
plt.figure(figsize=(12,4))

# ref. https://www.kaggle.com/kyakovlev/ieee-columns-scaling
plt.scatter(train_trn_f0['TransactionDT'], train_trn_f0['D15'], s=2)
plt.scatter(train_trn_f1['TransactionDT'], train_trn_f1['D15'], s=2, c='r')
plt.scatter(test_trn['TransactionDT'], test_trn['D15'], s=2, c='g')
plt.figure(figsize=(10,5))

corr = train_trn[['isFraud'] + dcols].corr()
sns.heatmap(corr, annot=True, fmt='.2f')
fig, ax = plt.subplots(1, 2, figsize=(15, 3))
train_trn.loc[train_trn['isFraud']==0, dcols].isnull().sum(axis=1).to_frame().hist(ax=ax[0], bins=20)
train_trn.loc[train_trn['isFraud']==1, dcols].isnull().sum(axis=1).to_frame().hist(ax=ax[1], bins=20)
cols = ['TransactionDT','TransactionAmt','isFraud'] + dcols
train_trn[train_trn['card1'] == 9500][cols].head(20)
cols = ['TransactionDT','TransactionAmt','isFraud'] + dcols
train_trn[train_trn['card1'] == 4774][cols].head(20)
train_trn[train_trn['card1'] == 14770][cols].head(20)
plotTrnCategoryRateBar('M1')
plotTrnCategoryRateBar('M2')
plotTrnCategoryRateBar('M3')
plotTrnCategoryRateBar('M4')
plotTrnCategoryRateBar('M5')
plotTrnCategoryRateBar('M6')
plotTrnCategoryRateBar('M7')
plotTrnCategoryRateBar('M8')
plotTrnCategoryRateBar('M9')
for f in ['V1','V14','V41','V65','V88','V107','V305']:
    plotTrnCategoryRateBar(f)
vsum0 = train_trn_f0[vcols].sum(axis=1)
vsum1 = train_trn_f1[vcols].sum(axis=1)
plt.scatter(train_trn_f0['_ymd'], vsum0, s=5)
plt.scatter(train_trn_f1['_ymd'], vsum1, s=5, c='r')
m = train_trn_f1[vcols].describe().T['max']
m[m >= 10000]
plt.scatter(train_trn_f0['_ymd'], train_trn_f0['V160'], s=5)
plt.scatter(train_trn_f1['_ymd'], train_trn_f1['V160'], s=5, c='r')
vcols_1 = [f'V{i}' for i in range(1,160)]+[f'V{i}' for i in range(161,340)]
vsum0 = train_trn_f0[vcols_1].sum(axis=1)
vsum1 = train_trn_f1[vcols_1].sum(axis=1)
plt.scatter(train_trn_f0['_ymd'], vsum0, s=5)
plt.scatter(train_trn_f1['_ymd'], vsum1, s=5, c='r')
train_trn[vcols].isnull().sum() / len(train_trn)
train_trn.loc[train_trn['V1'].isnull(), vcols].head(10)
train_trn.loc[train_trn['V1'].isnull() == False, vcols].head(10)
fig, ax = plt.subplots(1, 2, figsize=(15, 3))
train_trn.loc[train_trn['isFraud']==0, vcols].isnull().sum(axis=1).to_frame().hist(ax=ax[0], bins=20)
train_trn.loc[train_trn['isFraud']==1, vcols].isnull().sum(axis=1).to_frame().hist(ax=ax[1], bins=20)
train_trn[vcols].describe().T[['min','max']].T
vcols = [f'V{i}' for i in range(1,340)]

pca = PCA()
pca.fit(train_trn[vcols].fillna(-1))
plt.xlabel('components')
plt.plot(np.add.accumulate(pca.explained_variance_ratio_))
plt.show()

pca = PCA(n_components=0.99)
vcol_pca = pca.fit_transform(train_trn[vcols].fillna(-1))
print(vcol_pca.ndim)
del train_trn_f0,train_trn_f1,train_id_f0,train_id_f1

print(pd.DataFrame([[val for val in dir()], [sys.getsizeof(eval(val)) for val in dir()]],
                   index=['name','size']).T.sort_values('size', ascending=False).reset_index(drop=True)[:10])
train_id = pd.read_csv('../input/train_identity.csv')
train_trn = pd.read_csv('../input/train_transaction.csv')
test_id = pd.read_csv('../input/test_identity.csv')
test_trn = pd.read_csv('../input/test_transaction.csv')

id_cols = list(train_id.columns.values)
trn_cols = list(train_trn.drop('isFraud', axis=1).columns.values)

X_train = pd.merge(train_trn[trn_cols + ['isFraud']], train_id[id_cols], how='left')
X_train = reduce_mem_usage(X_train)
X_test = pd.merge(test_trn[trn_cols], test_id[id_cols], how='left')
X_test = reduce_mem_usage(X_test)

X_train_id = X_train.pop('TransactionID')
X_test_id = X_test.pop('TransactionID')
del train_id,train_trn,test_id,test_trn

all_data = X_train.append(X_test, sort=False).reset_index(drop=True)
vcols = [f'V{i}' for i in range(1,340)]

sc = preprocessing.MinMaxScaler()

pca = PCA(n_components=2) #0.99
vcol_pca = pca.fit_transform(sc.fit_transform(all_data[vcols].fillna(-1)))

all_data['_vcol_pca0'] = vcol_pca[:,0]
all_data['_vcol_pca1'] = vcol_pca[:,1]
all_data['_vcol_nulls'] = all_data[vcols].isnull().sum(axis=1)

all_data.drop(vcols, axis=1, inplace=True)
import datetime

START_DATE = '2017-12-01'
startdate = datetime.datetime.strptime(START_DATE, "%Y-%m-%d")
all_data['Date'] = all_data['TransactionDT'].apply(lambda x: (startdate + datetime.timedelta(seconds=x)))
all_data['_weekday'] = all_data['Date'].dt.dayofweek
all_data['_hour'] = all_data['Date'].dt.hour
all_data['_day'] = all_data['Date'].dt.day

all_data['_weekday'] = all_data['_weekday'].astype(str)
all_data['_hour'] = all_data['_hour'].astype(str)
all_data['_weekday__hour'] = all_data['_weekday'] + all_data['_hour']

cnt_day = all_data['_day'].value_counts()
cnt_day = cnt_day / cnt_day.mean()
all_data['_count_rate'] = all_data['_day'].map(cnt_day.to_dict())

all_data.drop(['TransactionDT','Date','_day'], axis=1, inplace=True)
all_data['_P_emaildomain__addr1'] = all_data['P_emaildomain'] + '__' + all_data['addr1'].astype(str)
all_data['_card1__card2'] = all_data['card1'].astype(str) + '__' + all_data['card2'].astype(str)
all_data['_card1__addr1'] = all_data['card1'].astype(str) + '__' + all_data['addr1'].astype(str)
all_data['_card2__addr1'] = all_data['card2'].astype(str) + '__' + all_data['addr1'].astype(str)
all_data['_card12__addr1'] = all_data['_card1__card2'] + '__' + all_data['addr1'].astype(str)
all_data['_card_all__addr1'] = all_data['_card1__card2'] + '__' + all_data['addr1'].astype(str)
all_data['_amount_decimal'] = ((all_data['TransactionAmt'] - all_data['TransactionAmt'].astype(int)) * 1000).astype(int)
all_data['_amount_decimal_len'] = all_data['TransactionAmt'].apply(lambda x: len(re.sub('0+$', '', str(x)).split('.')[1]))
all_data['_amount_fraction'] = all_data['TransactionAmt'].apply(lambda x: float('0.'+re.sub('^[0-9]|\.|0+$', '', str(x))))
all_data[['TransactionAmt','_amount_decimal','_amount_decimal_len','_amount_fraction']].head(10)
cols = ['ProductCD','card1','card2','card5','card6','P_emaildomain','_card_all__addr1']
#,'card3','card4','addr1','dist2','R_emaildomain'

# amount mean&std
for f in cols:
    all_data[f'_amount_mean_{f}'] = all_data['TransactionAmt'] / all_data.groupby([f])['TransactionAmt'].transform('mean')
    all_data[f'_amount_std_{f}'] = all_data['TransactionAmt'] / all_data.groupby([f])['TransactionAmt'].transform('std')
    all_data[f'_amount_pct_{f}'] = (all_data['TransactionAmt'] - all_data[f'_amount_mean_{f}']) / all_data[f'_amount_std_{f}']

# freq encoding
for f in cols:
    vc = all_data[f].value_counts(dropna=False)
    all_data[f'_count_{f}'] = all_data[f].map(vc)
print('features:', all_data.shape[1])
_='''
cat_cols = ['ProductCD','card1','card2','card3','card4','card5','card6','addr1','addr2','P_emaildomain','R_emaildomain',
            'M1','M2','M3','M4','M5','M6','M7','M8','M9','DeviceType','DeviceInfo'] + [f'id_{i}' for i in range(12,39)]
'''
cat_cols = [f'id_{i}' for i in range(12,39)]
for i in cat_cols:
    if i in all_data.columns:
        all_data[i] = all_data[i].astype(str)
        all_data[i].fillna('unknown', inplace=True)

enc_cols = []
for i, t in all_data.loc[:, all_data.columns != 'isFraud'].dtypes.iteritems():
    if t == object:
        enc_cols.append(i)
        #df = pd.concat([df, pd.get_dummies(df[i].astype(str), prefix=i)], axis=1)
        #df.drop(i, axis=1, inplace=True)
        all_data[i] = pd.factorize(all_data[i])[0]
        #all_data[i] = all_data[i].astype('category')
print(enc_cols)
X_train = all_data[all_data['isFraud'].notnull()]
X_test = all_data[all_data['isFraud'].isnull()].drop('isFraud', axis=1)
Y_train = X_train.pop('isFraud')
del all_data
### %time

import lightgbm as lgb

params={'learning_rate': 0.01,
        'objective': 'binary',
        'metric': 'auc',
        'num_threads': -1,
        'num_leaves': 256,
        'verbose': 1,
        'random_state': 42,
        'bagging_fraction': 1,
        'feature_fraction': 0.85
       }

oof_preds = np.zeros(X_train.shape[0])
sub_preds = np.zeros(X_test.shape[0])

clf = lgb.LGBMClassifier(**params, n_estimators=3000)
clf.fit(X_train, Y_train)
oof_preds = clf.predict_proba(X_train, num_iteration=clf.best_iteration_)[:,1]
sub_preds = clf.predict_proba(X_test, num_iteration=clf.best_iteration_)[:,1]
fpr, tpr, thresholds = metrics.roc_curve(Y_train, oof_preds)
auc = metrics.auc(fpr, tpr)

plt.plot(fpr, tpr, label='ROC curve (area = %.3f)'%auc)
plt.legend()
plt.title('ROC curve')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.grid(True)
# Plot feature importance
feature_importance = clf.feature_importances_
feature_importance = 100.0 * (feature_importance / feature_importance.max())
sorted_idx = np.argsort(feature_importance)
sorted_idx = sorted_idx[len(feature_importance) - 50:]
pos = np.arange(sorted_idx.shape[0]) + .5

plt.figure(figsize=(10,12))
plt.barh(pos, feature_importance[sorted_idx], align='center')
plt.yticks(pos, X_train.columns[sorted_idx])
plt.xlabel('Relative Importance')
plt.title('Variable Importance')
plt.show()
submission = pd.DataFrame()
submission['TransactionID'] = X_test_id
submission['isFraud'] = sub_preds
submission.to_csv('submission.csv', index=False)