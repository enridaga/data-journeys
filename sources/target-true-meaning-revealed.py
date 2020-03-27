
import numpy as np
import pandas as pd
pd.set_option('display.float_format', '{:.10f}'.format)

train = pd.read_csv('../input/train.csv')
historical_transactions = pd.read_csv('../input/historical_transactions.csv')
new_merchant_transactions = pd.read_csv('../input/new_merchant_transactions.csv')
new_merchant_transactions['purchase_amount_new'] = np.round(new_merchant_transactions['purchase_amount'] / 0.00150265118 + 497.06,2)
historical_transactions['purchase_amount_new'] = np.round(historical_transactions['purchase_amount'] / 0.00150265118 + 497.06,2)
train['target_raw'] = 2**train['target']
merchant_counts = historical_transactions.groupby(['card_id'])['merchant_id'].nunique().reset_index(name = 'merchant_n')
one_merchant = merchant_counts[merchant_counts['merchant_n']==1].reset_index(drop=True)
dat = historical_transactions.loc[historical_transactions['card_id'].isin(one_merchant['card_id'])]
dat = dat.loc[~dat['card_id'].isin(new_merchant_transactions['card_id'])]
dat = dat.loc[dat.merchant_id=='M_ID_fc7d7969c3'].reset_index(drop=True)
historical_transactions[historical_transactions.card_id=='C_ID_b3c7ff9e19'].sort_values('purchase_date')
new_merchant_transactions[new_merchant_transactions.card_id=='C_ID_b3c7ff9e19'].sort_values('purchase_date')
historical_transactions.loc[historical_transactions.merchant_id=='M_ID_fc7d7969c3'].groupby('purchase_amount_new')['card_id'].count()
dat = dat.merge(train, on = 'card_id')
dat.head()
dat.groupby('target_raw')['card_id'].count().reset_index(name='n')
dat.loc[dat['target_raw']==1,['card_id','purchase_date','purchase_amount_new','target_raw']].sort_values(['card_id','purchase_date'])
prices = [19.90, 22.90, 27.90, 29.90, 37.90]
sorted({ i/j for j in prices for i in prices})
dat.groupby('target_raw')['card_id'].count().reset_index(name='n')
dat[dat.card_id=='C_ID_2c8d99614f'].sort_values('purchase_date')