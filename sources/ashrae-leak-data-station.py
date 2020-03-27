
import pandas as pd
from pathlib import Path
from matplotlib import pyplot as plt
import os
os.listdir('../input/')
train_df = pd.read_csv('../input/ashrae-energy-prediction/train.csv')
train_df['timestamp'] = pd.to_datetime(train_df['timestamp'])
def plot_meter(train, leak, start=0, n=100, bn=10):
    for bid in leak.building_id.unique()[:bn]:    
        tr = train[train.building_id == bid]
        lk = leak[leak.building_id == bid]
        
        for m in lk.meter.unique():
            plt.figure(figsize=[10,2])
            trm = tr[tr.meter == m]
            lkm = lk[lk.meter == m]
            
            plt.plot(trm.timestamp[start:start+n], trm.meter_reading.values[start:start+n], label='train')    
            plt.plot(lkm.timestamp[start:start+n], lkm.meter_reading.values[start:start+n], '--', label='leak')
            plt.title('bid:{}, meter:{}'.format(bid, m))
            plt.legend()
# load site 0 data
ucf_root = Path('../input/ashrae-ucf-spider-and-eda-full-test-labels')
leak0_df = pd.read_pickle(ucf_root/'site0.pkl') 
leak0_df['meter_reading'] = leak0_df.meter_reading_scraped
leak0_df.drop(['meter_reading_original','meter_reading_scraped'], axis=1, inplace=True)
leak0_df.fillna(0, inplace=True)
leak0_df.loc[leak0_df.meter_reading < 0, 'meter_reading'] = 0
print(len(leak0_df))
leak0_df.head()
leak0_df.tail()
plot_meter(train_df, leak0_df, start=5000)
# load site 1 data
ucl_root = Path('../usr/lib/ucl_data_leakage_episode_2')
leak1_df = pd.read_pickle(ucl_root/'site1.pkl') 
leak1_df['meter_reading'] = leak1_df.meter_reading_scraped
leak1_df.drop(['meter_reading_scraped'], axis=1, inplace=True)
leak1_df.fillna(0, inplace=True)
leak1_df.loc[leak1_df.meter_reading < 0, 'meter_reading'] = 0
print(len(leak1_df))
leak1_df.head()
leak1_df.tail()
plot_meter(train_df, leak1_df, start=0)
# load site 2 data
leak2_df = pd.read_csv('/kaggle/input/asu-buildings-energy-consumption/asu_2016-2018.csv')
leak2_df['timestamp'] = pd.to_datetime(leak2_df['timestamp'])

leak2_df.fillna(0, inplace=True)
leak2_df.loc[leak2_df.meter_reading < 0, 'meter_reading'] = 0

leak2_df = leak2_df[leak2_df.building_id!=245] # building 245 is missing now.

#leak2_df = leak2_df[leak2_df.timestamp.dt.year > 2016]
print(len(leak2_df))
leak2_df.head()
leak2_df.tail()
plot_meter(train_df, leak2_df, start=0)
# load site 4 data
# its looks better to use threshold ...
leak4_df = pd.read_csv('../input/ucb-data-leakage-site-4/site4.csv')

leak4_df['timestamp'] = pd.to_datetime(leak4_df['timestamp'])
leak4_df.rename(columns={'meter_reading_scraped': 'meter_reading'}, inplace=True)
leak4_df.fillna(0, inplace=True)
leak4_df.loc[leak4_df.meter_reading < 0, 'meter_reading'] = 0
leak4_df['meter'] = 0

print('before remove dupilicate', leak4_df.duplicated(subset=['building_id','timestamp']).sum())
leak4_df.drop_duplicates(subset=['building_id','timestamp'],inplace=True)
print('after remove dupilicate', leak4_df.duplicated(subset=['building_id','timestamp']).sum())
print(len(leak4_df))
leak4_df.head()
leak4_df.tail() # its include 2019. i will delete them later
len(leak4_df.building_id.unique())
plot_meter(train_df, leak4_df, start=0)
train_df[train_df.building_id == 621].timestamp.min() # some train data is missing
# this data does not include 2016.
leak15_df = pd.read_csv('../input/ashrae-site15-cornell/site15_leakage.csv')

leak15_df['timestamp'] = pd.to_datetime(leak15_df['timestamp'])
leak15_df.fillna(0, inplace=True)
leak15_df.loc[leak15_df.meter_reading < 0, 'meter_reading'] = 0

print(leak15_df.duplicated().sum())
print(len(leak15_df))
leak15_df.head()
leak15_df.tail()
df = pd.concat([leak0_df, leak1_df, leak2_df, leak4_df, leak15_df])
df.drop('score', axis=1, inplace=True)
df = df[(df.timestamp.dt.year >= 2016) & (df.timestamp.dt.year < 2019)]
df.reset_index(inplace=True, drop=True)
print(len(df))
df.timestamp.min(), df.timestamp.max() 
df.head()
df.to_feather('leak.feather')
leak_df = pd.read_feather('leak.feather')
leak_df.head()
leak_df.meter.value_counts()
len(leak_df.building_id.unique())
# Wow!! What do you think ? it's really huge now !! 
len(leak_df) / len(train_df)