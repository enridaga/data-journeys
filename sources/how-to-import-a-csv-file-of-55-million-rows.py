
import pandas as pd 
import dask.dataframe as dd
import os
from tqdm import tqdm

TRAIN_PATH = '../input/train.csv'
%%time
# Assume we only know that the csv file is somehow large, but not the exact size
# we want to know the exact number of rows

# Method 1, using file.readlines. Takes about 20 seconds.
with open(TRAIN_PATH) as file:
    n_rows = len(file.readlines())

print (f'Exact number of rows: {n_rows}')
%%time

# Method 2 by @danlester, using wc unix command. Takes only 3 seconds!
s = !wc -l {TRAIN_PATH}

# add one because the file isn't quite correctly formatted as a CSV, should have a final newline char
n_rows = int(s[0].split(' ')[0])+1

print (f'Exact number of rows: {n_rows}')
%%time

# Same method but more 'pythonic'
import subprocess

def file_len(fname):
    p = subprocess.Popen(['wc', '-l', fname], stdout=subprocess.PIPE, 
                                              stderr=subprocess.PIPE)
    result, err = p.communicate()
    if p.returncode != 0:
        raise IOError(err)
    return int(result.strip().split()[0])+1

n_rows = file_len(TRAIN_PATH)
print (f'Exact number of rows: {n_rows}')
# Peep at the training file header
df_tmp = pd.read_csv(TRAIN_PATH, nrows=5)
df_tmp.head()
df_tmp.info()
# Set columns to most suitable type to optimize for memory usage
traintypes = {'fare_amount': 'float32',
              'pickup_datetime': 'str', 
              'pickup_longitude': 'float32',
              'pickup_latitude': 'float32',
              'dropoff_longitude': 'float32',
              'dropoff_latitude': 'float32',
              'passenger_count': 'uint8'}

cols = list(traintypes.keys())
chunksize = 5_000_000 # 5 million rows at one go. Or try 10 million
%%time
df_list = [] # list to hold the batch dataframe

for df_chunk in tqdm(pd.read_csv(TRAIN_PATH, usecols=cols, dtype=traintypes, chunksize=chunksize)):
     
    # Neat trick from https://www.kaggle.com/btyuhas/bayesian-optimization-with-xgboost
    # Using parse_dates would be much slower!
    df_chunk['pickup_datetime'] = df_chunk['pickup_datetime'].str.slice(0, 16)
    df_chunk['pickup_datetime'] = pd.to_datetime(df_chunk['pickup_datetime'], utc=True, format='%Y-%m-%d %H:%M')
    
    # Can process each chunk of dataframe here
    # clean_data(), feature_engineer(),fit()
    
    # Alternatively, append the chunk to list and merge all
    df_list.append(df_chunk) 
# Merge all dataframes into one dataframe
train_df = pd.concat(df_list)

# Delete the dataframe list to release memory
del df_list

# See what we have loaded
train_df.info()
display(train_df.head())
display(train_df.tail())
%%time
# Save into feather format, about 1.5Gb. 
train_df.to_feather('nyc_taxi_data_raw.feather')
%%time
# load the same dataframe next time directly, without reading the csv file again!
train_df_new = pd.read_feather('nyc_taxi_data_raw.feather')
# print the dataframe info to verify we have indeed loaded the saved dataframe of 55 million rows
train_df_new.info()
%%time

# dask's read_csv takes no time at all!
ddf = dd.read_csv(TRAIN_PATH, usecols=cols, dtype=traintypes)
# no info?
ddf.info()
# nothing to describe?
ddf.describe()
%%time

# dask is lazy. It only works when it is asked explicitly with compute()
ddf.describe().compute()
%%time

# Again, it only works when it is asked :)
len(ddf)
del ddf
%%time

# using panda read_csv to read the entire file in one shot
df = pd.read_csv(TRAIN_PATH, usecols=cols, dtype=traintypes)
%%time
df['pickup_datetime'] = df['pickup_datetime'].str.slice(0, 16)
df['pickup_datetime'] = pd.to_datetime(df['pickup_datetime'], utc=True, format='%Y-%m-%d %H:%M')
df.info()
del df
%%time

# using dask read_csv followed by compute() to create a panda dataframe
ddf_pd = dd.read_csv(TRAIN_PATH, usecols=cols, dtype=traintypes).compute()

# RangeIndex is an optimized version of Int64Index that can represent a monotonic ordered set
# Source: https://pandas-docs.github.io/pandas-docs-travis/advanced.html#int64index-and-rangeindex
# Furthermore, without conversion, the resulting dataframe takes up more memory usage (1.9GB)
ddf_pd.index = pd.RangeIndex(start=0, stop=len(ddf_pd)) 
%%time
ddf_pd['pickup_datetime'] = ddf_pd['pickup_datetime'].str.slice(0, 16)
ddf_pd['pickup_datetime'] = pd.to_datetime(ddf_pd['pickup_datetime'], utc=True, format='%Y-%m-%d %H:%M')
%%time
ddf_pd.info()
del ddf_pd