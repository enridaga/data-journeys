
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import matplotlib.image as mpimg

import warnings
warnings.filterwarnings("ignore")

import seaborn as sns

from functools import reduce


import os 
import gc
import psutil

#%matplotlib inline
print(os.listdir("../input/nfl-playing-surface-analytics/"))
#InjuryRecord = pd.read_csv("../input/nfl-playing-surface-analytics/InjuryRecord.csv")
#PlayList = pd.read_csv("../input/nfl-playing-surface-analytics/PlayList.csv")
#PlayerTrackData = pd.read_csv("../input/nfl-playing-surface-analytics/PlayerTrackData.csv")
# Sample 100 rows of data to determine dtypes.
#df_test = pd.read_csv("../input/nfl-playing-surface-analytics/PlayerTrackData.csv", nrows=100)

#float_cols = [c for c in df_test if df_test[c].dtype == "float64"]
#float32_cols = {c: np.float32 for c in float_cols}
#float16_cols = {c: np.float16 for c in float_cols}

#PlayerTrackData = pd.read_csv("../input/nfl-playing-surface-analytics/PlayerTrackData.csv", engine='c', dtype=float16_cols)
def reduce_mem_usage(df):
    """ iterate through all the columns of a dataframe and modify the data type
        to reduce memory usage.        
    """
    start_mem = df.memory_usage().sum() / 1024**2
    print('Memory usage of dataframe is {:.2f} MB'.format(start_mem))
    
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
        else:
            df[col] = df[col].astype('category')

    end_mem = df.memory_usage().sum() / 1024**2
    print('Memory usage after optimization is: {:.2f} MB'.format(end_mem))
    print('Decreased by {:.1f}%'.format(100 * (start_mem - end_mem) / start_mem))
    
    return df


def import_data(file):
    """create a dataframe and optimize its memory usage"""
    df = pd.read_csv(file, parse_dates=True, keep_date_col=True)
    df = reduce_mem_usage(df)
    return df
InjuryRecord = import_data("../input/nfl-playing-surface-analytics/InjuryRecord.csv")
PlayList = import_data("../input/nfl-playing-surface-analytics/PlayList.csv")
PlayerTrackData = import_data("../input/nfl-playing-surface-analytics/PlayerTrackData.csv")
#float_types = ["float16", "float32", "float64"]
#for it in float_types:
#    print(np.finfo(it))
#InjuryRecord.info(memory_usage='deep')#memory usage: 35.7 KB
#PlayList.info(memory_usage='deep') #memory usage: 162.2 MB
PlayerTrackData.info(memory_usage='deep')#memory usage: 11.1 GB
# memory usage: 9.1 GB after converting float64 to float32
# memory usage: 8.1 GB after converting float64 to float32
# memory usage: 3.6 GB after converting object to category for "PlayKey" column
# memory usage: 1.4 GB after converting object to category for "event" column
#PlayerTrackData["PlayKey"] = PlayerTrackData.PlayKey.astype('category')
# memory usage: 3.6 GB after converting object to category for "PlayKey" column
#PlayerTrackData["event"] = PlayerTrackData.event.astype('category')
print("available RAM:", psutil.virtual_memory())
gc.collect()
print("available RAM:", psutil.virtual_memory())
#PlayerTrackData.describe()
print(InjuryRecord.columns)
print(PlayList.columns)
print(PlayerTrackData.columns)
print(PlayerTrackData.shape)
print(PlayList.shape)
print(InjuryRecord.shape)
PlayerTrackData.head(4)
PlayList.head(2)
InjuryRecord.head(3)
sns.catplot(x="BodyPart", data=InjuryRecord, kind="count")
sns.catplot(x="Surface", data=InjuryRecord, kind="count")
sns.catplot(x="RosterPosition", data=PlayList, kind="count")
plt.gcf().set_size_inches(16, 8)
sns.catplot(x="PlayerDay", data=PlayList, kind="count")
plt.gcf().set_size_inches(100, 8)
sns.catplot(x="StadiumType", data=PlayList, kind="count")
plt.gcf().set_size_inches(50, 8)
sns.catplot(x="Weather", data=PlayList, kind="count")
plt.gcf().set_size_inches(450, 8)
sns.catplot(x="PlayType", data=PlayList, kind="count")
plt.gcf().set_size_inches(16, 8)
sns.catplot(x="Position", data=PlayList, kind="count")
plt.gcf().set_size_inches(16, 8)
sns.catplot(x="PlayerGamePlay", data=PlayList, kind="count")
plt.gcf().set_size_inches(50, 8)
sns.catplot(x="PositionGroup", data=PlayList, kind="count")
plt.gcf().set_size_inches(16, 8)
PlayerTrackData.event.value_counts()
print(PlayerTrackData.event.unique())
InjuryRecord.BodyPart.value_counts()
sns.catplot(x="BodyPart", data=InjuryRecord, kind="count")
plt.gcf().set_size_inches(16, 8)
InjuryRecord.DM_M1.value_counts()
sns.catplot(x="DM_M1", hue='BodyPart', data=InjuryRecord, kind="count")
InjuryRecord.DM_M7.value_counts()
#sns.catplot(x="DM_M7", hue='BodyPart', data=InjuryRecord, kind="count")
sns.catplot(x="DM_M7", hue='BodyPart', data=InjuryRecord[(InjuryRecord.DM_M7==1)], kind="count")

InjuryRecord.DM_M28.value_counts()
sns.catplot(x="DM_M28", hue='BodyPart', data=InjuryRecord[(InjuryRecord.DM_M28==1)], kind="count")
InjuryRecord.DM_M42.value_counts()
sns.catplot(x="DM_M42", hue='BodyPart', data=InjuryRecord[(InjuryRecord.DM_M42==1)], kind="count")
PlayList.StadiumType = PlayList.StadiumType.str.lower()
PlayList.StadiumType = PlayList.StadiumType.str.strip()
PlayList.StadiumType = PlayList.StadiumType.str.rstrip()
PlayList.StadiumType = PlayList.StadiumType.str.lstrip()
stadium_name = PlayList.StadiumType.unique()
print(len(stadium_name))
# printing the list using * and sep operator 
print("printing lists separated by commas") 
  
print(*stadium_name, sep = "/") 
#Similar word array:
array_outdoors = ["outdoor", "oudoor", "ourdoor",
                 "outddors", "outdor", "outside", "heinz field"]
array_indoors = ["indoor", "indoor", "indoor", "retractable roof"]
array_open = [" open", "outdoor retr roof-open", "retr. roof-open",
             "open roof", "retr. roof - open", " open roof", " open roof ", "indoor, open roof", "domed, open"] #1 issue
array_closed = ["closed dome"," closed","  closed", "domed", "dome", "retr. roof-closed",
               "bowl", "closed", "retr. roof closed", " roof closed","roof closed", "retr. roof - closed", "  roof closed",
               "domed, closed", "indoor, roof closed", "dome, closed"]
#array_
#df.replace(0, 5)
PlayList["StadiumType"] = PlayList["StadiumType"].replace(array_outdoors, "outdoors" )
PlayList["StadiumType"] = PlayList["StadiumType"].replace(array_indoors, "indoors" )
PlayList["StadiumType"] = PlayList["StadiumType"].replace(array_open, "open" )
PlayList["StadiumType"] = PlayList["StadiumType"].replace(array_closed, "closed" )
InjuryRecord.head(2)
PlayList.head(2)
PlayList.loc[PlayList['PlayKey']=="39873-4-32"]
#result = pd.merge(InjuryRecord,
#                 PlayList,
#                 on='PlayKey', 
#                 how='left')
Injury_games_play = InjuryRecord.merge(PlayList,
                  on='PlayKey',
                  how='left')
Injury_games_play.tail()
Injury_games_play.isna().sum()
#Drop rows with Nan in PlayKey column
Injury_games_play = Injury_games_play[pd.notnull(Injury_games_play['PlayKey'])]
Injury_games_play.tail()
Injury_games_play.shape
Injury_games_play.columns
sns.catplot(x="BodyPart", hue='StadiumType', data=Injury_games_play, kind="count")
Injury_games_play.isna().sum()
Injury_games_play=Injury_games_play.dropna()
data=Injury_games_play[['BodyPart', 'Surface', 'RosterPosition', 'PlayerDay', 'PlayerGame', 'StadiumType', 'FieldType',
       'Temperature', 'Weather', 'PlayType', 'PlayerGamePlay', 'Position',
       'PositionGroup']]

ax = sns.pairplot(data, hue='BodyPart', kind="reg")
sns.catplot(x="BodyPart", hue='Surface', data=Injury_games_play, kind="count")
Injury_games_play.head()
Injury_games_play.info()
Injury_games_play.loc[Injury_games_play['PlayerKey_x'] == 39873]
Injury_games_play.shape
print(InjuryRecord.shape)
print(PlayList.shape)
print(PlayerTrackData.shape)
PlayList.head(4)
dfs = [PlayList, InjuryRecord, PlayerTrackData]
df_final = reduce(lambda left,right: pd.merge(left,right,on='PlayKey'), dfs)
df_final.shape
print(df_final.head(10))
df_final.tail(10)
df_final.isna().sum()
