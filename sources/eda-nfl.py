
# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.
train = pd.read_csv('/kaggle/input/nfl-big-data-bowl-2020/train.csv')
train.head()
train.shape
import matplotlib.pyplot as plt
train.columns
len(train['GameId'].unique())
len(train['PlayId'].unique())
pla_by_game = list(map(lambda x:len(x),train.groupby(['GameId'])['PlayId'].unique()))
plt.hist(pla_by_game,bins=100)
plt.show()
train['Yards'].hist(bins=100)
plt.show()
numeric_cols = ['X','Y','S','A','Dis','Orientation','Dir','Distance','HomeScoreBeforePlay','VisitorScoreBeforePlay','DefendersInTheBox',
               'Temperature','Humidity']
categorical_cols = ['Team','Season','YardLine','Quarter','GameClock','PossessionTeam','Down','FieldPosition','OffenceFormation','OffencePersonnel',
                    'DefensePersonnel','PlayDirection','PlayerHeight','PlayerWeight','Position','Week','Stadium','Location','StadiumType','Turf',
                   'GameWeather','WindSpeed','WindDirection']
target = ['Yards']
id_cols = ['GameId','PlayId','NfId','DisplayName','JerseyNumber','NflIdRusher','TimeHandoff','TimeSnap','PlayerCollegeName','VisitorTeamAbbr']
w=10
h=10
fig=plt.figure(figsize=(20, 20))
columns = 4
rows = 4
for i in range(1,len(numeric_cols)+1):
    plt.xlabel(numeric_cols[i-1])
    plt.ylabel('Yards')
    ax = fig.add_subplot(rows, columns, i)
    ax.scatter(train[numeric_cols[i-1]],train['Yards'])
plt.show()
#for col in numeric_cols:
#    plt.scatter(train[col],train['Yards'])
#    plt.show()
