
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
appDf = pd.read_csv("/kaggle/input/google-play-store-apps/googleplaystore.csv")
reviewDf = pd.read_csv("/kaggle/input/google-play-store-apps/googleplaystore_user_reviews.csv")
appDf.head()
for i in range(len(appDf['Reviews'])):
    if 'M' in appDf['Reviews'][i]:
        appDf['Reviews'][i] = appDf['Reviews'][i][:-1]
        #print(appDf['Reviews'][i])
        appDf['Reviews'][i] = float(appDf['Reviews'][i]) * 1e6
    else:
        appDf['Reviews'][i] = float(appDf['Reviews'][i]) 
appDf.sort_values(by = 'Reviews', ascending = False).iloc[0]['App']
freeApps = len(appDf[appDf['Type'] == 'Free'])
print("No. of free apps = ", freeApps)
paidApps = len(appDf) - freeApps
print("No. of paid apps = ", paidApps)
type(appDf['Installs'][0])
for i in range(len(appDf['Installs'])):
    if("+" in appDf['Installs'][i]):
        appDf['Installs'][i] =appDf['Installs'][i][:-1]
        while(',' in appDf['Installs'][i]):
            idx = appDf['Installs'][i].index(',')
            appDf['Installs'][i] = appDf['Installs'][i][:idx] + appDf['Installs'][i][idx+1:]
        appDf['Installs'][i] = int(appDf['Installs'][i])
    if(appDf['Installs'][i] == 'Free'):
        appDf['Installs'][i] = 0
    appDf['Installs'][i] = int(appDf['Installs'][i])
appDf.head()
appInst = appDf.sort_values(by = 'Installs', ascending = False).head(10)
import matplotlib.pyplot as plt
appInst
