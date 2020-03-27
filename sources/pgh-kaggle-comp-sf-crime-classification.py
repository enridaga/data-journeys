
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
sfc = pd.read_csv('/kaggle/input/sf-crime/train.csv.zip')
sfc.head()
pd.set_option('display.max_rows', None)
pd.options.display.max_rows
sfc.head()
sfc.Category.value_counts()
sfc[(sfc['Category'] == 'WEAPON LAWS')]['Descript'].value_counts()
sfc[(sfc['Category'] == 'GAMBLING')]['Address'].value_counts()
sfc[(sfc['Category'] == 'DRUNKENNESS')]['Dates'].value_counts()
# sfc[(sfc['Category'] == 'BURGLARY')]['Descript'].value_counts()
sfc[(sfc['Category'] == 'NON-CRIMINAL')]['Address'].value_counts()
sfc[(sfc['Category'] == 'NON-CRIMINAL') & (sfc['Resolution'].str.contains('ARREST'))]
# sfc[(sfc['Descript'] == 'LOST PROPERTY')]['Address'].value_counts()
# sfc[(sfc['Category'] == 'TREA')]['Descript'].value_counts()
# sfc.Descript.value_counts()
# count how many of each category, and each descript