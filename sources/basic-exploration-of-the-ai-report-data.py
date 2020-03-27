
# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input/the-2019-ai-index-report/AI INDEX 2019 PUBLIC DATA/4. The Economy'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

pd.set_option('max_rows',120)
# Any results you write to the current directory are saved as output.
df = pd.read_excel('/kaggle/input/the-2019-ai-index-report/AI INDEX 2019 PUBLIC DATA/5. Education/Udacity/Copy of Copy of Data to Share with Stanford, SG.xlsx',index_col=[0])
df['node_title'].value_counts()

df[df['node_title'] == 'Intro to Machine Learning']['enrollments']
df[df['node_title'] == ' Intro to AI']['enrollments']
df_jobs = pd.read_excel('/kaggle/input/the-2019-ai-index-report/AI INDEX 2019 PUBLIC DATA/4. The Economy/4.1. Jobs/Burning Glass/AI Postings USA monthly data.xlsx')
df_jobs.groupby('date')[['Postings']].sum()
df_jobs.groupby('date')[['Postings']].sum().pct_change(12)[-20:-1].plot()
