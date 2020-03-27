
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
test = pd.read_csv('../input/data-science-bowl-2019/test.csv')

sample_submission = pd.read_csv('../input/data-science-bowl-2019/sample_submission.csv')
test_benchmark = test[(test.type=='Assessment') & ((test.event_code==4100) | (test.event_code==4100))]
test_benchmark['accuracy'] = test_benchmark['event_data'].str.contains('"correct":true')
accuracy = test_benchmark.groupby('installation_id')['accuracy'].mean().reset_index()
sub = sample_submission.merge(accuracy, how='left', on='installation_id')
test = test[['installation_id', 'title']].drop_duplicates('installation_id', keep='last')
test.reset_index(drop=True, inplace=True)

di = {'Bird Measurer (Assessment)': 0,
 'Cart Balancer (Assessment)': 3,
 'Cauldron Filler (Assessment)': 3,
 'Chest Sorter (Assessment)': 0,
 'Mushroom Sorter (Assessment)': 3}

test['accuracy_group'] = test.title.map(di)
sub = sub.merge(test, how='left', on='installation_id')
def accuracy_to_cls(x):
    if x == 0: 
        return 0
    elif x == 1:
        return 3
    elif x == 0.5:
        return 2
    else:
        return 1
sub['accuracy_group'] = sub['accuracy'].map(lambda x: accuracy_to_cls(x))

sub.loc[sub.accuracy.isnull(), 'accuracy_group']  = sub.loc[sub.accuracy.isnull(), 'accuracy_group_y']
sub[['installation_id', 'accuracy_group']].to_csv('submission.csv', index=False)
