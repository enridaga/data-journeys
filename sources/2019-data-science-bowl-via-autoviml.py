
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from tqdm import tqdm_notebook
from tqdm import tqdm
from collections import Counter
import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
url1 = '../input/ram-reduce/reduce_train.csv'
url2 = '../input/ram-reduce/reduce_test.csv'
reduce_train = pd.read_csv(url1,index_col=None)
reduce_test = pd.read_csv(url2,index_col=None)
print(reduce_train.shape,reduce_test.shape)
target='accuracy_group'
select = ['session_title', 'Cart Balancer (Assessment)_3121', 'Bird Measurer (Assessment)_3120', 'Bird Measurer (Assessment)_3021', 'e4f1efe6', 'acc_Chest Sorter (Assessment)', 'accumulated_accuracy', '7525289a', 'Scrub-A-Dub_3021', 'a52b92d5', 'Crystal Caves - Level 3', 'acc_Bird Measurer (Assessment)', '3afb49e6', 'Tree Top City - Level 3', 'Chow Time_2030', 'Clip', 'b74258a0', 'Mushroom Sorter (Assessment)_3121', 'acc_Mushroom Sorter (Assessment)', 'c7f7f0e1', '5290eab1', '3393b68b', 'Mushroom Sorter (Assessment)_4070', '0a08139c', '8f094001', 'a5be6304', 'Tree Top City - Level 3_2000', 'Mushroom Sorter (Assessment)_3010', 'Chest Sorter (Assessment)_2010', 'ecaab346', 'Cart Balancer (Assessment)_3110', 'Fireworks (Activity)_2000', 'c51d8688', '6c930e6e', '070a5291', 'Dino Drink_3020', '04df9b66', 'Happy Camel_2030', '222660ff', 'Mushroom Sorter (Assessment)_3120', 'Scrub-A-Dub_2000', 'Chest Sorter (Assessment)_2030', 'Cauldron Filler (Assessment)_3120', 'All Star Sorting_3121', '65a38bf7', 'Cart Balancer (Assessment)_3021', 'Tree Top City - Level 2', 'All Star Sorting_2000', 'Pan Balance_3120', 'Air Show_3121', 'Chest Sorter (Assessment)_4030', 'c7fe2a55', 'Bird Measurer (Assessment)_2000', '37937459', 'Mushroom Sorter (Assessment)_2010', 'ab4ec3a4', 'Mushroom Sorter (Assessment)_4100', 'Egg Dropper (Activity)_2020', 'Cart Balancer (Assessment)_2020', 'Scrub-A-Dub_2050', 'acc_Cauldron Filler (Assessment)', 'Egg Dropper (Activity)_2000', 'Crystal Caves - Level 2', 'Chest Sorter (Assessment)_4025', 'Ordering Spheres_2000', 'Cart Balancer (Assessment)_2000', 'f54238ee', '3a4be871', 'Bird Measurer (Assessment)_2020', 'acc_Cart Balancer (Assessment)', 'Chow Time_4035', 'Air Show_2000', 'Mushroom Sorter (Assessment)_2035', 'd3640339', '77c76bc5', 'Air Show_2030', 'installation_title_nunique', 'Bubble Bath_4090', '92687c59']
len(select)
from catboost import CatBoostClassifier, CatBoostRegressor
cat =  CatBoostClassifier(verbose=0,n_estimators=1000,
                                random_state=99,one_hot_max_size=100,
                                loss_function='MultiClass', eval_metric='AUC',
                                subsample=0.7,bootstrap_type='Bernoulli',
                               early_stopping_rounds=25,boosting_type='Plain')

cat.fit(reduce_train[select],reduce_train[target])
testm = cat.predict(reduce_test[select]).astype(int)
testm.shape
subm = pd.DataFrame(index=range(1000))
subm['installation_id'] = reduce_test['installation_id'].values[:1000]
subm[target] = testm
print(subm.shape)
subm.head()
subm.to_csv('submission.csv',index=False)
