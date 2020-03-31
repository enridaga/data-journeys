
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
print(os.listdir("../input"))
import sys
import matplotlib.pyplot as plt
### matplotlib inline
### git clone https://github.com/recursionpharma/rxrx1-utils
print ('rxrx1-utils cloned!')
### ls
sys.path.append('rxrx1-utils')
import rxrx.io as rio
t = rio.load_site('train', 'RPE-05', 3, 'D19', 2)
t.shape
fig, axes = plt.subplots(2, 3, figsize=(24, 16))

for i, ax in enumerate(axes.flatten()):
  ax.axis('off')
  ax.set_title('channel {}'.format(i + 1))
  _ = ax.imshow(t[:, :, i], cmap='gray')
x = rio.convert_tensor_to_rgb(t)
x.shape
plt.figure(figsize=(8, 8))
plt.axis('off')

_ = plt.imshow(x)
y = rio.load_site_as_rgb('train', 'HUVEC-08', 4, 'K09', 1)

plt.figure(figsize=(8, 8))
plt.axis('off')

_ = plt.imshow(y)
md = rio.combine_metadata()
md.head()
import seaborn as sns
md.head(10)
md.index
for i in md.columns:
    print (">> ",i,"\t", md[i].unique())
for col in ['cell_type', 'dataset', 'experiment', 'plate',  'site', 'well_type']:
    print (col)
    print (md[col].value_counts())
    sns.countplot(y = col,
              data = md,
              order = md[col].value_counts().index)
    plt.show()
    
missing_values_count = md.isnull().sum()
missing_values_count
md = md.fillna(0)
md.head()
train_df = md[md['dataset'] == 'train']
test_df = md[md['dataset'] == 'test']

train_df.shape, test_df.shape
plt.figure(figsize=(16,6))
plt.title("Distribution of SIRNA in the train and test set")
sns.distplot(train_df.sirna,color="green", kde=True,bins='auto', label='train')
sns.distplot(test_df.sirna,color="blue", kde=True, bins='auto', label='test')
plt.legend()
plt.show()
feat1 = 'sirna'
fig = plt.subplots(figsize=(15, 5))

# train
plt.subplot(1, 2, 1)
sns.kdeplot(train_df[feat1][train_df['site'] == 1], shade=False, color="b", label = 'site 1')
sns.kdeplot(train_df[feat1][train_df['site'] == 2], shade=False, color="r", label = 'site 2')
plt.title(feat1)
plt.xlabel('Feature Values')
plt.ylabel('Probability')

# test
plt.subplot(1, 2, 2)
sns.kdeplot(test_df[feat1][test_df['site'] == 1], shade=False, color="b", label = 'site 1')
sns.kdeplot(test_df[feat1][test_df['site'] == 2], shade=False, color="r", label = 'site 2')
plt.title(feat1)
plt.xlabel('Feature Values')
plt.ylabel('Probability')
plt.show()

# Prevent: Output path '/rxrx1-utils/.git/logs/refs/remotes/origin/HEAD' contains too many nested subdirectories (max 6)
### rm -r  rxrx1-utils
### ls