
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
from fastai.vision import *
import fastai
import imageio
from sklearn.model_selection import train_test_split

defaults.device = torch.device('cuda')
train_df = pd.read_csv('../input/Kannada-MNIST/train.csv')
train_df.head()
test_df = pd.read_csv('../input/Kannada-MNIST/test.csv')
test_df.head()
tmp_df = pd.read_csv('../input/Kannada-MNIST/sample_submission.csv')
tmp_df.head()
dig_df = pd.read_csv('../input/Kannada-MNIST/Dig-MNIST.csv')
dig_df.head()
new_df = train_df.append(dig_df, ignore_index=True)###
new_df
def save_imgs(path:Path, imgs, labels):
    imgs =  np.array(imgs).reshape(-1,28,28)
    imgs = np.stack((imgs,)*3, axis=-1)
    if len(labels) > 0:
        labels = np.array(labels)
        
    for i in range(len(imgs)):
        if len(labels) > 0:
            imageio.imsave(str(path/str(labels[i])/(str(i)+'.jpg') ), imgs[i])
        else:
            imageio.imsave(str(path/(str(i)+'.jpg') ), imgs[i]) 
def get_imgs_labels(path:Path, df):
    labels = df['label']
    path.mkdir(parents=True,exist_ok=True)
    imgs = df.loc[:, 'pixel0':'pixel783']
    return imgs, labels
train_path1 = Path('../input1/train')
valid_path1 = Path('../input1/valid')
test_path1 = Path('../input1/test')
imgs1, labels1 = get_imgs_labels(train_path1, new_df)
X_train1, X_test1, y_train1, y_test1 \
= train_test_split(imgs1, labels1, test_size=0.15, random_state=17, stratify=labels1)
len(X_train1), len(y_train1), len(X_test1), len(y_test1)
def create_subdirs(path:Path, num_dirs):
    for i in range(num_dirs):
        Path(path/str(i)).mkdir(parents=True,exist_ok=True)
create_subdirs(train_path1, labels1.nunique())
create_subdirs(valid_path1, labels1.nunique())
### mkdir ../input1/test
save_imgs(train_path1, X_train1, y_train1)
save_imgs(valid_path1, X_test1, y_test1)
save_imgs(test_path1, np.array(test_df.drop(['id'], axis=1)), [])
tfms = get_transforms(do_flip=False, flip_vert=False, max_rotate= 5.0, 
                      max_zoom=1.1, max_lighting=0.15, max_warp=0.15,
                      p_affine=0.75, p_lighting=0.75)
db1 = (ImageList.from_folder('../input1/') 
        .split_by_folder()          
        .label_from_folder()        
        .add_test_folder(test_path1)
        .transform(tfms, size=64)
        .databunch(bs=512)
        .normalize(mnist_stats))
db1.show_batch(rows=4, figsize=(6,6))
learn1 = cnn_learner(db1, models.resnet50, metrics=[error_rate, accuracy], model_dir="/tmp/model/", pretrained=False)
learn1.unfreeze()
learn1.lr_find()
learn1.recorder.plot(suggestion=True)
learn1.fit_one_cycle(50, max_lr=slice(5e-4, 5e-3, 5e-2))
learn1.save('unfreeze_resnet50_learn1')
learn1.recorder.plot_losses()
interp = ClassificationInterpretation.from_learner(learn1)
interp.plot_confusion_matrix()
interp.plot_top_losses(9, figsize=(6, 6))
test_df.drop('id',axis = 'columns',inplace = True)
tmp_df = tmp_df[0:0]
img_arr = np.array(test_df)
for i in range(len(img_arr)):
    img_data = img_arr[i].reshape(28,28)/255.
    img_data = np.stack((img_data,)*3, axis=0)
    img = Image(FloatTensor(img_data))
    tmp_df.loc[i]=[i+1,int(learn1.predict(img)[1])]
tmp_df
tmp_df['id'] = tmp_df['id'].apply(lambda k: k-1)
tmp_df
tmp_df.to_csv('submission.csv',index=False)