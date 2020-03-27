
import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler

import seaborn as sns
import matplotlib.pyplot as plt
#%matplotlib inline

from tqdm import tqdm_notebook
import re
import nltk
from nltk.stem import SnowballStemmer

from keras.preprocessing.text import Tokenizer

from keras.layers import Dense, Activation, Dropout, Flatten, Input
from keras.layers import Embedding, Conv1D, MaxPooling1D, GlobalMaxPooling1D, GlobalAveragePooling1D
from keras.models import Model, Sequential 
from keras.layers.recurrent import LSTM

import tensorflow as tf
from keras import backend as K

from keras.layers.merge import concatenate
from keras.utils import plot_model

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))
train = pd.read_csv('../input/train.csv', low_memory=False, index_col='id')
test = pd.read_csv('../input/test.csv', low_memory=False, index_col='id')

res = pd.read_csv('../input/resources.csv', low_memory=False, index_col='id')
train['is_train'] = 1
test['is_train'] = 0
df = pd.concat([train, test], axis=0)
sum_res = pd.pivot_table(res, index=res.index, aggfunc='sum', values=['price', 'quantity'])
mean_res = pd.pivot_table(res, index=res.index, aggfunc='mean', values=['price', 'quantity'])
median_res = pd.pivot_table(res, index=res.index, aggfunc='median', values=['price', 'quantity'])

df = pd.merge(df, sum_res,left_index=True, right_index=True)
df = pd.merge(df, mean_res,left_index=True, right_index=True, suffixes=('_sum', ''))
df = pd.merge(df, median_res,left_index=True, right_index=True, suffixes=('_mean', '_median'))
df.columns
cat_feature = ['school_state', 'teacher_prefix', 
               'project_subject_categories', 'project_subject_subcategories', 'project_grade_category']

target = 'project_is_approved'

text_feature = ['project_title', 'project_resource_summary', 'project_essay_1', 'project_essay_2', 'project_essay_3',
       'project_essay_4' ]

real_feature = ['teacher_number_of_previously_posted_projects', 'price_sum', 'quantity_sum', 'price_mean', 'quantity_mean',
       'price_median', 'quantity_median' ]


for i in cat_feature:
    df[i] = pd.factorize(df[i])[0]

trn_cat = df[cat_feature].values[:182080]
tst_cat = df[cat_feature].values[182080:]
SS = StandardScaler()
df_scale = SS.fit_transform(df[real_feature])

trn_real = df_scale[:182080]
tst_real = df_scale[182080:]
df_text = df[text_feature].fillna(' ')
df_text['full_text'] = ''
for f in text_feature:
    df_text['full_text'] = df_text['full_text'] + df_text[f]
stemmer = SnowballStemmer('english')

def clean(text):
    return re.sub('[!@#$:]', '', ' '.join(re.findall('\w{3,}', str(text).lower())))

def stem(text):
    return ' '.join([stemmer.stem(w) for w in text.split()])
df_text['full_text'] = df_text['full_text'].apply(lambda x: clean(x))
#df_text['full_text'] = df_text['full_text'].apply(lambda x: stem(x)) - don't think about it :)
max_words = 500 #more words for more accuracy
tokenizer = Tokenizer(num_words=max_words)
tokenizer.fit_on_texts(df_text['full_text'])

trn_text = tokenizer.texts_to_matrix(df_text['full_text'][:182080], mode='binary')
tst_text = tokenizer.texts_to_matrix(df_text['full_text'][182080:], mode='binary')
y = df[target].values[:182080]
len_cat = trn_cat.shape[1]
len_real = trn_real.shape[1]
len_text = trn_text.shape[1]


size_embedding = 5000
# categorical channel 
inputs1 = Input(shape=(len_cat,))
dense_cat_1 = Dense(256, activation='relu')(inputs1)
dense_cat_2 = Dense(128, activation='relu')(dense_cat_1)
dense_cat_3 = Dense(64, activation='relu')(dense_cat_2)
dense_cat_4 = Dense(32, activation='relu')(dense_cat_3)
flat1 = Dense(32, activation='relu')(dense_cat_4)



# real channel
inputs2 = Input(shape=(len_real,))
dense_real_1 = Dense(256, activation='relu')(inputs2)
dense_real_2 = Dense(128, activation='relu')(dense_real_1)
dense_real_3 = Dense(64, activation='relu')(dense_real_2)
dense_real_4 = Dense(32, activation='relu')(dense_real_3)
flat2 = Dense(32, activation='relu')(dense_real_4)


# text chanel
inputs3 = Input(shape=(len_text,))
embedding3 = Embedding(size_embedding, 36)(inputs3)
conv3 = Conv1D(filters=32, kernel_size=8, activation='relu')(embedding3)
drop3 = Dropout(0.1)(conv3)
pool3 = MaxPooling1D(pool_size=2)(drop3)
flat3 = Flatten()(pool3)

# merge
merged = concatenate([flat1, flat2, flat3])

# interpretation
dense1 = Dense(200, activation='relu')(merged)
dense2 = Dense(20, activation='relu')(dense1)
outputs = Dense(1, activation='sigmoid')(dense2)
model = Model(inputs=[inputs1, inputs2, inputs3], outputs=outputs)

model.summary()
# AUC for a binary classifier
def auc(y_true, y_pred):   
    ptas = tf.stack([binary_PTA(y_true,y_pred,k) for k in np.linspace(0, 1, 1000)],axis=0)
    pfas = tf.stack([binary_PFA(y_true,y_pred,k) for k in np.linspace(0, 1, 1000)],axis=0)
    pfas = tf.concat([tf.ones((1,)) ,pfas],axis=0)
    binSizes = -(pfas[1:]-pfas[:-1])
    s = ptas*binSizes
    return K.sum(s, axis=0)
#-----------------------------------------------------------------------------------------------------------------------------------------------------
# PFA, prob false alert for binary classifier
def binary_PFA(y_true, y_pred, threshold=K.variable(value=0.5)):
    y_pred = K.cast(y_pred >= threshold, 'float32')
    # N = total number of negative labels
    N = K.sum(1 - y_true)
    # FP = total number of false alerts, alerts from the negative class labels
    FP = K.sum(y_pred - y_pred * y_true)    
    return FP/N
#-----------------------------------------------------------------------------------------------------------------------------------------------------
# P_TA prob true alerts for binary classifier
def binary_PTA(y_true, y_pred, threshold=K.variable(value=0.5)):
    y_pred = K.cast(y_pred >= threshold, 'float32')
    # P = total number of positive labels
    P = K.sum(y_true)
    # TP = total number of correct alerts, alerts from the positive class labels
    TP = K.sum(y_pred * y_true)    
    return TP/P
model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy', auc])
batch_size = 1000
model.fit([trn_cat, trn_real, trn_text], y, batch_size=batch_size, epochs=3, validation_split=0.2)
submit = model.predict([tst_cat, tst_real, tst_text], batch_size=batch_size,verbose=1)
submission = pd.read_csv('../input/sample_submission.csv')
submission['project_is_approved'] = submit
submission.to_csv('mi_nn.csv', index=False)
trn_all = np.hstack((trn_cat, trn_real, trn_text))
trn_all.shape
model2 = Sequential()
model2.add(Dense(256, input_shape=(trn_all.shape[1],), activation='relu'))
model2.add(Dense(128, activation='relu'))
model2.add(Dense(1, activation='sigmoid'))
model2.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy', auc])
batch_size = 2000
model2.fit(trn_all, y, batch_size=batch_size, epochs=3, validation_split=0.2)
