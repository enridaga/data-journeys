
import numpy as np # linear algebra
# np.random.seed(8)
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
print(os.listdir("../input"))

from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import MinMaxScaler, StandardScaler, scale
from sklearn.metrics import roc_auc_score

from keras.layers import Activation, Dropout, Flatten, Dense, GlobalMaxPooling2D, BatchNormalization, Input, Conv2D
from keras.callbacks import ModelCheckpoint
from keras import metrics
from keras.optimizers import Adam 
from keras import backend as K
import keras
from keras.models import Model
from keras import regularizers
import tensorflow as tf
from keras.losses import binary_crossentropy
import gc
import scipy.special
from tqdm import *
from scipy.stats import norm, rankdata

from keras.callbacks import ModelCheckpoint, LearningRateScheduler, EarlyStopping, ReduceLROnPlateau


BATCH_SIZE = 1024
NUM_FEATURES = 1200
train = pd.read_csv('../input/santander-customer-transaction-prediction/train.csv')
test = pd.read_csv('../input/santander-customer-transaction-prediction/test.csv')
train_knowledge = pd.read_csv('../input/santander-2019-distillation/lgbm_train.csv')
y = train['target']
y_knowledge = train_knowledge['target']
id_code_train = train['ID_code']
id_code_test = test['ID_code']
features = [c for c in train.columns if c not in ['ID_code', 'target']]
for feature in features:
    # train['mean_'+feature] = (train[feature].mean()-train[feature])
    # train['z_'+feature] = (train[feature] - train[feature].mean())/train[feature].std(ddof=0)
    train['sq_'+feature] = (train[feature])**2
    # train['sqrt_'+feature] = np.abs(train[feature])**(1/2)
    train['c_'+feature] = (train[feature])**3
    # train['p4_'+feature] = (train[feature])**4
    # train['r1_'+feature] = np.round(train[feature], 1)
    train['r2_'+feature] = np.round(train[feature], 2)
    
for feature in features:
    # test['mean_'+feature] = (train[feature].mean()-test[feature])
    # test['z_'+feature] = (test[feature] - train[feature].mean())/train[feature].std(ddof=0)
    test['sq_'+feature] = (test[feature])**2
    # test['sqrt_'+feature] = np.abs(test[feature])**(1/2)
    test['c_'+feature] = (test[feature])**3
    # test['p4_'+feature] = (test[feature])**4
    # test['r1_'+feature] = np.round(test[feature], 1)
    test['r2_'+feature] = np.round(test[feature], 2)
class GaussRankScaler():

    def __init__( self ):
        self.epsilon = 1e-9
        self.lower = -1 + self.epsilon
        self.upper =  1 - self.epsilon
        self.range = self.upper - self.lower

    def fit_transform( self, X ):

        i = np.argsort( X, axis = 0 )
        j = np.argsort( i, axis = 0 )

        assert ( j.min() == 0 ).all()
        assert ( j.max() == len( j ) - 1 ).all()

        j_range = len( j ) - 1
        self.divider = j_range / self.range

        transformed = j / self.divider
        transformed = transformed - self.upper
        transformed = scipy.special.erfinv( transformed )
        ############
        # transformed = transformed - np.mean(transformed)

        return transformed
SPLIT = len(train)
train = train.append(test)
del test; gc.collect()
# print(train.shape)
scaler = GaussRankScaler()
sc = StandardScaler()
for feat in tqdm(features):
    # train[feat] = scaler.fit_transform(train[feat])
    train[feat] = sc.fit_transform(train[feat].values.reshape(-1, 1))
    train[feat+'_r'] = rankdata(train[feat]).astype('float32')
    train[feat+'_n'] = norm.cdf(train[feat]).astype('float32')

feats = [c for c in train.columns if c not in (['ID_code', 'target'] + features)]
for feat in tqdm(feats):
    train[feat] = sc.fit_transform(train[feat].values.reshape(-1, 1))

train = train.drop(['target', 'ID_code'], axis=1)
test = train[SPLIT:].values
train = train[:SPLIT].values
# test = test.drop(['ID_code'], axis=1)
print('Done!!')
print(train.shape)
# train.head()
# train[0:5]
train = np.reshape(train, (-1, NUM_FEATURES, 1))
test = np.reshape(test, (-1, NUM_FEATURES, 1))
x_train, x_valid, y_train, y_valid, y_knowledge_train, y_knowledge_valid  = train_test_split(train, y, y_knowledge, stratify=y, test_size=0.2, random_state=8)
function = 'relu'
# function = keras.layers.advanced_activations.LeakyReLU(alpha=.001)

def create_model(input_shape, n_out):
    input_tensor = Input(shape=input_shape)
    x = Dense(16, activation=function)(input_tensor)
    x = Flatten()(x)
    out_put = Dense(n_out, activation='sigmoid')(x)
    model = Model(input_tensor, out_put)
    
    return model
def auc(y_true, y_pred):
    return tf.py_func(roc_auc_score, (y_true, y_pred), tf.double)
gamma = 2.0
alpha=.25
epsilon = K.epsilon()

def focal_loss(y_true, y_pred):
    pt_1 = y_pred * y_true
    pt_1 = K.clip(pt_1, epsilon, 1-epsilon)
    CE_1 = -K.log(pt_1)
    FL_1 = alpha* K.pow(1-pt_1, gamma) * CE_1
    
    pt_0 = (1-y_pred) * (1-y_true)
    pt_0 = K.clip(pt_0, epsilon, 1-epsilon)
    CE_0 = -K.log(pt_0)
    FL_0 = (1-alpha)* K.pow(1-pt_0, gamma) * CE_0
    
    loss = K.sum(FL_1, axis=1) + K.sum(FL_0, axis=1)
    return loss
def mixup_data(x, y, alpha=1.0):
    # y = np.array(y)
    # print(y)
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    sample_size = x.shape[0]
    index_array = np.arange(sample_size)
    np.random.shuffle(index_array)
    
    mixed_x = lam * x + (1 - lam) * x[index_array]
    mixed_y = (lam * y) + ((1 - lam) * y[index_array])
    # print((1 - lam) * y[index_array])
    # print((lam * y).shape,((1 - lam) * y[index_array]).shape)
    return mixed_x, mixed_y

def make_batches(size, batch_size):
    nb_batch = int(np.ceil(size/float(batch_size)))
    return [(i*batch_size, min(size, (i+1)*batch_size)) for i in range(0, nb_batch)]


def batch_generator(X,y,batch_size=128,shuffle=True,mixup=False):
    y = np.array(y)
    # print(X.shape[0], y.shape[0])
    sample_size = X.shape[0]
    index_array = np.arange(sample_size)
    
    while True:
        if shuffle:
            np.random.shuffle(index_array)
        batches = make_batches(sample_size, batch_size)
        for batch_index, (batch_start, batch_end) in enumerate(batches):
            batch_ids = index_array[batch_start:batch_end]
            X_batch = X[batch_ids]
            y_batch = y[batch_ids]
            
            if mixup:
                # print('before', X_batch.shape, y_batch.shape)
                X_batch,y_batch = mixup_data(X_batch,y_batch,alpha=1.0)
            # print('*****************')    
            yield X_batch,y_batch
# model = create_model((train.shape[1],), 1)
model = create_model((NUM_FEATURES,1), 1)
# model.compile(loss='binary_crossentropy', optimizer='adam', metrics=[auc])
model.compile(loss='binary_crossentropy', optimizer='adam')
model.summary()

checkpoint = ModelCheckpoint('feed_forward_model.h5', monitor='val_loss', verbose=1, 
                             save_best_only=True, mode='min', save_weights_only = True)
reduceLROnPlat = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=4, 
                                   verbose=1, mode='min', epsilon=0.0001)
early = EarlyStopping(monitor="val_loss", 
                      mode="min", 
                      patience=9)
callbacks_list = [checkpoint, reduceLROnPlat, early]
tr_gen = batch_generator(x_train,y_train,batch_size=BATCH_SIZE,shuffle=True,mixup=True)

history = model.fit_generator(# x_train,y_train,
                                tr_gen,
                                steps_per_epoch=np.ceil(float(len(x_train)) / float(BATCH_SIZE)),
                                epochs=10,
                                verbose=1,
                                callbacks=callbacks_list,
                                validation_data=(x_valid, y_valid))
model.load_weights('feed_forward_model.h5')
prediction = model.predict(x_valid, batch_size=512, verbose=1)
roc_auc_score(y_valid, prediction)
y_train = np.vstack((y_train, y_knowledge_train)).T
y_valid = np.vstack((y_valid, y_knowledge_valid)).T

print(y_train.shape)
y_train[0]
def knowledge_distillation_loss_withBE(y_true, y_pred, beta=0.1):

    # Extract the groundtruth from dataset and the prediction from teacher model
    y_true, y_pred_teacher = y_true[: , :1], y_true[: , 1:]
    
    # Extract the prediction from student model
    y_pred, y_pred_stu = y_pred[: , :1], y_pred[: , 1:]

    loss = beta*binary_crossentropy(y_true,y_pred) + (1-beta)*binary_crossentropy(y_pred_teacher, y_pred_stu)

    return loss
def auc_2(y_true, y_pred):
    y_true = y_true[:, :1]
    y_pred = y_pred[:, :1]
    return tf.py_func(roc_auc_score, (y_true, y_pred), tf.double)

def auc_3(y_true, y_pred):
    y_true = y_true[:, :1]
    y_pred = y_pred[:, 1:]
    return tf.py_func(roc_auc_score, (y_true, y_pred), tf.double)
# model = create_model((train.shape[1],), 2)
model = create_model((NUM_FEATURES,1), 2)
model.compile(loss=knowledge_distillation_loss_withBE, optimizer='adam', metrics=[auc_2])

checkpoint = ModelCheckpoint('student_model_BE.h5', monitor='val_auc_2', verbose=1, 
                             save_best_only=True, mode='max', save_weights_only = True)
reduceLROnPlat = ReduceLROnPlateau(monitor='val_auc_2', factor=0.5, patience=4, 
                                   verbose=1, mode='max', epsilon=0.0001)
early = EarlyStopping(monitor="val_auc_2", 
                      mode="max", 
                      patience=9)
callbacks_list = [checkpoint, reduceLROnPlat, early]

history = model.fit(x_train,y_train,
                    epochs=10,
                    batch_size = BATCH_SIZE,
                    validation_data=(x_valid, y_valid))
def knowledge_distillation_loss_withFL(y_true, y_pred, beta=0.1):

    # Extract the groundtruth from dataset and the prediction from teacher model
    y_true, y_pred_teacher = y_true[: , :1], y_true[: , 1:]
    
    # Extract the prediction from student model
    y_pred, y_pred_stu = y_pred[: , :1], y_pred[: , 1:]

    loss = beta*focal_loss(y_true,y_pred) + (1-beta)*binary_crossentropy(y_pred_teacher, y_pred_stu)

    return loss
# model = create_model((train.shape[1],), 2)
model = create_model((NUM_FEATURES,1), 2)
model.compile(loss=knowledge_distillation_loss_withFL, optimizer='adam', metrics=[auc_2, auc_3])

checkpoint = ModelCheckpoint('student_model_FL.h5', monitor='val_auc_2', verbose=1, 
                             save_best_only=True, mode='max', save_weights_only = True)
reduceLROnPlat = ReduceLROnPlateau(monitor='val_auc_2', factor=0.5, patience=4, 
                                   verbose=1, mode='max', epsilon=0.0001)
early = EarlyStopping(monitor="val_auc_2", 
                      mode="max", 
                      patience=9)
callbacks_list = [checkpoint, reduceLROnPlat, early]

history = model.fit(x_train,y_train,
                    epochs=10,
                    batch_size = BATCH_SIZE,
                    callbacks=callbacks_list,
                    validation_data=(x_valid, y_valid))
from scipy.special import logit

def sigmoid(x, derivative=False):
  return x*(1-x) if derivative else 1/(1+np.exp(-x))

TEMPERATURE = 2

y_knowledge_logit = logit(y_knowledge)
y_temperature = sigmoid(y_knowledge_logit/TEMPERATURE)

# del x_train, x_valid; gc.collect()

x_train, x_valid, y_train, y_valid, y_knowledge_train, y_knowledge_valid  = train_test_split(train, y, y_temperature,
                                                                                             stratify=y, test_size=0.2, random_state=8)


y_train = np.vstack((y_train, y_knowledge_train)).T
y_valid = np.vstack((y_valid, y_knowledge_valid)).T

print(y_train.shape)
y_train[0]
# model = create_model((train.shape[1],), 2)
model = create_model((NUM_FEATURES,1), 2)
model.compile(loss=knowledge_distillation_loss_withFL, optimizer='adam', metrics=[auc_2,auc_3])

checkpoint = ModelCheckpoint('student_model_FL.h5', monitor='val_auc_2', verbose=1, 
                             save_best_only=True, mode='max', save_weights_only = True)
reduceLROnPlat = ReduceLROnPlateau(monitor='val_auc_2', factor=0.5, patience=4, 
                                   verbose=1, mode='max', epsilon=0.0001)
early = EarlyStopping(monitor="val_auc_2", 
                      mode="max", 
                      patience=9)
callbacks_list = [checkpoint, reduceLROnPlat, early]

history = model.fit(x_train,y_train,
                    epochs=10,
                    batch_size = 1024,
                    callbacks=callbacks_list,
                    validation_data=(x_valid, y_valid))
# run k-fold
num_fold = 5
folds = list(StratifiedKFold(n_splits=num_fold, shuffle=True, random_state=7).split(train, y))
# del x_train, x_valid; gc.collect()

y_test_pred_log = np.zeros(len(train))
y_train_pred_log = np.zeros(len(train))
print(y_test_pred_log.shape)
print(y_train_pred_log.shape)
score = []

for j, (train_idx, valid_idx) in enumerate(folds):
    print('\n===================FOLD=',j)
    x_train, x_valid = train[train_idx], train[valid_idx]
    y_train, y_valid = y[train_idx], y[valid_idx]
    y_knowledge_train, y_knowledge_valid = y_temperature[train_idx], y_temperature[valid_idx]
    
    y_train = np.vstack((y_train, y_knowledge_train)).T
    y_valid = np.vstack((y_valid, y_knowledge_valid)).T
    
    # model = create_model((train.shape[1],), 2)
    model = create_model((NUM_FEATURES,1), 2)
    model.compile(loss=knowledge_distillation_loss_withFL, optimizer='adam', metrics=[auc_2, auc_3])

    checkpoint = ModelCheckpoint('student_model_FL.h5', monitor='val_auc_2', verbose=1, 
                                 save_best_only=True, mode='max', save_weights_only = True)
    reduceLROnPlat = ReduceLROnPlateau(monitor='val_auc_2', factor=0.5, patience=4, 
                                       verbose=1, mode='max', epsilon=0.0001)
    early = EarlyStopping(monitor="val_auc_2", 
                          mode="max", 
                          patience=9)
    callbacks_list = [checkpoint, reduceLROnPlat, early]
    history = model.fit(x_train,y_train,
                        epochs=100,
                        batch_size = BATCH_SIZE,
                        callbacks=callbacks_list,
                        validation_data=(x_valid, y_valid))
    
    model.load_weights('student_model_FL.h5')
    prediction = model.predict(x_valid,
                               batch_size=512,
                               verbose=1)
    # print(prediction.shape)
    # prediction = np.sum(prediction, axis=1)/2
    score.append(roc_auc_score(y_valid[:,0], prediction[:,1]))
    # score.append(roc_auc_score(y_valid[:,0], prediction))
    prediction = model.predict(test,
                               batch_size=512,
                               verbose=1)
    # y_test_pred_log += np.sum(prediction, axis=1)/2
    y_test_pred_log += np.squeeze(prediction[:, 1])
    
    prediction = model.predict(x_valid,
                               batch_size=512,
                               verbose=1)
    # y_train_pred_log += np.sum(prediction, axis=1)/2
    y_train_pred_log[valid_idx] += np.squeeze(prediction[:, 1])
    
    del x_train, x_valid, y_train, y_valid, y_knowledge_train, y_knowledge_valid
    gc.collect()



print("OOF score: ", roc_auc_score(y, y_train_pred_log/num_fold))
print("average {} folds score: ".format(num_fold), np.sum(score)/num_fold)

# make submission
submit = pd.read_csv('../input/santander-customer-transaction-prediction/sample_submission.csv')
submit['ID_code'] = id_code_test
submit['target'] = y_test_pred_log/num_fold
submit.to_csv('submission.csv', index=False)
submit.head()