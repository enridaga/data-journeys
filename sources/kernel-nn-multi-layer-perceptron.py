
# Importar os principais pacotes
import numpy as np
import pandas as pd
import seaborn as sns
sns.set()

import matplotlib.pyplot as plt
#%matplotlib inline

from tqdm import tqdm_notebook as tqdm
import re
import random as rd
import os
import codecs
import time
import datetime
import gc
from numba import jit
from collections import Counter
import copy
from typing import Any

seed = 12345
np.random.seed(seed)
rd.seed(seed)
os.environ['PYTHONHASHSEED'] = str(seed)

# Evitar que aparece os warnings
import warnings
warnings.filterwarnings("ignore")

# Seta algumas opções no Jupyter para exibição dos datasets
pd.set_option('display.max_columns', 200)
pd.set_option('display.max_rows', 200)

# Variavel para controlar o treinamento no Kaggle
TRAIN_OFFLINE = False
# Importa os pacotes de algoritmos
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
import lightgbm as lgb
import xgboost as xgb 

# Importa os pacotes de algoritmos de redes neurais (Keras)
import keras
from keras.losses import binary_crossentropy, categorical_crossentropy
from keras.utils import to_categorical
from keras.layers import Dense,Input,Flatten,concatenate,Dropout,Lambda,BatchNormalization
from keras.layers import Activation
from keras.models import Sequential, Model
from keras.callbacks import Callback,EarlyStopping,ModelCheckpoint,ReduceLROnPlateau
import keras.backend as K
from keras.optimizers import Adam
from keras import optimizers
from keras.utils import np_utils

# Importa pacotes do sklearn
from sklearn import preprocessing
import sklearn.metrics as mtr
from sklearn.model_selection import KFold, GridSearchCV
from sklearn.metrics import mean_squared_error, log_loss, confusion_matrix
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.preprocessing import scale, MinMaxScaler, StandardScaler
from sklearn import model_selection
from sklearn.utils import class_weight

from sklearn.preprocessing import StandardScaler as SS
from sklearn.model_selection import train_test_split as TTS

from keras.callbacks import ReduceLROnPlateau as RLRP
from keras.callbacks import EarlyStopping as ES
def read_data():
    
    if TRAIN_OFFLINE:
        print('Carregando arquivo dataset_treino.csv....')
        train = pd.read_csv('../dataset/dataset_treino.csv')
        print('dataset_treino.csv tem {} linhas and {} colunas'.format(train.shape[0], train.shape[1]))

        print('Carregando arquivo dataset_teste.csv....')
        test = pd.read_csv('../dataset/dataset_teste.csv')
        print('dataset_teste.csv tem {} linhas and {} colunas'.format(test.shape[0], test.shape[1]))

        
    else:
        print('Carregando arquivo dataset_treino.csv....')
        train = pd.read_csv('/kaggle/input/competicao-dsa-machine-learning-dec-2019/dataset_treino.csv')
        print('dataset_treino.csv tem {} linhas and {} colunas'.format(train.shape[0], train.shape[1]))
        
        print('Carregando arquivo dataset_treino.csv....')
        test = pd.read_csv('/kaggle/input/competicao-dsa-machine-learning-dec-2019/dataset_teste.csv')
        print('dataset_teste.csv tem {} linhas and {} colunas'.format(test.shape[0], test.shape[1]))
    
    return train, test
# Leitura dos dados
train, test = read_data()
# Removendo as top 5 colunas com mais dados missing
drop_columns = ['v30', 'v113', 'v56', 'v3', 'v31']
train = train.drop(drop_columns, axis=1)
test = test.drop(drop_columns, axis=1)
train.shape, test.shape
# Label encoder nas features categoricas
for c in train.columns[train.dtypes == 'object']:
    train[c] = train[c].factorize()[0]
    
for c in test.columns[test.dtypes == 'object']:
    test[c] = test[c].factorize()[0]
# Preenche os dados missing com a media
train.fillna(train.mean(),inplace=True)
test.fillna(train.mean(),inplace=True)
from boruta import BorutaPy

# Separando features preditoras e target
X = train.drop(['ID', 'target'], axis=1)
y = train['target']

X = X.values
y = y.values
y = y.ravel()

# Define o classificador Random Forest
rf = RandomForestClassifier(n_jobs=-1, class_weight='balanced', max_depth=5)
rf.fit(X, y)

# Define o metodo de feature selection
feat_selector = BorutaPy(rf, n_estimators=100, verbose=2, random_state=1)

# Procura por todas as features relevantes
feat_selector.fit(X, y)

# Check as features selecionadas
feat_selector.support_

# Check o ranking das features
feat_selector.ranking_

# Chama o call transform() nos dados de treino para filtrar as features
X_filtered = feat_selector.transform(X)

# Mostra no final o shape do dataset
X_filtered.shape
# Separando features preditoras e target
train_x = X_filtered.copy()
train_y = train['target']
train_y = to_categorical(train_y)

# Padronizando os dados
scaler = StandardScaler()
train_x = scaler.fit_transform(train_x)

# Verificando o shape dos datasets depois dos ajustes
# Neste momento está pronto para ser usado pelo treinamento
train_x.shape, train_y.shape
# Limpeza da memória
gc.collect()
# Criando função para treinar a rede neural
def get_nn(x_tr,y_tr,x_val,y_val,shape):
    K.clear_session()
    
    # Cria a estrutura da rede neural com 3 camadas ocultas
    inp = Input(shape = (x_tr.shape[1],))

    x = Dense(512, input_dim=x_tr.shape[1], activation='relu')(inp)
    x = Dropout(0.5)(x)    
    x = BatchNormalization()(x)

    x = Dense(256, activation='relu')(x)
    x = Dropout(0.5)(x)    
    x = BatchNormalization()(x)
    
    x = Dense(32, activation='relu')(x)
    x = Dropout(0.1)(x)    
    x = BatchNormalization()(x)
    
    x = Dense(16, activation='relu')(x)
    x = Dropout(.25)(x)
    x = BatchNormalization()(x)
    
    out = Dense(2, activation='softmax')(x)
    model = Model(inp,out)
    
    model.compile(optimizer = 'Adam',
                  loss='categorical_crossentropy', 
                  metrics=['categorical_accuracy'])
    
    # Realiza a parada mais cedo quando percebe overfitting
    es = EarlyStopping(monitor='val_loss', 
                       mode='min',
                       restore_best_weights=True, 
                       verbose=1, 
                       patience=20)

    # Realiza checkpoint durante o treinamento
    mc = ModelCheckpoint('best_model.h5',
                         monitor='val_loss',
                         mode='min',
                         save_best_only=True, 
                         verbose=1, 
                         save_weights_only=True)

    # Realize o ajuste na Learning Rate durante o treinamento
    rl = ReduceLROnPlateau(monitor='val_loss', 
                           factor=0.1, 
                           patience=10, 
                           verbose=1, 
                           epsilon=1e-4, 
                           mode='min')

    # Realiza o fit do modelo
    model.fit(x_tr, y_tr,
              validation_data=[x_val, y_val],
              callbacks=[es,mc,rl],
              epochs=250, 
              batch_size=1024,
              verbose=1,
              shuffle=True)
    
    # Carrega os melhores pesos
    model.load_weights("best_model.h5")
    
    # Realiza as previsões
    y_pred = model.predict(x_val)
    y_valid = y_val
             
    # Calcula o log loss
    logloss = log_loss(y_valid, y_pred, eps=1e-15)

    return model, logloss
%%time

# Bloco para executar a rede neural a cada passada do KFold
# Vamos realizar 2 loops com 5 kfolds e apurar a média
loop = 2
fold = 5

# Definindo listas que serão preenchidas durante o loop for
oof_nn = np.zeros([loop, train_y.shape[0], train_y.shape[1]])
models_nn = []
logloss_csv_nn = []

# Treinando o modelo
for k in range(loop):
    kfold = KFold(fold, random_state = 42 + k, shuffle = True)
    for k_fold, (tr_inds, val_inds) in enumerate(kfold.split(train_y)):
        print("-----------")
        print(f'Loop {k+1}/{loop}' + f' Fold {k_fold+1}/{fold}')
        print("-----------")
        
        tr_x, tr_y = train_x[tr_inds], train_y[tr_inds]
        val_x, val_y = train_x[val_inds], train_y[val_inds]
        
        # Train NN
        nn, logloss_nn = get_nn(tr_x, tr_y, val_x, val_y, shape=val_x.shape[0])
        models_nn.append(nn)
        print("the %d fold Log-Loss (NN) is %f"%((k_fold+1), logloss_nn))
        logloss_csv_nn.append(logloss_nn)
        
        #Predict OOF
        oof_nn[k, val_inds, :] = nn.predict(val_x)
        
    print("PARTIAL: mean Log-Loss (NN) is %f"%np.mean(logloss_csv_nn))        
# Verificando o resultado médio do Log Loss para cada passada do Kfold
loss_oof_nn = []

for k in range(loop):
    loss_oof_nn.append(log_loss(train_y, oof_nn[k,...], eps=1e-15))
    
print("Média log-loss:  %f"%np.mean(logloss_csv_nn))
print("Média OOF log-loss: %f"%np.mean(loss_oof_nn))
# Apenas para acompanhar o resultado visual
# Exibir o treinamento somente do primeiro kfold
plt.figure(figsize=(18, 12))
plt.subplot(2, 1, 1)
plt.plot(models_nn[0].history.history["loss"], "o-", alpha=.9, label="loss")
plt.plot(models_nn[0].history.history["val_loss"], "o-", alpha=.9, label="val_loss")
plt.axhline(1, linestyle="--", c="C2")
plt.legend()
plt.subplot(2, 1, 2)
plt.plot(models_nn[0].history.history["categorical_accuracy"], "o-", alpha=.9, label="accuracy")
plt.plot(models_nn[0].history.history["val_categorical_accuracy"], "o-", alpha=.9, label="val_accuracy")
plt.axhline(.7, linestyle="--", c="C2")
plt.legend()
plt.show()
# Preparando os dados de teste
new_test = test.drop(['ID'], axis=1).values
test_filtered = feat_selector.transform(new_test)
test_filtered = scaler.fit_transform(test_filtered)
test_filtered.shape
# Funcao para realizar as previsoes baseado em todos os modelos do Kfold
def predict_proba(model, x, batch_size=32, verbose=0):
    preds = model.predict(x, batch_size, verbose)
    if preds.min() < 0. or preds.max() > 1.:
        warnings.warn('Network returning invalid probability values.')
    return preds

def predict(x_te, models_nn):
    model_num_nn = len(models_nn)

    for k,m in enumerate(models_nn):
        if k==0:
            y_pred_nn = predict_proba(m, x_te, batch_size=1024)
        else:
            y_pred_nn += predict_proba(m, x_te, batch_size=1024)
            
    y_pred_nn = y_pred_nn / model_num_nn
    return y_pred_nn
# Realizando as previsões no dataset de teste
test_pred = predict(test_filtered, models_nn)
test_pred[:,1]
# Carrega o dataset de exemplo de submission e carrega as previsões das probabilidades
submission = pd.read_csv('/kaggle/input/competicao-dsa-machine-learning-dec-2019/sample_submission.csv')
submission['PredictedProb'] = test_pred[:,1]
print(submission.shape)
submission.head()
# Gera o arquivo de saída para submeter no Kaggle
submission.to_csv('submission_nn_v1.0.2.csv', index=False)
# Apenas para visualizar a distribuição das previsões
submission['PredictedProb'].value_counts(normalize=True)
# Histograma com as previsões
plt.hist(submission.PredictedProb)
plt.show()
### Continua....