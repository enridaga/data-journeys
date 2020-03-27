
# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import itertools
import seaborn as sns
sns.set()

import matplotlib.pyplot as plt
#%matplotlib inline

import time
import datetime
import gc

# Evitar que aparece os warnings
import warnings
warnings.filterwarnings("ignore")

# Importa os pacotes de algoritmos
import xgboost as xgb
from xgboost.sklearn import XGBClassifier

# Importa pacotes do sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn import preprocessing
from sklearn.model_selection import train_test_split, GridSearchCV, KFold
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_auc_score, log_loss
from sklearn.preprocessing import scale, MinMaxScaler, StandardScaler

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.
#get the train and test data
treino = pd.read_csv('../input/competicao-dsa-machine-learning-dec-2019/dataset_treino.csv')
teste = pd.read_csv('../input/competicao-dsa-machine-learning-dec-2019/dataset_teste.csv')
treino.isna().sum()/len(treino)
treino.head()
## Retirando do dataset todas as colunas com mais 30% de missing
pct_null =treino.isna().sum()/len(treino)
missing_features = pct_null[pct_null > 0.30].index
treino.drop(missing_features, axis=1, inplace=True)
#Selecionando as variáveis numéricas
num=['int64','float64']
numericas=list(treino.select_dtypes(include=num).columns)
numerica=treino[numericas]
numerica.info()
# Preenchendo os valores faltantes com zero
numerica.fillna(numerica.mean(),inplace=True)
numerica.info()
## Selecionando as variáveis categóricas 
cat=['object']
categoricas=list(treino.select_dtypes(include=cat).columns)
categ=treino[categoricas]
categ.info()

categ.info()
# Separando features preditoras e target somente com as variáveis numericas
train_x = numerica.drop(['ID','target'], axis=1)
train_y = numerica['target']

# Padronizando os dados
scaler = StandardScaler()
train_x = scaler.fit_transform(train_x)
# Criando uma funcao para criação, execução e validação do modelo
def run_model(modelo, X_tr, y_tr, useTrainCV=True, cv_folds=5, early_stopping_rounds=10):
    
    # Utilização do Cross-Validation
    if useTrainCV:
        xgb_param = modelo.get_xgb_params()
        xgtrain = xgb.DMatrix(X_tr, label=y_tr)
        
        print ('Start cross validation')
        cvresult = xgb.cv(xgb_param, 
                          xgtrain, 
                          num_boost_round=modelo.get_params()['n_estimators'], 
                          nfold=cv_folds,
                          metrics=['logloss'],
                          stratified=True,
                          seed=42,
                          #verbose_eval=True,
                          early_stopping_rounds=early_stopping_rounds)

        modelo.set_params(n_estimators=cvresult.shape[0])
        best_tree = cvresult.shape[0]
        print('Best number of trees = {}'.format(best_tree))
    
    # Fit do modelo
    modelo.fit(X_tr, y_tr, eval_metric='logloss')
        
    # Predição no dataset de treino
    train_pred = modelo.predict(X_tr)
    train_pred_prob = modelo.predict_proba(X_tr)[:,1]
    
    # Exibir o relatorio do modelo
    #print("Acurácia : %.4g" % accuracy_score(y_tr, train_pred))
    #print("AUC Score (Treino): %f" % roc_auc_score(y_tr, train_pred_prob))
    print("Log Loss (Treino): %f" % log_loss(y_tr, train_pred_prob))
    print("Log Loss (Test): %f" % cvresult['test-logloss-mean'][best_tree-1])
    
    feature_imp = pd.Series(modelo.feature_importances_.astype(float)).sort_values(ascending=False)
    
    plt.figure(figsize=(18,8))
    feature_imp[:25].plot(kind='bar', title='Feature Importances')
    plt.ylabel('Feature Importance Score')
    plt.tight_layout()
%%time

# Criando o primeiro modelo XGB
modeloXGB = XGBClassifier(learning_rate = 0.1,
                          n_estimators = 200,
                          max_depth = 5,
                          min_child_weight = 1,
                          gamma = 0,
                          subsample = 0.8,
                          colsample_bytree = 0.8,
                          objective = 'binary:logistic',
                          n_jobs = -1,
                          scale_pos_weight = 1,
                          seed = 42)

run_model(modeloXGB, train_x, train_y)
gc.collect()
'''%%time

# Definindo os parametros que serão testados no GridSearch
param_v1 = {
 'max_depth':range(2,5),
 'min_child_weight':range(1,2)
}

grid_1 = GridSearchCV(estimator = XGBClassifier(learning_rate = 0.1, 
                                                n_estimators = 200, 
                                                max_depth = 5,
                                                min_child_weight = 1, 
                                                gamma = 0, 
                                                subsample = 0.8, 
                                                colsample_bytree = 0.8,
                                                objective = 'binary:logistic', 
                                                nthread = 4, 
                                                scale_pos_weight = 1, 
                                                seed = 42),
                      param_grid = param_v1, 
                      scoring = 'neg_log_loss',
                      n_jobs = -1,
                      iid = False, 
                      cv = 5)

# Realizando o fit e obtendo os melhores parametros do grid
grid_1.fit(train_x, train_y)
grid_1.best_params_, grid_1.best_score_'''
'''%%time

# Definindo os parametros que serão testados no GridSearch
param_v2 = {
 'gamma':[i/10.0 for i in range(0,2)]
}

grid_2 = GridSearchCV(estimator = XGBClassifier(learning_rate = 0.1, 
                                                n_estimators = 200, 
                                                max_depth = grid_1.best_params_['max_depth'],
                                                min_child_weight = grid_1.best_params_['min_child_weight'], 
                                                gamma = 0, 
                                                subsample = 0.8, 
                                                colsample_bytree = 0.8,
                                                objective = 'binary:logistic', 
                                                nthread = 4, 
                                                scale_pos_weight = 1, 
                                                seed = 42),
                      param_grid = param_v2, 
                      scoring = 'neg_log_loss',
                      n_jobs = -1,
                      iid = False, 
                      cv = 5)

# Realizando o fit e obtendo os melhores parametros do grid
grid_2.fit(train_x, train_y)
grid_2.best_params_, grid_2.best_score_'''
'''%%time

# Definindo os parametros que serão testados no GridSearch
param_v3 = {
 'subsample':[i/10.0 for i in range(6,8)],
 'colsample_bytree':[i/10.0 for i in range(6,8)]
}

grid_3 = GridSearchCV(estimator = XGBClassifier(learning_rate = 0.1, 
                                                n_estimators = 200, 
                                                max_depth = grid_1.best_params_['max_depth'],
                                                min_child_weight = grid_1.best_params_['min_child_weight'], 
                                                gamma = grid_2.best_params_['gamma'], 
                                                subsample = 0.8, 
                                                colsample_bytree = 0.8,
                                                objective = 'binary:logistic', 
                                                nthread = 4, 
                                                scale_pos_weight = 1, 
                                                seed = 42),
                      param_grid = param_v3, 
                      scoring = 'neg_log_loss',
                      n_jobs = -1,
                      iid = False, 
                      cv = 5)

grid_3.fit(train_x, train_y)
grid_3.best_params_, grid_3.best_score_'''
'''%%time

# Definindo os parametros que serão testados no GridSearch
param_v4 = {
 'reg_alpha':[0, 0.001, 0.005]
}

grid_4 = GridSearchCV(estimator = XGBClassifier(learning_rate = 0.1, 
                                                n_estimators = 200, 
                                                max_depth = grid_1.best_params_['max_depth'],
                                                min_child_weight = grid_1.best_params_['min_child_weight'], 
                                                gamma = grid_2.best_params_['gamma'], 
                                                subsample = grid_3.best_params_['subsample'], 
                                                colsample_bytree = grid_3.best_params_['colsample_bytree'],
                                                objective = 'binary:logistic', 
                                                nthread = 4, 
                                                scale_pos_weight = 1, 
                                                seed = 42),
                      param_grid = param_v4, 
                      scoring = 'neg_log_loss',
                      n_jobs = -1,
                      iid = False, 
                      cv = 5)

# Realizando o fit e obtendo os melhores parametros do grid
grid_4.fit(train_x, train_y)
grid_4.best_params_, grid_4.best_score_'''
'''%%time

# Criando o modelo XGB com todas as otimizações
modeloXGB_v2 = XGBClassifier(learning_rate = 0.01, 
                             n_estimators = 1000, 
                             max_depth = 4,
                             min_child_weight = 1,
                             gamma = 0.04, 
                             subsample = 0.6,
                             colsample_bytree = 0.8,
                             reg_alpha = 0, 
                             objective = 'binary:logistic', 
                             n_jobs = -1,
                             scale_pos_weight = 1, 
                             seed = 42)

run_model(modeloXGB_v2, train_x, train_y)'''
# Visualizando o modelo XGBoost
print(modeloXGB)
# Colocando o dataset de teste conforme o modelo treinado
# Neste caso é necessário aplicar a Feature Engineering usada para gerar o modelo
teste.drop(missing_features, axis=1, inplace=True)
text_x = teste.drop(['ID'], axis=1)

# Removendo todas as variaveis categoricas
drop_features = []
for col in text_x.columns:
    if text_x[col].dtype =='object':
        drop_features.append(col)
text_x = text_x.drop(drop_features, axis=1)
# Preenche os dados missing com 0 (zero)
text_x.fillna(text_x.mean(),inplace=True)
# Aplicando escala aos dados
text_x = scaler.fit_transform(text_x)

# Realizando as previsoes
test_pred_prob = modeloXGB.predict_proba(text_x)[:,1]
# Criando dataset de submissao
submission = pd.DataFrame({'ID': teste["ID"], 'PredictedProb': test_pred_prob.reshape((test_pred_prob.shape[0]))})
print(submission.head(10))
submission.to_csv('submission.csv', index=False)
plt.hist(submission.PredictedProb)
plt.show()