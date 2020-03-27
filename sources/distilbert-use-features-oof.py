
!pip install ../input/sacremoses/sacremoses-master/ > /dev/null

import os
import sys
import glob
import torch

sys.path.insert(0, "../input/transformers/transformers-master/")
import transformers
import numpy as np
import pandas as pd
import math
def chunks(l, n):
    """Yield successive n-sized chunks from l."""
    for i in range(0, len(l), n):
        yield l[i:i + n]
def fetch_vectors(string_list, batch_size=64):
    # inspired by https://jalammar.github.io/a-visual-guide-to-using-bert-for-the-first-time/
    DEVICE = torch.device("cuda")
    tokenizer = transformers.DistilBertTokenizer.from_pretrained("../input/distilbertbaseuncased/")
    model = transformers.DistilBertModel.from_pretrained("../input/distilbertbaseuncased/")
    model.to(DEVICE)

    fin_features = []
    for data in chunks(string_list, batch_size):
        tokenized = []
        for x in data:
            x = " ".join(x.strip().split()[:300])
            tok = tokenizer.encode(x, add_special_tokens=True)
            tokenized.append(tok[:512])

        max_len = 512
        padded = np.array([i + [0] * (max_len - len(i)) for i in tokenized])
        attention_mask = np.where(padded != 0, 1, 0)
        input_ids = torch.tensor(padded).to(DEVICE)
        attention_mask = torch.tensor(attention_mask).to(DEVICE)

        with torch.no_grad():
            last_hidden_states = model(input_ids, attention_mask=attention_mask)

        features = last_hidden_states[0][:, 0, :].cpu().numpy()
        fin_features.append(features)

    fin_features = np.vstack(fin_features)
    return fin_features
df_train = pd.read_csv("../input/google-quest-challenge/train.csv").fillna("none")
df_test = pd.read_csv("../input/google-quest-challenge/test.csv").fillna("none")

sample = pd.read_csv("../input/google-quest-challenge/sample_submission.csv")
target_cols = list(sample.drop("qa_id", axis=1).columns)

train_question_body_dense = fetch_vectors(df_train.question_body.values)
train_answer_dense = fetch_vectors(df_train.answer.values)

test_question_body_dense = fetch_vectors(df_test.question_body.values)
test_answer_dense = fetch_vectors(df_test.answer.values)
import os
import re
import gc
import pickle  
import random
import keras

import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow_hub as hub
import keras.backend as K

from keras.models import Model
from keras.layers import Dense, Input, Dropout, Lambda
from keras.optimizers import Adam
from keras.callbacks import Callback
from scipy.stats import spearmanr, rankdata
from os.path import join as path_join
from numpy.random import seed
from urllib.parse import urlparse
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import KFold
from sklearn.linear_model import MultiTaskElasticNet

seed(42)
tf.random.set_seed(42)
random.seed(42)
data_dir = '../input/google-quest-challenge/'
train = pd.read_csv(path_join(data_dir, 'train.csv'))
test = pd.read_csv(path_join(data_dir, 'test.csv'))
print(train.shape, test.shape)
train.head()
targets = [
        'question_asker_intent_understanding',
        'question_body_critical',
        'question_conversational',
        'question_expect_short_answer',
        'question_fact_seeking',
        'question_has_commonly_accepted_answer',
        'question_interestingness_others',
        'question_interestingness_self',
        'question_multi_intent',
        'question_not_really_a_question',
        'question_opinion_seeking',
        'question_type_choice',
        'question_type_compare',
        'question_type_consequence',
        'question_type_definition',
        'question_type_entity',
        'question_type_instructions',
        'question_type_procedure',
        'question_type_reason_explanation',
        'question_type_spelling',
        'question_well_written',
        'answer_helpful',
        'answer_level_of_information',
        'answer_plausible',
        'answer_relevance',
        'answer_satisfaction',
        'answer_type_instructions',
        'answer_type_procedure',
        'answer_type_reason_explanation',
        'answer_well_written'    
    ]

input_columns = ['question_title', 'question_body', 'answer']
find = re.compile(r"^[^.]*")

train['netloc'] = train['url'].apply(lambda x: re.findall(find, urlparse(x).netloc)[0])
test['netloc'] = test['url'].apply(lambda x: re.findall(find, urlparse(x).netloc)[0])

features = ['netloc', 'category']
merged = pd.concat([train[features], test[features]])
ohe = OneHotEncoder()
ohe.fit(merged)

features_train = ohe.transform(train[features]).toarray()
features_test = ohe.transform(test[features]).toarray()
module_url = "../input/universalsentenceencoderlarge4/"
embed = hub.load(module_url)
embeddings_train = {}
embeddings_test = {}
for text in input_columns:
    print(text)
    train_text = train[text].str.replace('?', '.').str.replace('!', '.').tolist()
    test_text = test[text].str.replace('?', '.').str.replace('!', '.').tolist()
    
    curr_train_emb = []
    curr_test_emb = []
    batch_size = 4
    ind = 0
    while ind*batch_size < len(train_text):
        curr_train_emb.append(embed(train_text[ind*batch_size: (ind + 1)*batch_size])["outputs"].numpy())
        ind += 1
        
    ind = 0
    while ind*batch_size < len(test_text):
        curr_test_emb.append(embed(test_text[ind*batch_size: (ind + 1)*batch_size])["outputs"].numpy())
        ind += 1    
        
    embeddings_train[text + '_embedding'] = np.vstack(curr_train_emb)
    embeddings_test[text + '_embedding'] = np.vstack(curr_test_emb)
    
del embed
K.clear_session()
gc.collect()
l2_dist = lambda x, y: np.power(x - y, 2).sum(axis=1)

cos_dist = lambda x, y: (x*y).sum(axis=1)

dist_features_train = np.array([
    l2_dist(embeddings_train['question_title_embedding'], embeddings_train['answer_embedding']),
    l2_dist(embeddings_train['question_body_embedding'], embeddings_train['answer_embedding']),
    l2_dist(embeddings_train['question_body_embedding'], embeddings_train['question_title_embedding']),
    cos_dist(embeddings_train['question_title_embedding'], embeddings_train['answer_embedding']),
    cos_dist(embeddings_train['question_body_embedding'], embeddings_train['answer_embedding']),
    cos_dist(embeddings_train['question_body_embedding'], embeddings_train['question_title_embedding'])
]).T

dist_features_test = np.array([
    l2_dist(embeddings_test['question_title_embedding'], embeddings_test['answer_embedding']),
    l2_dist(embeddings_test['question_body_embedding'], embeddings_test['answer_embedding']),
    l2_dist(embeddings_test['question_body_embedding'], embeddings_test['question_title_embedding']),
    cos_dist(embeddings_test['question_title_embedding'], embeddings_test['answer_embedding']),
    cos_dist(embeddings_test['question_body_embedding'], embeddings_test['answer_embedding']),
    cos_dist(embeddings_test['question_body_embedding'], embeddings_test['question_title_embedding'])
]).T

X_train = np.hstack([item for k, item in embeddings_train.items()] + [features_train, dist_features_train])
X_test = np.hstack([item for k, item in embeddings_test.items()] + [features_test, dist_features_test])
y_train = train[targets].values
X_train = np.hstack((X_train, train_question_body_dense, train_answer_dense))
X_test = np.hstack((X_test, test_question_body_dense, test_answer_dense))
# Compatible with tensorflow backend
class SpearmanRhoCallback(Callback):
    def __init__(self, training_data, validation_data, patience, model_name):
        self.x = training_data[0]
        self.y = training_data[1]
        self.x_val = validation_data[0]
        self.y_val = validation_data[1]
        
        self.patience = patience
        self.value = -1
        self.bad_epochs = 0
        self.model_name = model_name

    def on_train_begin(self, logs={}):
        return

    def on_train_end(self, logs={}):
        return

    def on_epoch_begin(self, epoch, logs={}):
        return

    def on_epoch_end(self, epoch, logs={}):
        y_pred_val = self.model.predict(self.x_val)
        rho_val = np.mean([spearmanr(self.y_val[:, ind], y_pred_val[:, ind] + np.random.normal(0, 1e-7, y_pred_val.shape[0])).correlation for ind in range(y_pred_val.shape[1])])
        if rho_val >= self.value:
            self.value = rho_val
        else:
            self.bad_epochs += 1
        if self.bad_epochs >= self.patience:
            print("Epoch %05d: early stopping Threshold" % epoch)
            self.model.stop_training = True
            #self.model.save_weights(self.model_name)
        print('\rval_spearman-rho: %s' % (str(round(rho_val, 4))), end=100*' '+'\n')
        return rho_val

    def on_batch_begin(self, batch, logs={}):
        return

    def on_batch_end(self, batch, logs={}):
        return
def create_model():
    inps = Input(shape=(X_train.shape[1],))
    x = Dense(512, activation='elu')(inps)
    x = Dropout(0.2)(x)
    x = Dense(y_train.shape[1], activation='sigmoid')(x)
    model = Model(inputs=inps, outputs=x)
    model.compile(
        optimizer=Adam(lr=1e-4),
        loss=['binary_crossentropy']
    )
    model.summary()
    return model
all_predictions = []

kf = KFold(n_splits=5, random_state=42, shuffle=True)
for ind, (tr, val) in enumerate(kf.split(X_train)):
    X_tr = X_train[tr]
    y_tr = y_train[tr]
    X_vl = X_train[val]
    y_vl = y_train[val]
    
    model = create_model()
    model.fit(
        X_tr, y_tr, epochs=100, batch_size=32, validation_data=(X_vl, y_vl), verbose=True, 
        callbacks=[SpearmanRhoCallback(training_data=(X_tr, y_tr), validation_data=(X_vl, y_vl),
                                       patience=5, model_name=f'best_model_batch{ind}.h5')]
    )
    all_predictions.append(model.predict(X_test))
    
model = create_model()
model.fit(X_train, y_train, epochs=33, batch_size=32, verbose=False)
all_predictions.append(model.predict(X_test))
    
kf = KFold(n_splits=5, random_state=2019, shuffle=True)
for ind, (tr, val) in enumerate(kf.split(X_train)):
    X_tr = X_train[tr]
    y_tr = y_train[tr]
    X_vl = X_train[val]
    y_vl = y_train[val]
    
    model = MultiTaskElasticNet(alpha=0.001, random_state=42, l1_ratio=0.5)
    model.fit(X_tr, y_tr)
    all_predictions.append(model.predict(X_test))
    
model = MultiTaskElasticNet(alpha=0.001, random_state=42, l1_ratio=0.5)
model.fit(X_train, y_train)
all_predictions.append(model.predict(X_test))
test_preds = np.array([np.array([rankdata(c) for c in p.T]).T for p in all_predictions]).mean(axis=0)
max_val = test_preds.max() + 1
test_preds = test_preds/max_val + 1e-12
submission = pd.read_csv(path_join(data_dir, 'sample_submission.csv'))
submission[targets] = test_preds
submission.to_csv("submission.csv", index = False)
submission.head()