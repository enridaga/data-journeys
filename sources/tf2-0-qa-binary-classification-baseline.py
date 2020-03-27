
import gc
import json
import numpy as np 
import pandas as pd
import re

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

from multiprocessing import Pool

from sklearn.metrics import accuracy_score, f1_score
from tqdm import tqdm_notebook as tqdm
import Levenshtein 

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction import text
from sklearn.model_selection import GroupKFold
from sklearn.linear_model import LogisticRegression

from scipy import spatial
import lightgbm as lgb
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import RandomOverSampler
%%time
dtypes = {
    'CorrectString': str,
    'CorrectString_short': str,
    'CandidateString': str
}
train = pd.read_csv('../input/tf20-qa-features/train_data.csv', dtype=dtypes)
# test = pd.read_csv('../input/tf20-qa-features/test_data.csv', dtype=dtypes)
train.head(20)
html_tags = ['<P>', '</P>', '<Table>', '</Table>', '<Tr>', '</Tr>', '<Ul>', '<Ol>', '<Dl>', '</Ul>', '</Ol>', \
             '</Dl>', '<Li>', '<Dd>', '<Dt>', '</Li>', '</Dd>', '</Dt>']
r_buf = ['is', 'are', 'do', 'does', 'did', 'was', 'were', 'will', 'can', 'the', 'a', 'an', 'of', 'in', 'and', 'on', \
         'what', 'where', 'when', 'which']


def clean(x, stop_words=[]):
    for r in html_tags:
        x = x.replace(r, '')
    for r in stop_words:
        x = x.replace(r, '')
    x = x.lower()
    x = re.sub(' +', ' ', x)
    return x


feature_names = [
    'qa_cos_d', 'qd_cos_d', 'ad_cos_d', 
    'qa_euc_d', 'qd_euc_d', 'ad_euc_d',
    'qa_lev_d', 'qa_lev_r', 'qa_jar_s', 'qa_jaw_s',
    'qa_tfidf_score', 'qd_tfidf_score', 'ad_tfidf_score', 
    'document_tfidf_sum', 'question_tfidf_sum', 'answer_tfidf_sum'
]

def extract_features(document_tfidf, question_tfidf, answer_tfidf, document, question, answer):
    qa_cos_d = spatial.distance.cosine(question_tfidf, answer_tfidf)
    qd_cos_d = spatial.distance.cosine(question_tfidf, document_tfidf)
    ad_cos_d = spatial.distance.cosine(answer_tfidf, document_tfidf)

    qa_euc_d = np.linalg.norm(question_tfidf - answer_tfidf)
    qd_euc_d = np.linalg.norm(question_tfidf - document_tfidf)
    ad_euc_d = np.linalg.norm(answer_tfidf - document_tfidf)
    
    qa_lev_d = Levenshtein.distance(question, answer)
    qa_lev_r = Levenshtein.ratio(question, answer)
    qa_jar_s = Levenshtein.jaro(question, answer) 
    qa_jaw_s = Levenshtein.jaro_winkler(question, answer)
    
    qa_tfidf_score = np.sum(question_tfidf*answer_tfidf.T)
    qd_tfidf_score = np.sum(question_tfidf*document_tfidf.T)
    ad_tfidf_score = np.sum(answer_tfidf*document_tfidf.T)
    
    document_tfidf_sum = np.sum(document_tfidf)
    question_tfidf_sum = np.sum(question_tfidf)
    answer_tfidf_sum = np.sum(answer_tfidf)
    
    f = [
        qa_cos_d, qd_cos_d, ad_cos_d, 
        qa_euc_d, qd_euc_d, ad_euc_d,
        qa_lev_d, qa_lev_r, qa_jar_s, qa_jaw_s,
        qa_tfidf_score, qd_tfidf_score, ad_tfidf_score, 
        document_tfidf_sum, question_tfidf_sum, answer_tfidf_sum
    ]       
    return f


def process_sample(args):
    json_data, annotated = args
    
    ids = []
    candidates_str = []
    targets = []
    targets_str = []
    targets_str_short = []
    features = []
    rank_features = []

    document = json_data['document_text']
        
    # TFIDF for document
    stop_words = text.ENGLISH_STOP_WORDS.union(['book'])
    tfidf = TfidfVectorizer(ngram_range=(1, 1), stop_words=stop_words)
    tfidf.fit([document])

    document_tfidf = tfidf.transform([document]).todense()

    # TFIDF for question
    question = json_data['question_text']
    question_tfidf = tfidf.transform([question]).todense()

    if annotated:
        # Collect annotations
        start_token_true = json_data['annotations'][0]['long_answer']['start_token']
        end_token_true = json_data['annotations'][0]['long_answer']['end_token']

        # Collect short annotations
        if json_data['annotations'][0]['yes_no_answer'] == 'NONE':
            if len(json_data['annotations'][0]['short_answers']) > 0:
                s_ans = str(json_data['annotations'][0]['short_answers'][0]['start_token']) + ':' + \
                    str(json_data['annotations'][0]['short_answers'][0]['end_token'])
            else:
                s_ans = ''
        else:
            s_ans = json_data['annotations'][0]['yes_no_answer']

    cos_d_buf = []
    euc_d_buf = []
    lev_d_buf = []

    doc_tokenized = json_data['document_text'].split(' ')
    candidates = json_data['long_answer_candidates']
    candidates = [c for c in candidates if c['top_level'] == True]

    if not annotated or start_token_true != -1:
        for c in candidates:
            ids.append(str(json_data['example_id']))

            # TFIDF for candidate answer
            start_token = c['start_token']
            end_token = c['end_token']
            answer = ' '.join(doc_tokenized[start_token:end_token])
            answer_tfidf = tfidf.transform([answer]).todense()

            # Extract some features
            f = extract_features(document_tfidf, question_tfidf, answer_tfidf, 
                                 clean(document), clean(question, stop_words=r_buf), clean(answer))

            cos_d_buf.append(f[0])
            euc_d_buf.append(f[3])
            lev_d_buf.append(f[6])

            features.append(f)

            if annotated:
                targets_str.append(str(start_token_true) + ':' + str(end_token_true))
                targets_str_short.append(s_ans)
                # Get target
                if start_token == start_token_true and end_token == end_token_true:
                    target = 1
                else:
                    target = 0
                targets.append(target)
                
            candidates_str.append(str(start_token) + ':' + str(end_token))
            
        features = np.array(features)
        
        rank_cos_d = np.argsort(np.argsort(cos_d_buf))
        rank_euc_d = np.argsort(np.argsort(euc_d_buf))
        rank_lev_d = np.argsort(np.argsort(lev_d_buf))
        rank_cos_d_ismin = (cos_d_buf == np.nanmin(cos_d_buf)).astype(int)
        rank_euc_d_ismin = (euc_d_buf == np.nanmin(euc_d_buf)).astype(int)
        rank_lev_d_ismin = (lev_d_buf == np.nanmin(lev_d_buf)).astype(int)
        rank_features = np.array([rank_cos_d, rank_euc_d, rank_lev_d, \
                                       rank_cos_d_ismin, rank_euc_d_ismin, rank_lev_d_ismin]).T

    return {
        'ids': ids,
        'candidates_str': candidates_str,
        'targets': targets,
        'targets_str': targets_str,
        'targets_str_short': targets_str_short,
        'features': features,
        'rank_features': rank_features
    }

def get_test():
    ids = []
    question_tfidfs = []
    answer_tfidfs = []
    candidates_str = []
    features = []
    rank_features = []

    with open('/kaggle/input/tensorflow2-question-answering/simplified-nq-test.jsonl', 'r') as json_file:   
        batch = []
        batch_cnt = 0
        for line in tqdm(json_file):
            json_data = json.loads(line) 
            
            r = process_sample((json_data, False))
            if len(r['ids']) > 0:
                ids += r['ids']
                candidates_str += r['candidates_str']
                features.append(r['features'])
                rank_features.append(r['rank_features'])
                

    test = pd.DataFrame()
    test['example_id'] = ids
    test['CandidateString'] = candidates_str

    features = np.concatenate(features, axis=0)
    features_df = pd.DataFrame(features)
    features_df.columns = feature_names
    test = pd.concat([test, features_df], axis=1)

    rank_features = np.concatenate(rank_features, axis=0)
    rank_features_df = pd.DataFrame(rank_features)
    rank_features_df.columns = [f'rank_feature_{i}' for i in range(rank_features.shape[1])]
    test = pd.concat([test, rank_features_df], axis=1)

    del features, features_df, rank_features, rank_features_df
    gc.collect()
    
    return test


test = get_test()
test.to_csv('test_data.csv', index=False)
print(f'test.shape: {test.shape}')
test.head(10)
p_buf = []
n_splits = 4

kf = GroupKFold(
    n_splits=n_splits)

err_buf = []   

cols_to_drop = ['example_id', 'target', 'CorrectString', 'CorrectString_short', 'CandidateString']

X = train.drop(cols_to_drop, axis=1, errors='ignore')
y = train['target'].values
g = train['example_id'].values

X_test = test.drop(cols_to_drop, axis=1, errors='ignore')
id_test = test['example_id'].values

print(f'X.shape: {X.shape}, y.shape: {y.shape}')
print(f'X_test.shape: {X_test.shape}')

n_features = X.shape[1]

lgb_params = {
    'boosting_type': 'gbdt',
    'objective': 'binary',
    'metric': 'binary_logloss',
    'max_depth': 16,
    'learning_rate': 0.005, 
    'feature_fraction': 0.9,
    'bagging_fraction': 0.9,
    'bagging_freq': 5,
    'verbose': -1,
    'num_threads': 4,
}

for fold_i, (train_index, valid_index) in enumerate(kf.split(X, y, g)):
    print('Fold {}/{}'.format(fold_i + 1, n_splits))
    params = lgb_params.copy() 
    
    X_train, y_train = X.iloc[train_index], y[train_index]
    X_valid, y_valid = X.iloc[valid_index], y[valid_index]

    print(f'X_train.shape: {X_train.shape}, X_valid.shape: {X_valid.shape}')
    feature_names = list(X_train.columns)

    lgb_train = lgb.Dataset(
        X_train, 
        y_train, 
        feature_name=feature_names,
        )
    lgb_train.raw_data = None

    lgb_valid = lgb.Dataset(
        X_valid, 
        y_valid,
        feature_name=feature_names,
        )
    lgb_valid.raw_data = None

    model = lgb.train(
        params,
        lgb_train,
        num_boost_round=4000,
        valid_sets=[lgb_train, lgb_valid],
        early_stopping_rounds=100, 
        verbose_eval=400, 
    )

    # Feature importance
    if fold_i == 0:
        importance = model.feature_importance()
        model_fnames = model.feature_name()
        tuples = sorted(zip(model_fnames, importance), key=lambda x: x[1])[::-1]
        tuples = [x for x in tuples if x[1] > 0]
        print('Important features:')
        for i in range(20):
            if i < len(tuples):
                print(i, tuples[i])
            else:
                break

    # Evaluate model
    p = model.predict(X.loc[valid_index], num_iteration=model.best_iteration)
    valid_df = train.loc[valid_index]
    valid_df['pred'] = p
    pred_df = valid_df.sort_values('pred', ascending=True).groupby('example_id').tail(1)

    pred_df_long = pred_df[['example_id', 'CorrectString', 'CandidateString']]
    pred_df_long.rename({'CandidateString': 'PredictionString'}, axis=1, inplace=True)
    pred_df_long['example_id'] = pred_df_long['example_id'].apply(lambda x: str(x) + '_long')

    pred_df_short = pred_df[['example_id', 'CorrectString_short', 'CandidateString']]
    pred_df_short.rename({'CorrectString_short': 'CorrectString', 'CandidateString': 'PredictionString'}, \
                         axis=1, inplace=True)
    pred_df_short['example_id'] = pred_df_short['example_id'].apply(lambda x: str(x) + '_short')
    pred_df_short['PredictionString'] = ''

    pred_df = pd.concat([pred_df_long, pred_df_short], axis=0).sort_values('example_id')
#     print(pred_df.head(20))

    err = f1_score(pred_df['CorrectString'].fillna('').values, 
                   pred_df['PredictionString'].fillna('').values, 
                   average='micro')
    print('{} F1: {}'.format(fold_i, err))
    
    # Inference on test data
    p_test = model.predict(X_test[feature_names], num_iteration=model.best_iteration)
    p_buf.append(p_test)
    err_buf.append(err)

#     if fold_i >= 0: # Comment this to run several folds
#         break

    del model, lgb_train, lgb_valid, p
    gc.collect()
err_mean = np.mean(err_buf)
err_std = np.std(err_buf)
print('F1 = {:.4f} +/- {:.4f}'.format(err_mean, err_std))
valid_df = train.loc[valid_index]
test['pred'] = np.mean(p_buf, axis=0)
pred_df = test.sort_values('pred', ascending=True).groupby('example_id').tail(1)

pred_df_long = pred_df[['example_id', 'CandidateString']]
pred_df_long.rename({'CandidateString': 'PredictionString'}, axis=1, inplace=True)
pred_df_long['example_id'] = pred_df_long['example_id'].apply(lambda x: str(x) + '_long')

pred_df_short = pred_df[['example_id', 'CandidateString']]
pred_df_short.rename({'CandidateString': 'PredictionString'}, axis=1, inplace=True)
pred_df_short['example_id'] = pred_df_short['example_id'].apply(lambda x: str(x) + '_short')
pred_df_short['PredictionString'] = ''
# pred_df_short['PredictionString'] = np.nan # https://www.kaggle.com/c/tensorflow2-question-answering/discussion/115836

subm = pd.concat([pred_df_long, pred_df_short], axis=0).sort_values('example_id')
subm.to_csv('submission.csv', index=False)
print(f'subm.shape: {subm.shape}')
subm.head(20)
