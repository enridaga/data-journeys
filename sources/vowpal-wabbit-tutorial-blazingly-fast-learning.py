
import warnings
warnings.filterwarnings('ignore')
import os
import re
import numpy as np
import pandas as pd
from tqdm import tqdm_notebook
from sklearn.datasets import fetch_20newsgroups, load_files
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score, log_loss
from sklearn.metrics import roc_auc_score, roc_curve, confusion_matrix
from scipy.sparse import csr_matrix
import matplotlib.pyplot as plt
#%matplotlib inline
import seaborn as sns
PATH_TO_ALL_DATA = '../input/spooky-vw-tutorial/'
data_demo = pd.read_csv(os.path.join(PATH_TO_ALL_DATA, 'weights_heights.csv'))
plt.scatter(data_demo['Weight'], data_demo['Height']);
plt.xlabel('Weight in lb')
plt.ylabel('Height in inches');
df = pd.read_csv(os.path.join(PATH_TO_ALL_DATA, 'bank_train.csv'))
labels = pd.read_csv(os.path.join(PATH_TO_ALL_DATA,
                                  'bank_train_target.csv'), header=None)

df.head()
df['education'].value_counts().plot.barh();
label_encoder = LabelEncoder()
mapped_education = pd.Series(label_encoder.fit_transform(df['education']))
mapped_education.value_counts().plot.barh()
print(dict(enumerate(label_encoder.classes_)))
df['education'] = mapped_education
df.head()
categorical_columns = df.columns[df.dtypes == 'object'].union(['education'])
for column in categorical_columns:
    df[column] = label_encoder.fit_transform(df[column])
df.head()
df.loc[1].job - df.loc[2].job
def logistic_regression_accuracy_on(dataframe, labels):
    features = dataframe.as_matrix()
    train_features, test_features, train_labels, test_labels = \
        train_test_split(features, labels)

    logit = LogisticRegression()
    logit.fit(train_features, train_labels)
    return classification_report(test_labels, logit.predict(test_features))

print(logistic_regression_accuracy_on(df[categorical_columns], labels))
one_hot_example = pd.DataFrame([{i: 0 for i in range(10)}])
one_hot_example.loc[0, 6] = 1
one_hot_example
onehot_encoder = OneHotEncoder(sparse=False)
encoded_categorical_columns = pd.DataFrame(onehot_encoder.fit_transform(df[categorical_columns]))
encoded_categorical_columns.head()
print(logistic_regression_accuracy_on(encoded_categorical_columns, labels))
for s in ('university.degree', 'high.school', 'illiterate'):
    print(s, '->', hash(s))
hash_space = 25
for s in ('university.degree', 'high.school', 'illiterate'):
    print(s, '->', hash(s) % hash_space)
hashing_example = pd.DataFrame([{i: 0.0 for i in range(hash_space)}])
for s in ('job=student', 'marital=single', 'day_of_week=mon'):
    print(s, '->', hash(s) % hash_space)
    hashing_example.loc[0, hash(s) % hash_space] = 1
hashing_example
assert hash('no') == hash('no')
assert hash('housing=no') != hash('loan=no')
# !vw --help
#! echo '1 1.0 |Subject WHAT car is this |Organization University of Maryland:0.5 College Park' | vw
# load data with sklearn's fubction 
newsgroups = fetch_20newsgroups(PATH_TO_ALL_DATA)
newsgroups['target_names']
text = newsgroups['data'][0]
target = newsgroups['target_names'][newsgroups['target'][0]]

print('-----')
print(target)
print('-----')
print(text.strip())
print('----')
def to_vw_format(document, label=None):
    return str(label or '') + ' |text ' + ' '.join(re.findall('\w{3,}', document.lower())) + '\n'

to_vw_format(text, 1 if target == 'rec.autos' else -1)
all_documents = newsgroups['data']
all_targets = [1 if newsgroups['target_names'][target] == 'rec.autos' 
               else -1 for target in newsgroups['target']]
train_documents, test_documents, train_labels, test_labels = \
    train_test_split(all_documents, all_targets, random_state=7)
    
# with open(os.path.join(PATH_TO_ALL_DATA, '20news_train.vw'), 'w') as vw_train_data:
#     for text, target in zip(train_documents, train_labels):
#         vw_train_data.write(to_vw_format(text, target))
# with open(os.path.join(PATH_TO_ALL_DATA, '20news_test.vw'), 'w') as vw_test_data:
#     for text in test_documents:
#         vw_test_data.write(to_vw_format(text))
# !vw -d $PATH_TO_ALL_DATA/20news_train.vw \
#  --loss_function hinge -f $PATH_TO_ALL_DATA/20news_model.vw
# !vw -i $PATH_TO_ALL_DATA/20news_model.vw -t -d $PATH_TO_ALL_DATA/20news_test.vw \
# -p $PATH_TO_ALL_DATA/20news_test_predictions.txt
with open(os.path.join(PATH_TO_ALL_DATA, '20news_test_predictions.txt')) as pred_file:
    test_prediction = [float(label) 
                             for label in pred_file.readlines()]

auc = roc_auc_score(test_labels, test_prediction)
roc_curve = roc_curve(test_labels, test_prediction)

with plt.xkcd():
    plt.plot(roc_curve[0], roc_curve[1]);
    plt.plot([0,1], [0,1])
    plt.xlabel('FPR'); plt.ylabel('TPR'); plt.title('test AUC = %f' % (auc)); plt.axis([-0.05,1.05,-0.05,1.05]);
all_documents = newsgroups['data']
topic_encoder = LabelEncoder()
all_targets_mult = topic_encoder.fit_transform(newsgroups['target']) + 1
train_documents, test_documents, train_labels_mult, test_labels_mult = \
    train_test_split(all_documents, all_targets_mult, random_state=7)
    
# with open(os.path.join(PATH_TO_ALL_DATA, '20news_train_mult.vw'), 'w') as vw_train_data:
#     for text, target in zip(train_documents, train_labels_mult):
#         vw_train_data.write(to_vw_format(text, target))
# with open(os.path.join(PATH_TO_ALL_DATA, '20news_test_mult.vw'), 'w') as vw_test_data:
#     for text in test_documents:
#         vw_test_data.write(to_vw_format(text))
# %%time
# !vw --oaa 20 $PATH_TO_ALL_DATA/20news_train_mult.vw -f $PATH_TO_ALL_DATA/20news_model_mult.vw \
# --loss_function=hinge
# %%time
# !vw -i $PATH_TO_ALL_DATA/20news_model_mult.vw -t -d $PATH_TO_ALL_DATA/20news_test_mult.vw \
# -p $PATH_TO_ALL_DATA/20news_test_predictions_mult.txt
with open(os.path.join(PATH_TO_ALL_DATA, '20news_test_predictions_mult.txt')) as pred_file:
    test_prediction_mult = [float(label) for label in pred_file.readlines()]
accuracy_score(test_labels_mult, test_prediction_mult)
M = confusion_matrix(test_labels_mult, test_prediction_mult)
for i in np.where(M[0,:] > 0)[0][1:]:
    print(newsgroups['target_names'][i], M[0,i])
import pickle
# path_to_movies = os.path.expanduser('/Users/y.kashnitsky/Documnents/imdb_reviews')
# reviews_train = load_files(os.path.join(path_to_movies, 'train'))
with open(os.path.join(PATH_TO_ALL_DATA, 'reviews_train.pkl'), 'rb') as reviews_train_pkl:
    reviews_train = pickle.load(reviews_train_pkl)
text_train, y_train = reviews_train.data, reviews_train.target
print("Number of documents in training data: %d" % len(text_train))
print(np.bincount(y_train))
# reviews_test = load_files(os.path.join(path_to_movies, 'test'))
with open(os.path.join(PATH_TO_ALL_DATA, 'reviews_test.pkl'), 'rb') as reviews_test_pkl:
    reviews_test = pickle.load(reviews_test_pkl)
text_test, y_test = reviews_test.data, reviews_train.target
print("Number of documents in test data: %d" % len(text_test))
print(np.bincount(y_test))
text_train[0]
y_train[0] # good review
text_train[1]
y_train[1] # bad review
to_vw_format(str(text_train[1]), 1 if y_train[0] == 1 else -1)
train_share = int(0.7 * len(text_train))
train, valid = text_train[:train_share], text_train[train_share:]
train_labels, valid_labels = y_train[:train_share], y_train[train_share:]
len(train_labels), len(valid_labels)
# with open('movie_reviews_train.vw', 'w') as vw_train_data:
#     for text, target in zip(train, train_labels):
#         vw_train_data.write(to_vw_format(str(text), 1 if target == 1 else -1))
# with open('movie_reviews_valid.vw', 'w') as vw_train_data:
#     for text, target in zip(valid, valid_labels):
#         vw_train_data.write(to_vw_format(str(text), 1 if target == 1 else -1))
# with open('movie_reviews_test.vw', 'w') as vw_test_data:
#     for text in text_test:
#         vw_test_data.write(to_vw_format(str(text)))
!head -2 $PATH_TO_ALL_DATA/movie_reviews_train.vw
!head -2 $PATH_TO_ALL_DATA/movie_reviews_valid.vw
!head -2 $PATH_TO_ALL_DATA/movie_reviews_test.vw
# !vw -d $PATH_TO_ALL_DATA/movie_reviews_train.vw --loss_function hinge \
# -f $PATH_TO_ALL_DATA/movie_reviews_model.vw --quiet
# !vw -i $PATH_TO_ALL_DATA/movie_reviews_model.vw -t \
# -d $PATH_TO_ALL_DATA/movie_reviews_valid.vw -p $PATH_TO_ALL_DATA/movie_valid_pred.txt --quiet
with open(os.path.join(PATH_TO_ALL_DATA, 'movie_valid_pred.txt')) as pred_file:
    valid_prediction = [float(label) 
                             for label in pred_file.readlines()]
print("Accuracy: {}".format(round(accuracy_score(valid_labels, 
               [int(pred_prob > 0) for pred_prob in valid_prediction]), 3)))
print("AUC: {}".format(round(roc_auc_score(valid_labels, valid_prediction), 3)))
# !vw -i $PATH_TO_ALL_DATA/movie_reviews_model.vw -t 
# -d $PATH_TO_ALL_DATA/movie_reviews_test.vw \
# -p $PATH_TO_ALL_DATA/movie_test_pred.txt --quiet
with open(os.path.join(PATH_TO_ALL_DATA, 'movie_test_pred.txt')) as pred_file:
    test_prediction = [float(label) 
                             for label in pred_file.readlines()]
print("Accuracy: {}".format(round(accuracy_score(y_test, 
               [int(pred_prob > 0) for pred_prob in test_prediction]), 3)))
print("AUC: {}".format(round(roc_auc_score(y_test, test_prediction), 3)))
# !vw -d $PATH_TO_ALL_DATA/movie_reviews_train.vw \
# --loss_function hinge --ngram 2 -f $PATH_TO_ALL_DATA/movie_reviews_model2.vw --quiet
# !vw -i$PATH_TO_ALL_DATA/ movie_reviews_model2.vw -t -d $PATH_TO_ALL_DATA/movie_reviews_valid.vw \
# -p $PATH_TO_ALL_DATA/movie_valid_pred2.txt --quiet
with open(os.path.join(PATH_TO_ALL_DATA, 'movie_valid_pred2.txt')) as pred_file:
    valid_prediction = [float(label) 
                             for label in pred_file.readlines()]
print("Accuracy: {}".format(round(accuracy_score(valid_labels, 
               [int(pred_prob > 0) for pred_prob in valid_prediction]), 3)))
print("AUC: {}".format(round(roc_auc_score(valid_labels, valid_prediction), 3)))
# !vw -i $PATH_TO_ALL_DATA/movie_reviews_model2.vw -t -d $PATH_TO_ALL_DATA/movie_reviews_test.vw \
# -p $PATH_TO_ALL_DATA/movie_test_pred2.txt --quiet
with open(os.path.join(PATH_TO_ALL_DATA, 'movie_test_pred2.txt')) as pred_file:
    test_prediction2 = [float(label) 
                             for label in pred_file.readlines()]
print("Accuracy: {}".format(round(accuracy_score(y_test, 
               [int(pred_prob > 0) for pred_prob in test_prediction2]), 3)))
print("AUC: {}".format(round(roc_auc_score(y_test, test_prediction2), 3)))
# change the path to data
# PATH_TO_STACKOVERFLOW_DATA = '/Users/y.kashnitsky/Documents/stackoverflow'
!head -3 $PATH_TO_ALL_DATA/stackoverflow_sample.vw
# !du -hs $PATH_TO_STACKOVERFLOW_DATA/stackoverflow_*.vw
# 4,7G stackoverflow_10mln.vw
# 1,6G stackoverflow_test.vw
# 3,1G stackoverflow_train.vw
# %%time
# !vw --oaa 10 -d $PATH_TO_STACKOVERFLOW_DATA/stackoverflow_train.vw \
# -f vw_model1_10mln.vw -b 28 --random_seed 17 --quiet
# CPU times: user 567 ms, sys: 212 ms, total: 778 ms
# Wall time: 36.5 s
# %%time
# !vw -t -i vw_model1_10mln.vw -d $PATH_TO_STACKOVERFLOW_DATA/stackoverflow_test.vw \
# -p vw_test_pred.csv --random_seed 17 --quiet
# CPU times: user 222 ms, sys: 86.4 ms, total: 308 ms
# Wall time: 14.4 s
vw_pred = np.loadtxt(os.path.join(PATH_TO_ALL_DATA, 'vw_test_pred.csv'))
test_labels = np.loadtxt(os.path.join(PATH_TO_ALL_DATA, 'stackoverflow_test_labels.txt'))
accuracy_score(test_labels, vw_pred)
train_texts = pd.read_csv('../input/spooky-author-identification/train.csv', index_col='id')
test_texts = pd.read_csv('../input/spooky-author-identification/test.csv', index_col='id')
sample_sub = pd.read_csv('../input/spooky-author-identification/sample_submission.csv', 
                         index_col='id')
author_code = {"EAP": 1, "MWS": 2,"HPL": 3}
train_texts["author_code"] = train_texts["author"].map(author_code)
train_texts_part, valid_texts = train_test_split(train_texts, test_size=0.3, random_state=17, 
                                                 stratify=train_texts["author_code"], shuffle=True)
train_texts_part.shape[0], valid_texts.shape[0]
def to_vw_only_text(out_vw, df, is_train=True):
    with open(out_vw, "w") as out:
        for i in range(df.shape[0]):
            
            if is_train:
                target = df["author_code"].iloc[i]
            else:
                # for the test set we can pick any target label – we don't need it actually
                target = 1 
                       
            # remove special VW symbols
            text = df["text"].iloc[i].strip().replace('|', '').replace(':', '').lower() 
            # leave only words of 3 and more chars
            words = re.findall("\w{3,}", text) 
            new_text = " ".join(words) 

            s = "{} |text {}\n".format(target, new_text)

            out.write(s)    
# to_vw_only_text("train_part_only_text.vw", train_texts_part)
!head -2 $PATH_TO_ALL_DATA/train_part_only_text.vw
# to_vw_only_text("valid_only_text.vw", valid_texts)
!head -2 $PATH_TO_ALL_DATA/valid_only_text.vw
# to_vw_only_text("train_only_text.vw", train_texts)
!head -2 $PATH_TO_ALL_DATA/train_only_text.vw
# to_vw_only_text("test_only_text.vw", test_texts, is_train=False)
!head -2 $PATH_TO_ALL_DATA/test_only_text.vw
# !vw --oaa 3 train_part_only_text.vw -f model_only_text_part.vw -b 28 --random_seed 17 \
# --loss_function logistic --ngram 2 --passes 10 -k -c
# %%time
# !vw -i model_only_text_part.vw -t -d valid_only_text.vw -p valid_pred1.txt --random_seed 17 -r valid_prob1.txt
def evaluate_vw_prediction(path_to_vw_pred_probs, is_test=False, target=None, write_submission=False,
                          submission_file=None, test_index=test_texts.index, columns=['EAP', 'MWS', 'HPL']):
    def sigmoid(z):
        return 1 / (1 + np.exp(-z)) 
    
    with open(path_to_vw_pred_probs) as pred_file:
        pred_probs =  np.array([[float(pair.split(':')[1]) for pair in line.strip().split()] 
                         for line in pred_file.readlines()])
        pred_probs  = sigmoid(pred_probs)
        
        if target is not None and not is_test:
            print(log_loss(target, pred_probs))
        
        if write_submission and submission_file is not None:
            subm_df = pd.DataFrame(pred_probs, columns=columns)
            subm_df.index = test_index
            subm_df.to_csv(submission_file)
evaluate_vw_prediction(os.path.join(PATH_TO_ALL_DATA, 'valid_prob1.txt'), 
                       target=valid_texts['author_code'])
# !vw --oaa 3 train_only_text.vw -f model_only_text.vw -b 28 --random_seed 17 \
# --loss_function logistic --ngram 2 --passes 10 -k -c --quiet
# %%time
# !vw -i model_only_text.vw -t -d test_only_text.vw -p test_pred1.txt --random_seed 17 -r test_prob1.txt --quiet
#evaluate_vw_prediction(os.path.join(PATH_TO_ALL_DATA, 'test_prob1.txt'), 
                       is_test=True, write_submission=True,
                       submission_file='submission1_only_text.csv')
#!head -3 submission1_only_text.csv
max_words_in_text = train_texts['text'].apply(lambda text: len(re.findall("\w{3,}", text.strip()))).max()
max_unique_words_in_text = train_texts['text'].apply(lambda text: len(set(re.findall("\w{3,}", text.strip())))).max()
max_aver_word_len_in_text = train_texts['text'].apply(lambda text: 
                                                      sum([len(w) for w in re.findall("\w{3,}", text.strip())]) / 
                                                      len(re.findall("\w{3,}", text.strip()))).max()
max_words_in_text, max_unique_words_in_text, max_aver_word_len_in_text
def to_vw_text_and_some_features(out_vw, df, is_train=True):
    with open(out_vw, "w") as out:
        for i in range(df.shape[0]):
            
            if is_train:
                target = df["author_code"].iloc[i]
            else:
                # for the test set we can pick any target label – we don't need it actually
                target = 1 
                       
            # remove special VW symbols
            text = df["text"].iloc[i].strip().replace('|', '').replace(':', '').lower() 
            # leave only words of 3 and more chars
            words = re.findall("\w{3,}", text) 
            new_text = " ".join(words)    
            
            num_words = round(len(words) / max_words_in_text, 4)
            num_uniq_words = round(len(set(words)) / max_unique_words_in_text, 4)
            aver_word_len = round(sum([len(w) for w in words]) / len(words) / max_aver_word_len_in_text, 4)

            features = [num_words, num_uniq_words, aver_word_len] 
            features_vw = ' '.join(['{}:{}'.format(i[0], i[1]) for i in zip(range(len(features)), features)])
            s = "{} |text {} |num {}\n".format(target, new_text, features_vw)

            out.write(s)   
 
# to_vw_text_and_some_features("train_part_text_feat.vw", train_texts_part)
!head -2 $PATH_TO_ALL_DATA/train_part_text_feat.vw
# to_vw_text_and_some_features("valid_text_feat.vw", valid_texts)
# to_vw_text_and_some_features("train_text_feat.vw", train_texts)
# to_vw_text_and_some_features("test_text_feat.vw", test_texts, is_train=False)
# !vw --oaa 3 train_part_text_feat.vw -f model_text_feat_part.vw -b 28 --random_seed 17 \
# --loss_function logistic --ngram 2 --passes 10 -k -c --quiet
# %%time
# !vw -i model_text_feat_part.vw -t -d valid_text_feat.vw -p valid_pred2.txt --random_seed 17 -r valid_prob2.txt --quiet
evaluate_vw_prediction(os.path.join(PATH_TO_ALL_DATA, 'valid_prob2.txt'), 
                       target=valid_texts['author_code'])
# !vw --oaa 3 train_text_feat.vw -f model_text_feat.vw -b 28 --random_seed 17 \
# --loss_function logistic --ngram 2 --passes 10 -k -c --quiet
# %%time
# !vw -i model_text_feat.vw -t -d test_text_feat.vw -p test_pred2.txt --random_seed 17 -r test_prob2.txt --quiet
evaluate_vw_prediction(os.path.join(PATH_TO_ALL_DATA, 'test_prob2.txt'), 
                       is_test=True, write_submission=True,
                       submission_file='submission2_text_feat.csv')
def validate_submission_local_and_lb_mix(local_score, public_lb_score, local_size=5874, public_lb_size=2517):
    return 1. / (local_size + public_lb_size) * (local_size * local_score +
                                                public_lb_size * public_lb_score)
# first submission
validate_submission_local_and_lb_mix(local_score=.47951, public_lb_score=.43187)
# second submission
validate_submission_local_and_lb_mix(local_score=.469, 
                                      public_lb_score=.43267)