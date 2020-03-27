
import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os

import gc

import matplotlib.pyplot as plt

import seaborn as sns

#%matplotlib inline



pal = sns.color_palette()



print('# File sizes')

for f in os.listdir('../input'):

    if 'zip' not in f:

        print(f.ljust(30) + str(round(os.path.getsize('../input/' + f) / 1000000, 2)) + 'MB')
df_train = pd.read_csv('../input/train.csv')

df_train.head()
print('Total number of question pairs for training: {}'.format(len(df_train)))

print('Duplicate pairs: {}%'.format(round(df_train['is_duplicate'].mean()*100, 2)))

qids = pd.Series(df_train['qid1'].tolist() + df_train['qid2'].tolist())

print('Total number of questions in the training data: {}'.format(len(

    np.unique(qids))))

print('Number of questions that appear multiple times: {}'.format(np.sum(qids.value_counts() > 1)))



plt.figure(figsize=(12, 5))

plt.hist(qids.value_counts(), bins=50)

plt.yscale('log', nonposy='clip')

plt.title('Log-Histogram of question appearance counts')

plt.xlabel('Number of occurences of question')

plt.ylabel('Number of questions')

print()
from sklearn.metrics import log_loss



p = df_train['is_duplicate'].mean() # Our predicted probability

print('Predicted score:', log_loss(df_train['is_duplicate'], np.zeros_like(df_train['is_duplicate']) + p))



df_test = pd.read_csv('../input/test.csv')

sub = pd.DataFrame({'test_id': df_test['test_id'], 'is_duplicate': p})

sub.to_csv('naive_submission.csv', index=False)

sub.head()
df_test = pd.read_csv('../input/test.csv')

df_test.head()
print('Total number of question pairs for testing: {}'.format(len(df_test)))
train_qs = pd.Series(df_train['question1'].tolist() + df_train['question2'].tolist()).astype(str)

test_qs = pd.Series(df_test['question1'].tolist() + df_test['question2'].tolist()).astype(str)



dist_train = train_qs.apply(len)

dist_test = test_qs.apply(len)

plt.figure(figsize=(15, 10))

plt.hist(dist_train, bins=200, range=[0, 200], color=pal[2], normed=True, label='train')

plt.hist(dist_test, bins=200, range=[0, 200], color=pal[1], normed=True, alpha=0.5, label='test')

plt.title('Normalised histogram of character count in questions', fontsize=15)

plt.legend()

plt.xlabel('Number of characters', fontsize=15)

plt.ylabel('Probability', fontsize=15)



print('mean-train {:.2f} std-train {:.2f} mean-test {:.2f} std-test {:.2f} max-train {:.2f} max-test {:.2f}'.format(dist_train.mean(), 

                          dist_train.std(), dist_test.mean(), dist_test.std(), dist_train.max(), dist_test.max()))
dist_train = train_qs.apply(lambda x: len(x.split(' ')))

dist_test = test_qs.apply(lambda x: len(x.split(' ')))



plt.figure(figsize=(15, 10))

plt.hist(dist_train, bins=50, range=[0, 50], color=pal[2], normed=True, label='train')

plt.hist(dist_test, bins=50, range=[0, 50], color=pal[1], normed=True, alpha=0.5, label='test')

plt.title('Normalised histogram of word count in questions', fontsize=15)

plt.legend()

plt.xlabel('Number of words', fontsize=15)

plt.ylabel('Probability', fontsize=15)



print('mean-train {:.2f} std-train {:.2f} mean-test {:.2f} std-test {:.2f} max-train {:.2f} max-test {:.2f}'.format(dist_train.mean(), 

                          dist_train.std(), dist_test.mean(), dist_test.std(), dist_train.max(), dist_test.max()))
from wordcloud import WordCloud

cloud = WordCloud(width=1440, height=1080).generate(" ".join(train_qs.astype(str)))

plt.figure(figsize=(20, 15))

plt.imshow(cloud)

plt.axis('off')
qmarks = np.mean(train_qs.apply(lambda x: '?' in x))

math = np.mean(train_qs.apply(lambda x: '[math]' in x))

fullstop = np.mean(train_qs.apply(lambda x: '.' in x))

capital_first = np.mean(train_qs.apply(lambda x: x[0].isupper()))

capitals = np.mean(train_qs.apply(lambda x: max([y.isupper() for y in x])))

numbers = np.mean(train_qs.apply(lambda x: max([y.isdigit() for y in x])))



print('Questions with question marks: {:.2f}%'.format(qmarks * 100))

print('Questions with [math] tags: {:.2f}%'.format(math * 100))

print('Questions with full stops: {:.2f}%'.format(fullstop * 100))

print('Questions with capitalised first letters: {:.2f}%'.format(capital_first * 100))

print('Questions with capital letters: {:.2f}%'.format(capitals * 100))

print('Questions with numbers: {:.2f}%'.format(numbers * 100))
from nltk.corpus import stopwords



stops = set(stopwords.words("english"))



def word_match_share(row):

    q1words = {}

    q2words = {}

    for word in str(row['question1']).lower().split():

        if word not in stops:

            q1words[word] = 1

    for word in str(row['question2']).lower().split():

        if word not in stops:

            q2words[word] = 1

    if len(q1words) == 0 or len(q2words) == 0:

        # The computer-generated chaff includes a few questions that are nothing but stopwords

        return 0

    shared_words_in_q1 = [w for w in q1words.keys() if w in q2words]

    shared_words_in_q2 = [w for w in q2words.keys() if w in q1words]

    R = (len(shared_words_in_q1) + len(shared_words_in_q2))/(len(q1words) + len(q2words))

    return R



plt.figure(figsize=(15, 5))

train_word_match = df_train.apply(word_match_share, axis=1, raw=True)

plt.hist(train_word_match[df_train['is_duplicate'] == 0], bins=20, normed=True, label='Not Duplicate')

plt.hist(train_word_match[df_train['is_duplicate'] == 1], bins=20, normed=True, alpha=0.7, label='Duplicate')

plt.legend()

plt.title('Label distribution over word_match_share', fontsize=15)

plt.xlabel('word_match_share', fontsize=15)
from collections import Counter



# If a word appears only once, we ignore it completely (likely a typo)

# Epsilon defines a smoothing constant, which makes the effect of extremely rare words smaller

def get_weight(count, eps=10000, min_count=2):

    if count < min_count:

        return 0

    else:

        return 1 / (count + eps)



eps = 5000 

words = (" ".join(train_qs)).lower().split()

counts = Counter(words)

weights = {word: get_weight(count) for word, count in counts.items()}
print('Most common words and weights: \n')

print(sorted(weights.items(), key=lambda x: x[1] if x[1] > 0 else 9999)[:10])

print('\nLeast common words and weights: ')

(sorted(weights.items(), key=lambda x: x[1], reverse=True)[:10])
def tfidf_word_match_share(row):

    q1words = {}

    q2words = {}

    for word in str(row['question1']).lower().split():

        if word not in stops:

            q1words[word] = 1

    for word in str(row['question2']).lower().split():

        if word not in stops:

            q2words[word] = 1

    if len(q1words) == 0 or len(q2words) == 0:

        # The computer-generated chaff includes a few questions that are nothing but stopwords

        return 0

    

    shared_weights = [weights.get(w, 0) for w in q1words.keys() if w in q2words] + [weights.get(w, 0) for w in q2words.keys() if w in q1words]

    total_weights = [weights.get(w, 0) for w in q1words] + [weights.get(w, 0) for w in q2words]

    

    R = np.sum(shared_weights) / np.sum(total_weights)

    return R
plt.figure(figsize=(15, 5))

tfidf_train_word_match = df_train.apply(tfidf_word_match_share, axis=1, raw=True)

plt.hist(tfidf_train_word_match[df_train['is_duplicate'] == 0].fillna(0), bins=20, normed=True, label='Not Duplicate')

plt.hist(tfidf_train_word_match[df_train['is_duplicate'] == 1].fillna(0), bins=20, normed=True, alpha=0.7, label='Duplicate')

plt.legend()

plt.title('Label distribution over tfidf_word_match_share', fontsize=15)

plt.xlabel('word_match_share', fontsize=15)
from sklearn.metrics import roc_auc_score

print('Original AUC:', roc_auc_score(df_train['is_duplicate'], train_word_match))

print('   TFIDF AUC:', roc_auc_score(df_train['is_duplicate'], tfidf_train_word_match.fillna(0)))
# First we create our training and testing data

x_train = pd.DataFrame()

x_test = pd.DataFrame()

x_train['word_match'] = train_word_match

x_train['tfidf_word_match'] = tfidf_train_word_match

x_test['word_match'] = df_test.apply(word_match_share, axis=1, raw=True)

x_test['tfidf_word_match'] = df_test.apply(tfidf_word_match_share, axis=1, raw=True)



y_train = df_train['is_duplicate'].values
pos_train = x_train[y_train == 1]

neg_train = x_train[y_train == 0]



# Now we oversample the negative class

# There is likely a much more elegant way to do this...

p = 0.165

scale = ((len(pos_train) / (len(pos_train) + len(neg_train))) / p) - 1

while scale > 1:

    neg_train = pd.concat([neg_train, neg_train])

    scale -=1

neg_train = pd.concat([neg_train, neg_train[:int(scale * len(neg_train))]])

print(len(pos_train) / (len(pos_train) + len(neg_train)))



x_train = pd.concat([pos_train, neg_train])

y_train = (np.zeros(len(pos_train)) + 1).tolist() + np.zeros(len(neg_train)).tolist()

del pos_train, neg_train
# Finally, we split some of the data off for validation

from sklearn.cross_validation import train_test_split



x_train, x_valid, y_train, y_valid = train_test_split(x_train, y_train, test_size=0.2, random_state=4242)
import xgboost as xgb



# Set our parameters for xgboost

params = {}

params['objective'] = 'binary:logistic'

params['eval_metric'] = 'logloss'

params['eta'] = 0.02

params['max_depth'] = 4



d_train = xgb.DMatrix(x_train, label=y_train)

d_valid = xgb.DMatrix(x_valid, label=y_valid)



watchlist = [(d_train, 'train'), (d_valid, 'valid')]



bst = xgb.train(params, d_train, 400, watchlist, early_stopping_rounds=50, verbose_eval=10)
d_test = xgb.DMatrix(x_test)

p_test = bst.predict(d_test)



sub = pd.DataFrame()

sub['test_id'] = df_test['test_id']

sub['is_duplicate'] = p_test

sub.to_csv('simple_xgb.csv', index=False)