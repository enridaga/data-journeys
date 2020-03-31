
import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns

from nltk.corpus import stopwords

from nltk import word_tokenize, ngrams

from sklearn import ensemble

from sklearn.model_selection import KFold

from sklearn.metrics import log_loss

import xgboost as xgb



eng_stopwords = set(stopwords.words('english'))

color = sns.color_palette()



### matplotlib inline



pd.options.mode.chained_assignment = None  # default='warn'
train_df = pd.read_csv("../input/train.csv")

test_df = pd.read_csv("../input/test.csv")

print(train_df.shape)

print(test_df.shape)
train_df.head()
test_df.head()
is_dup = train_df['is_duplicate'].value_counts()



plt.figure(figsize=(8,4))

sns.barplot(is_dup.index, is_dup.values, alpha=0.8, color=color[1])

plt.ylabel('Number of Occurrences', fontsize=12)

plt.xlabel('Is Duplicate', fontsize=12)

plt.show()
is_dup / is_dup.sum()
all_ques_df = pd.DataFrame(pd.concat([train_df['question1'], train_df['question2']]))

all_ques_df.columns = ["questions"]



all_ques_df["num_of_words"] = all_ques_df["questions"].apply(lambda x : len(str(x).split()))
cnt_srs = all_ques_df['num_of_words'].value_counts()



plt.figure(figsize=(12,6))

sns.barplot(cnt_srs.index, cnt_srs.values, alpha=0.8, color=color[0])

plt.ylabel('Number of Occurrences', fontsize=12)

plt.xlabel('Number of words in the question', fontsize=12)

plt.xticks(rotation='vertical')

plt.show()
all_ques_df["num_of_chars"] = all_ques_df["questions"].apply(lambda x : len(str(x)))

cnt_srs = all_ques_df['num_of_chars'].value_counts()



plt.figure(figsize=(50,8))

sns.barplot(cnt_srs.index, cnt_srs.values, alpha=0.8, color=color[3])

plt.ylabel('Number of Occurrences', fontsize=12)

plt.xlabel('Number of characters in the question', fontsize=12)

plt.xticks(rotation='vertical')

plt.show()      



del all_ques_df
def get_unigrams(que):

    return [word for word in word_tokenize(que.lower()) if word not in eng_stopwords]



def get_common_unigrams(row):

    return len( set(row["unigrams_ques1"]).intersection(set(row["unigrams_ques2"])) )



def get_common_unigram_ratio(row):

    return float(row["unigrams_common_count"]) / max(len( set(row["unigrams_ques1"]).union(set(row["unigrams_ques2"])) ),1)



train_df["unigrams_ques1"] = train_df['question1'].apply(lambda x: get_unigrams(str(x)))

train_df["unigrams_ques2"] = train_df['question2'].apply(lambda x: get_unigrams(str(x)))

train_df["unigrams_common_count"] = train_df.apply(lambda row: get_common_unigrams(row),axis=1)

train_df["unigrams_common_ratio"] = train_df.apply(lambda row: get_common_unigram_ratio(row), axis=1)
cnt_srs = train_df['unigrams_common_count'].value_counts()



plt.figure(figsize=(12,6))

sns.barplot(cnt_srs.index, cnt_srs.values, alpha=0.8)

plt.ylabel('Number of Occurrences', fontsize=12)

plt.xlabel('Common unigrams count', fontsize=12)

plt.show()
plt.figure(figsize=(12,6))

sns.boxplot(x="is_duplicate", y="unigrams_common_count", data=train_df)

plt.xlabel('Is duplicate', fontsize=12)

plt.ylabel('Common unigrams count', fontsize=12)

plt.show()
plt.figure(figsize=(12,6))

sns.boxplot(x="is_duplicate", y="unigrams_common_ratio", data=train_df)

plt.xlabel('Is duplicate', fontsize=12)

plt.ylabel('Common unigrams ratio', fontsize=12)

plt.show()
ques = pd.concat([train_df[['question1', 'question2']], \

        test_df[['question1', 'question2']]], axis=0).reset_index(drop='index')

ques.shape
from collections import defaultdict

q_dict = defaultdict(set)

for i in range(ques.shape[0]):

        q_dict[ques.question1[i]].add(ques.question2[i])

        q_dict[ques.question2[i]].add(ques.question1[i])
def q1_freq(row):

    return(len(q_dict[row['question1']]))

    

def q2_freq(row):

    return(len(q_dict[row['question2']]))

    

def q1_q2_intersect(row):

    return(len(set(q_dict[row['question1']]).intersection(set(q_dict[row['question2']]))))



train_df['q1_q2_intersect'] = train_df.apply(q1_q2_intersect, axis=1, raw=True)

train_df['q1_freq'] = train_df.apply(q1_freq, axis=1, raw=True)

train_df['q2_freq'] = train_df.apply(q2_freq, axis=1, raw=True)
cnt_srs = train_df['q1_q2_intersect'].value_counts()



plt.figure(figsize=(12,6))

sns.barplot(cnt_srs.index, np.log1p(cnt_srs.values), alpha=0.8)

plt.xlabel('Q1-Q2 neighbor intersection count', fontsize=12)

plt.ylabel('Log of Number of Occurrences', fontsize=12)

plt.xticks(rotation='vertical')

plt.show()
grouped_df = train_df.groupby('q1_q2_intersect')['is_duplicate'].aggregate(np.mean).reset_index()

plt.figure(figsize=(12,8))

sns.pointplot(grouped_df["q1_q2_intersect"].values, grouped_df["is_duplicate"].values, alpha=0.8, color=color[2])

plt.ylabel('Mean is_duplicate', fontsize=12)

plt.xlabel('Q1-Q2 neighbor intersection count', fontsize=12)

plt.xticks(rotation='vertical')

plt.show()
cnt_srs = train_df['q1_freq'].value_counts()



plt.figure(figsize=(12,8))

sns.barplot(cnt_srs.index, cnt_srs.values, alpha=0.8)

plt.xlabel('Q1 frequency', fontsize=12)

plt.ylabel('Number of Occurrences', fontsize=12)

plt.xticks(rotation='vertical')

plt.show()
plt.figure(figsize=(12,8))

grouped_df = train_df.groupby('q1_freq')['is_duplicate'].aggregate(np.mean).reset_index()

sns.barplot(grouped_df["q1_freq"].values, grouped_df["is_duplicate"].values, alpha=0.8, color=color[4])

plt.ylabel('Mean is_duplicate', fontsize=12)

plt.xlabel('Q1 frequency', fontsize=12)

plt.xticks(rotation='vertical')

plt.show()
pvt_df = train_df.pivot_table(index="q1_freq", columns="q2_freq", values="is_duplicate")

plt.figure(figsize=(12,12))

sns.heatmap(pvt_df)

plt.title("Mean is_duplicate value distribution across q1 and q2 frequency")

plt.show()
cols_to_use = ['q1_q2_intersect', 'q1_freq', 'q2_freq']

temp_df = train_df[cols_to_use]

corrmat = temp_df.corr(method='spearman')

f, ax = plt.subplots(figsize=(8, 8))



# Draw the heatmap using seaborn

sns.heatmap(corrmat, vmax=1., square=True)

plt.title("Leaky variables correlation map", fontsize=15)

plt.show()