
import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns

import nltk

from nltk.corpus import stopwords

import string

import xgboost as xgb

from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer

from sklearn.decomposition import TruncatedSVD

from sklearn import ensemble, metrics, model_selection, naive_bayes

color = sns.color_palette()



#%matplotlib inline



eng_stopwords = set(stopwords.words("english"))

pd.options.mode.chained_assignment = None
## Read the train and test dataset and check the top few lines ##

train_df = pd.read_csv("../input/train.csv")

test_df = pd.read_csv("../input/test.csv")

print("Number of rows in train dataset : ",train_df.shape[0])

print("Number of rows in test dataset : ",test_df.shape[0])
train_df.head()
cnt_srs = train_df['author'].value_counts()



plt.figure(figsize=(8,4))

sns.barplot(cnt_srs.index, cnt_srs.values, alpha=0.8)

plt.ylabel('Number of Occurrences', fontsize=12)

plt.xlabel('Author Name', fontsize=12)

plt.show()
grouped_df = train_df.groupby('author')

for name, group in grouped_df:

    print("Author name : ", name)

    cnt = 0

    for ind, row in group.iterrows():

        print(row["text"])

        cnt += 1

        if cnt == 5:

            break

    print("\n")
## Number of words in the text ##

train_df["num_words"] = train_df["text"].apply(lambda x: len(str(x).split()))

test_df["num_words"] = test_df["text"].apply(lambda x: len(str(x).split()))



## Number of unique words in the text ##

train_df["num_unique_words"] = train_df["text"].apply(lambda x: len(set(str(x).split())))

test_df["num_unique_words"] = test_df["text"].apply(lambda x: len(set(str(x).split())))



## Number of characters in the text ##

train_df["num_chars"] = train_df["text"].apply(lambda x: len(str(x)))

test_df["num_chars"] = test_df["text"].apply(lambda x: len(str(x)))



## Number of stopwords in the text ##

train_df["num_stopwords"] = train_df["text"].apply(lambda x: len([w for w in str(x).lower().split() if w in eng_stopwords]))

test_df["num_stopwords"] = test_df["text"].apply(lambda x: len([w for w in str(x).lower().split() if w in eng_stopwords]))



## Number of punctuations in the text ##

train_df["num_punctuations"] =train_df['text'].apply(lambda x: len([c for c in str(x) if c in string.punctuation]) )

test_df["num_punctuations"] =test_df['text'].apply(lambda x: len([c for c in str(x) if c in string.punctuation]) )



## Number of title case words in the text ##

train_df["num_words_upper"] = train_df["text"].apply(lambda x: len([w for w in str(x).split() if w.isupper()]))

test_df["num_words_upper"] = test_df["text"].apply(lambda x: len([w for w in str(x).split() if w.isupper()]))



## Number of title case words in the text ##

train_df["num_words_title"] = train_df["text"].apply(lambda x: len([w for w in str(x).split() if w.istitle()]))

test_df["num_words_title"] = test_df["text"].apply(lambda x: len([w for w in str(x).split() if w.istitle()]))



## Average length of the words in the text ##

train_df["mean_word_len"] = train_df["text"].apply(lambda x: np.mean([len(w) for w in str(x).split()]))

test_df["mean_word_len"] = test_df["text"].apply(lambda x: np.mean([len(w) for w in str(x).split()]))
train_df['num_words'].loc[train_df['num_words']>80] = 80 #truncation for better visuals

plt.figure(figsize=(12,8))

sns.violinplot(x='author', y='num_words', data=train_df)

plt.xlabel('Author Name', fontsize=12)

plt.ylabel('Number of words in text', fontsize=12)

plt.title("Number of words by author", fontsize=15)

plt.show()
train_df['num_punctuations'].loc[train_df['num_punctuations']>10] = 10 #truncation for better visuals

plt.figure(figsize=(12,8))

sns.violinplot(x='author', y='num_punctuations', data=train_df)

plt.xlabel('Author Name', fontsize=12)

plt.ylabel('Number of puntuations in text', fontsize=12)

plt.title("Number of punctuations by author", fontsize=15)

plt.show()
## Prepare the data for modeling ###

author_mapping_dict = {'EAP':0, 'HPL':1, 'MWS':2}

train_y = train_df['author'].map(author_mapping_dict)

train_id = train_df['id'].values

test_id = test_df['id'].values



### recompute the trauncated variables again ###

train_df["num_words"] = train_df["text"].apply(lambda x: len(str(x).split()))

test_df["num_words"] = test_df["text"].apply(lambda x: len(str(x).split()))

train_df["mean_word_len"] = train_df["text"].apply(lambda x: np.mean([len(w) for w in str(x).split()]))

test_df["mean_word_len"] = test_df["text"].apply(lambda x: np.mean([len(w) for w in str(x).split()]))



cols_to_drop = ['id', 'text']

train_X = train_df.drop(cols_to_drop+['author'], axis=1)

test_X = test_df.drop(cols_to_drop, axis=1)
def runXGB(train_X, train_y, test_X, test_y=None, test_X2=None, seed_val=0, child=1, colsample=0.3):

    param = {}

    param['objective'] = 'multi:softprob'

    param['eta'] = 0.1

    param['max_depth'] = 3

    param['silent'] = 1

    param['num_class'] = 3

    param['eval_metric'] = "mlogloss"

    param['min_child_weight'] = child

    param['subsample'] = 0.8

    param['colsample_bytree'] = colsample

    param['seed'] = seed_val

    num_rounds = 2000



    plst = list(param.items())

    xgtrain = xgb.DMatrix(train_X, label=train_y)



    if test_y is not None:

        xgtest = xgb.DMatrix(test_X, label=test_y)

        watchlist = [ (xgtrain,'train'), (xgtest, 'test') ]

        model = xgb.train(plst, xgtrain, num_rounds, watchlist, early_stopping_rounds=50, verbose_eval=20)

    else:

        xgtest = xgb.DMatrix(test_X)

        model = xgb.train(plst, xgtrain, num_rounds)



    pred_test_y = model.predict(xgtest, ntree_limit = model.best_ntree_limit)

    if test_X2 is not None:

        xgtest2 = xgb.DMatrix(test_X2)

        pred_test_y2 = model.predict(xgtest2, ntree_limit = model.best_ntree_limit)

    return pred_test_y, pred_test_y2, model
kf = model_selection.KFold(n_splits=5, shuffle=True, random_state=2017)

cv_scores = []

pred_full_test = 0

pred_train = np.zeros([train_df.shape[0], 3])

for dev_index, val_index in kf.split(train_X):

    dev_X, val_X = train_X.loc[dev_index], train_X.loc[val_index]

    dev_y, val_y = train_y[dev_index], train_y[val_index]

    pred_val_y, pred_test_y, model = runXGB(dev_X, dev_y, val_X, val_y, test_X, seed_val=0)

    pred_full_test = pred_full_test + pred_test_y

    pred_train[val_index,:] = pred_val_y

    cv_scores.append(metrics.log_loss(val_y, pred_val_y))

    break

print("cv scores : ", cv_scores)
### Plot the important variables ###

fig, ax = plt.subplots(figsize=(12,12))

xgb.plot_importance(model, max_num_features=50, height=0.8, ax=ax)

plt.show()
### Fit transform the tfidf vectorizer ###

tfidf_vec = TfidfVectorizer(stop_words='english', ngram_range=(1,3))

full_tfidf = tfidf_vec.fit_transform(train_df['text'].values.tolist() + test_df['text'].values.tolist())

train_tfidf = tfidf_vec.transform(train_df['text'].values.tolist())

test_tfidf = tfidf_vec.transform(test_df['text'].values.tolist())
def runMNB(train_X, train_y, test_X, test_y, test_X2):

    model = naive_bayes.MultinomialNB()

    model.fit(train_X, train_y)

    pred_test_y = model.predict_proba(test_X)

    pred_test_y2 = model.predict_proba(test_X2)

    return pred_test_y, pred_test_y2, model
cv_scores = []

pred_full_test = 0

pred_train = np.zeros([train_df.shape[0], 3])

kf = model_selection.KFold(n_splits=5, shuffle=True, random_state=2017)

for dev_index, val_index in kf.split(train_X):

    dev_X, val_X = train_tfidf[dev_index], train_tfidf[val_index]

    dev_y, val_y = train_y[dev_index], train_y[val_index]

    pred_val_y, pred_test_y, model = runMNB(dev_X, dev_y, val_X, val_y, test_tfidf)

    pred_full_test = pred_full_test + pred_test_y

    pred_train[val_index,:] = pred_val_y

    cv_scores.append(metrics.log_loss(val_y, pred_val_y))

print("Mean cv score : ", np.mean(cv_scores))

pred_full_test = pred_full_test / 5.
### Function to create confusion matrix ###

import itertools

from sklearn.metrics import confusion_matrix



### From http://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html#sphx-glr-auto-examples-model-selection-plot-confusion-matrix-py #

def plot_confusion_matrix(cm, classes,

                          normalize=False,

                          title='Confusion matrix',

                          cmap=plt.cm.Blues):

    """

    This function prints and plots the confusion matrix.

    Normalization can be applied by setting `normalize=True`.

    """

    if normalize:

        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

        #print("Normalized confusion matrix")

    #else:

    #    print('Confusion matrix, without normalization')



    #print(cm)



    plt.imshow(cm, interpolation='nearest', cmap=cmap)

    plt.title(title)

    plt.colorbar()

    tick_marks = np.arange(len(classes))

    plt.xticks(tick_marks, classes, rotation=45)

    plt.yticks(tick_marks, classes)



    fmt = '.2f' if normalize else 'd'

    thresh = cm.max() / 2.

    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):

        plt.text(j, i, format(cm[i, j], fmt),

                 horizontalalignment="center",

                 color="white" if cm[i, j] > thresh else "black")



    plt.tight_layout()

    plt.ylabel('True label')

    plt.xlabel('Predicted label')
cnf_matrix = confusion_matrix(val_y, np.argmax(pred_val_y,axis=1))

np.set_printoptions(precision=2)



# Plot non-normalized confusion matrix

plt.figure(figsize=(8,8))

plot_confusion_matrix(cnf_matrix, classes=['EAP', 'HPL', 'MWS'],

                      title='Confusion matrix, without normalization')

plt.show()

n_comp = 20

svd_obj = TruncatedSVD(n_components=n_comp, algorithm='arpack')

svd_obj.fit(full_tfidf)

train_svd = pd.DataFrame(svd_obj.transform(train_tfidf))

test_svd = pd.DataFrame(svd_obj.transform(test_tfidf))

    

train_svd.columns = ['svd_word_'+str(i) for i in range(n_comp)]

test_svd.columns = ['svd_word_'+str(i) for i in range(n_comp)]

train_df = pd.concat([train_df, train_svd], axis=1)

test_df = pd.concat([test_df, test_svd], axis=1)

del full_tfidf, train_tfidf, test_tfidf, train_svd, test_svd
### Fit transform the count vectorizer ###

tfidf_vec = CountVectorizer(stop_words='english', ngram_range=(1,3))

tfidf_vec.fit(train_df['text'].values.tolist() + test_df['text'].values.tolist())

train_tfidf = tfidf_vec.transform(train_df['text'].values.tolist())

test_tfidf = tfidf_vec.transform(test_df['text'].values.tolist())
cv_scores = []

pred_full_test = 0

pred_train = np.zeros([train_df.shape[0], 3])

kf = model_selection.KFold(n_splits=5, shuffle=True, random_state=2017)

for dev_index, val_index in kf.split(train_X):

    dev_X, val_X = train_tfidf[dev_index], train_tfidf[val_index]

    dev_y, val_y = train_y[dev_index], train_y[val_index]

    pred_val_y, pred_test_y, model = runMNB(dev_X, dev_y, val_X, val_y, test_tfidf)

    pred_full_test = pred_full_test + pred_test_y

    pred_train[val_index,:] = pred_val_y

    cv_scores.append(metrics.log_loss(val_y, pred_val_y))

print("Mean cv score : ", np.mean(cv_scores))

pred_full_test = pred_full_test / 5.



# add the predictions as new features #

train_df["nb_cvec_eap"] = pred_train[:,0]

train_df["nb_cvec_hpl"] = pred_train[:,1]

train_df["nb_cvec_mws"] = pred_train[:,2]

test_df["nb_cvec_eap"] = pred_full_test[:,0]

test_df["nb_cvec_hpl"] = pred_full_test[:,1]

test_df["nb_cvec_mws"] = pred_full_test[:,2]
cnf_matrix = confusion_matrix(val_y, np.argmax(pred_val_y,axis=1))

np.set_printoptions(precision=2)



# Plot non-normalized confusion matrix

plt.figure(figsize=(8,8))

plot_confusion_matrix(cnf_matrix, classes=['EAP', 'HPL', 'MWS'],

                      title='Confusion matrix of NB on word count, without normalization')

plt.show()
### Fit transform the tfidf vectorizer ###

tfidf_vec = CountVectorizer(ngram_range=(1,7), analyzer='char')

tfidf_vec.fit(train_df['text'].values.tolist() + test_df['text'].values.tolist())

train_tfidf = tfidf_vec.transform(train_df['text'].values.tolist())

test_tfidf = tfidf_vec.transform(test_df['text'].values.tolist())



cv_scores = []

pred_full_test = 0

pred_train = np.zeros([train_df.shape[0], 3])

kf = model_selection.KFold(n_splits=5, shuffle=True, random_state=2017)

for dev_index, val_index in kf.split(train_X):

    dev_X, val_X = train_tfidf[dev_index], train_tfidf[val_index]

    dev_y, val_y = train_y[dev_index], train_y[val_index]

    pred_val_y, pred_test_y, model = runMNB(dev_X, dev_y, val_X, val_y, test_tfidf)

    pred_full_test = pred_full_test + pred_test_y

    pred_train[val_index,:] = pred_val_y

    cv_scores.append(metrics.log_loss(val_y, pred_val_y))

print("Mean cv score : ", np.mean(cv_scores))

pred_full_test = pred_full_test / 5.



# add the predictions as new features #

train_df["nb_cvec_char_eap"] = pred_train[:,0]

train_df["nb_cvec_char_hpl"] = pred_train[:,1]

train_df["nb_cvec_char_mws"] = pred_train[:,2]

test_df["nb_cvec_char_eap"] = pred_full_test[:,0]

test_df["nb_cvec_char_hpl"] = pred_full_test[:,1]

test_df["nb_cvec_char_mws"] = pred_full_test[:,2]
### Fit transform the tfidf vectorizer ###

tfidf_vec = TfidfVectorizer(ngram_range=(1,5), analyzer='char')

full_tfidf = tfidf_vec.fit_transform(train_df['text'].values.tolist() + test_df['text'].values.tolist())

train_tfidf = tfidf_vec.transform(train_df['text'].values.tolist())

test_tfidf = tfidf_vec.transform(test_df['text'].values.tolist())



cv_scores = []

pred_full_test = 0

pred_train = np.zeros([train_df.shape[0], 3])

kf = model_selection.KFold(n_splits=5, shuffle=True, random_state=2017)

for dev_index, val_index in kf.split(train_X):

    dev_X, val_X = train_tfidf[dev_index], train_tfidf[val_index]

    dev_y, val_y = train_y[dev_index], train_y[val_index]

    pred_val_y, pred_test_y, model = runMNB(dev_X, dev_y, val_X, val_y, test_tfidf)

    pred_full_test = pred_full_test + pred_test_y

    pred_train[val_index,:] = pred_val_y

    cv_scores.append(metrics.log_loss(val_y, pred_val_y))

print("Mean cv score : ", np.mean(cv_scores))

pred_full_test = pred_full_test / 5.



# add the predictions as new features #

train_df["nb_tfidf_char_eap"] = pred_train[:,0]

train_df["nb_tfidf_char_hpl"] = pred_train[:,1]

train_df["nb_tfidf_char_mws"] = pred_train[:,2]

test_df["nb_tfidf_char_eap"] = pred_full_test[:,0]

test_df["nb_tfidf_char_hpl"] = pred_full_test[:,1]

test_df["nb_tfidf_char_mws"] = pred_full_test[:,2]
n_comp = 20

svd_obj = TruncatedSVD(n_components=n_comp, algorithm='arpack')

svd_obj.fit(full_tfidf)

train_svd = pd.DataFrame(svd_obj.transform(train_tfidf))

test_svd = pd.DataFrame(svd_obj.transform(test_tfidf))

    

train_svd.columns = ['svd_char_'+str(i) for i in range(n_comp)]

test_svd.columns = ['svd_char_'+str(i) for i in range(n_comp)]

train_df = pd.concat([train_df, train_svd], axis=1)

test_df = pd.concat([test_df, test_svd], axis=1)

del full_tfidf, train_tfidf, test_tfidf, train_svd, test_svd
cols_to_drop = ['id', 'text']

train_X = train_df.drop(cols_to_drop+['author'], axis=1)

test_X = test_df.drop(cols_to_drop, axis=1)



kf = model_selection.KFold(n_splits=5, shuffle=True, random_state=2017)

cv_scores = []

pred_full_test = 0

pred_train = np.zeros([train_df.shape[0], 3])

for dev_index, val_index in kf.split(train_X):

    dev_X, val_X = train_X.loc[dev_index], train_X.loc[val_index]

    dev_y, val_y = train_y[dev_index], train_y[val_index]

    pred_val_y, pred_test_y, model = runXGB(dev_X, dev_y, val_X, val_y, test_X, seed_val=0, colsample=0.7)

    pred_full_test = pred_full_test + pred_test_y

    pred_train[val_index,:] = pred_val_y

    cv_scores.append(metrics.log_loss(val_y, pred_val_y))

    break

print("cv scores : ", cv_scores)



out_df = pd.DataFrame(pred_full_test)

out_df.columns = ['EAP', 'HPL', 'MWS']

out_df.insert(0, 'id', test_id)

out_df.to_csv("sub_fe.csv", index=False)
### Plot the important variables ###

fig, ax = plt.subplots(figsize=(12,12))

xgb.plot_importance(model, max_num_features=50, height=0.8, ax=ax)

plt.show()
cnf_matrix = confusion_matrix(val_y, np.argmax(pred_val_y,axis=1))

np.set_printoptions(precision=2)



# Plot non-normalized confusion matrix

plt.figure(figsize=(8,8))

plot_confusion_matrix(cnf_matrix, classes=['EAP', 'HPL', 'MWS'],

                      title='Confusion matrix of XGB, without normalization')

plt.show()
import af


**More to come. Stay tuned.!**