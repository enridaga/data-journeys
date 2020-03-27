
import numpy as np

import pandas as pd

from matplotlib import pyplot as plt

#%matplotlib inline

import seaborn as sns

import re
train_df = pd.read_csv('../input/train.csv')

test_df = pd.read_csv('../input/test.csv')
train_df.sample(5)
cols_target = ['obscene','insult','toxic','severe_toxic','identity_hate','threat']
# check missing values in numeric columns

train_df.describe()
# check for any 'null' comment

no_comment = train_df[train_df['comment_text'].isnull()]

len(no_comment)
test_df.head()
no_comment = test_df[test_df['comment_text'].isnull()]

no_comment
# fill NaN with string "unknown"

test_df.fillna('unknown',inplace=True)
# let's see the total rows in train, test data and the numbers for the various categories

print('Total rows in train is {}'.format(len(train_df)))

print('Total rows in test is {}'.format(len(test_df)))
print(train_df[cols_target].sum())
# Let's look at the character length for the rows and record these

train_df['char_length'] = train_df['comment_text'].str.len()
# look at the histogram plot for text length

sns.set()

train_df['char_length'].hist()

plt.show()
data = train_df[cols_target]
colormap = plt.cm.magma

plt.figure(figsize=(7,7))

plt.title('Correlation of features & targets',y=1.05,size=14)

sns.heatmap(data.astype(float).corr(),linewidths=0.1,vmax=1.0,square=True,cmap=colormap,

           linecolor='white',annot=True)
test_df['char_length'] = test_df['comment_text'].str.len()
plt.figure(figsize=(20,5))

plt.hist(test_df['char_length'])

plt.show()
test_df[test_df['char_length']>5000]
test_comment = test_df['comment_text'].apply(lambda x: x[:5000])

char_length = test_comment.str.len()
plt.figure()

plt.hist(char_length)

plt.show()
def clean_text(text):

    text = re.sub(r"what's", "what is ", text)

    text = re.sub(r"\'s", " ", text)

    text = re.sub(r"\'ve", " have ", text)

    text = re.sub(r"can't", "cannot ", text)

    text = re.sub(r"n't", " not ", text)

    text = re.sub(r"i'm", "i am ", text)

    text = re.sub(r"\'re", " are ", text)

    text = re.sub(r"\'d", " would ", text)

    text = re.sub(r"\'ll", " will ", text)

    text = re.sub(r"\'scuse", " excuse ", text)

    text = re.sub(r"\.", " ", text)

    text = re.sub('\W', ' ', text)

    text = re.sub('\s+', ' ', text)

    text = text.strip()

    return text
# clean the comment_text in train_df

cleaned_train_comment = []

for i in range(0,len(train_df)):

    cleaned_comment = clean_text(train_df['comment_text'][i])

    cleaned_train_comment.append(cleaned_comment)

train_df['comment_text'] = pd.Series(cleaned_train_comment).astype(str)
# clean the comment_text in test_df

cleaned_test_comment = []

for i in range(0,len(test_df)):

    cleaned_comment = clean_text(test_df['comment_text'][i])

    cleaned_test_comment.append(cleaned_comment)

test_df['comment_text'] = pd.Series(cleaned_test_comment).astype(str)
train_df = train_df.drop('char_length',axis=1)
X = train_df.comment_text

test_X = test_df.comment_text
print(X.shape, test_X.shape)
# import and instantiate CountVectorizer

from sklearn.feature_extraction.text import CountVectorizer

from sklearn.feature_extraction.text import TfidfVectorizer

vect = TfidfVectorizer(max_features=20000,min_df=2)

vect
# learn the vocabulary in the training data, then use it to create a document-term matrix

X_dtm = vect.fit_transform(X)

# examine the document-term matrix created from X_train

X_dtm
# transform the test data using the earlier fitted vocabulary, into a document-term matrix

test_X_dtm = vect.transform(test_X)

# examine the document-term matrix from X_test

test_X_dtm
# import and instantiate the Logistic Regression model

from sklearn.linear_model import LogisticRegression

from sklearn.metrics import accuracy_score

logreg = LogisticRegression(C=6.0,random_state=123)



# create submission file

submission_binary = pd.read_csv('../input/sample_submission.csv')



for label in cols_target:

    print('... Processing {}'.format(label))

    y = train_df[label]

    # train the model using X_dtm & y

    logreg.fit(X_dtm, y)

    # compute the training accuracy

    y_pred_X = logreg.predict(X_dtm)

    print('Training accuracy is {}'.format(accuracy_score(y, y_pred_X)))

    # compute the predicted probabilities for X_test_dtm

    test_y_prob = logreg.predict_proba(test_X_dtm)[:,1]

    submission_binary[label] = test_y_prob
submission_binary.head()
# generate submission file

submission_binary.to_csv('submission_binary.csv',index=False)
# create submission file

submission_chains = pd.read_csv('../input/sample_submission.csv')



# create a function to add features

def add_feature(X, feature_to_add):

    '''

    Returns sparse feature matrix with added feature.

    feature_to_add can also be a list of features.

    '''

    from scipy.sparse import csr_matrix, hstack

    return hstack([X, csr_matrix(feature_to_add).T], 'csr')
for label in cols_target:

    print('... Processing {}'.format(label))

    y = train_df[label]

    # train the model using X_dtm & y

    logreg.fit(X_dtm,y)

    # compute the training accuracy

    y_pred_X = logreg.predict(X_dtm)

    print('Training Accuracy is {}'.format(accuracy_score(y,y_pred_X)))

    # make predictions from test_X

    test_y = logreg.predict(test_X_dtm)

    test_y_prob = logreg.predict_proba(test_X_dtm)[:,1]

    submission_chains[label] = test_y_prob

    # chain current label to X_dtm

    X_dtm = add_feature(X_dtm, y)

    print('Shape of X_dtm is now {}'.format(X_dtm.shape))

    # chain current label predictions to test_X_dtm

    test_X_dtm = add_feature(test_X_dtm, test_y)

    print('Shape of test_X_dtm is now {}'.format(test_X_dtm.shape))
submission_chains.head()
# generate submission file

submission_chains.to_csv('submission_chains.csv', index=False)