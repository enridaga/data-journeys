
# Importing libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()

import os
#%matplotlib inline
print(os.listdir('../input'))
# Loading in dataset
df = pd.read_csv('../input/rotten-tomatoes/reviews.tsv', sep='\t', encoding = 'ISO-8859-1')
df.head()
# Way too many types of reviews from critics with each reviewer using their own set of review rating system
print('List of Reviews:')
print(df['rating'].unique())
print('\n')
print('Number of unique reviews:')
print(df['rating'].nunique())
# We'll stick with Rotten Tomatoes' final review classification of "Fresh" vs. "Rotten" when training our model
df['fresh'].value_counts()
df['fresh'].unique()
# Distribution of "Fresh" vs "Rotten" reviews are roughly balanced. 
sns.countplot(df['fresh'])
plt.show()
# Checking for missing values
df.isnull().sum()
# Since we cannot work with missing data or find a viable way to replace missing text reviews, we will drop these missings rows under reviews.
df = df.dropna(subset=['review'])
df.isnull().sum()
df.describe(include='all')
df_fresh = df[['fresh', 'review']]
# Checking for reviews with no text in review
blank_reviews = []

# (index, label, review text)
for i, label, review in df_fresh.itertuples():
    if type(review) == str:
        if review.isspace():
            blank_reviews.append(i)
# All remaining reviews contain text
blank_reviews
# Addining in a new feature to see if there is any correlation to the length of the review to the fresh rating.
df_fresh['review length'] = df_fresh['review'].apply(lambda review: len(review))
df_fresh.head()
bins = 20
plt.hist(df_fresh[df_fresh['fresh']=='fresh']['review length'],bins=bins,alpha=0.5)
plt.hist(df_fresh[df_fresh['fresh']=='rotten']['review length'],bins=bins,alpha=0.5)
plt.legend(('fresh','rotten'))
plt.show()
# Splitting data into training and testing datasets
from sklearn.model_selection import train_test_split

X = df_fresh['review']
y = df_fresh['fresh']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
# Building a simple pipeline to preprocess text data
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC

text_clf_svc = Pipeline([('tfidf', TfidfVectorizer()),
                    ('clf', LinearSVC())])
# Fitting and generating predictions
text_clf_svc.fit(X_train, y_train)
y_pred_svc = text_clf_svc.predict(X_test)
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
print(classification_report(y_test, y_pred_svc))
print(confusion_matrix(y_test, y_pred_svc))
from sklearn.linear_model import LogisticRegression
# Building pipeline
text_clf_lr = Pipeline([('tfidf', TfidfVectorizer()),
                    ('clf', LogisticRegression())])
# Fitting and generating predictions
text_clf_lr.fit(X_train, y_train)
y_pred_lr = text_clf_lr.predict(X_test)
print(classification_report(y_test, y_pred_lr))
print(confusion_matrix(y_test, y_pred_lr))
from xgboost import XGBClassifier
# Building pipeline
text_clf_xgb = Pipeline([('tfidf', TfidfVectorizer()),
                    ('clf', XGBClassifier())])
# Fitting and generating predictions
text_clf_xgb.fit(X_train, y_train)
y_pred_xgb = text_clf_xgb.predict(X_test)
print(classification_report(y_test, y_pred_xgb))
print(confusion_matrix(y_test, y_pred_xgb))
from sklearn.ensemble import RandomForestClassifier
# Building pipeline
text_clf_rf = Pipeline([('tfidf', TfidfVectorizer()),
                    ('clf', RandomForestClassifier())])
# Fitting and generating predictions
text_clf_rf.fit(X_train, y_train)
y_pred_rf = text_clf_rf.predict(X_test)
print(classification_report(y_test, y_pred_rf))
print(confusion_matrix(y_test, y_pred_rf))
model_performance = [accuracy_score(y_test, y_pred_svc),accuracy_score(y_test, y_pred_lr),accuracy_score(y_test, y_pred_xgb),accuracy_score(y_test, y_pred_rf)]
models = ['Linear SVC', 'Logistic Regression', 'XGBoost', 'Random Forest']
df_model = pd.DataFrame(model_performance, columns=['Accuracy'])
df_model['Model'] = models
df_model
plt.figure(figsize=(8,6))
plt.ylim(0.5,0.8)
sns.barplot(x='Model', y='Accuracy', data=df_model)
plt.show()
np.random.seed(42)
rand_sample_1 = int(np.random.randint(0, len(X_test), size=1))
list(X_test)[rand_sample_1]
y_pred_1 = text_clf_svc.predict([list(X_test)[rand_sample_1]])
y_pred_1
df[df['review'] == 'As a work of cinema, The Passion of the Christ possesses a majestic beauty within its horror, one that comes most effectively through a tiny, solitary teardrop.']['fresh']
np.random.seed(43)
rand_sample_2 = int(np.random.randint(0, len(X_test), size=1))
list(X_test)[rand_sample_2]
y_pred_2 = text_clf_svc.predict([list(X_test)[rand_sample_2]])
y_pred_2
df[df['review'] == 'A character-driven dramedy with equal parts humor and heart, Safety Not Guaranteed is a magical film about the human spirit whose charm is impossible to ignore.']['fresh']
np.random.seed(44)
rand_sample_3 = int(np.random.randint(0, len(X_test), size=1))
list(X_test)[rand_sample_3]
y_pred_3 = text_clf_svc.predict([list(X_test)[rand_sample_3]])
y_pred_3
df[df['review'] == 'My mother is going to love this movie. ']['fresh']