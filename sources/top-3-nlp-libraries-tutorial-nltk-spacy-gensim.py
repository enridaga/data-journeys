
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib
import warnings
import sklearn
import gensim
import scipy
import numpy
import json
import nltk
import sys
import csv
import os
print('matplotlib: {}'.format(matplotlib.__version__))
print('scipy: {}'.format(scipy.__version__))
print('seaborn: {}'.format(sns.__version__))
print('pandas: {}'.format(pd.__version__))
print('numpy: {}'.format(np.__version__))
print('Python: {}'.format(sys.version))
sns.set(style='white', context='notebook', palette='deep')
warnings.filterwarnings('ignore')
sns.set_style('white')
### matplotlib inline
print(os.listdir("../input/"))
gendered_pronoun_df = pd.read_csv('../input/test_stage_1.tsv', delimiter='\t')
submission = pd.read_csv('../input/sample_submission_stage_1.csv')
gendered_pronoun_df.shape
submission.shape
gendered_pronoun_df.head()
gendered_pronoun_df.info()
print(gendered_pronoun_df.Text.head())
print("Shape of train set : ",gendered_pronoun_df.shape)
gendered_pronoun_df.columns
def check_missing_data(df):
    flag=df.isna().sum().any()
    if flag==True:
        total = df.isnull().sum()
        percent = (df.isnull().sum())/(df.isnull().count()*100)
        output = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
        data_type = []
        # written by MJ Bahmani
        for col in df.columns:
            dtype = str(df[col].dtype)
            data_type.append(dtype)
        output['Types'] = data_type
        return(np.transpose(output))
    else:
        return(False)
check_missing_data(gendered_pronoun_df)
gendered_pronoun_df["num_words"] = gendered_pronoun_df["Text"].apply(lambda x: len(str(x).split()))
#MJ Bahmani
print('maximum of num_words in data_df',gendered_pronoun_df["num_words"].max())
print('min of num_words in data_df',gendered_pronoun_df["num_words"].min())
gendered_pronoun_df["num_unique_words"] = gendered_pronoun_df["Text"].apply(lambda x: len(set(str(x).split())))
print('maximum of num_unique_words in train',gendered_pronoun_df["num_unique_words"].max())
print('mean of num_unique_words in data_df',gendered_pronoun_df["num_unique_words"].mean())
gendered_pronoun_df["num_chars"] = gendered_pronoun_df["Text"].apply(lambda x: len(str(x)))
print('maximum of num_chars in data_df',gendered_pronoun_df["num_chars"].max())
from nltk.corpus import stopwords
eng_stopwords = set(stopwords.words("english"))
gendered_pronoun_df["num_stopwords"] = gendered_pronoun_df["Text"].apply(lambda x: len([w for w in str(x).lower().split() if w in eng_stopwords]))

print('maximum of num_stopwords in data_df',gendered_pronoun_df["num_stopwords"].max())
import string
gendered_pronoun_df["num_punctuations"] =gendered_pronoun_df['Text'].apply(lambda x: len([c for c in str(x) if c in string.punctuation]) )
print('maximum of num_punctuations in data_df',gendered_pronoun_df["num_punctuations"].max())
gendered_pronoun_df["num_words_upper"] = gendered_pronoun_df["Text"].apply(lambda x: len([w for w in str(x).split() if w.isupper()]))
print('maximum of num_words_upper in data_df',gendered_pronoun_df["num_words_upper"].max())
print(gendered_pronoun_df.columns)
gendered_pronoun_df.head(1)
pronoun=gendered_pronoun_df["Pronoun"]
np.unique(pronoun)
## is suggested by  https://www.kaggle.com/aavella77
binary = {
    "He": 0,
    "he": 0,
    "She": 1,
    "she": 1,
    "His": 2,
    "his": 2,
    "Him": 3,
    "him": 3,
    "Her": 4,
    "her": 4
}
for index in range(len(gendered_pronoun_df)):
    key = gendered_pronoun_df.iloc[index]['Pronoun']
    gendered_pronoun_df.at[index, 'Pronoun_binary'] = binary[key]
gendered_pronoun_df.head(30)
from wordcloud import WordCloud as wc
from nltk.corpus import stopwords
def generate_wordcloud(text): 
    wordcloud = wc(relative_scaling = 1.0,stopwords = eng_stopwords).generate(text)
    fig,ax = plt.subplots(1,1,figsize=(10,10))
    ax.imshow(wordcloud, interpolation='bilinear')
    ax.axis("off")
    ax.margins(x=0, y=0)
    plt.show()
from nltk.corpus import stopwords
eng_stopwords = set(stopwords.words("english"))
text =" ".join(gendered_pronoun_df.Text)
generate_wordcloud(text)
gendered_pronoun_df.hist();
pd.plotting.scatter_matrix(gendered_pronoun_df,figsize=(10,10))
plt.figure();
sns.jointplot(x='Pronoun-offset',y='A-offset' ,data=gendered_pronoun_df, kind='reg')
sns.swarmplot(x='Pronoun-offset',y='B-offset',data=gendered_pronoun_df);
sns.distplot(gendered_pronoun_df["Pronoun-offset"])
sns.violinplot(data=gendered_pronoun_df,x="Pronoun_binary", y="num_words")
from nltk.tokenize import sent_tokenize, word_tokenize
gendered_pronoun_df.Text[0]
our_text=gendered_pronoun_df.Text[0]
print(word_tokenize(our_text))
from nltk.tokenize import sent_tokenize, word_tokenize
print(sent_tokenize(our_text))
from nltk.tokenize import sent_tokenize, word_tokenize
 
phrases = sent_tokenize(our_text)
words = word_tokenize(our_text)
print(phrases)
print(words)
type(words)
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
 

stopWords = set(stopwords.words('english'))
words = word_tokenize(our_text)
wordsFiltered = []
 
for w in words:
    if w not in stopWords:
        wordsFiltered.append(w)
 
print(wordsFiltered)
from nltk.corpus import stopwords

stopWords = set(stopwords.words('english'))

print(len(stopWords))
print(stopWords)
for w in words:
    if w not in stopWords:
        wordsFiltered.append(w)
our_text=gendered_pronoun_df.Text[0]

from nltk.stem import PorterStemmer
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.stem import PorterStemmer
from nltk.tokenize import sent_tokenize, word_tokenize


ps = PorterStemmer()
 
for word in word_tokenize(our_text):
    print(ps.stem(word))
import nltk
from nltk.tokenize import PunktSentenceTokenizer
 

sentences = nltk.sent_tokenize(our_text)   
for sent in sentences:
    print(nltk.pos_tag(nltk.word_tokenize(sent)))
import nltk
from nltk.corpus import state_union
from nltk.tokenize import PunktSentenceTokenizer
 

sentences = nltk.sent_tokenize(our_text)   
 
data = []
for sent in sentences:
    data = data + nltk.pos_tag(nltk.word_tokenize(sent))
 
for word in data: 
    if 'NNP' in word[1]: 
        print(word)
#https://pythonspot.com/natural-language-processing-prediction/
from nltk.corpus import names
 
# Load data and training 
names = ([(name, 'male') for name in names.words('male.txt')] + [(name, 'female') for name in names.words('female.txt')])
[(u'Aaron', 'male'), (u'Abbey', 'male'), (u'Abbie', 'male')]
[(u'Zorana', 'female'), (u'Zorina', 'female'), (u'Zorine', 'female')]
def gender_features(word): 
    return {'last_letter': word[-1]}
#Based on https://pythonspot.com/category/nltk/
import nltk.classify.util
from nltk.classify import NaiveBayesClassifier
from nltk.corpus import names
 
def gender_features(word): 
    return {'last_letter': word[-1]} 
 
# Load data and training 
names = ([(name, 'male') for name in names.words('male.txt')] + 
	 [(name, 'female') for name in names.words('female.txt')])
 
featuresets = [(gender_features(n), g) for (n,g) in names] 
train_set = featuresets
classifier = nltk.NaiveBayesClassifier.train(train_set) 
 
# Predict
print(classifier.classify(gender_features('Frank')))
# Predict, you can change name
name = 'Sarah'
print(classifier.classify(gender_features(name)))
positive_vocab = [ 'awesome', 'outstanding', 'fantastic', 'terrific', 'good', 'nice', 'great', ':)' ]
negative_vocab = [ 'bad', 'terrible','useless', 'hate', ':(' ]
neutral_vocab = [ 'movie','the','sound','was','is','actors','did','know','words','not' ]
def word_feats(words):
    return dict([(word, True) for word in words])
 
positive_features = [(word_feats(pos), 'pos') for pos in positive_vocab]
negative_features = [(word_feats(neg), 'neg') for neg in negative_vocab]
neutral_features = [(word_feats(neu), 'neu') for neu in neutral_vocab]
train_set = negative_features + positive_features + neutral_features
#Based on https://pythonspot.com/category/nltk/
import nltk.classify.util
from nltk.classify import NaiveBayesClassifier
from nltk.corpus import names
 
def word_feats(words):
    return dict([(word, True) for word in words])
 
positive_vocab = [ 'awesome', 'outstanding', 'fantastic', 'terrific', 'good', 'nice', 'great', ':)' ]
negative_vocab = [ 'bad', 'terrible','useless', 'hate', ':(' ]
neutral_vocab = [ 'movie','the','sound','was','is','actors','did','know','words','not' ]
 
positive_features = [(word_feats(pos), 'pos') for pos in positive_vocab]
negative_features = [(word_feats(neg), 'neg') for neg in negative_vocab]
neutral_features = [(word_feats(neu), 'neu') for neu in neutral_vocab]
 
train_set = negative_features + positive_features + neutral_features
 
classifier = NaiveBayesClassifier.train(train_set) 
 
# Predict
neg = 0
pos = 0
##sentence = "Awesome movie, I liked it"
our_text = our_text.lower()
words = our_text.split(' ')
for word in words:
    classResult = classifier.classify( word_feats(word))
    if classResult == 'neg':
        neg = neg + 1
    if classResult == 'pos':
        pos = pos + 1
 
print('Positive: ' + str(float(pos)/len(words)))
print('Negative: ' + str(float(neg)/len(words)))
import spacy
nlp = spacy.load('en')
doc = nlp(our_text)
i=0
for token in doc:
    i=i+1;
    if i<20:
        print('"' + token.text + '"')
nlp = spacy.load('en')
doc=nlp(our_text)
i=0
for sent in doc.sents:
    i=i+1
    print(i,' - ',sent)
doc = nlp( our_text)
print([(token.text, token.tag_) for token in doc])
doc = nlp(our_text)
for ent in doc.ents:
    print(ent.text, ent.label_)
from spacy import displacy
 
doc = nlp(our_text )
displacy.render(doc, style='ent', jupyter=True)
from spacy import displacy
 
doc = nlp(our_text)
displacy.render(doc, style='dep', jupyter=True, options={'distance': 90})
import gensim
from gensim import corpora
from pprint import pprint
# How to create a dictionary from a list of sentences?
documents = [" Zoe Telford  played the police officer girlfriend of Simon, Maggie.",
             "Dumped by Simon in the final episode of series 1, after he slept with Jenny, and is not seen again.",
             "Phoebe Thomas played Cheryl Cassidy, Paulines friend and also a year 11 pupil in Simons class.", 
             "Dumped her boyfriend following Simons advice after he wouldnt ",
             "have sex with her but later realised this was due to him catching crabs off her friend Pauline."]

documents_2 = ["One source says the report will likely conclude that", 
                "the operation was carried out without clearance and", 
                "transparency and that those involved will be held", 
                "responsible. One of the sources acknowledged that the", 
                "report is still being prepared and cautioned that", 
                "things could change."]

texts = [[text for text in doc.split()] for doc in documents]

# Create dictionary
dictionary = corpora.Dictionary(texts)

# Get information about the dictionary
print(dictionary)
# 
print(dictionary.token2id)