
# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
# lets create a text
text = "No woman no cry"

# length of text ( includes spaces)
print("length of text: ",len(text))

# split the text
splitted_text = text.split() # default split methods splits text according to spaces
print("Splitted text: ",splitted_text)    # splitted_text is a list that includes words of text sentence
# each word is called token in text maning world.

# find specific words with list comprehension method
specific_words = [word for word in splitted_text if(len(word)>2)]
print("Words which are more than 3 letter: ",specific_words)

# capitalized words with istitle() method that finds capitalized words
capital_words = [ word for word in splitted_text if word.istitle()]
print("Capitalized words: ",capital_words)

# words which end with "o": endswith() method finds last letter of word
words_end_with_o =  [word for word in splitted_text if word.endswith("o")]
print("words end with o: ",words_end_with_o) 

# words which starts with "w": startswith() method
words_start_with_w = [word for word in splitted_text if word.startswith("w")]
print("words start with w: ",words_start_with_w) 
# unique with set() method
print("unique words: ",set(splitted_text))  # actually the word "no" is occured twice bc one word is "no" and others "No" there is a capital letter at first letter

# make all letters lowercase with lower() method
lowercase_text = [word.lower() for word in splitted_text]

# then find uniques again with set() method
print("unique words: ",set(lowercase_text))
# chech words includes or not includes particular substring or letter
print("Is w letter in woman word:", "w" in "woman")

# check words are upper case or lower case
print("Is word uppercase:", "WOMAN".isupper())
print("Is word lowercase:", "cry".islower())

# check words are made of by digits or not
print("Is word made of by digits: ","12345".isdigit())

# get rid of from white space characters like spaces and tabs or from unwanted letters with strip() method
print("00000000No cry: ","00000000No cry".strip("0"))

# find particular letter from front 
print("Find particular letter from back: ","No cry no".find("o"))  # at index 1

# find particular letter from back  rfind = reverse find
print("Find particular letter from back: ","No cry no".rfind("o"))  # at index 8

# replace letter with letter
print("Replace o with 3 ", "No cry no".replace("o","3"))

# find each letter and store them in list
print("Each letter: ",list("No cry"))
# Cleaning text
text1 = "    Be fair and tolerant    "
print("Split text: ",text1.split(" "))   # as you can see there are unnecessary white space in list

# get rid of from these unnecassary white spaces with strip() method then split
print("Cleaned text: ",text1.strip().split(" "))
# reading files line by line
f = open("../input/religious-and-philosophical-texts/35895-0.txt","r")

# read first line
print(f.readline())

# length of text
text3=f.read()
print("Length of text: ",len(text3))

# Number of lines with splitlines() method
lines = text3.splitlines()
print("Number of lines: ",len(lines))


# read data
data = pd.read_csv(r"../input/ben-hamners-tweets/benhamner.csv", encoding='latin-1')
data.head()
# find which entries contain the word 'appointment'
print("In his tweets, the rate of occuring kaggle word is: ",sum(data.text.str.contains('kaggle'))/len(data))
# text
text = data.text[1]
print(text)
# find regular expression on text
# import regular expression package
import re
# find callouts that starts with @
callouts = [word for word in text.split(" ") if re.search("@[A-Za-z0-9_]+",word)]
print("callouts: ",callouts)
# continue finding regular expressions
# [A-Za-z0-9_] =\w
# We will use "\w" to find callouts and our result will be same because \w matches with [A-Za-z0-9_] 
callouts1 = [word for word in text.split(" ") if re.search("@\w+",word)]
print("callouts: ",callouts1)
# find specific characters like "w"
print(re.findall(r"[w]",text))
# "w"ith, "w"indo"w", sho"w"ing, s"w"itches 

# do not find specific character like "w". We will use "^" symbol
print(re.findall(r"[^w]",text))
# Regular expressions for Dates
date = "15-10-2000\n09/10/2005\n15-05-1999\n05/05/99\n\n05/05/199\n\n05/05/9"
re.findall(r"\d{1,2}[/-]\d{1,2}[/-]\d{1,4}",date)
# import natural language tool kit
import nltk as nlp

# counting vocabulary of words
text = data.text[1]
splitted = text.split(" ")
print("number of words: ",len(splitted))

# counting unique vocabulary of words
text = data.text[1]
print("number of unique words: ",len(set(splitted)))

# print first five unique words
print("first 5 unique words: ",list(set(splitted))[:5])

# frequency of words 
dist = nlp.FreqDist(splitted)
print("frequency of words: ",dist)

# look at keys in dist
print("words in text: ",dist.keys())

# count how many time a particalar value occurs. Lets look at "box"
print("the word box is occured how many times:",dist["box"])
# normalization
words = "task Tasked tasks tasking"
words_list = words.lower().split(" ")
print("normalized words: ",words_list)

# stemming
porter_stemmer = nlp.PorterStemmer()
roots = [porter_stemmer.stem(each) for each in words_list]
print("roots of task Tasked tasks tasking: ",roots)
# stemming
stemming_word_list = ["Universal","recognition","Become","being","happened"]
porter_stemmer = nlp.PorterStemmer()
roots = [porter_stemmer.stem(each) for each in stemming_word_list]
print("result of stemming: ",roots)

# lemmatization
lemma = nlp.WordNetLemmatizer()
lemma_roots = [lemma.lemmatize(each) for each in stemming_word_list]
print("result of lemmatization: ",lemma_roots)
text_t = "You’re in the right place!"
print("split the sentece: ", text_t.split(" "))  # 5 words

# tokenization with nltk
print("tokenize with nltk: ",nlp.word_tokenize(text_t))

# %% import data
data = pd.read_csv(r"../input/twitter-user-gender-classification/gender-classifier-DFE-791531.csv",encoding='latin1')

# concat gender and description
data = pd.concat([data.gender,data.description],axis=1)

# drop nan values
data.dropna(inplace=True,axis=0)

# convert genders from female and male to 1 and 0 respectively
data.gender = [1 if each == "female" else 0 for each in data.gender] 
# import re # regular expression library
# # %% remove non important word a, the, that, and, in 
# import nltk as nlp
# import nltk
# nltk.download("stopwords")  # stopwords = (irrelavent words)
# from nltk.corpus import stopwords 

# %% creating bag of words model
from sklearn.feature_extraction.text import CountVectorizer  # for bag of words 
max_features = 150 # max_features dimension reduction 
count_vectorizer = CountVectorizer(stop_words = "english",max_features = max_features)  
# stop_words parameter = automatically remove all stopwords 
# lowercase parameter 
# token_pattern removing other karakters like .. !

sparce_matrix = count_vectorizer.fit_transform(review_list).toarray() # sparce matrix yaratir bag of words model = sparce matrix

print("Most used {} words: {}".format(max_features,count_vectorizer.get_feature_names()))

y = data.iloc[:,0].values  # positive or negative comment

#sparce matrix includes independent variable
# train test split
from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(sparce_matrix,y,test_size = 0.1,random_state = 0)
# %% naive bayes
from sklearn.naive_bayes import GaussianNB
nb = GaussianNB()
nb.fit(sparce_matrix,y)
# %% predict
y_pred = nb.predict(sparce_matrix)
#%%
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y,y_pred)
cm
import matplotlib.pyplot as plt
import seaborn as sns
f,ax = plt.subplots(figsize=(5, 5))
sns.heatmap(cm, annot=True, linewidths=0.5,linecolor="red", fmt= '.1f',ax=ax)
plt.show()
plt.savefig('graph.png')