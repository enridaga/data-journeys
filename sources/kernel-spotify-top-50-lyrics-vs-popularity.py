
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns
import chardet
import nltk
from nltk.probability import FreqDist
from nltk.corpus import stopwords
import string
with open('/kaggle/input/top50spotify2019/top50.csv', 'rb') as f:
    result = chardet.detect(f.read())
    

raw_data = pd.read_csv("../input/top50spotify2019/top50.csv", encoding=result['encoding'])
raw_data_lyrics = pd.read_csv("../input/songs-lyrics/Lyrics.csv")
raw_data.info()
raw_data = raw_data.drop('Unnamed: 0', axis = 1 )
raw_data.shape
raw_data.head(3)
raw_data_lyrics.info()
raw_data_lyrics.head(3)
raw_data_lyrics.shape
raw_data.columns = ['Track_Name','Artist_Name','Genre','BPM','Energy','Danceability','Loudness', 'Liveness', 'Valence','Length', 'Acousticness', 'Speechiness','Popularity']
raw_data.describe()
sns.boxplot( y = raw_data["Popularity"])
fig, ax = plt.subplots(1,3)
fig.subplots_adjust(hspace=0.6, wspace=0.6)

sns.boxplot( y = raw_data["BPM"], ax=ax[0])
sns.boxplot( y = raw_data["Energy"], ax=ax[1])
sns.boxplot( y = raw_data["Danceability"], ax=ax[2])

fig.show()
fig, ax = plt.subplots(1,3)
fig.subplots_adjust(hspace=0.6, wspace=0.6)

sns.boxplot( y = raw_data["Loudness"], ax=ax[0])
sns.boxplot( y = raw_data["Liveness"], ax=ax[1])
sns.boxplot( y = raw_data["Valence"], ax=ax[2])

fig.show()
fig, ax = plt.subplots(1,3)
fig.subplots_adjust(hspace=0.8, wspace=0.8)

sns.boxplot( y = raw_data["Length"], ax=ax[0])
sns.boxplot( y = raw_data["Acousticness"], ax=ax[1])
sns.boxplot( y = raw_data["Speechiness"], ax=ax[2])

fig.show()
sns.catplot(y = "Genre", kind = "count",
            palette = "pastel", edgecolor = ".6",
            data = raw_data)
sns.catplot(x = "Popularity", y = "Genre", kind = "bar" ,
            palette = "pastel", edgecolor = ".6",
            data = raw_data)
sns.pairplot(raw_data)
dataset = pd.merge(raw_data, raw_data_lyrics, left_on='Track_Name', right_on='Track.Name')
del dataset['Track.Name']
dataset['Lyrics'] = dataset['Lyrics'].astype(str)
dataset
dataset = dataset.drop(dataset.index[[30, 22]])
dataset['Lyrics'] = dataset['Lyrics'].str.lower().replace(r'\n',' ')
dataset['Lyrics']
tokens = dataset['Lyrics'].fillna("").map(nltk.word_tokenize)
allWords = []
for wordList in tokens:
    allWords += wordList

fdist = FreqDist(allWords)
fdist.plot(30,cumulative=False)
plt.show()
stop_words_en = set(stopwords.words("english"))
stop_words_es = set(stopwords.words("spanish"))
punctuations = list(string.punctuation)
allWords = [i for i in allWords if i not in punctuations]
forbidden = ['oh',"'s", 'yo',"'ll", 'el', "'re","'m","oh-oh","'d", "n't", "``", "ooh", "uah", "'em", "'ve", "eh", "pa", "brr", "yeah"] 
filtered_sent=[]
for w in allWords:
    if (w not in stop_words_en) and (w not in stop_words_es):
        filtered_sent.append(w)
filter_=[]
for w in filtered_sent:
    if (w not in forbidden):
        filter_.append(w)
fdist = FreqDist(filter_)
fdist.plot(30,cumulative=False)
plt.show()