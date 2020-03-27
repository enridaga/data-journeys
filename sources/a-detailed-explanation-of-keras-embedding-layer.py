

# Ignore  the warnings
import warnings
warnings.filterwarnings('always')
warnings.filterwarnings('ignore')

# data visualisation and manipulation
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import style
import seaborn as sns
#configure
# sets matplotlib to inline and displays graphs below the corressponding cell.
#%matplotlib inline  
style.use('fivethirtyeight')
sns.set(style='whitegrid',color_codes=True)

#nltk
import nltk

#stop-words
from nltk.corpus import stopwords
stop_words=set(nltk.corpus.stopwords.words('english'))

# tokenizing
from nltk import word_tokenize,sent_tokenize

#keras
import keras
from keras.preprocessing.text import one_hot,Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Dense , Flatten ,Embedding,Input
from keras.models import Model

sample_text_1="bitty bought a bit of butter"
sample_text_2="but the bit of butter was a bit bitter"
sample_text_3="so she bought some better butter to make the bitter butter better"

corp=[sample_text_1,sample_text_2,sample_text_3]
no_docs=len(corp)

vocab_size=50 
encod_corp=[]
for i,doc in enumerate(corp):
    encod_corp.append(one_hot(doc,50))
    print("The encoding for document",i+1," is : ",one_hot(doc,50))
# length of maximum document. will be nedded whenever create embeddings for the words
maxlen=-1
for doc in corp:
    tokens=nltk.word_tokenize(doc)
    if(maxlen<len(tokens)):
        maxlen=len(tokens)
print("The maximum number of words in any document is : ",maxlen)
# now to create embeddings all of our docs need to be of same length. hence we can pad the docs with zeros.
pad_corp=pad_sequences(encod_corp,maxlen=maxlen,padding='post',value=0.0)
print("No of padded documents: ",len(pad_corp))
for i,doc in enumerate(pad_corp):
     print("The padded encoding for document",i+1," is : ",doc)
# specifying the input shape
input=Input(shape=(no_docs,maxlen),dtype='float64')
'''
shape of input. 
each document has 12 element or words which is the value of our maxlen variable.

'''
word_input=Input(shape=(maxlen,),dtype='float64')  

# creating the embedding
word_embedding=Embedding(input_dim=vocab_size,output_dim=8,input_length=maxlen)(word_input)

word_vec=Flatten()(word_embedding) # flatten
embed_model =Model([word_input],word_vec) # combining all into a Keras model
embed_model.compile(optimizer=keras.optimizers.Adam(lr=1e-3),loss='binary_crossentropy',metrics=['acc']) 
# compiling the model. parameters can be tuned as always.
print(type(word_embedding))
print(word_embedding)
print(embed_model.summary()) # summary of the model
embeddings=embed_model.predict(pad_corp) # finally getting the embeddings.
print("Shape of embeddings : ",embeddings.shape)
print(embeddings)
embeddings=embeddings.reshape(-1,maxlen,8)
print("Shape of embeddings : ",embeddings.shape) 
print(embeddings)


for i,doc in enumerate(embeddings):
    for j,word in enumerate(doc):
        print("The encoding for ",j+1,"th word","in",i+1,"th document is : \n\n",word)


