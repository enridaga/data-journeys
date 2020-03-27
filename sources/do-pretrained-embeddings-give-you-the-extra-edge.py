
import sys, os, re, csv, codecs, numpy as np, pandas as pd

import matplotlib.pyplot as plt

#%matplotlib inline

from keras.preprocessing.text import Tokenizer

from keras.preprocessing.sequence import pad_sequences

from keras.layers import Dense, Input, LSTM, Embedding, Dropout, Activation

from keras.layers import Bidirectional, GlobalMaxPool1D,Bidirectional

from keras.models import Model

from keras import initializers, regularizers, constraints, optimizers, layers

import matplotlib.pyplot as plt

#%matplotlib inline

import gensim.models.keyedvectors as word2vec

import gc
train = pd.read_csv('../input/jigsaw-toxic-comment-classification-challenge/train.csv')

test = pd.read_csv('../input/jigsaw-toxic-comment-classification-challenge/test.csv')

embed_size=0
list_classes = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]

y = train[list_classes].values

list_sentences_train = train["comment_text"]

list_sentences_test = test["comment_text"]
max_features = 20000

tokenizer = Tokenizer(num_words=max_features)

tokenizer.fit_on_texts(list(list_sentences_train))

list_tokenized_train = tokenizer.texts_to_sequences(list_sentences_train)

list_tokenized_test = tokenizer.texts_to_sequences(list_sentences_test)
maxlen = 200

X_t = pad_sequences(list_tokenized_train, maxlen=maxlen)

X_te = pad_sequences(list_tokenized_test, maxlen=maxlen)
def loadEmbeddingMatrix(typeToLoad):

        #load different embedding file from Kaggle depending on which embedding 

        #matrix we are going to experiment with

        if(typeToLoad=="glove"):

            EMBEDDING_FILE='../input/glove-twitter/glove.twitter.27B.25d.txt'

            embed_size = 25

        elif(typeToLoad=="word2vec"):

            word2vecDict = word2vec.KeyedVectors.load_word2vec_format("../input/googlenewsvectorsnegative300/GoogleNews-vectors-negative300.bin", binary=True)

            embed_size = 300

        elif(typeToLoad=="fasttext"):

            EMBEDDING_FILE='../input/fasttext/wiki.simple.vec'

            embed_size = 300



        if(typeToLoad=="glove" or typeToLoad=="fasttext" ):

            embeddings_index = dict()

            #Transfer the embedding weights into a dictionary by iterating through every line of the file.

            f = open(EMBEDDING_FILE)

            for line in f:

                #split up line into an indexed array

                values = line.split()

                #first index is word

                word = values[0]

                #store the rest of the values in the array as a new array

                coefs = np.asarray(values[1:], dtype='float32')

                embeddings_index[word] = coefs #50 dimensions

            f.close()

            print('Loaded %s word vectors.' % len(embeddings_index))

        else:

            embeddings_index = dict()

            for word in word2vecDict.wv.vocab:

                embeddings_index[word] = word2vecDict.word_vec(word)

            print('Loaded %s word vectors.' % len(embeddings_index))

            

        gc.collect()

        #We get the mean and standard deviation of the embedding weights so that we could maintain the 

        #same statistics for the rest of our own random generated weights. 

        all_embs = np.stack(list(embeddings_index.values()))

        emb_mean,emb_std = all_embs.mean(), all_embs.std()

        

        nb_words = len(tokenizer.word_index)

        #We are going to set the embedding size to the pretrained dimension as we are replicating it.

        #the size will be Number of Words in Vocab X Embedding Size

        embedding_matrix = np.random.normal(emb_mean, emb_std, (nb_words, embed_size))

        gc.collect()



        #With the newly created embedding matrix, we'll fill it up with the words that we have in both 

        #our own dictionary and loaded pretrained embedding. 

        embeddedCount = 0

        for word, i in tokenizer.word_index.items():

            i-=1

            #then we see if this word is in glove's dictionary, if yes, get the corresponding weights

            embedding_vector = embeddings_index.get(word)

            #and store inside the embedding matrix that we will train later on.

            if embedding_vector is not None: 

                embedding_matrix[i] = embedding_vector

                embeddedCount+=1

        print('total embedded:',embeddedCount,'common words')

        

        del(embeddings_index)

        gc.collect()

        

        #finally, return the embedding matrix

        return embedding_matrix

embedding_matrix = loadEmbeddingMatrix('word2vec')
embedding_matrix.shape
inp = Input(shape=(maxlen, )) #maxlen=200 as defined earlier
x = Embedding(len(tokenizer.word_index), embedding_matrix.shape[1],weights=[embedding_matrix],trainable=False)(inp)
x = Bidirectional(LSTM(60, return_sequences=True,name='lstm_layer',dropout=0.1,recurrent_dropout=0.1))(x)
x = GlobalMaxPool1D()(x)
x = Dropout(0.1)(x)
x = Dense(50, activation="relu")(x)
x = Dropout(0.1)(x)
x = Dense(6, activation="sigmoid")(x)
model = Model(inputs=inp, outputs=x)

model.compile(loss='binary_crossentropy',

                  optimizer='adam',

                  metrics=['accuracy'])
model.summary()
#batch_size = 32

#epochs = 4

#hist = model.fit(X_t,y, batch_size=batch_size, epochs=epochs, validation_split=0.1)
#loadEmbeddingMatrix('word2vec')
#loadEmbeddingMatrix('glove') #for GLOVE or

#loadEmbeddingMatrix('fasttext') #for fasttext
all_losses = {

'word2vec_loss': [0.084318213647104789,

  0.057314205012433353,

  0.051338302593577821,

  0.047672802178572039],

 'word2vec_val_loss': [0.063002561892695971,

  0.057253835496480658,

  0.051085027624451551,

  0.049801279793734249],

'glove_loss': [0.11598931579683543,

  0.088738223480436862,

  0.079895263566000005,

  0.075343037429358703],

 'glove_val_loss': [0.093467933030432285,

  0.080007083813922117,

  0.075349041991106688,

  0.072366507668134517],

 'fasttext_loss': [0.079714499498945865,

  0.056074704045674786,

  0.050703874653286324,

  0.047420131195761134],

 'fasttext_val_loss': [0.058888281775148932,

  0.054906051694414926,

  0.054768857866843601,

  0.050697043558286421],

 'baseline_loss': [0.063304489498915865,

  0.044864004045674786,

  0.039013874651286124,

  0.038630130175761134],

 'baseline_val_loss': [0.048044281075148932,

  0.046414051594414926,

  0.047058757860843601,

  0.047886043558285421]

}
#f, ax = plt.subplots(1)

epochRange = np.arange(1,5,1)

plt.plot(epochRange,all_losses['word2vec_loss'])

plt.plot(epochRange,all_losses['glove_loss'])

plt.plot(epochRange,all_losses['fasttext_loss'])

plt.plot(epochRange,all_losses['baseline_loss'])

plt.title('Training loss for different embeddings')

plt.ylabel('loss')

plt.xlabel('epoch')

plt.legend(['Word2Vec', 'GLOVE','FastText','Baseline'], loc='upper left')

plt.show()
epochRange = np.arange(1,5,1)

plt.plot(epochRange,all_losses['baseline_loss'])

plt.plot(epochRange,all_losses['baseline_val_loss'])

plt.title('Training Vs Validation loss for baseline model')

plt.ylabel('loss')

plt.xlabel('epoch')

plt.legend(['Training', 'Validation'], loc='upper left')

plt.show()
f, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, sharex='col', sharey='row',figsize=(20, 20))



plt.title('Training Vs Validation loss for all embeddings')

ax1.plot(epochRange,all_losses['baseline_loss'])

ax1.plot(epochRange,all_losses['baseline_val_loss'])

ax1.set_title('Baseline')

ax1.set_ylim(0.03, 0.12)



ax2.plot(epochRange,all_losses['word2vec_loss'])

ax2.plot(epochRange,all_losses['word2vec_val_loss'])

ax2.set_title('Word2Vec')

ax2.set_ylim(0.03, 0.12)



ax3.plot(epochRange,all_losses['glove_loss'])

ax3.plot(epochRange,all_losses['glove_val_loss'])

ax3.set_title('GLOVE')

ax3.set_ylim(0.03, 0.12)





ax4.plot(epochRange,all_losses['fasttext_loss'])

ax4.plot(epochRange,all_losses['fasttext_val_loss'])

ax4.set_title('FastText')

ax4.set_ylim(0.03, 0.12)



plt.show()
wordCount = {'word2vec':66078,'glove':81610,'fasttext':59613,'baseline':210337}
ind = np.arange(0,4,1)  # the x locations for the groups

width = 0.35       # the width of the bars



plt.title('Number of common words used in different embeddings')

embNames = list(wordCount.keys())

embVals = list(wordCount.values())

plt.barh(ind,embVals,align='center', height=0.5, color='m',tick_label=embNames)

plt.show()