
# Usual Imports
import numpy as np
import pandas as pd
from tqdm import tqdm
import string
import random
import operator
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import NMF, LatentDirichletAllocation, TruncatedSVD
from statistics import *
from sklearn.feature_extraction.text import CountVectorizer
import concurrent.futures
import time
import pyLDAvis.sklearn
from pylab import bone, pcolor, colorbar, plot, show, rcParams, savefig
import textstat
import warnings
import nltk
warnings.filterwarnings('ignore')

#%matplotlib inline
import os
print(os.listdir("../input"))

# Plotly based imports for visualization
from plotly import tools
import plotly.plotly as py
from plotly.offline import init_notebook_mode, iplot
init_notebook_mode(connected=True)
import plotly.graph_objs as go
import plotly.figure_factory as ff

# spaCy based imports
import spacy
from spacy.lang.en.stop_words import STOP_WORDS
from spacy.lang.en import English

import warnings
warnings.filterwarnings("ignore")
quora_train = pd.read_csv("../input/train.csv")
quora_train.head()
# SpaCy Parser for questions
punctuations = string.punctuation
stopwords = list(STOP_WORDS)

parser = English()
def spacy_tokenizer(sentence):
    mytokens = parser(sentence)
    mytokens = [ word.lemma_.lower().strip() if word.lemma_ != "-PRON-" else word.lower_ for word in mytokens ]
    mytokens = [ word for word in mytokens if word not in stopwords and word not in punctuations ]
    mytokens = " ".join([i for i in mytokens])
    return mytokens
tqdm.pandas()
sincere_questions = quora_train["question_text"][quora_train["target"] == 0].progress_apply(spacy_tokenizer)
insincere_questions = quora_train["question_text"][quora_train["target"] == 1].progress_apply(spacy_tokenizer)
# One function for all plots
def plot_readability(a,b,title,bins=0.1,colors=['#3A4750', '#F64E8B']):
    trace1 = ff.create_distplot([a,b], ["Sincere questions","Insincere questions"], bin_size=bins, colors=colors, show_rug=False)
    trace1['layout'].update(title=title)
    iplot(trace1, filename='Distplot')
    table_data= [["Statistical Measures","Sincere questions","Insincere questions"],
                ["Mean",mean(a),mean(b)],
                ["Standard Deviation",pstdev(a),pstdev(b)],
                ["Variance",pvariance(a),pvariance(b)],
                ["Median",median(a),median(b)],
                ["Maximum value",max(a),max(b)],
                ["Minimum value",min(a),min(b)]]
    trace2 = ff.create_table(table_data)
    iplot(trace2, filename='Table')
syllable_sincere = np.array(quora_train["question_text"][quora_train["target"] == 0].progress_apply(textstat.syllable_count))
syllable_insincere = np.array(quora_train["question_text"][quora_train["target"] == 1].progress_apply(textstat.syllable_count))
plot_readability(syllable_sincere,syllable_insincere,"Syllable Analysis",5)
    
lexicon_sincere = np.array(quora_train["question_text"][quora_train["target"] == 0].progress_apply(textstat.lexicon_count))
lexicon_insincere = np.array(quora_train["question_text"][quora_train["target"] == 1].progress_apply(textstat.lexicon_count))
plot_readability(lexicon_sincere,lexicon_insincere,"Lexicon Analysis",4,['#C65D17','#DDB967'])
length_sincere = np.array(quora_train["question_text"][quora_train["target"] == 0].progress_apply(len))
length_insincere = np.array(quora_train["question_text"][quora_train["target"] == 1].progress_apply(len))
plot_readability(length_sincere,length_insincere,"Question Length",40,['#C65D17','#DDB967'])
spw_sincere = np.array(quora_train["question_text"][quora_train["target"] == 0].progress_apply(textstat.avg_syllables_per_word))
spw_insincere = np.array(quora_train["question_text"][quora_train["target"] == 1].progress_apply(textstat.avg_syllables_per_word))
plot_readability(spw_sincere,spw_insincere,"Average syllables per word",0.2,['#8D99AE','#EF233C'])
lpw_sincere = np.array(quora_train["question_text"][quora_train["target"] == 0].progress_apply(textstat.avg_letter_per_word))
lpw_insincere = np.array(quora_train["question_text"][quora_train["target"] == 1].progress_apply(textstat.avg_letter_per_word))
plot_readability(lpw_sincere,lpw_insincere,"Average letters per word",2,['#8491A3','#2B2D42'])
fre_sincere = np.array(quora_train["question_text"][quora_train["target"] == 0].progress_apply(textstat.flesch_reading_ease))
fre_insincere = np.array(quora_train["question_text"][quora_train["target"] == 1].progress_apply(textstat.flesch_reading_ease))
plot_readability(fre_sincere,fre_insincere,"Flesch Reading Ease",20)
fkg_sincere = np.array(quora_train["question_text"][quora_train["target"] == 0].progress_apply(textstat.flesch_kincaid_grade))
fkg_insincere = np.array(quora_train["question_text"][quora_train["target"] == 1].progress_apply(textstat.flesch_kincaid_grade))
plot_readability(fkg_sincere,fkg_insincere,"Flesch Kincaid Grade",4,['#C1D37F','#491F21'])
fog_sincere = np.array(quora_train["question_text"][quora_train["target"] == 0].progress_apply(textstat.gunning_fog))
fog_insincere = np.array(quora_train["question_text"][quora_train["target"] == 1].progress_apply(textstat.gunning_fog))
plot_readability(fog_sincere,fog_insincere,"The Fog Scale (Gunning FOG Formula)",4,['#E2D58B','#CDE77F'])
ari_sincere = np.array(quora_train["question_text"][quora_train["target"] == 0].progress_apply(textstat.automated_readability_index))
ari_insincere = np.array(quora_train["question_text"][quora_train["target"] == 1].progress_apply(textstat.automated_readability_index))
plot_readability(ari_sincere,ari_insincere,"Automated Readability Index",10,['#488286','#FF934F'])
cli_sincere = np.array(quora_train["question_text"][quora_train["target"] == 0].progress_apply(textstat.coleman_liau_index))
cli_insincere = np.array(quora_train["question_text"][quora_train["target"] == 1].progress_apply(textstat.coleman_liau_index))
plot_readability(cli_sincere,cli_insincere,"The Coleman-Liau Index",10,['#8491A3','#2B2D42'])
lwf_sincere = np.array(quora_train["question_text"][quora_train["target"] == 0].progress_apply(textstat.linsear_write_formula))
lwf_insincere = np.array(quora_train["question_text"][quora_train["target"] == 1].progress_apply(textstat.linsear_write_formula))
plot_readability(lwf_sincere,lwf_insincere,"Linsear Write Formula",2,['#8D99AE','#EF233C'])
dcr_sincere = np.array(quora_train["question_text"][quora_train["target"] == 0].progress_apply(textstat.dale_chall_readability_score))
dcr_insincere = np.array(quora_train["question_text"][quora_train["target"] == 1].progress_apply(textstat.dale_chall_readability_score))
plot_readability(dcr_sincere,dcr_insincere,"Dale-Chall Readability Score",1,['#C65D17','#DDB967'])
def consensus_all(text):
    return textstat.text_standard(text,float_output=True)

con_sincere = np.array(quora_train["question_text"][quora_train["target"] == 0].progress_apply(consensus_all))
con_insincere = np.array(quora_train["question_text"][quora_train["target"] == 1].progress_apply(consensus_all))
plot_readability(con_sincere,con_insincere,"Readability Consensus based upon all the above tests",2)
def word_generator(text):
    word = list(text.split())
    return word
def bigram_generator(text):
    bgram = list(nltk.bigrams(text.split()))
    bgram = [' '.join((a, b)) for (a, b) in bgram]
    return bgram
def trigram_generator(text):
    tgram = list(nltk.trigrams(text.split()))
    tgram = [' '.join((a, b, c)) for (a, b, c) in tgram]
    return tgram
sincere_words = sincere_questions.progress_apply(word_generator)
insincere_words = insincere_questions.progress_apply(word_generator)
sincere_bigrams = sincere_questions.progress_apply(bigram_generator)
insincere_bigrams = insincere_questions.progress_apply(bigram_generator)
sincere_trigrams = sincere_questions.progress_apply(trigram_generator)
insincere_trigrams = insincere_questions.progress_apply(trigram_generator)

color_brewer = ['#57B8FF','#B66D0D','#009FB7','#FBB13C','#FE6847','#4FB5A5','#8C9376','#F29F60','#8E1C4A','#85809B','#515B5D','#9EC2BE','#808080','#9BB58E','#5C0029','#151515','#A63D40','#E9B872','#56AA53','#CE6786','#449339','#2176FF','#348427','#671A31','#106B26','#008DD5','#034213','#BC2F59','#939C44','#ACFCD9','#1D3950','#9C5414','#5DD9C1','#7B6D49','#8120FF','#F224F2','#C16D45','#8A4F3D','#616B82','#443431','#340F09']

def ngram_visualizer(v,t):
    X = v.values
    Y = v.index
    trace = [go.Bar(
                y=Y,
                x=X,
                orientation = 'h',
                marker=dict(color=color_brewer, line=dict(color='rgb(8,48,107)',width=1.5,)),
                opacity = 0.6
    )]
    layout = go.Layout(
        title=t,
        margin = go.Margin(
            l = 200,
            r = 400
        )
    )

    fig = go.Figure(data=trace, layout=layout)
    iplot(fig, filename='horizontal-bar')
    
def ngram_plot(ngrams,title):
    ngram_list = []
    for i in tqdm(ngrams.values, total=ngrams.shape[0]):
        ngram_list.extend(i)
    random.shuffle(color_brewer)
    ngram_visualizer(pd.Series(ngram_list).value_counts()[:20],title)
# Top Sincere words
ngram_plot(sincere_words,"Top Sincere Words")
# Top Insincere words
ngram_plot(insincere_words,"Top Insincere Words")
# Sincere Bigrams
ngram_plot(sincere_bigrams,"Top 20 Sincere Bigrams")
# Insincere Bigrams
ngram_plot(insincere_bigrams,"Top 20 Insincere Bigrams")
# Sincere Trigrams
ngram_plot(sincere_trigrams,"Top 20 Sincere Trigrams")
# Insincere Trigrams
ngram_plot(insincere_trigrams,"Top 20 Insincere Trigrams")
vectorizer_sincere = CountVectorizer(min_df=5, max_df=0.9, stop_words='english', lowercase=True, token_pattern='[a-zA-Z\-][a-zA-Z\-]{2,}')
sincere_questions_vectorized = vectorizer_sincere.fit_transform(sincere_questions)
vectorizer_insincere = CountVectorizer(min_df=5, max_df=0.9, stop_words='english', lowercase=True, token_pattern='[a-zA-Z\-][a-zA-Z\-]{2,}')
insincere_questions_vectorized = vectorizer_insincere.fit_transform(insincere_questions)
# Latent Dirichlet Allocation Model
lda_sincere = LatentDirichletAllocation(n_components=10, max_iter=5, learning_method='online',verbose=True)
sincere_lda = lda_sincere.fit_transform(sincere_questions_vectorized)
lda_insincere = LatentDirichletAllocation(n_components=10, max_iter=5, learning_method='online',verbose=True)
insincere_lda = lda_insincere.fit_transform(insincere_questions_vectorized)
# Functions for printing keywords for each topic
def selected_topics(model, vectorizer, top_n=10):
    for idx, topic in enumerate(model.components_):
        print("Topic %d:" % (idx))
        print([(vectorizer.get_feature_names()[i], topic[i])
                        for i in topic.argsort()[:-top_n - 1:-1]]) 
# Keywords for topics clustered by Latent Dirichlet Allocation
print("Sincere questions LDA Model:")
selected_topics(lda_sincere, vectorizer_sincere)
print("Insincere questions LDA Model:")
selected_topics(lda_insincere, vectorizer_insincere)
pyLDAvis.enable_notebook()
dash = pyLDAvis.sklearn.prepare(lda_sincere, sincere_questions_vectorized, vectorizer_sincere, mds='tsne')
dash
pyLDAvis.enable_notebook()
dash = pyLDAvis.sklearn.prepare(lda_insincere, insincere_questions_vectorized, vectorizer_insincere, mds='tsne')
dash