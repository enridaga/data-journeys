
## import libraries 
from collections import Counter 
import pandas as pd 
import numpy as np 
import string 

from plotly.offline import init_notebook_mode, iplot
import plotly.graph_objs as go
import matplotlib.pyplot as plt
import plotly.plotly as py
from plotly import tools
import seaborn as sns
init_notebook_mode(connected=True)
from itertools import zip_longest
import string 
import re

from nltk.corpus import stopwords 
from nltk.util import ngrams
import nltk 
stopwords = stopwords.words('english')

## dataset preparation
messages = pd.read_csv("../input/ForumMessages.csv")
messages['CreationDate'] = pd.to_datetime(messages['PostDate'])
messages['CreationYear'] = messages['CreationDate'].dt.year
messages['CreationMonth'] = messages['CreationDate'].dt.month
messages['CreationMonth'] = messages['CreationMonth'].apply(lambda x : "0"+str(x) if len(str(x)) < 2 else x)
messages['CreationDay'] = "29"
messages['KernelDate'] = messages["CreationYear"].astype(str) +"-"+ messages["CreationMonth"].astype(str) +"-"+ messages["CreationDay"].astype(str)
messages['Message'] = messages['Message'].fillna(" ")

## function to remove html entities from text
def striphtml(data):
    p = re.compile(r'<.*?>')
    return p.sub('', data)

## function to clean a text
def clntxt(text):
    text = text.lower()
    text = striphtml(text)
    text = " ".join([c for c in text.split() if c not in stopwords])
    for c in string.punctuation:
        text = text.replace(c, " ")
    text = " ".join([c for c in text.split() if c not in stopwords])
    
    words = []
    ignorewords = ["&nbsp;", "quot", "quote", "www", "http", "com"]
    for wrd in text.split():
        if len(wrd) <= 2: 
            continue
        if wrd in ignorewords:
            continue
        words.append(wrd)
    text = " ".join(words)    
    return text

## function to get top ngrams for a given year
def get_top_ngrams(yr, n, limit):
    # get relevant text
    temp = messages[messages['CreationYear'] == yr]
    text = " ".join(temp['Message']).lower()
    
    # cleaning
    text = striphtml(text)
    text = " ".join([c for c in text.split() if c not in stopwords])
    for c in string.punctuation:
        text = text.replace(c, " ")
    text = " ".join([c for c in text.split() if c not in stopwords])
    
    # ignore 
    words = []
    ignorewords = ["&nbsp;", "quot", "quote", "www", "http", "com"]
    for wrd in text.split():
        if len(wrd) <= 2: 
            continue
        if wrd in ignorewords:
            continue
        words.append(wrd)
    text = " ".join(words)
    
    # tokenize
    token = nltk.word_tokenize(text)
    grams = ngrams(token, n)
    grams = [" ".join(c) for c in grams]
    return dict(Counter(grams).most_common(limit))

def check_presence(txt, wrds):    
    cnt = 0
    txt = " "+txt+" "
    for wrd in wrds.split("|"):
        if " "+wrd+" " in txt:
            cnt += 1 
    return cnt

messages['CMessage'] = messages['Message'].apply(lambda x : clntxt(x))

messages['CreationDay'] = "21"
messages['KernelDate'] = messages["CreationYear"].astype(str) +"-"+ messages["CreationMonth"].astype(str) +"-"+ messages["CreationDay"].astype(str)
def plotthem(listed, title):    
    traces = []
    for model in listed:
        temp = messages.groupby('KernelDate').agg({model : "sum"}).reset_index()
        trace = go.Scatter(x = temp["KernelDate"], y = temp[model], name=model.split("|")[0].title(), line=dict(shape="spline", width=2), mode = "lines")
        traces.append(trace)

    layout = go.Layout(
        paper_bgcolor='#fff',
        plot_bgcolor="#fff",
        legend=dict(orientation="h", y=1.1),
        title=title,
        xaxis=dict(
            gridcolor='rgb(255,255,255)',
            range = ['2010-01-01','2018-06-01'],
            showgrid=True,
            showline=False,
            showticklabels=True,
            tickcolor='rgb(127,127,127)',
            ticks='outside',
            zeroline=False
        ),
        yaxis=dict(
            title="Number of Kaggle Discussions",
            gridcolor='rgb(255,255,255)',
            showgrid=False,
            showline=False,
            showticklabels=True,
            tickcolor='rgb(127,127,127)',
            ticks='outside',
            zeroline=False
        ),
    )

    fig = go.Figure(data=traces, layout=layout)
    iplot(fig)
    
## linear vs logistic regression
models = ["linear regression", "logistic regression"]
for col in models:
    messages[col] = messages["CMessage"].apply(lambda x : check_presence(x, col))
plotthem(models, "Kaggle Discussions: Linear vs Logistic")    

models = ["decision tree","random forest", "xgboost|xgb", "lightgbm|lgb", "catboost"]
for col in models:
    messages[col] = messages["CMessage"].apply(lambda x : check_presence(x, col))
plotthem(models, "Kaggle Discussions: Tree based models")    

models = ["neural network", "deep learning"]
for col in models:
    messages[col] = messages["CMessage"].apply(lambda x : check_presence(x, col))
plotthem(models, "Kaggle Discussions: Neural Networks vs Deep Learning")    
models = ["scikit", "tensorflow|tensor flow", "keras", "pytorch"]
for col in models:
    messages[col] = messages["CMessage"].apply(lambda x : check_presence(x, col))
plotthem(models, "Kaggle Discussions: ML Tools")    
models = ["xgboost|xgb", "keras"]
for col in models:
    messages[col] = messages["CMessage"].apply(lambda x : check_presence(x, col))
plotthem(models, "Kaggle Discussions: Xgboost vs Deep Learning")    

models = ["cnn|convolution", "lstm|rnn|gru"]
for col in models:
    messages[col] = messages["CMessage"].apply(lambda x : check_presence(x, col))
plotthem(models, "Kaggle Discussions: CNN and LSTM")    
models = ["matplotlib", "seaborn", "plotly"]
for col in models:
    messages[col] = messages["CMessage"].apply(lambda x : check_presence(x, col))
plotthem(models, "Kaggle Discussions: Python Data Visualization Libraries")    

models = ["ggplot", "highchart", "leaflet"]
for col in models:
    messages[col] = messages["CMessage"].apply(lambda x : check_presence(x, col))
plotthem(models, "Kaggle Discussions: R Data Visualization Libraries")    
models = ["exploration|explore|eda" , 'feature engineering', 'parameter tuning|hyperparameter tuning|model tuning|tuning', "ensembling|ensemble"]
for col in models:
    messages[col] = messages["CMessage"].apply(lambda x : check_presence(x, col))
plotthem(models, "Kaggle Discussions: Important Data Science Techniques")    
models = ["dataset" , 'kernel', 'competition', 'learn']
for col in models:
    messages[col] = messages["CMessage"].apply(lambda x : check_presence(x, col))
plotthem(models, "What is hottest on Kaggle")    