
### pip install ../input/sacremoses/sacremoses-master/ > /dev/null
### pip install "../input/kerasswa/keras-swa-0.1.2"  > /dev/null
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import re
import gc
import glob
import os
import sys
import string
import random
from tqdm import tqdm_notebook
from urllib.parse import urlparse
from sklearn.preprocessing import OneHotEncoder, minmax_scale, MultiLabelBinarizer
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score
from sklearn.linear_model import MultiTaskElasticNet
from scipy.stats import spearmanr, rankdata
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD

#os.environ['CUDA_VISIBLE_DEVICES'] = '-1' # -1 = CPU only

import nltk
from nltk.tag import pos_tag
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk.corpus import stopwords

from gensim.models.doc2vec import TaggedDocument
from gensim import utils
import torch

sys.path.insert(0, "../input/transformers/transformers-master/")
import transformers


import tensorflow as tf

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        # Currently, memory growth needs to be the same across GPUs
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    except RuntimeError as e:
        # Memory growth must be set before GPUs have been initialized
        print(e)

import tensorflow_hub as hub

import keras
import keras.backend as K
from keras.layers import *
from keras.callbacks import *
from keras.optimizers import *
from keras import Model
from swa.keras import SWA

import lightgbm as lgb
import catboost as cb
import xgboost as xgb

import pickle    

def save_obj(obj, name ):
    with open(name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)
def load_obj(name ):
    with open(name + '.pkl', 'rb') as f:
        return pickle.load(f)                

#INPUT_PATH="/kaggle/input/"
INPUT_PATH="../input/"
train = pd.read_csv(INPUT_PATH+'google-quest-challenge/train.csv')
test = pd.read_csv(INPUT_PATH+'google-quest-challenge/test.csv')
submission = pd.read_csv(INPUT_PATH+'google-quest-challenge/sample_submission.csv')
targets = [
        'question_asker_intent_understanding',
        'question_body_critical',
        'question_conversational',
        'question_expect_short_answer',
        'question_fact_seeking',
        'question_has_commonly_accepted_answer',
        'question_interestingness_others',
        'question_interestingness_self',
        'question_multi_intent',
        'question_not_really_a_question',
        'question_opinion_seeking',
        'question_type_choice',
        'question_type_compare',
        'question_type_consequence',
        'question_type_definition',
        'question_type_entity',
        'question_type_instructions',
        'question_type_procedure',
        'question_type_reason_explanation',
        'question_type_spelling',
        'question_well_written',
        'answer_helpful',
        'answer_level_of_information',
        'answer_plausible',
        'answer_relevance',
        'answer_satisfaction',
        'answer_type_instructions',
        'answer_type_procedure',
        'answer_type_reason_explanation',
        'answer_well_written'    
    ]

input_columns = ['question_title','question_body','answer']
#clean data
puncts = [',', '.', '"', ':', ')', '(', '-', '!', '?', '|', ';', "'", '$', '&', '/', '[', ']', '>', '%', '=', '#', '*', '+', '\\', '•',  '~', '@', '£',
 '·', '_', '{', '}', '©', '^', '®', '`',  '<', '→', '°', '€', '™', '›',  '♥', '←', '×', '§', '″', '′', 'Â', '█', '½', 'à', '…', '\xa0', '\t',
 '“', '★', '”', '–', '●', 'â', '►', '−', '¢', '²', '¬', '░', '¶', '↑', '±', '¿', '▾', '═', '¦', '║', '―', '¥', '▓', '—', '‹', '─', '\u3000', '\u202f',
 '▒', '：', '¼', '⊕', '▼', '▪', '†', '■', '’', '▀', '¨', '▄', '♫', '☆', 'é', '¯', '♦', '¤', '▲', 'è', '¸', '¾', 'Ã', '⋅', '‘', '∞', '«',
 '∙', '）', '↓', '、', '│', '（', '»', '，', '♪', '╩', '╚', '³', '・', '╦', '╣', '╔', '╗', '▬', '❤', 'ï', 'Ø', '¹', '≤', '‡', '√', ]

mispell_dict = {"aren't" : "are not",
"can't" : "cannot",
"couldn't" : "could not",
"couldnt" : "could not",
"didn't" : "did not",
"doesn't" : "does not",
"doesnt" : "does not",
"don't" : "do not",
"hadn't" : "had not",
"hasn't" : "has not",
"haven't" : "have not",
"havent" : "have not",
"he'd" : "he would",
"he'll" : "he will",
"he's" : "he is",
"i'd" : "I would",
"i'd" : "I had",
"i'll" : "I will",
"i'm" : "I am",
"isn't" : "is not",
"it's" : "it is",
"it'll":"it will",
"i've" : "I have",
"let's" : "let us",
"mightn't" : "might not",
"mustn't" : "must not",
"shan't" : "shall not",
"she'd" : "she would",
"she'll" : "she will",
"she's" : "she is",
"shouldn't" : "should not",
"shouldnt" : "should not",
"that's" : "that is",
"thats" : "that is",
"there's" : "there is",
"theres" : "there is",
"they'd" : "they would",
"they'll" : "they will",
"they're" : "they are",
"theyre":  "they are",
"they've" : "they have",
"we'd" : "we would",
"we're" : "we are",
"weren't" : "were not",
"we've" : "we have",
"what'll" : "what will",
"what're" : "what are",
"what's" : "what is",
"what've" : "what have",
"where's" : "where is",
"who'd" : "who would",
"who'll" : "who will",
"who're" : "who are",
"who's" : "who is",
"who've" : "who have",
"won't" : "will not",
"wouldn't" : "would not",
"you'd" : "you would",
"you'll" : "you will",
"you're" : "you are",
"you've" : "you have",
"'re": " are",
"wasn't": "was not",
"we'll":" will",
"didn't": "did not",
"tryin'":"trying"}


def clean_text(x):
    x = str(x).replace("\n","")
    for punct in puncts:
        x = x.replace(punct, f' {punct} ')
    return x


def clean_numbers(x):
    x = re.sub('[0-9]{5,}', '#####', x)
    x = re.sub('[0-9]{4}', '####', x)
    x = re.sub('[0-9]{3}', '###', x)
    x = re.sub('[0-9]{2}', '##', x)
    return x


def _get_mispell(mispell_dict):
    mispell_re = re.compile('(%s)' % '|'.join(mispell_dict.keys()))
    return mispell_dict, mispell_re


def replace_typical_misspell(text):
    mispellings, mispellings_re = _get_mispell(mispell_dict)

    def replace(match):
        return mispellings[match.group(0)]

    return mispellings_re.sub(replace, text)


def clean_data(df, columns: list):
    for col in columns:
        df[col] = df[col].apply(lambda x: clean_numbers(x))
        df[col] = df[col].apply(lambda x: clean_text(x.lower()))
        df[col] = df[col].apply(lambda x: replace_typical_misspell(x))

    return df
train = clean_data(train, input_columns)
test = clean_data(test, input_columns)
train.head(3)
train['question_body'][0]
from transformers import BertTokenizer, AdamW, BertModel, BertForSequenceClassification,BertPreTrainedModel, BertConfig 
from transformers import DistilBertConfig, DistilBertModel, DistilBertTokenizer, DistilBertForSequenceClassification
from transformers import RobertaConfig, RobertaModel, RobertaTokenizer, RobertaForSequenceClassification

from transformers import AdamW, XLNetTokenizer, XLNetModel, XLNetLMHeadModel, XLNetConfig

from transformers.optimization import get_linear_schedule_with_warmup
# word count in title, body and answer
for colname in ['question_title', 'question_body', 'answer']:
    newname = colname + '_word_len'
    
    train[newname] = train[colname].str.split().str.len()
    test[newname] = test[colname].str.split().str.len()

    
del newname, colname
for colname in ['question', 'answer']:

    # check for nonames, i.e. users with logins like user12389
    train['is_'+colname+'_no_name_user'] = train[colname +'_user_name'].str.contains('^user\d+$') + 0
    test['is_'+colname+'_no_name_user'] = test[colname +'_user_name'].str.contains('^user\d+$') + 0
    

colname = 'answer'
# check lexical diversity (unique words count vs total )
train[colname+'_div'] = train[colname].apply(lambda s: len(set(s.split())) / len(s.split()) )
test[colname+'_div'] = test[colname].apply(lambda s: len(set(s.split())) / len(s.split()) )
train.head(2)
## domain components
train['domcom'] = train['question_user_page'].apply(lambda s: s.split('://')[1].split('/')[0].split('.'))
test['domcom'] = test['question_user_page'].apply(lambda s: s.split('://')[1].split('/')[0].split('.'))

# count components
train['dom_cnt'] = train['domcom'].apply(lambda s: len(s))
test['dom_cnt'] = test['domcom'].apply(lambda s: len(s))

# extend length
train['domcom'] = train['domcom'].apply(lambda s: s + ['none', 'none'])
test['domcom'] = test['domcom'].apply(lambda s: s + ['none', 'none'])

# components
for ii in range(0,4):
    train['dom_'+str(ii)] = train['domcom'].apply(lambda s: s[ii])
    test['dom_'+str(ii)] = test['domcom'].apply(lambda s: s[ii])
    
# clean up
train.drop('domcom', axis = 1, inplace = True)
test.drop('domcom', axis = 1, inplace = True)
import nltk
from nltk.corpus import stopwords
eng_stopwords = set(stopwords.words("english"))

# shared elements
train['q_words'] = train['question_body'].apply(lambda s: [f for f in s.split() if f not in eng_stopwords] )
train['a_words'] = train['answer'].apply(lambda s: [f for f in s.split() if f not in eng_stopwords] )
train['qa_word_overlap'] = train.apply(lambda s: len(np.intersect1d(s['q_words'], s['a_words'])), axis = 1)
train['qa_word_overlap_norm1'] = train.apply(lambda s: s['qa_word_overlap']/(1 + len(s['a_words'])), axis = 1)
train['qa_word_overlap_norm2'] = train.apply(lambda s: s['qa_word_overlap']/(1 + len(s['q_words'])), axis = 1)
train.drop(['q_words', 'a_words'], axis = 1, inplace = True)

test['q_words'] = test['question_body'].apply(lambda s: [f for f in s.split() if f not in eng_stopwords] )
test['a_words'] = test['answer'].apply(lambda s: [f for f in s.split() if f not in eng_stopwords] )
test['qa_word_overlap'] = test.apply(lambda s: len(np.intersect1d(s['q_words'], s['a_words'])), axis = 1)
test['qa_word_overlap_norm1'] = test.apply(lambda s: s['qa_word_overlap']/(1 + len(s['a_words'])), axis = 1)
test['qa_word_overlap_norm2'] = test.apply(lambda s: s['qa_word_overlap']/(1 + len(s['q_words'])), axis = 1)
test.drop(['q_words', 'a_words'], axis = 1, inplace = True)
## Number of characters in the text ##
train["question_title_num_chars"] = train["question_title"].apply(lambda x: len(str(x)))
test["question_title_num_chars"] = test["question_title"].apply(lambda x: len(str(x)))
train["question_body_num_chars"] = train["question_body"].apply(lambda x: len(str(x)))
test["question_body_num_chars"] = test["question_body"].apply(lambda x: len(str(x)))
train["answer_num_chars"] = train["answer"].apply(lambda x: len(str(x)))
test["answer_num_chars"] = test["answer"].apply(lambda x: len(str(x)))

## Number of stopwords in the text ##
train["question_title_num_stopwords"] = train["question_title"].apply(lambda x: len([w for w in str(x).lower().split() if w in eng_stopwords]))
test["question_title_num_stopwords"] = test["question_title"].apply(lambda x: len([w for w in str(x).lower().split() if w in eng_stopwords]))
train["question_body_num_stopwords"] = train["question_body"].apply(lambda x: len([w for w in str(x).lower().split() if w in eng_stopwords]))
test["question_body_num_stopwords"] = test["question_body"].apply(lambda x: len([w for w in str(x).lower().split() if w in eng_stopwords]))
train["answer_num_stopwords"] = train["answer"].apply(lambda x: len([w for w in str(x).lower().split() if w in eng_stopwords]))
test["answer_num_stopwords"] = test["answer"].apply(lambda x: len([w for w in str(x).lower().split() if w in eng_stopwords]))

## Number of punctuations in the text ##
train["question_title_num_punctuations"] =train['question_title'].apply(lambda x: len([c for c in str(x) if c in string.punctuation]) )
test["question_title_num_punctuations"] =test['question_title'].apply(lambda x: len([c for c in str(x) if c in string.punctuation]) )
train["question_body_num_punctuations"] =train['question_body'].apply(lambda x: len([c for c in str(x) if c in string.punctuation]) )
test["question_body_num_punctuations"] =test['question_body'].apply(lambda x: len([c for c in str(x) if c in string.punctuation]) )
train["answer_num_punctuations"] =train['answer'].apply(lambda x: len([c for c in str(x) if c in string.punctuation]) )
test["answer_num_punctuations"] =test['answer'].apply(lambda x: len([c for c in str(x) if c in string.punctuation]) )

## Number of title case words in the text ##
train["question_title_num_words_upper"] = train["question_title"].apply(lambda x: len([w for w in str(x).split() if w.isupper()]))
test["question_title_num_words_upper"] = test["question_title"].apply(lambda x: len([w for w in str(x).split() if w.isupper()]))
train["question_body_num_words_upper"] = train["question_body"].apply(lambda x: len([w for w in str(x).split() if w.isupper()]))
test["question_body_num_words_upper"] = test["question_body"].apply(lambda x: len([w for w in str(x).split() if w.isupper()]))
train["answer_num_words_upper"] = train["answer"].apply(lambda x: len([w for w in str(x).split() if w.isupper()]))
test["answer_num_words_upper"] = test["answer"].apply(lambda x: len([w for w in str(x).split() if w.isupper()]))
list(train.columns)
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)
train.head(4)
cont_cols = [
'question_title_word_len',
'question_body_word_len',
'answer_div',
'answer_word_len',
'qa_word_overlap',
 'qa_word_overlap_norm1',
 'qa_word_overlap_norm2',
 'question_title_num_chars',
 'question_body_num_chars',
 'answer_num_chars',
 'question_title_num_punctuations',
 'question_body_num_punctuations',
 'answer_num_punctuations',
 'question_title_num_stopwords',
 'question_body_num_stopwords',
 'answer_num_stopwords',
 'question_title_num_words_upper',
 'question_body_num_words_upper',
 'answer_num_words_upper'
 
   
]

cat_cols = [
 'is_question_no_name_user',
 'is_answer_no_name_user',    
'dom_cnt',
 'dom_0',
 'dom_1',
 'dom_2',
 'dom_3' 
       
]
train.shape[1] == len(cat_cols+cont_cols+targets+input_columns)+8
def constructLabeledSentences(data):
    sentences=[]
    for index, row in data.iteritems():
        sentences.append(TaggedDocument(utils.to_unicode(row).split(), ['Text' + '_%s' % str(index)]))
    return sentences

def textClean(text):
    text = re.sub(r"[^A-Za-z0-9^,!.\/'+-=]", " ", text)
    text = text.lower().split()
    stops = set(stopwords.words("english"))
    text = [w for w in text if not w in stops]    
    text = " ".join(text)
    return(text)
    
def cleanup(text):
    text = textClean(text)
    text= text.translate(str.maketrans("","", string.punctuation))
    return text

train_question_body_sentences = constructLabeledSentences(train['question_body'])
train_question_title_sentences = constructLabeledSentences(train['question_title'])
train_answer_sentences = constructLabeledSentences(train['answer'])

test_question_body_sentences = constructLabeledSentences(test['question_body'])
test_question_title_sentences = constructLabeledSentences(test['question_title'])
test_answer_sentences = constructLabeledSentences(test['answer'])
from gensim.models import Doc2Vec

all_sentences = train_question_body_sentences + \
                train_answer_sentences + \
                test_question_body_sentences + \
                test_answer_sentences

Text_INPUT_DIM=128
text_model = Doc2Vec(min_count=1, window=5, vector_size=Text_INPUT_DIM, sample=1e-4, negative=5, workers=4, epochs=5,seed=1)
text_model.build_vocab(all_sentences)
text_model.train(all_sentences, total_examples=text_model.corpus_count, epochs=text_model.iter)
from nltk.stem.wordnet import WordNetLemmatizer

def normalize_sentence(tokens):
    lemmatizer = WordNetLemmatizer()
    lemmatized_sentence = []
    for word, tag in pos_tag(tokens):
        if tag.startswith('NN') or tag.startswith('PRP'):
            pos = 'n'
        elif tag.startswith('VB'):
            pos = 'v'
        else:
            continue
            pos = 'a'
        lemmatized_sentence.append(lemmatizer.lemmatize(word, pos).lower())
    return lemmatized_sentence
def normalize_vectorize(df, columns: list):
    for col in columns:
        print(col)
        df[col+'_norm'] = df[col].apply(lambda x: ' '.join(set(normalize_sentence(word_tokenize(x)))))
        df[col+'_vec'] = df[col].apply(lambda x: text_model.infer_vector([x]))

    return df

train = normalize_vectorize(train, input_columns)
test = normalize_vectorize(test, input_columns)
train.head()
#bert embedings
try:
    pbe = load_obj("../input/questembeddings/precomputed_bert_embeddings")
    train_question_body_dense = pbe['train_question_body_dense']
    train_answer_dense = pbe['train_answer_dense']
    train_question_title_dense = pbe['train_question_title_dense']
    test_question_body_dense = pbe['test_question_body_dense']
    test_answer_dense = pbe['test_answer_dense']
    test_question_title_dense = pbe['test_question_title_dense']
except:
    print("Load failed, build embedding")
    def sigmoid(x):
        return 1 / (1 + math.exp(-x))

    def chunks(l, n):
        """Yield successive n-sized chunks from l."""
        for i in range(0, len(l), n):
            yield l[i:i + n]

    def fetch_vectors(string_list, batch_size=64):
        
#         DEVICE = torch.device("cuda")
#         tokenizer = transformers.DistilBertTokenizer.from_pretrained("../input/distilbertbaseuncased/")
#         model = transformers.DistilBertModel.from_pretrained("../input/distilbertbaseuncased/")
#         model.to(DEVICE)
        # inspired by https://jalammar.github.io/a-visual-guide-to-using-bert-for-the-first-time/
        DEVICE = torch.device("cuda")
        tokenizer = transformers.BertTokenizer.from_pretrained("../input/pretrained-bert-models-for-pytorch/bert-base-uncased-vocab.txt")
        bert_config = BertConfig.from_json_file('../input/pretrained-bert-models-for-pytorch/bert-base-uncased/bert_config.json')
        bert_config.num_labels =30

        model_path = os.path.join('../input/pretrained-bert-models-for-pytorch/bert-base-uncased/')

        model = BertModel.from_pretrained(model_path, config=bert_config)
        model.to(DEVICE)

            

        fin_features = []
        for data in tqdm_notebook(chunks(string_list, batch_size)):
            tokenized = []
            for x in data:
                x = " ".join(x.strip().split()[:300])
                tok = tokenizer.encode(x, add_special_tokens=True)
                tokenized.append(tok[:512])

            max_len = 512
            padded = np.array([i + [0] * (max_len - len(i)) for i in tokenized], dtype='int64')
            attention_mask = np.where(padded != 0, 1, 0)
            input_ids = torch.tensor(padded).to(DEVICE)
            attention_mask = torch.tensor(attention_mask).to(DEVICE)

            with torch.no_grad():
                last_hidden_states = model(input_ids, attention_mask=attention_mask)

            features = last_hidden_states[0][:, 0, :].cpu().numpy()
            fin_features.append(features)

        fin_features = np.vstack(fin_features)
        return fin_features

    train_question_body_dense = fetch_vectors(train.question_body.values)
    train_answer_dense = fetch_vectors(train.answer.values)
    train_question_title_dense = fetch_vectors(train.question_title.values)


    test_question_body_dense = fetch_vectors(test.question_body.values)
    test_answer_dense = fetch_vectors(test.answer.values)
    test_question_title_dense = fetch_vectors(test.question_title.values)

    precomputed_bert_embeddings = {
        'train_question_body_dense': train_question_body_dense,
        'train_answer_dense': train_answer_dense,
        'train_question_title_dense': train_question_title_dense,
        'test_question_body_dense': test_question_body_dense,
        'test_answer_dense': test_answer_dense,
        'test_question_title_dense': test_question_title_dense,

    }

    #save_obj(precomputed_bert_embeddings,"../input/questembeddings/precomputed_bert_embeddings")
# From the Ref Kernel's
from math import floor, ceil

def _get_masks(tokens, max_seq_length):
    """Mask for padding"""
    if len(tokens)>max_seq_length:
        raise IndexError("Token length more than max seq length!")
    return [1]*len(tokens) + [0] * (max_seq_length - len(tokens))

def _get_segments(tokens, max_seq_length):
    """Segments: 0 for the first sequence, 1 for the second"""
    
    if len(tokens) > max_seq_length:
        raise IndexError("Token length more than max seq length!")
        
    segments = []
    first_sep = True
    current_segment_id = 0
    
    for token in tokens:
        segments.append(current_segment_id)
        if token == "[SEP]":
            if first_sep:
                first_sep = False 
            else:
                current_segment_id = 1
    return segments + [0] * (max_seq_length - len(tokens))

def _get_ids(tokens, tokenizer, max_seq_length):
    """Token ids from Tokenizer vocab"""
    
    token_ids = tokenizer.convert_tokens_to_ids(tokens)
    input_ids = token_ids + [0] * (max_seq_length-len(token_ids))
    return input_ids

def _trim_input(title, question, answer, max_sequence_length=512, t_max_len=30, q_max_len=239, a_max_len=239):
    
    #293+239+30 = 508 + 4 = 512
    t = tokenizer.tokenize(title)
    q = tokenizer.tokenize(question)
    a = tokenizer.tokenize(answer)
    
    t_len = len(t)
    q_len = len(q)
    a_len = len(a)

    if (t_len+q_len+a_len+4) > max_sequence_length:
        
        if t_max_len > t_len:
            t_new_len = t_len
            a_max_len = a_max_len + floor((t_max_len - t_len)/2)
            q_max_len = q_max_len + ceil((t_max_len - t_len)/2)
        else:
            t_new_len = t_max_len
      
        if a_max_len > a_len:
            a_new_len = a_len 
            q_new_len = q_max_len + (a_max_len - a_len)
        elif q_max_len > q_len:
            a_new_len = a_max_len + (q_max_len - q_len)
            q_new_len = q_len
        else:
            a_new_len = a_max_len
            q_new_len = q_max_len
            
            
        if t_new_len+a_new_len+q_new_len+4 != max_sequence_length:
            raise ValueError("New sequence length should be %d, but is %d"%(max_sequence_length, (t_new_len + a_new_len + q_new_len + 4)))
        
        t = t[:t_new_len]
        q = q[:q_new_len]
        a = a[:a_new_len]
    
    return t, q, a

def _convert_to_bert_inputs(title, question, answer, tokenizer, max_sequence_length):
    """Converts tokenized input to ids, masks and segments for BERT"""
    
    stoken = ["[CLS]"] + title + ["[SEP]"] + question + ["[SEP]"] + answer + ["[SEP]"]

    input_ids = _get_ids(stoken, tokenizer, max_sequence_length)
    input_masks = _get_masks(stoken, max_sequence_length)
    input_segments = _get_segments(stoken, max_sequence_length)

    return [input_ids, input_masks, input_segments]

def compute_input_arays(df, columns, tokenizer, max_sequence_length):
    
    input_ids, input_masks, input_segments = [], [], []
    for _, instance in tqdm_notebook(df[columns].iterrows()):
        t, q, a = instance.question_title, instance.question_body, instance.answer
        t, q, a = _trim_input(t, q, a, max_sequence_length)
        ids, masks, segments = _convert_to_bert_inputs(t, q, a, tokenizer, max_sequence_length)
        input_ids.append(ids)
        input_masks.append(masks)
        input_segments.append(segments)
    return [
        np.asarray(input_ids, dtype=np.int32),
        np.asarray(input_masks, dtype=np.int32),
        np.asarray(input_segments, dtype=np.int32),
    ]

def compute_output_arrays(df, columns):
    return np.asarray(df[columns])
tokenizer = transformers.BertTokenizer.from_pretrained("../input/pretrained-bert-models-for-pytorch/bert-base-uncased-vocab.txt")
input_categories = list(train.columns[[1,2,5]]); input_categories
### %time
outputs = compute_output_arrays(train, columns = targets)
inputs = compute_input_arays(train, input_categories, tokenizer, max_sequence_length=512)
test_inputs = compute_input_arays(test, input_categories, tokenizer, max_sequence_length=512)
inputs[0], inputs[1], inputs[2]
test_inputs[0], test_inputs[1], test_inputs[2]
inputs[0].shape, inputs[1].shape, inputs[2].shape, test_inputs[0].shape, test_inputs[1].shape, test_inputs[2].shape
inputs_feats = np.hstack((inputs[0], inputs[1], inputs[2]))
test_inputs_feats = np.hstack((test_inputs[0], test_inputs[1], test_inputs[2]))
inputs_feats.shape, test_inputs_feats.shape
import re, string
re_tok = re.compile(f'([{string.punctuation}“”¨«»®´·º½¾¿¡§£₤‘’])')
def tokenize(s): return re_tok.sub(r' \1 ', s).split()
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn import linear_model
from sklearn.model_selection import train_test_split
import scipy
from sklearn.metrics import log_loss
import xgboost as xgb
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_auc_score
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.sparse import hstack
from sklearn.decomposition import TruncatedSVD

train_text_1 = train['question_body']
test_text_1 = test['question_body']
all_text_1 = pd.concat([train_text_1, test_text_1])

train_text_2 = train['answer']
test_text_2 = test['answer']
all_text_2 = pd.concat([train_text_2, test_text_2])

train_text_3 = train['question_title']
test_text_3 = test['question_title']
all_text_3 = pd.concat([train_text_3, test_text_3])

word_vectorizer = TfidfVectorizer(
    sublinear_tf=True,
    strip_accents='unicode',
    analyzer='word',
    token_pattern=r'\w{1,}',
    stop_words='english',
    ngram_range=(1, 2),
    max_features=80000,
    tokenizer=tokenize)
word_vectorizer.fit(all_text_1)

train_word_features_1 = word_vectorizer.transform(train_text_1)
test_word_features_1 = word_vectorizer.transform(test_text_1)

word_vectorizer = TfidfVectorizer(
    sublinear_tf=True,
    strip_accents='unicode',
    analyzer='word',
    token_pattern=r'\w{1,}',
    stop_words='english',
    ngram_range=(1, 2),
    max_features=80000,
    tokenizer=tokenize)
word_vectorizer.fit(all_text_2)

train_word_features_2 = word_vectorizer.transform(train_text_2)
test_word_features_2 = word_vectorizer.transform(test_text_2)

word_vectorizer = TfidfVectorizer(
    sublinear_tf=True,
    strip_accents='unicode',
    analyzer='word',
    token_pattern=r'\w{1,}',
    stop_words='english',
    ngram_range=(1, 2),
    max_features=80000,
    tokenizer=tokenize)
word_vectorizer.fit(all_text_3)

train_word_features_3 = word_vectorizer.transform(train_text_3)
test_word_features_3 = word_vectorizer.transform(test_text_3)

char_vectorizer = TfidfVectorizer(
    sublinear_tf=True,
    strip_accents='unicode',
    analyzer='char',
    stop_words='english',
    token_pattern=r'\w{1,}',
    ngram_range=(1, 4),
    max_features=50000,
    tokenizer=tokenize)
char_vectorizer.fit(all_text_1)

train_char_features_1 = char_vectorizer.transform(train_text_1)
test_char_features_1 = char_vectorizer.transform(test_text_1)

char_vectorizer = TfidfVectorizer(
    sublinear_tf=True,
    strip_accents='unicode',
    analyzer='char',
    stop_words='english',
    token_pattern=r'\w{1,}',
    ngram_range=(1, 4),
    max_features=50000,
    tokenizer=tokenize)
char_vectorizer.fit(all_text_2)

train_char_features_2 = char_vectorizer.transform(train_text_2)
test_char_features_2 = char_vectorizer.transform(test_text_2)

char_vectorizer = TfidfVectorizer(
    sublinear_tf=True,
    strip_accents='unicode',
    analyzer='char',
    stop_words='english',
    ngram_range=(1, 4),
    max_features=50000,
    tokenizer=tokenize)
char_vectorizer.fit(all_text_3)

train_char_features_3 = char_vectorizer.transform(train_text_3)
test_char_features_3 = char_vectorizer.transform(test_text_3)

train_features_tfidf = hstack([train_char_features_1, train_word_features_1, train_char_features_2, train_word_features_2,train_char_features_3, train_word_features_3])
test_features_tfidf = hstack([test_char_features_1, test_word_features_1, test_char_features_2, test_word_features_2,test_char_features_3, test_word_features_3])

pca = TruncatedSVD(n_components=128, n_iter=10)
tf_idf_text_train = pca.fit_transform(train_features_tfidf)
tf_idf_text_test = pca.fit_transform(test_features_tfidf)
tf_idf_text_train.shape, tf_idf_text_test.shape
torch.cuda.empty_cache() # release all gpu memory from pytorch
# universal sentence encoder

try:
    embeddings_train = load_obj("../input/questembeddings/use_embeddings_train")
    embeddings_test = load_obj("../input/questembeddings/use_embeddings_test")
except:
    print("Load failed, build embedding")
    try:
        module_url = INPUT_PATH+'universalsentenceencoderlarge4/'
        embed = hub.load(module_url)
        def UniversalEmbedding(x):
            results = embed(tf.squeeze(tf.cast(x, tf.string)))["outputs"]
            return keras.backend.concatenate([results])
    except:
        module_url = INPUT_PATH+'universalsentenceencoderlarge3/'
        embed = hub.Module(module_url)
        def UniversalEmbedding(x):
            results = embed(tf.squeeze(tf.cast(x, tf.string)), signature="default", as_dict=True)["default"]
            return keras.backend.concatenate([results])

    embeddings_train = {}
    embeddings_test = {}
    for text in input_columns:
        print(text)
        train_text = train[text].str.replace('?', '.').str.replace('!', '.').tolist()
        test_text = test[text].str.replace('?', '.').str.replace('!', '.').tolist()

        curr_train_emb = []
        curr_test_emb = []
        batch_size = 4
        ind = 0
        while ind*batch_size < len(train_text):
            curr_train_emb.append(embed(train_text[ind*batch_size: (ind + 1)*batch_size])["outputs"].numpy())
            ind += 1

        ind = 0
        while ind*batch_size < len(test_text):
            curr_test_emb.append(embed(test_text[ind*batch_size: (ind + 1)*batch_size])["outputs"].numpy())
            ind += 1    

        embeddings_train[text + '_embedding'] = np.vstack(curr_train_emb)
        embeddings_test[text + '_embedding'] = np.vstack(curr_test_emb)

    del embed
    K.clear_session()
    gc.collect()
    
    #save_obj(embeddings_train,"../input/questembeddings/use_embeddings_train")
    #save_obj(embeddings_test,"../input/questembeddings/use_embeddings_test")
embeddings_train
find = re.compile(r"^[^.]*")
train['netloc_1'] = train['url'].apply(lambda x: re.findall(find, urlparse(x).netloc)[0])
test['netloc_1'] = test['url'].apply(lambda x: re.findall(find, urlparse(x).netloc)[0])

train['netloc_2'] = train['question_user_page'].apply(lambda x: re.findall(find, urlparse(x).netloc)[0])
test['netloc_2'] = test['question_user_page'].apply(lambda x: re.findall(find, urlparse(x).netloc)[0])

train['netloc_3'] = train['answer_user_page'].apply(lambda x: re.findall(find, urlparse(x).netloc)[0])
test['netloc_3'] = test['answer_user_page'].apply(lambda x: re.findall(find, urlparse(x).netloc)[0])

tabular_cols = ['question_user_name', 'answer_user_name', 
               'netloc_1', 'netloc_2', 'netloc_3',
               'category', 'host']


    
l2_dist = lambda x, y: np.power(x - y, 2).sum(axis=1)
cos_dist = lambda x, y: (x*y).sum(axis=1)
abs_dist = lambda x, y: np.abs(x-y).sum(axis=1)
sum_dist = lambda x, y: (x+y).sum(axis=1)

dist_features_train = np.array([
    l2_dist(embeddings_train['question_title_embedding'], embeddings_train['answer_embedding']),
    l2_dist(embeddings_train['question_body_embedding'], embeddings_train['answer_embedding']),
    l2_dist(embeddings_train['question_body_embedding'], embeddings_train['question_title_embedding']),
    
    l2_dist(np.array([x for x in train.question_body_vec.values]), np.array([x for x in train.answer_vec.values])),
    l2_dist(np.array([x for x in train.question_title_vec.values]), np.array([x for x in train.answer_vec.values])),
    l2_dist(np.array([x for x in train.question_body_vec.values]), np.array([x for x in train.question_title_vec.values])),

    cos_dist(embeddings_train['question_title_embedding'], embeddings_train['answer_embedding']),
    cos_dist(embeddings_train['question_body_embedding'], embeddings_train['answer_embedding']),
    cos_dist(embeddings_train['question_body_embedding'], embeddings_train['question_title_embedding']),
    
    cos_dist(np.array([x for x in train.question_body_vec.values]), np.array([x for x in train.answer_vec.values])),
    cos_dist(np.array([x for x in train.question_title_vec.values]), np.array([x for x in train.answer_vec.values])),
    cos_dist(np.array([x for x in train.question_body_vec.values]), np.array([x for x in train.question_title_vec.values])),
    
    abs_dist(embeddings_train['question_title_embedding'], embeddings_train['answer_embedding']),
    abs_dist(embeddings_train['question_body_embedding'], embeddings_train['answer_embedding']),
    abs_dist(embeddings_train['question_body_embedding'], embeddings_train['question_title_embedding']),
    
    abs_dist(np.array([x for x in train.question_body_vec.values]), np.array([x for x in train.answer_vec.values])),
    abs_dist(np.array([x for x in train.question_title_vec.values]), np.array([x for x in train.answer_vec.values])),
    abs_dist(np.array([x for x in train.question_body_vec.values]), np.array([x for x in train.question_title_vec.values])),
  
    sum_dist(embeddings_train['question_title_embedding'], embeddings_train['answer_embedding']),
    sum_dist(embeddings_train['question_body_embedding'], embeddings_train['answer_embedding']),
    sum_dist(embeddings_train['question_body_embedding'], embeddings_train['question_title_embedding']),
    
    sum_dist(np.array([x for x in train.question_body_vec.values]), np.array([x for x in train.answer_vec.values])),
    sum_dist(np.array([x for x in train.question_title_vec.values]), np.array([x for x in train.answer_vec.values])),
    sum_dist(np.array([x for x in train.question_body_vec.values]), np.array([x for x in train.question_title_vec.values])),

    l2_dist(train_question_body_dense, train_answer_dense),
    cos_dist(train_question_body_dense, train_answer_dense),
    abs_dist(train_question_body_dense, train_answer_dense),
    sum_dist(train_question_body_dense, train_answer_dense),
    
    l2_dist(train_question_body_dense, train_question_title_dense),
    cos_dist(train_question_body_dense, train_question_title_dense),
    abs_dist(train_question_body_dense, train_question_title_dense),
    sum_dist(train_question_body_dense, train_question_title_dense),
    
    l2_dist(train_answer_dense, train_question_title_dense),
    cos_dist(train_answer_dense, train_question_title_dense),
    abs_dist(train_answer_dense, train_question_title_dense),
    sum_dist(train_answer_dense, train_question_title_dense),
]).T


dist_features_test = np.array([
    l2_dist(embeddings_test['question_title_embedding'], embeddings_test['answer_embedding']),
    l2_dist(embeddings_test['question_body_embedding'], embeddings_test['answer_embedding']),
    l2_dist(embeddings_test['question_body_embedding'], embeddings_test['question_title_embedding']),
    
    l2_dist(np.array([x for x in test.question_body_vec.values]), np.array([x for x in test.answer_vec.values])),
    l2_dist(np.array([x for x in test.question_title_vec.values]), np.array([x for x in test.answer_vec.values])),
    l2_dist(np.array([x for x in test.question_body_vec.values]), np.array([x for x in test.question_title_vec.values])),
    
    
    cos_dist(embeddings_test['question_title_embedding'], embeddings_test['answer_embedding']),
    cos_dist(embeddings_test['question_body_embedding'], embeddings_test['answer_embedding']),
    cos_dist(embeddings_test['question_body_embedding'], embeddings_test['question_title_embedding']),
    
    cos_dist(np.array([x for x in test.question_body_vec.values]), np.array([x for x in test.answer_vec.values])),
    cos_dist(np.array([x for x in test.question_title_vec.values]), np.array([x for x in test.answer_vec.values])),
    cos_dist(np.array([x for x in test.question_body_vec.values]), np.array([x for x in test.question_title_vec.values])),
   
    
    abs_dist(embeddings_test['question_title_embedding'], embeddings_test['answer_embedding']),
    abs_dist(embeddings_test['question_body_embedding'], embeddings_test['answer_embedding']),
    abs_dist(embeddings_test['question_body_embedding'], embeddings_test['question_title_embedding']),
    
    abs_dist(np.array([x for x in test.question_body_vec.values]), np.array([x for x in test.answer_vec.values])),
    abs_dist(np.array([x for x in test.question_title_vec.values]), np.array([x for x in test.answer_vec.values])),
    abs_dist(np.array([x for x in test.question_body_vec.values]), np.array([x for x in test.question_title_vec.values])),

    
    sum_dist(embeddings_test['question_title_embedding'], embeddings_test['answer_embedding']),
    sum_dist(embeddings_test['question_body_embedding'], embeddings_test['answer_embedding']),
    sum_dist(embeddings_test['question_body_embedding'], embeddings_test['question_title_embedding']),
    
    sum_dist(np.array([x for x in test.question_body_vec.values]), np.array([x for x in test.answer_vec.values])),
    sum_dist(np.array([x for x in test.question_title_vec.values]), np.array([x for x in test.answer_vec.values])),
    sum_dist(np.array([x for x in test.question_body_vec.values]), np.array([x for x in test.question_title_vec.values])),

    
    l2_dist(test_question_body_dense, test_answer_dense),
    cos_dist(test_question_body_dense, test_answer_dense),
    abs_dist(test_question_body_dense, test_answer_dense),
    sum_dist(test_question_body_dense, test_answer_dense),
    
    l2_dist(test_question_body_dense, test_question_title_dense),
    cos_dist(test_question_body_dense, test_question_title_dense),
    abs_dist(test_question_body_dense, test_question_title_dense),
    sum_dist(test_question_body_dense, test_question_title_dense),
    
    l2_dist(test_answer_dense, test_question_title_dense),
    cos_dist(test_answer_dense, test_question_title_dense),
    abs_dist(test_answer_dense, test_question_title_dense),
    sum_dist(test_answer_dense, test_question_title_dense),
]).T
dist_features_train.shape, tf_idf_text_train.shape
for k, item in embeddings_train.items():
    print(len(item))
train_question_body_dense.shape, train_answer_dense.shape
possible_features_train = [
    [item for k, item in embeddings_train.items()],
    [ dist_features_train ],
    [ train_question_body_dense ],
    [ train_answer_dense ],
    [ inputs_feats ],
    [ [x for x in train.question_body_vec.values] ],
    [ [x for x in train.question_title_vec.values] ],
    [ [x for x in train.answer_vec.values] ],
    
    [ tf_idf_text_train ]
    
]
possible_features_test = [
    [item for k, item in embeddings_test.items()],
    [ dist_features_test ],
    [ test_question_body_dense ],
    [ test_answer_dense ],
    [ test_inputs_feats ],
    [ [x for x in test.question_body_vec.values] ],
    [ [x for x in test.question_title_vec.values] ],
    [ [x for x in test.answer_vec.values] ],
    
    [ tf_idf_text_test ]
]

def get_train_test(split=0.8):
    total_len = len(possible_features_train)
    r_idx = random.sample(range(total_len), int(total_len * split))
    
    train = [ train_question_title_dense ]

    test =  [ test_question_title_dense ]

    for i in r_idx:
        train += possible_features_train[i]
        test += possible_features_test[i]
        
    return np.hstack(train),np.hstack(test)

X_train,X_test = get_train_test()
y_train = train[targets].values
X_train.shape, X_test.shape
X_train = pd.DataFrame(data = X_train)
X_train.columns = [str(col) + '_col' for col in X_train.columns]

X_test = pd.DataFrame(data=X_test)
X_test.columns = [str(col) + '_col' for col in X_test.columns]
train[tabular_cols].head()
pd_concat_train = [X_train, train[targets], train[tabular_cols], train[cat_cols], train[cont_cols]]
pd_concat_test = [X_test, test[tabular_cols], test[cat_cols], test[cont_cols]]


final_train_df = pd.concat(pd_concat_train, axis=1)
final_test_df = pd.concat(pd_concat_test, axis=1)
final_train_df.shape, final_test_df.shape
valid_sz = 1000
valid_idx = range(len(final_train_df)-valid_sz, len(final_train_df))
valid_idx
### reload_ext autoreload
### autoreload 2
### matplotlib inline

from fastai import *
from fastai.text import *
from fastai.tabular import *

from pathlib import Path
from typing import *

import torch
import torch.optim as optim

from scipy.special import erfinv
import matplotlib.pyplot as plt
import torch
from torch.utils.data import *
from torch.optim import *
from fastai.tabular import *
import torch.utils.data as Data
from fastai.basics import *
from fastai.callbacks.hooks import *
from tqdm import tqdm_notebook as tqdm
def to_gauss(x): return np.sqrt(2)*erfinv(x)  #from scipy

def normalize(data, exclude=None):
    # if not binary, normalize it
    norm_cols = [n for n, c in data.drop(exclude, 1).items() if len(np.unique(c)) > 2]
    n = data.shape[0]
    for col in norm_cols:
        sorted_idx = data[col].sort_values().index.tolist()# list of sorted index
        uniform = np.linspace(start=-0.99, stop=0.99, num=n) # linsapce
        normal = to_gauss(uniform) # apply gauss to linspace
        normalized_col = pd.Series(index=sorted_idx, data=normal) # sorted idx and normalized space
        data[col] = normalized_col # column receives its corresponding rank
    return data
list(final_train_df[tabular_cols+targets+cat_cols].columns)
exclude = list(final_train_df[tabular_cols+targets+cat_cols].columns)
norm_data_train = normalize(final_train_df, exclude=exclude)

exclude = list(final_test_df[tabular_cols+cat_cols].columns)
norm_data_test = normalize(final_test_df, exclude=exclude)
gc.collect()
tabular_cols+cat_cols
cont_names = list(X_train.columns)+cont_cols
cat_names = tabular_cols+cat_cols
dep_var = targets
procs = [FillMissing, Categorify]

test_tab = TabularList.from_df(norm_data_test, cat_names=cat_names, cont_names=cont_names, procs=procs)

data = (TabularList.from_df(norm_data_train, procs = procs, cont_names=cont_names, cat_names=cat_names)
        .split_by_idx(valid_idx)
        .label_from_df(cols=dep_var)
        .add_test(test_tab)
        .databunch(bs=32))
data.show_batch()
import seaborn as sns
from matplotlib import pyplot as plt
import matplotlib.style as style
style.use('seaborn-poster')
style.use('ggplot')
def _get_ranks(arr: np.ndarray) -> np.ndarray:
    """
        Efficiently calculates the ranks of the data.
        Only sorts once to get the ranked data.
        
        :param arr: A 1D NumPy Array
        :return: A 1D NumPy Array containing the ranks of the data
    """
    temp = arr.argsort()
    ranks = np.empty_like(temp)
    ranks[temp] = np.arange(len(arr))
    return ranks
def spearmans_rho_custom(y_true: np.ndarray, y_pred: np.ndarray) -> np.float64:
    """
        Efficiently calculates the Spearman's Rho correlation using only NumPy
        
        :param y_true: The ground truth labels
        :param y_pred: The predicted labels
    """
    # Get ranked data
    true_rank = _get_ranks(y_true)
    pred_rank = _get_ranks(y_pred)
    
    cost = np.corrcoef(true_rank, pred_rank)[1][0]
    loss = cost.mean()
    loss = loss*(-1)
    return loss
class Regress_Loss(torch.nn.Module):
    
    def __init__(self):
        super(Regress_Loss,self).__init__()
        
    def forward(self,x,y):
        true_rank = _get_ranks(x)
        pred_rank = _get_ranks(y)
        
        cost = np.corrcoef(true_rank, pred_rank)[1][0]
        loss = cost.mean()
        loss = loss*(-1)
        return loss
def myloss(input,target):
    #alpha of 0.5 means half weight goes to first, remaining half split by remaining 15
    x = input
    y = target
    vx = x - torch.mean(x)
    vy = y - torch.mean(y)

    cost = torch.sum(vx * vy) / (torch.sqrt(torch.sum(vx ** 2)) * torch.sqrt(torch.sum(vy ** 2)))
    loss = cost.mean() 
    return loss
class Regress_Loss_1(torch.nn.Module):
    
    def __init__(self):
        super(Regress_Loss_1,self).__init__()
        
    def forward(self,x,y):
        x = input
        y = target
        vx = x - torch.mean(x)
        vy = y - torch.mean(y)
        
        cost = torch.sum(vx * vy) / (torch.sqrt(torch.sum(vx ** 2)) * torch.sqrt(torch.sum(vy ** 2)))
        loss = cost.mean()
        loss = loss*(-1)
        return loss
from fastai.callbacks import *

learn_tfidf = tabular_learner(data, 
                              layers=[64, 128], 
                              ps=[0.30, 0.3], 
                              emb_drop=0.10, use_bn=False,
                              callback_fns=ReduceLROnPlateauCallback)

learn_tfidf.crit = Regress_Loss
# learn_tfidf.crit = Regress_Loss_1
learn_tfidf.lr_find()
learn_tfidf.recorder.plot(suggestion=True)
lr = 5e-2
learn_tfidf.fit_one_cycle(10, max_lr=lr,  pct_start=0.3, wd = 1.)
learn_tfidf.recorder.plot_losses()
pred_test_tfidf, lbl_test_tfidf = learn_tfidf.get_preds(ds_type=DatasetType.Test)
pred_test_tfidf = np.clip(pred_test_tfidf, 0.00001, 0.999999)
pred_test_tfidf.shape
pred_test_tfidf
learn_tfidf_2 = tabular_learner(data, 
                              layers=[512, 256], 
                              ps=[0.30, 0.3], 
                              emb_drop=0.10, use_bn=False,
                              callback_fns=ReduceLROnPlateauCallback)
#learn_tfidf.crit = Regress_Loss
learn_tfidf_2.crit = Regress_Loss_1
learn_tfidf_2.lr_find()
learn_tfidf_2.recorder.plot(suggestion=True)
lr = 5e-2
learn_tfidf_2.fit_one_cycle(10, max_lr=lr,  pct_start=0.3, wd = 1.)
learn_tfidf_2.recorder.plot_losses()
pred_test_tfidf_2, lbl_test_tfidf_2 = learn_tfidf_2.get_preds(ds_type=DatasetType.Test)
pred_test_tfidf_2 = np.clip(pred_test_tfidf_2, 0.00001, 0.999999)
pred_test_tfidf_2.shape
pred_test_tfidf_2
learn_tfidf_3 = tabular_learner(data, 
                              layers=[512], 
                              ps=[0.30], 
                              emb_drop=0.10, use_bn=False,
                              callback_fns=ReduceLROnPlateauCallback)
#learn_tfidf.crit = Regress_Loss
learn_tfidf_3.crit = Regress_Loss_1
learn_tfidf_3.lr_find()
learn_tfidf_3.recorder.plot(suggestion=True)
lr = 3e-2
learn_tfidf_3.fit_one_cycle(10, max_lr=lr,  pct_start=0.3, wd = 1.)
learn_tfidf_3.recorder.plot_losses()
pred_test_tfidf_3, lbl_test_tfidf_3 = learn_tfidf_3.get_preds(ds_type=DatasetType.Test)
pred_test_tfidf_3 = np.clip(pred_test_tfidf_3, 0.00001, 0.999999)
pred_test_tfidf_3.shape
pred_test_tfidf_3
test_preds_fastai = (pred_test_tfidf+ pred_test_tfidf_2 + pred_test_tfidf_3)/3
test_preds_fastai
# submission = pd.read_csv(INPUT_PATH+'google-quest-challenge/sample_submission.csv')
# submission[targets] = test_preds_fastai
# submission.head(20)
# submission.to_csv("submission.csv", index = False)
data_init = data
x = int(len(norm_data_train)*.9)
train_df = norm_data_train.iloc[:x]
test_df = norm_data_train.iloc[x:]
skf = KFold(n_splits=10, shuffle=True, random_state=42)
for train_index, test_index in skf.split(norm_data_train.index):
    
    X_train, X_test = norm_data_train.iloc[train_index], norm_data_train.iloc[test_index]
    y_train, y_test = norm_data_train[targets].iloc[train_index], norm_data_train[targets].iloc[test_index]    
X_train.head()

val_pct = []
test_pct = []


for train_index, val_index in skf.split(train_df.index, train_df[dep_var]):
    data_fold = (TabularList.from_df(train_df, cat_names=cat_names.copy(),
                                  cont_names=cont_names.copy(), procs=procs,
                                  processor=data_init.processor) # Very important
              .split_by_idxs(train_index, val_index)
              .label_from_df(cols=dep_var)
              .databunch())
    
    data_test = (TabularList.from_df(test_df, cat_names=cat_names.copy(),
                                  cont_names=cont_names.copy(), procs=procs,
                                  processor=data_init.processor) # Very important
              .split_none()
              .label_from_df(cols=dep_var))
    
    data_test.valid = data_test.train
    data_test = data_test.databunch()
    
    learn_f = tabular_learner(data_fold, layers=[200, 100], 
                        ps=[0.3, 0.3], emb_drop=0.3, use_bn=False)
    
    learn_f.fit_one_cycle(10, max_lr=1e-3,  pct_start=0.5, wd = 1, callbacks = [SaveModelCallback(learn_f)])
    
    val = learn_f.validate()
    
    learn_f.data.valid_dl = data_test.valid_dl
    
    test = learn_f.validate()
    
    val_pct.append(val)
    test_pct.append(test)
print(f'Validation\nmean: {np.mean(val_pct)}\nstd: {np.std(val_pct)}')
print(f'Test\nmean: {np.mean(test_pct)}\nstd: {np.std(test_pct)}')
data = (TabularList.from_df(norm_data_train, procs = procs, cont_names=cont_names, cat_names=cat_names)
        .split_by_idx(valid_idx)
        .label_from_df(cols=dep_var)
        .add_test(test_tab)
        .databunch(bs=32))
learn_f = tabular_learner(data, layers=[200, 100], 
                        ps=[0.3, 0.3], emb_drop=0.3, use_bn=False)
learn_f.load('bestmodel')
learn_f.fit_one_cycle(1, max_lr=1e-3,  pct_start=0.5, wd = 1)
pred_test_tfidf, lbl_test_tfidf = learn_f.get_preds(ds_type=DatasetType.Test)
pred_test_tfidf = np.clip(pred_test_tfidf, 0.00001, 0.999999)
pred_test_tfidf.shape
pred_test_tfidf
# submission = pd.read_csv(INPUT_PATH+'google-quest-challenge/sample_submission.csv')
# submission[targets] = pred_test_tfidf
# submission.head(20)
# submission.to_csv("submission.csv", index = False)
import os
import re
import gc
import pickle  
import random
import keras

import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow_hub as hub
import keras.backend as K

from keras.models import Model
from keras.layers import Dense, Input, Dropout, Lambda
from keras.optimizers import Adam
from keras.callbacks import Callback
from scipy.stats import spearmanr, rankdata
from os.path import join as path_join
from numpy.random import seed
from urllib.parse import urlparse
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import KFold
from sklearn.linear_model import MultiTaskElasticNet
from sklearn.multioutput import MultiOutputRegressor
from lightgbm import LGBMRegressor

seed(42)
tf.random.set_seed(42)
random.seed(42)
# Categorical boolean mask
categorical_feature_mask = norm_data_train.dtypes=='object'
# filter categorical columns using mask and turn it into a list
categorical_cols = norm_data_train.columns[categorical_feature_mask].tolist()
categorical_cols
type(norm_data_train.question_user_name[0])
# Categorical boolean mask
categorical_feature_mask = norm_data_train.dtypes=='object'
# filter categorical columns using mask and turn it into a list
categorical_cols = norm_data_train.columns[categorical_feature_mask].tolist()

# import labelencoder
from sklearn.preprocessing import LabelEncoder
# instantiate labelencoder object
le = LabelEncoder()

# apply le on categorical feature columns
norm_data_train[categorical_cols] = norm_data_train[categorical_cols].apply(lambda col: le.fit_transform(col))
norm_data_train[categorical_cols].head(10)

# Categorical boolean mask
categorical_feature_mask = norm_data_test.dtypes=='object'
# filter categorical columns using mask and turn it into a list
categorical_cols = norm_data_test.columns[categorical_feature_mask].tolist()

# import labelencoder
from sklearn.preprocessing import LabelEncoder
# instantiate labelencoder object
le = LabelEncoder()

# apply le on categorical feature columns
norm_data_test[categorical_cols] = norm_data_test[categorical_cols].apply(lambda col: le.fit_transform(col))
norm_data_test[categorical_cols].head(10)
X_train = norm_data_train.loc[:, ~norm_data_train.columns.isin(targets)].values
y_train = norm_data_train[targets].values
X_test = norm_data_test.values
X_train.shape, y_train.shape, X_test.shape
norm_data_train.head()
# Compatible with tensorflow backend
class SpearmanRhoCallback(Callback):
    def __init__(self, training_data, validation_data, patience, model_name):
        self.x = training_data[0]
        self.y = training_data[1]
        self.x_val = validation_data[0]
        self.y_val = validation_data[1]
        
        self.patience = patience
        self.value = -1
        self.bad_epochs = 0
        self.model_name = model_name

    def on_train_begin(self, logs={}):
        return

    def on_train_end(self, logs={}):
        return

    def on_epoch_begin(self, epoch, logs={}):
        return

    def on_epoch_end(self, epoch, logs={}):
        y_pred_val = self.model.predict(self.x_val)
        rho_val = np.mean([spearmanr(self.y_val[:, ind], y_pred_val[:, ind] + np.random.normal(0, 1e-7, y_pred_val.shape[0])).correlation for ind in range(y_pred_val.shape[1])])
        if rho_val >= self.value:
            self.value = rho_val
        else:
            self.bad_epochs += 1
        if self.bad_epochs >= self.patience:
            print("Epoch %05d: early stopping Threshold" % epoch)
            self.model.stop_training = True
            #self.model.save_weights(self.model_name)
        print('\rval_spearman-rho: %s' % (str(round(rho_val, 4))), end=100*' '+'\n')
        return rho_val

    def on_batch_begin(self, batch, logs={}):
        return

    def on_batch_end(self, batch, logs={}):
        return
from keras.losses import *
def bce(t,p):
    return binary_crossentropy(t,p)

def custom_loss(true,pred):
    bce = binary_crossentropy(true,pred)
    return bce + logcosh(true,pred)

def swish(x):
    return K.sigmoid(x) * x

def relu1(x):
    return keras.activations.relu(x, alpha=0.0, max_value=1., threshold=0.0)

def create_model1():
    input1 = Input(shape=(X_train.shape[1],))
    x = Dense(64, activation='elu',kernel_initializer='lecun_normal')(input1)
    x = Dense(128, activation='elu',kernel_initializer='lecun_normal')(x)
    x = Dropout(0.4)(x)
    output = Dense(len(targets),activation='sigmoid',name='output')(x)
    model = Model(inputs=input1, outputs=output)
    optimizer = Adam(lr=5e-4,clipnorm=1.4)
    model.compile(optimizer=optimizer, loss=custom_loss, metrics=[bce,logcosh])
    return model


def create_model2():
    input1 = Input(shape=(X_train.shape[1],))
    x = Dense(512, activation='elu',kernel_initializer='lecun_normal')(input1)
    x = Dense(256, activation='elu',
              kernel_initializer='lecun_normal', 
              #kernel_regularizer=keras.regularizers.l2(0.01)
             )(x)
    output = Dense(len(targets),activation='sigmoid',name='output')(x)
    model = Model(inputs=input1, outputs=output)
    optimizer = Adam(lr=5e-4,clipnorm=1.4)
    model.compile(optimizer=optimizer, loss=custom_loss, metrics=[bce,logcosh])
    return model


def create_model3():
    input1 = Input(shape=(X_train.shape[1],))
    x = Dense(200, activation='selu',kernel_initializer='lecun_normal')(input1)
    x = Dense(512, activation='selu',
              kernel_initializer='lecun_normal', 
              )(x)
    x = Dropout(0.4)(x)
    output = Dense(len(targets),activation='sigmoid',name='output')(x)
    model = Model(inputs=input1, outputs=output)
    optimizer = Adam(lr=5e-4,clipnorm=1.4)
    model.compile(optimizer=optimizer, loss=custom_loss, metrics=[bce,logcosh])
    return model

def create_model4():
    input1 = Input(shape=(X_train.shape[1],))
    x = Dense(512, activation='elu',
              kernel_initializer='lecun_normal', 
              )(input1)
    x = Dropout(0.2)(x)
    output = Dense(len(targets),activation='sigmoid',name='output')(x)
    model = Model(inputs=input1, outputs=output)
    optimizer = Adam(lr=5e-4,clipnorm=1.4)
    model.compile(optimizer=optimizer, loss=custom_loss, metrics=[bce,logcosh])
    return model

def create_model5():
    input1 = Input(shape=(X_train.shape[1],))
    x = Dense(4096, activation='selu',kernel_initializer='lecun_normal')(input1)
    x = Dense(512, activation='selu',
              kernel_initializer='lecun_normal', 
              kernel_regularizer=keras.regularizers.l2(0.01)
             )(x)
    output = Dense(len(targets),activation='sigmoid',name='output')(x)
    model = Model(inputs=input1, outputs=output)
    optimizer = Adam(lr=5e-4,clipnorm=1.4)
    model.compile(optimizer=optimizer, loss=custom_loss, metrics=[bce,logcosh])
    return model
all_predictions = []

kf = KFold(n_splits=5, random_state=42, shuffle=True)
for ind, (tr, val) in enumerate(kf.split(X_train)):
    X_tr = X_train[tr]
    y_tr = y_train[tr]
    X_vl = X_train[val]
    y_vl = y_train[val]
    
    model = create_model1()
    model.fit(
        X_tr, y_tr, epochs=100, batch_size=32, validation_data=(X_vl, y_vl), verbose=True, 
        callbacks=[SpearmanRhoCallback(training_data=(X_tr, y_tr), validation_data=(X_vl, y_vl),
                                       patience=5, model_name=f'best_model_batch{ind}.h5')]
    )
    all_predictions.append(model.predict(X_test))
    
model = create_model1()
model.fit(X_train, y_train, epochs=33, batch_size=32, verbose=False)
all_predictions.append(model.predict(X_test))
    
kf = KFold(n_splits=5, random_state=2019, shuffle=True)
for ind, (tr, val) in enumerate(kf.split(X_train)):
    X_tr = X_train[tr]
    y_tr = y_train[tr]
    X_vl = X_train[val]
    y_vl = y_train[val]
    
    model = create_model2()
    model.fit(
        X_tr, y_tr, epochs=100, batch_size=32, validation_data=(X_vl, y_vl), verbose=True, 
        callbacks=[SpearmanRhoCallback(training_data=(X_tr, y_tr), validation_data=(X_vl, y_vl),
                                       patience=5, model_name=f'best_model_batch{ind}.h5')]
    )
    all_predictions.append(model.predict(X_test))
    
model = create_model2()
model.fit(X_train, y_train, epochs=33, batch_size=32, verbose=False)
all_predictions.append(model.predict(X_test))

kf = KFold(n_splits=5, random_state=2019, shuffle=True)
for ind, (tr, val) in enumerate(kf.split(X_train)):
    X_tr = X_train[tr]
    y_tr = y_train[tr]
    X_vl = X_train[val]
    y_vl = y_train[val]
    
    model = create_model3()
    model.fit(
        X_tr, y_tr, epochs=100, batch_size=32, validation_data=(X_vl, y_vl), verbose=True, 
        callbacks=[SpearmanRhoCallback(training_data=(X_tr, y_tr), validation_data=(X_vl, y_vl),
                                       patience=5, model_name=f'best_model_batch{ind}.h5')]
    )
    all_predictions.append(model.predict(X_test))
    
model = create_model3()
model.fit(X_train, y_train, epochs=33, batch_size=32, verbose=False)
all_predictions.append(model.predict(X_test))

kf = KFold(n_splits=5, random_state=2019, shuffle=True)
for ind, (tr, val) in enumerate(kf.split(X_train)):
    X_tr = X_train[tr]
    y_tr = y_train[tr]
    X_vl = X_train[val]
    y_vl = y_train[val]
    
    model = create_model4()
    model.fit(
        X_tr, y_tr, epochs=100, batch_size=32, validation_data=(X_vl, y_vl), verbose=True, 
        callbacks=[SpearmanRhoCallback(training_data=(X_tr, y_tr), validation_data=(X_vl, y_vl),
                                       patience=5, model_name=f'best_model_batch{ind}.h5')]
    )
    all_predictions.append(model.predict(X_test))
    
model = create_model4()
model.fit(X_train, y_train, epochs=33, batch_size=32, verbose=False)
all_predictions.append(model.predict(X_test))

kf = KFold(n_splits=5, random_state=2019, shuffle=True)
for ind, (tr, val) in enumerate(kf.split(X_train)):
    X_tr = X_train[tr]
    y_tr = y_train[tr]
    X_vl = X_train[val]
    y_vl = y_train[val]
    
    model = create_model5()
    model.fit(
        X_tr, y_tr, epochs=100, batch_size=32, validation_data=(X_vl, y_vl), verbose=True, 
        callbacks=[SpearmanRhoCallback(training_data=(X_tr, y_tr), validation_data=(X_vl, y_vl),
                                       patience=5, model_name=f'best_model_batch{ind}.h5')]
    )
    all_predictions.append(model.predict(X_test))
    
model = create_model5()
model.fit(X_train, y_train, epochs=33, batch_size=32, verbose=False)
all_predictions.append(model.predict(X_test))
test_preds = np.array([np.array([rankdata(c) for c in p.T]).T for p in all_predictions]).mean(axis=0)
max_val = test_preds.max() + 1
test_preds = test_preds/max_val + 1e-12
test_preds
test_preds = torch.from_numpy(test_preds) * 0.50 + pred_test_tfidf * 0.50
test_preds
submission = pd.read_csv(INPUT_PATH+'google-quest-challenge/sample_submission.csv')
submission[targets] = test_preds
submission.head(20)
submission.to_csv("submission.csv", index = False)
import torch
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler, DataLoader
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
import os
import math
from torch.nn import BCEWithLogitsLoss
from torch import nn
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from tqdm.notebook import tqdm
from tqdm import trange
type(X_train), type(y_train), type(X_test)
# train valid split
X_train, X_valid, y_train, y_valid = train_test_split(X_train, y_train, test_size=0.2, random_state=42)
X_train = torch.tensor(X_train)
y_train = torch.tensor(y_train, dtype=torch.float32)

X_val = torch.tensor(X_valid)
y_val = torch.tensor(y_valid, dtype=torch.float32)

X_test = torch.tensor(X_test)
batch_size = 32

train_data = TensorDataset(X_train, y_train)
train_sampler = RandomSampler(train_data)
train_dataloader = DataLoader(train_data, sampler = train_sampler, batch_size=batch_size)

val_data = TensorDataset(X_val, y_val)
val_sampler = SequentialSampler(val_data)
val_dataloader = DataLoader(val_data, sampler = val_sampler, batch_size = batch_size)
class py_net(torch.nn.Module):
    
    def __init__(self):
        super(py_net, self).__init__()
        self.fc1 = nn.Linear(X_train.shape[1], 200)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(0.25)
        self.fc2 = nn.Linear(200, 100)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(0.25)
        self.fc3 = nn.Linear(100, 30)
        self.prelu3 = nn.PReLU()
        self.out = nn.Linear(30, len(targets))
        self.out_act = nn.Sigmoid()
    
    def forward(self, input_):
        a1 = self.fc1(input_)
        h1 = self.relu1(a1)
        dout1 = self.dropout1(h1)
        a2 = self.fc2(dout1)
        h2 = self.relu2(a2)
        dout2 = self.dropout2(h2)
        a3 = self.fc3(dout2)
        h3 = self.prelu3(a3)
        a4 = self.out(h3)
        y = self.out_act(a4)
        return y
    
    def predict(self, x):
        pred = self.forward(x)
        return torch.tensor(pred)          
    
model = py_net()
model.cuda()
criterion = nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1)
from scipy.stats import spearmanr
def train(model, num_epochs,\
          optimizer,\
          train_dataloader, valid_dataloader,\
          train_loss_set=[], valid_loss_set = [],\
          lowest_eval_loss=None, start_epoch=0,\
          device="cuda"
          ):
    
    
    crit_function = nn.BCEWithLogitsLoss()
    model.to(device)
    
    for i in trange(num_epochs, desc = 'Epoch'):
        actual_epoch = start_epoch + i
        
        model.train()
        
        tr_loss = 0
        
        num_train_samples = 0
        
        t = tqdm(total = len(train_data), desc = 'Training:', position=0)
        
        for step, batch in enumerate(train_dataloader):
            batch = tuple(t.to(device) for t in batch)
            X_batch, y_batch = batch
            
            optimizer.zero_grad()
            
            y_pred = model(X_batch.float())
            
            loss = crit_function(y_pred, y_batch)
            
            tr_loss += loss.item()
            
            num_train_samples = y_batch.size(0)
            
            loss.backward()
            
            optimizer.step()
            
            t.update(n = X_batch.shape[0])
            
        t.close()
        
        epoch_train_loss = tr_loss/num_train_samples
        train_loss_set.append(epoch_train_loss)

        print("Train loss: {}".format(epoch_train_loss))
        
        # validation
        
        model.eval()
        eval_loss = 0
        num_eval_samples = 0
        
        v_preds = []
        v_labels = []
        
        # Evaluate data for one epoch
        t = tqdm(total=len(val_data), desc="Validating: ", position=0)
        for batch in val_dataloader:
            # Add batch to GPU
            batch = tuple(t.to(device) for t in batch)
            # Unpack the inputs from our dataloader
            X_batch, y_batch = batch
            # Telling the model not to compute or store gradients,
            # saving memory and speeding up validation
            with torch.no_grad():
                # Forward pass, calculate validation loss
                preds = model(X_batch.float())
                loss = crit_function(preds, y_batch)
                v_labels.append(y_batch.cpu().numpy())
                v_preds.append(preds.cpu().numpy())
                # store valid loss
                eval_loss += loss.item()
                num_eval_samples += y_batch.size(0)
            t.update(n=y_batch.shape[0])
        t.close()
        
        v_labels = np.vstack(v_labels)
        v_preds = np.vstack(v_preds)
        print(v_labels.shape)
        print(v_preds.shape)
        rho_val = np.mean([spearmanr(v_labels[:, ind] + np.random.normal(0, 1e-7, v_preds.shape[0]),
                                            v_preds[:, ind] + np.random.normal(0, 1e-7, v_preds.shape[0])).correlation for ind in range(v_preds.shape[1])]
                                )
        epoch_eval_loss = eval_loss/num_eval_samples
        valid_loss_set.append(epoch_eval_loss)

        print("Epoch #{}, training BCE loss: {}, validation BCE loss: ~{}, validation spearmanr: {}"\
                .format(0, epoch_train_loss, epoch_eval_loss, rho_val))

        if lowest_eval_loss == None:
            lowest_eval_loss = epoch_eval_loss
           
        else:
            if epoch_eval_loss < lowest_eval_loss:
                lowest_eval_loss = epoch_eval_loss
         
        print("\n")

    return model, train_loss_set, valid_loss_set
model, train_loss_set, valid_loss_set = train(model=model,\
                                              num_epochs = 100,
                                              optimizer = optimizer,
                                              train_dataloader = train_dataloader,
                                              valid_dataloader = val_dataloader,
                                              device='cuda'
                                              )
