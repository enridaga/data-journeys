
import numpy as np
import pandas as pd
import os
import sys
import random
import keras
import tensorflow as tf
import json
import gc
from tqdm import tqdm

sys.path.insert(0, '../input/pretrained-bert-including-scripts/master/bert-master')
### cp -r '../input/kerasbert/keras_bert' '/kaggle/working'
BERT_PRETRAINED_DIR = '../input/pretrained-bert-including-scripts/uncased_l-12_h-768_a-12/uncased_L-12_H-768_A-12'
print('***** BERT pretrained directory: {} *****'.format(BERT_PRETRAINED_DIR))
import tokenization  #Actually keras_bert contains tokenization part, here just for convenience
from tensorflow.keras.models import Model       # Keras is the new high level API for TensorFlow
from tensorflow.python.ops.rnn_cell import DropoutWrapper
from tensorflow.python.ops import variable_scope as vs
from tensorflow.keras.layers import GRU, Bidirectional
from tensorflow.keras.layers import Layer
from tensorflow.keras.layers import Dense
from keras_bert.keras_bert.bert import get_model
from keras_bert.keras_bert.loader import load_trained_model_from_checkpoint
from keras.optimizers import Adam
adam = Adam(lr=2e-5,decay=0.01)
print('begin_build')
config_file = os.path.join(BERT_PRETRAINED_DIR, 'bert_config.json')
checkpoint_file = os.path.join(BERT_PRETRAINED_DIR, 'bert_model.ckpt')
sys.path.insert(0, '../input/bert-and-bidaf/model_bidaf.h5')
### ls -l ../input/bert-and-bidaf
cont_len = 512
ques_len = 126

cont_model = load_trained_model_from_checkpoint(config_file, checkpoint_file, training=False,seq_len=cont_len)
ques_model = load_trained_model_from_checkpoint(config_file, checkpoint_file, training=False,seq_len=ques_len)

cont_model.trainable = False
ques_model.trainable = False

dict_path = os.path.join(BERT_PRETRAINED_DIR, 'vocab.txt')
tokenizer = tokenization.FullTokenizer(vocab_file=dict_path, do_lower_case=True)
def get_masks(tokens, max_seq_length):
    """Mask for padding"""
    if len(tokens)>max_seq_length:
        print("Token length more than max seq length!")
        return max_seq_length*[1]
    return [1]*len(tokens) + [0] * (max_seq_length - len(tokens))


def get_segments(tokens, max_seq_length):
    """Segments: 0 for the first sequence, 1 for the second"""
    if len(tokens)>max_seq_length:
        #raise IndexError("Token length more than max seq length!")
        return max_seq_length*[1]
    segments = []
    current_segment_id = 0
    for token in tokens:
        segments.append(current_segment_id)
        if token == "[SEP]":
            current_segment_id = 1
    return segments + [0] * (max_seq_length - len(tokens))


def get_ids(tokens, tokenizer, max_seq_length):
    """Token ids from Tokenizer vocab"""
    token_ids = tokenizer.convert_tokens_to_ids(tokens)
    if len(tokens)>max_seq_length:
        return token_ids[:max_seq_length]
    input_ids = token_ids + [0] * (max_seq_length-len(token_ids))
    return input_ids
import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
path = '/kaggle/input/tensorflow2-question-answering/'
train_path = 'simplified-nq-train.jsonl'
test_path = 'simplified-nq-test.jsonl'
sample_submission_path = 'sample_submission.csv'

def read_data(path, sample = True, chunksize = 30000):
    if sample == True:
        df = []
        with open(path, 'rt') as reader:
            for i in range(chunksize):
                df.append(json.loads(reader.readline()))
        df = pd.DataFrame(df)
        print('Our sampled dataset have {} rows and {} columns'.format(df.shape[0], df.shape[1]))
    else:
        df = pd.read_json(path, orient = 'records', lines = True)
        print('Our dataset have {} rows and {} columns'.format(df.shape[0], df.shape[1]))
        gc.collect()
    return df

train = read_data(path+train_path, sample = True)
test = read_data(path+test_path, sample = False)
train.head()
class QA_Layer_partA(Layer):
    def __init__(self,**kwargs):
        super(QA_Layer_partA, self).__init__(**kwargs)
        self.supports_masking = False
        init_op = tf.global_variables_initializer
    def build(self, input_shape):
        #inputs: [q, qmask, c, cmask] in that order
        #input shapes: q.shape = (batch,seq_len,emb_size) = (batch, ques_len, 768)
        print(input_shape)
        self.S_W = self.add_weight(name = 'S_W', shape = (768*3, ), initializer='uniform', trainable=True)
        init_op = tf.global_variables_initializer
        super(QA_Layer_partA, self).build(input_shape)
        
    def call(self, ccmqqm):
        #input must be a list of four list(ccmqqm):
        #cont_embs, cont_mask, ques_embs, ques_mask     
        c = ccmqqm[0]
        c_mask = ccmqqm[1]
        q = ccmqqm[2]
        q_mask = ccmqqm[3]
        
        # Calculating similarity matrix
        c_expand = tf.expand_dims(c,2)  #[batch,N,1,2h] ; 2h = bert emb size
        q_expand = tf.expand_dims(q,1)  #[batch,1,M,2h]
        c_pointWise_q = c_expand * q_expand  #[batch,N,M,2h]

        c_input = tf.tile(c_expand, [1, 1, tf.shape(q)[1], 1])
        q_input = tf.tile(q_expand, [1, tf.shape(c)[1], 1, 1])

        concat_input = tf.concat([c_input, q_input, c_pointWise_q], -1) # [batch,N,M,6h]
        print("concat_in", concat_input.shape)
        similarity=tf.reduce_sum(concat_input * self.S_W, axis=-1)  #[batch,N,M]
        print("similarity", similarity.shape)
        # Calculating context to question attention
        similarity_mask = tf.expand_dims(q_mask, 1) # shape (batch_size, 1, M)
        print("sim mask", similarity_mask.shape)
        exp_mask_c2q = (1 - tf.cast(similarity_mask, 'float')) * (-1e30) # -large where there's padding, 0 elsewhere
        print("exp mask", exp_mask_c2q.shape)
        masked_logits_c2q = tf.add(similarity, exp_mask_c2q) # where there's padding, set logits to -large
        c2q_dist = tf.nn.softmax(masked_logits_c2q, 1)  # dim = 1
                
        # Use attention distribution to take weighted sum of values
        c2q = tf.matmul(c2q_dist, q) # shape (batch_size, N, vec_size)
        
        # Calculating question to context attention c_dash
        S_max = tf.reduce_max(similarity, axis=2) # shape (batch, N)

        exp_mask_S = (1 - tf.cast(c_mask, 'float')) * (-1e30) # -large where there's padding, 0 elsewhere
        masked_logits_S = tf.add(S_max, exp_mask_S) # where there's padding, set logits to -large
        c_dash_dist = tf.nn.softmax(masked_logits_S, 1)  # dim = 1
        
        c_dash_dist_expand = tf.expand_dims(c_dash_dist, 1) # shape (batch, 1, N)
        c_dash = tf.matmul(c_dash_dist_expand, c) # shape (batch_size, 1, vec_size)
        
        c_c2q = c * c2q # shape (batch, N, vec_size)
        print("c_c2q", c_c2q.shape)
        c_c_dash = c * c_dash # shape (batch, N, vec_size)
        print("cdash", c_c_dash.shape)
        # concatenate the output
        output = tf.concat([c2q, c_c2q, c_c_dash], axis=2) # (batch_size, N, vec_size * 3)

        # Apply dropout
        attn_output = tf.nn.dropout(output, 0.9)

        blended_reps = tf.concat([c, attn_output], axis=2) #attn out has len same as c_mask
        print("blended", blended_reps.shape)
        
        return blended_reps
        
class QA_Layer_partB(Layer):
    def __init__(self,**kwargs):
        super(QA_Layer_partB, self).__init__(**kwargs)
        self.supports_masking = False
    def build(self, input_shape):
        #inputs: [logits, cmask] in that order
        #input shapes: q.shape = (batch,seq_len,emb_size) = (batch, ques_len, 768)
        #input shapes: q.shape = (batch,seq_len,emb_size) = (batch, cont_len, 768)
        print(input_shape)
        super(QA_Layer_partB, self).build(input_shape)
    def call(self, lcm):
        #input must be a list of two list(logits, cmask):
        logits_start = lcm[0] 
        c_mask = lcm[1]
        logits_start = tf.squeeze(logits_start, axis=[2]) # shape (batch_size, seq_len)
        exp_mask_start = (1 - tf.cast(c_mask, 'float')) * (-1e30) # -large where there's padding, 0 elsewhere
        masked_logits_start = tf.add(logits_start, exp_mask_start) # where there's padding, set logits to -large
        start_prob = tf.nn.softmax(masked_logits_start, 1)  # dim = 1
        return start_prob
#Model for short sentences.
#Xplainin: So, this is a quite simplified version of: 
#https://github.com/priya-dwivedi/cs224n-Squad-Project/blob/master/code/modules.py
#Priya explains that it's better to make bert trainable, add ques and context together and feed as such.
#Here we don't. But again, we are using bert embs, which is hopefully better than their glove embs. We'll see.

cont_embs = tf.keras.layers.Input(shape=(cont_len, 768, ), dtype=tf.float32,
                                       name="cont_embs")
cont_masks = tf.keras.layers.Input(shape=(cont_len,), dtype=tf.int32,
                                   name="cont_masks")
ques_embs = tf.keras.layers.Input(shape=(ques_len, 768, ), dtype=tf.float32,
                                       name="ques_embs")
ques_masks = tf.keras.layers.Input(shape=(ques_len,), dtype=tf.int32,
                                   name="ques_masks")

bidaf_model_partA = QA_Layer_partA()
bidaf_startlogits = QA_Layer_partB()
bidaf_endlogits = QA_Layer_partB()
bidaf_presentlogits = QA_Layer_partB()

blended_reps = bidaf_model_partA([cont_embs, cont_masks, ques_embs, ques_masks])

blended_reps_final = Bidirectional(GRU(128, kernel_initializer = 'glorot_uniform', return_sequences = True, dropout = 0.1))(blended_reps)
blended_reps_final = Bidirectional(GRU(128, kernel_initializer = 'glorot_uniform', return_sequences = True, dropout = 0.1))(blended_reps_final)

logits_start = GRU(1, kernel_initializer = 'glorot_uniform', return_sequences = True, dropout = 0.1)(blended_reps_final)
logits_end = GRU(1, kernel_initializer = 'glorot_uniform', return_sequences = True, dropout = 0.1)(blended_reps_final)
logits_present = GRU(1, kernel_initializer = 'glorot_uniform', return_sequences = True, dropout = 0.1)(blended_reps_final)

start_prob = bidaf_startlogits([logits_start, cont_masks])
end_prob = bidaf_endlogits([logits_end, cont_masks])
present_prob = bidaf_presentlogits([logits_present, cont_masks])

list_of_inputs = [cont_embs, cont_masks, ques_embs, ques_masks]
dist_probs = [start_prob, end_prob, present_prob]

model_bidaf = Model(inputs=list_of_inputs, outputs=dist_probs)

print(model_bidaf.summary())

model_bidaf.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
from keras.models import Model

cont_embs_output  = cont_model.layers[-6].output
cont_bert = Model(inputs=cont_model.input, outputs=cont_embs_output)
cont_bert.compile(loss='binary_crossentropy', optimizer=adam)
cont_bert.summary()

ques_embs_output  = ques_model.layers[-6].output
ques_bert = Model(inputs=ques_model.input, outputs=ques_embs_output)
ques_bert.compile(loss='binary_crossentropy', optimizer=adam)
ques_bert.summary()
model_bidaf.load_weights('../input/bert-and-bidaf/model_bidaf.h5') #works only with cpu
#target present is an auxilliary target. not used for predictions.

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    model_bidaf.load_weights('../input/bert-and-bidaf/model_bidaf.h5')
    
    for i_main in range(15,35):
        
        sample_weight = []

        row = train.iloc[i_main]

        if i_main == 3:
            print('row no ', i_main)
            print('annotation short ans', row['annotations'][0]['short_answers'])
        document_text = row['document_text'].split()
        question_text = row['question_text']

        ques_ans_tokens = tokenizer.tokenize(question_text)

        ques_ids = np.asarray([get_ids(ques_ans_tokens, tokenizer, ques_len)])
        ques_masks = np.asarray([get_masks(ques_ans_tokens, ques_len)])

        batch_of_cont_embs = []
        batch_of_cont_masks = []
        batch_of_ques_embs = []
        batch_of_ques_masks = []

        batch_of_target_st = []
        batch_of_target_en = []
        batch_of_target_pr = []

        for candidate_no, long_answer_candidate in enumerate(row['long_answer_candidates']):

            target_start = [0] * cont_len
            target_end = [0] * cont_len
            target_present = [0] * cont_len

            long_ans_start_tok = long_answer_candidate['start_token']
            long_ans_end_tok = long_answer_candidate['end_token']
            long_sentence = " ".join(document_text[long_ans_start_tok:long_ans_end_tok])

            if long_ans_start_tok == row['annotations'][0]['long_answer']['start_token'] and \
                len(row['annotations'][0]['short_answers']) > 0:

                #print("this is correct long answer")

                short_answer_start_token = row['annotations'][0]['short_answers'][0]['start_token']
                short_answer_end_token = row['annotations'][0]['short_answers'][0]['end_token']
                short_start_idx = short_answer_start_token-long_ans_start_tok
                short_end_idx = short_answer_end_token-long_ans_start_tok

                if short_end_idx < cont_len:
                    target_start[short_start_idx] = 1
                    target_end[short_end_idx] = 1
                    sample_weight.append(900)
                    for i in range(short_start_idx,short_end_idx):
                        target_present[i] = 1
                else:
                    print("short answer beyond maximum len")
                    sample_weight.append(1)
            else:
                sample_weight.append(1)

            long_ans_tokens = tokenizer.tokenize(long_sentence)
            long_ids = np.asarray([get_ids(long_ans_tokens, tokenizer, cont_len)])
            cont_masks = np.asarray([get_masks(long_ans_tokens, cont_len)])

            cont_embs = cont_bert.predict([long_ids,cont_masks],verbose=1,batch_size=1)
            ques_embs = ques_bert.predict([ques_ids,ques_masks],verbose=1,batch_size=1)

            batch_of_cont_embs.append(np.squeeze(cont_embs))
            batch_of_cont_masks.append(np.squeeze(cont_masks))
            batch_of_ques_embs.append(np.squeeze(ques_embs))
            batch_of_ques_masks.append(np.squeeze(ques_masks))

            batch_of_target_st.append(target_start)
            batch_of_target_en.append(target_end)
            batch_of_target_pr.append(target_present)

            #we will train after every line due to the massive size of embs.
        
        sample_weight = np.asarray(sample_weight)
        
        xcont_embs = np.asarray(batch_of_cont_embs)
        xcont_masks = np.asarray(batch_of_cont_masks)
        xques_embs = np.asarray(batch_of_ques_embs)
        xques_masks = np.array(batch_of_ques_masks)

        print(xcont_embs.shape)
        print(xques_embs.shape)

        ytarget_st = np.asarray(batch_of_target_st)
        ytarget_en = np.asarray(batch_of_target_en)
        ytarget_pr = np.asarray(batch_of_target_pr)

        train_x = [xcont_embs, xcont_masks, xques_embs, xques_masks]
        train_y = [ytarget_st, ytarget_en, ytarget_pr]


        model_bidaf.fit(train_x, train_y, batch_size=1, epochs=1,sample_weight=[sample_weight,sample_weight,sample_weight])
        model_bidaf.save_weights("model_bidaf.h5")
def bidaf_pred(row):
    
    threshold = 0.03 #random selection

    document_text = row['document_text'].split()
    question_text = row['question_text']
    
    ques_ans_tokens = tokenizer.tokenize(question_text)

    ques_ids = np.asarray([get_ids(ques_ans_tokens, tokenizer, ques_len)])
    ques_masks = np.asarray([get_masks(ques_ans_tokens, ques_len)])
    
    highest_combined_score = 0
    best_short_tokens = (0,0)
    best_long_tokens = (0,0)
    
    for candidate_no, long_answer_candidate in enumerate(row['long_answer_candidates']):

        long_ans_start_tok = long_answer_candidate['start_token']
        long_ans_end_tok = long_answer_candidate['end_token']
        long_sentence = " ".join(document_text[long_ans_start_tok:long_ans_end_tok])

        batch_of_cont_embs = []
        batch_of_cont_masks = []
        batch_of_ques_embs = []
        batch_of_ques_masks = []

        cont_embs = cont_bert.predict([long_ids,cont_masks],verbose=1,batch_size=1)
        ques_embs = ques_bert.predict([ques_ids,ques_masks],verbose=1,batch_size=1)

        batch_of_cont_embs.append(np.squeeze(cont_embs))
        batch_of_cont_masks.append(np.squeeze(cont_masks))
        batch_of_ques_embs.append(np.squeeze(ques_embs))
        batch_of_ques_masks.append(np.squeeze(ques_masks))
    
        xcont_embs = np.asarray(batch_of_cont_embs)
        xcont_masks = np.asarray(batch_of_cont_masks)
        xques_embs = np.asarray(batch_of_ques_embs)
        xques_masks = np.array(batch_of_ques_masks)
        
        output = model_bidaf.predict([xcont_embs, xcont_masks, xques_embs, xques_masks], batch_size = 1)
    
        start_pred_scores = output[0]
        end_pred_scores = output[1]
        present_pred_scores = output[2]

        start_tok_pred = np.argmax(start_pred_scores, axis=1)[0]
        end_tok_pred = np.argmax(end_pred_scores, axis=1)[0]
        present_tok_pred = np.argmax(present_pred_scores, axis=1)[0]

        start_pred_score = start_pred_scores[0][start_tok_pred]
        end_pred_score = end_pred_scores[0][end_tok_pred]
        present_pred_score = present_pred_scores[0][present_tok_pred]
        
        #print(start_pred_score)
        #print(start_tok_pred)
        
        if start_pred_score + end_pred_score > highest_combined_score and start_pred_score > threshold and end_pred_score > threshold:
            best_short_tokens = (start_tok_pred + long_ans_start_tok, end_tok_pred + long_ans_start_tok)
            best_long_tokens = (long_ans_start_tok, long_ans_end_tok)
            print("found one!")

    return best_short_tokens, best_long_tokens
        
### ls
TEST_TOTAL = 346

def get_joined_tokens(answer: dict) -> str:
    return '%d:%d' % (answer['start_token'], answer['end_token'])

def get_pred(json_data: dict, count) -> dict:
    ret = {'short': 'YES', 'long': ''}
    candidates = json_data['long_answer_candidates']
    
    paragraphs = []
    tokens = json_data['document_text'].split(' ')
    for cand in candidates:
        start_token = tokens[cand['start_token']]
        if start_token == '<P>' and cand['top_level'] and cand['end_token']-cand['start_token']>35:
            break
    else:
        cand = candidates[0]
    
    ret['long'] = get_joined_tokens(cand)

    best_short_tokens = (0,0)
    best_long_tokens = (0,0)
    
    if count < 15: #it takes a long time, so we just show few samples here.
        best_short_tokens, best_long_tokens = bidaf_pred(json_data)

    #if bidaf doesn't return good pred, it falls back to the first paragraph method.
    if best_short_tokens != (0,0):
        ret['short'] = '%d:%d' % (best_short_tokens[0], best_short_tokens[1])
        ret['long'] = '%d:%d' % (best_long_tokens[0], best_long_tokens[1])
    
    id_ = str(json_data['example_id'])
    ret = {id_+'_'+k: v for k, v in ret.items()} 
    return ret

preds = dict()

with open(path + test_path, 'r') as f:
    count = 0
    for line in tqdm(f, total=TEST_TOTAL):
        count += 1
        json_data = json.loads(line) 
        model_bidaf.load_weights('../input/bert-and-bidaf/model_bidaf.h5')
        prediction = get_pred(json_data, count)
        preds.update(prediction)
            
submission = pd.read_csv(path + 'sample_submission.csv')
submission['PredictionString'] = submission['example_id'].map(lambda x: preds[x])
submission.to_csv('submission.csv', index=False)