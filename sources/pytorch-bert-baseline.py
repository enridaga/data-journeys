
!pip install ../input/sacremoses/sacremoses-master/
!pip install ../input/transformers/transformers-master/
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
DATA_DIR = '../input/google-quest-challenge'
!ls ../input
sub = pd.read_csv(f'{DATA_DIR}/sample_submission.csv')
sub.head()
target_columns = sub.columns.values[1:].tolist()
target_columns
train = pd.read_csv(f'{DATA_DIR}/train.csv')
train.head()
test = pd.read_csv(f'{DATA_DIR}/test.csv')
test.head()
import torch
#import torch.utils.data as data
from torchvision import datasets, models, transforms
from transformers import *
from sklearn.utils import shuffle
import random
from math import floor, ceil
from sklearn.model_selection import GroupKFold

MAX_LEN = 512
#MAX_Q_LEN = 250
#MAX_A_LEN = 259
SEP_TOKEN_ID = 102

class QuestDataset(torch.utils.data.Dataset):
    def __init__(self, df, train_mode=True, labeled=True):
        self.df = df
        self.train_mode = train_mode
        self.labeled = labeled
        #self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.tokenizer = BertTokenizer.from_pretrained('../input/bert-base-uncased/')

    def __getitem__(self, index):
        row = self.df.iloc[index]
        token_ids, seg_ids = self.get_token_ids(row)
        if self.labeled:
            labels = self.get_label(row)
            return token_ids, seg_ids, labels
        else:
            return token_ids, seg_ids

    def __len__(self):
        return len(self.df)

    def select_tokens(self, tokens, max_num):
        if len(tokens) <= max_num:
            return tokens
        if self.train_mode:
            num_remove = len(tokens) - max_num
            remove_start = random.randint(0, len(tokens)-num_remove-1)
            return tokens[:remove_start] + tokens[remove_start + num_remove:]
        else:
            return tokens[:max_num//2] + tokens[-(max_num - max_num//2):]

    def trim_input(self, title, question, answer, max_sequence_length=MAX_LEN, 
                t_max_len=30, q_max_len=239, a_max_len=239):
        t = self.tokenizer.tokenize(title)
        q = self.tokenizer.tokenize(question)
        a = self.tokenizer.tokenize(answer)

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
                raise ValueError("New sequence length should be %d, but is %d" 
                                 % (max_sequence_length, (t_new_len+a_new_len+q_new_len+4)))

            t = t[:t_new_len]
            q = q[:q_new_len]
            a = a[:a_new_len]

        return t, q, a
        
    def get_token_ids(self, row):
        t_tokens, q_tokens, a_tokens = self.trim_input(row.question_title, row.question_body, row.answer)

        tokens = ['[CLS]'] + t_tokens + ['[SEP]'] + q_tokens + ['[SEP]'] + a_tokens + ['[SEP]']
        token_ids = self.tokenizer.convert_tokens_to_ids(tokens)
        if len(token_ids) < MAX_LEN:
            token_ids += [0] * (MAX_LEN - len(token_ids))
        ids = torch.tensor(token_ids)
        seg_ids = self.get_seg_ids(ids)
        return ids, seg_ids
    
    def get_seg_ids(self, ids):
        seg_ids = torch.zeros_like(ids)
        seg_idx = 0
        first_sep = True
        for i, e in enumerate(ids):
            seg_ids[i] = seg_idx
            if e == SEP_TOKEN_ID:
                if first_sep:
                    first_sep = False
                else:
                    seg_idx = 1
        pad_idx = torch.nonzero(ids == 0)
        seg_ids[pad_idx] = 0

        return seg_ids

    def get_label(self, row):
        #print(row[target_columns].values)
        return torch.tensor(row[target_columns].values.astype(np.float32))

    def collate_fn(self, batch):
        token_ids = torch.stack([x[0] for x in batch])
        seg_ids = torch.stack([x[1] for x in batch])
    
        if self.labeled:
            labels = torch.stack([x[2] for x in batch])
            return token_ids, seg_ids, labels
        else:
            return token_ids, seg_ids

def get_test_loader(batch_size=4):
    df = pd.read_csv(f'{DATA_DIR}/test.csv')
    ds_test = QuestDataset(df, train_mode=False, labeled=False)
    loader = torch.utils.data.DataLoader(ds_test, batch_size=batch_size, shuffle=False, num_workers=0, collate_fn=ds_test.collate_fn, drop_last=False)
    loader.num = len(df)
    
    return loader
        
def get_train_val_loaders(batch_size=4, val_batch_size=4, ifold=0):
    df = pd.read_csv(f'{DATA_DIR}/train.csv')
    df = shuffle(df, random_state=1234)
    #split_index = int(len(df) * (1-val_percent))
    gkf = GroupKFold(n_splits=5).split(X=df.question_body, groups=df.question_body)
    for fold, (train_idx, valid_idx) in enumerate(gkf):
        if fold == ifold:
            df_train = df.iloc[train_idx]
            df_val = df.iloc[valid_idx]
            break

    #print(df_val.head())
    #df_train = df[:split_index]
    #df_val = df[split_index:]

    print(df_train.shape)
    print(df_val.shape)

    ds_train = QuestDataset(df_train)
    train_loader = torch.utils.data.DataLoader(ds_train, batch_size=batch_size, shuffle=True, num_workers=0, collate_fn=ds_train.collate_fn, drop_last=True)
    train_loader.num = len(df_train)

    ds_val = QuestDataset(df_val, train_mode=False)
    val_loader = torch.utils.data.DataLoader(ds_val, batch_size=val_batch_size, shuffle=False, num_workers=0, collate_fn=ds_val.collate_fn, drop_last=False)
    val_loader.num = len(df_val)
    val_loader.df = df_val

    return train_loader, val_loader

def test_train_loader():
    loader, _ = get_train_val_loaders(4, 4, 1)
    for ids, seg_ids, labels in loader:
        print(ids)
        print(seg_ids.numpy())
        print(labels)
        break
def test_test_loader():
    loader = get_test_loader(4)
    for ids, seg_ids in loader:
        print(ids)
        print(seg_ids)
        break
test_train_loader()
from transformers import *
import torch
import torch.nn as nn
import torch.nn.functional as F


class QuestModel(nn.Module):
    def __init__(self, n_classes=30):
        super(QuestModel, self).__init__()
        self.model_name = 'QuestModel'
        self.bert_model = BertModel.from_pretrained('../input/bert-base-uncased/')    
        self.fc = nn.Linear(768, n_classes)

    def forward(self, ids, seg_ids):
        attention_mask = (ids > 0)
        layers, pool_out = self.bert_model(input_ids=ids, token_type_ids=seg_ids, attention_mask=attention_mask)
        #print(layers[-1][0].size())
        #print(pool_out.size())

        #out = F.dropout(layers[-1][:, 0, :], p=0.2, training=self.training)
        out =  F.dropout(pool_out, p=0.2, training=self.training)
        logit = self.fc(out)
        return logit
    
def test_model():
    x = torch.tensor([[1,2,3,4,5, 0, 0], [1,2,3,4,5, 0, 0]])
    seg_ids = torch.tensor([[0,0,0,0,0, 0, 0], [0,0,0,0,0, 0, 0]])
    model = QuestModel()

    y = model(x, seg_ids)
    print(y)
!ls ../input
def create_model(model_file):
    model = QuestModel()
    model.load_state_dict(torch.load(model_file))
    model = model.cuda()
    #model = DataParallel(model)
    return model

def create_models():
    models = []
    for i in range(5):
        model = create_model(f'../input/quest-models/best_{i}.pth')
        model.eval()
        models.append(model)
    return models
from tqdm import tqdm
import torch
def predict(models, test_loader):
    all_scores = []
    with torch.no_grad():
        for ids, seg_ids in tqdm(test_loader, total=test_loader.num // test_loader.batch_size):
            ids, seg_ids = ids.cuda(), seg_ids.cuda()
            scores = []
            for model in models:
                outputs = torch.sigmoid(model(ids, seg_ids)).cpu()
                scores.append(outputs)
            all_scores.append(torch.mean(torch.stack(scores), 0))

    all_scores = torch.cat(all_scores, 0).numpy()
    
    return all_scores
test_loader = get_test_loader(batch_size=32)
models = create_models()
preds = predict(models, test_loader)
preds[:1]
sub[target_columns] = preds
sub.to_csv('submission.csv', index=False)
sub.head()
