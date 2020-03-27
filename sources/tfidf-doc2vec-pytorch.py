
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from category_encoders.one_hot import OneHotEncoder
import numpy as np
import torch
import torch.nn as nn

from torch.nn import Module
from torch.nn import Linear
from torch.nn import Sigmoid

from torch.nn import MSELoss
from torch.optim import Adam

from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from scipy.stats import spearmanr
import gc

from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from gensim.test.utils import common_texts
from nltk.tokenize import word_tokenize
"""
train = pd.read_csv("/kaggle/input/google-quest-challenge/train.csv")
texts = list(train['answer'].values)
"""
class MyDoc2Vec():
    def __init__(self):
        self.tadded_data = None
        self.model = None
        
    def fit_transform(self, df):
        texts = list(df.values)
        tagged_data = [TaggedDocument(words=word_tokenize(_d.lower()), tags=[str(i)]) for i, _d in enumerate(texts)]
        self.model = Doc2Vec(tagged_data, vector_size=20, epochs=30)
        feature = np.array([self.model.docvecs[i] for i in range(df.shape[0])]) # convert type docvecs to ndarray
        return feature
        
    def transform(self, df):
        texts = list(df.values)
        words = [word_tokenize(texts[i].lower()) for i in range(df.shape[0])]
        feature = np.array([self.model.infer_vector(words[i], epochs=30) for i in range(df.shape[0])])
        return feature
"""
my_doc2vec = MyDoc2Vec()
feature = my_doc2vec.fit_transform(train['answer'])
"""
"""
test_feature = my_doc2vec.transform(train['answer'])
test_feature2 = my_doc2vec.transform(train['answer'])
"""
def cos_sim(v1, v2):
    return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
# cos_sim(test_feature2[0], test_feature[0])
def load():
    train = pd.read_csv("/kaggle/input/google-quest-challenge/train.csv")
    test = pd.read_csv("/kaggle/input/google-quest-challenge/test.csv")
    
    target_cols = ['question_asker_intent_understanding',
                   'question_body_critical', 'question_conversational',
                   'question_expect_short_answer', 'question_fact_seeking',
                   'question_has_commonly_accepted_answer',
                   'question_interestingness_others', 'question_interestingness_self',
                   'question_multi_intent', 'question_not_really_a_question',
                   'question_opinion_seeking', 'question_type_choice',
                   'question_type_compare', 'question_type_consequence',
                   'question_type_definition', 'question_type_entity',
                   'question_type_instructions', 'question_type_procedure',
                   'question_type_reason_explanation', 'question_type_spelling',
                   'question_well_written', 'answer_helpful',
                   'answer_level_of_information', 'answer_plausible', 'answer_relevance',
                   'answer_satisfaction', 'answer_type_instructions',
                   'answer_type_procedure', 'answer_type_reason_explanation',
                   'answer_well_written']
    
    data_cols = ['question_title', 'question_body', 'answer', 'category']
    
    y_train = train[target_cols].copy()
    print(type(y_train))
    x_train = train[data_cols].copy()
    del train

    x_test = test.copy()
    del test    

    
    question_body_doc2vec = MyDoc2Vec()
    answer_doc3vec = MyDoc2Vec()
    
    x_train_question_vec = question_body_doc2vec.fit_transform(x_train['question_body'])
    x_test_question_vec = question_body_doc2vec.transform(x_test['question_body'])
    x_train_answer_vec = question_body_doc2vec.fit_transform(x_train['answer'])
    x_test_answer_vec = question_body_doc2vec.transform(x_test['answer'])
    
    print(x_train_question_vec.shape)
    
    text_encoder = Pipeline([
        ('Text-TF-IDF', TfidfVectorizer(ngram_range=(1, 1))),
        ('Text-SVD', TruncatedSVD(n_components = 100))], verbose=True)
    
    ohe = OneHotEncoder(cols=['category'])
    
    preprocessor = ColumnTransformer([
        ('Q-T', text_encoder, 'question_title'),
        ('Q-B', text_encoder, 'question_body'),
        ('A', text_encoder, 'answer'),
        ('Category', ohe, 'category'),
        ])
    
    x_train = preprocessor.fit_transform(x_train).astype(np.float32)
    x_test = preprocessor.transform(x_test).astype(np.float32)
    y_train = y_train.values.astype(np.float32)
    
    x_train = np.concatenate([x_train, x_train_question_vec, x_train_answer_vec], axis=1)
    x_test = np.concatenate([x_test, x_test_question_vec, x_test_answer_vec], axis=1)
    
    return x_train, y_train, x_test

def calc_score(prediction, label):

    score = 0
    for col_index in range(30):
        score += spearmanr(prediction[:, col_index], label[col_index]).correlation/len(target_cols)
    return score
class Net(Module):
    def __init__(self):
        super(Net, self).__init__()
        self.hidden_feature = 600
        self.linear1 = Linear(345, self.hidden_feature)
        self.sigmoid1 = Sigmoid()
        self.linear2 = Linear(self.hidden_feature, 30)
        # self.sigmoid2 = Sigmoid()

    def forward(self, x):
        x = self.linear1(x)
        x = self.sigmoid1(x)
        x = self.linear2(x)
        # x = self.sigmoid2(x)
        return x
class MyDataset(Dataset):

    def __init__(self, data, label, device, transform=None):
        self.transform = transform
        self.data = torch.from_numpy(data).to(device)
        self.label = torch.from_numpy(label).to(device)
        self.data_num = self.data.shape[0]
        
    def __len__(self):
        return self.data_num

    def __getitem__(self, idx):
        out_data = self.data[idx]
        out_label =  self.label[idx]

        #if self.transform:
        #    out_data = self.transform(out_data)

        return out_data, out_label

    def get_numpy_label(self):
        return self.label.detach().cpu().numpy()
    
class TestDataset(Dataset):

    def __init__(self, data, device, transform=None):
        self.transform = transform
        self.data = torch.from_numpy(data).to(device)
        self.data_num = self.data.shape[0]

    def __len__(self):
        return self.data_num

    def __getitem__(self, idx):
        out_data = self.data[idx]

        #if self.transform:
        #    out_data = self.transform(out_data)

        return out_data

if torch.cuda.is_available():
   device = 'cuda'
else:
    device = 'cpu'


train_data, train_label, test_data = load()
dataset = MyDataset(train_data, train_label, device)
test_dataset = TestDataset(test_data, device)
train_size = int(len(dataset)*1.0)
val_size = len(dataset) - train_size
train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

train_dataloader = DataLoader(train_dataset, batch_size=20, shuffle=True)

model = Net().to(device)


epoch_size = 100

criterion = MSELoss()
optimizer = Adam(model.parameters(), lr=1e-5)

loss_list = np.zeros(epoch_size)

for epoch in range(epoch_size):
    for n_batch, data_batched in enumerate(train_dataloader):
        data, label = data_batched
        optimizer.zero_grad()

        output = model(data)
        loss = criterion(output, label)
        loss.backward()
        optimizer.step()
        tmp = loss.detach().cpu()
        
        loss_list[epoch] += tmp.detach().numpy()

    loss_list[epoch] /= n_batch
    print(f"{epoch}, loss is {loss_list[epoch]}")
    
# torch.save(model.state_dict(), 'weight.pth')
plt.plot(loss_list)
plt.show()
target_cols = ['question_asker_intent_understanding',
               'question_body_critical', 'question_conversational',
               'question_expect_short_answer', 'question_fact_seeking',
               'question_has_commonly_accepted_answer',
               'question_interestingness_others', 'question_interestingness_self',
               'question_multi_intent', 'question_not_really_a_question',
               'question_opinion_seeking', 'question_type_choice',
               'question_type_compare', 'question_type_consequence',
               'question_type_definition', 'question_type_entity',
               'question_type_instructions', 'question_type_procedure',
               'question_type_reason_explanation', 'question_type_spelling',
               'question_well_written', 'answer_helpful',
               'answer_level_of_information', 'answer_plausible', 'answer_relevance',
               'answer_satisfaction', 'answer_type_instructions',
               'answer_type_procedure', 'answer_type_reason_explanation',
               'answer_well_written']
train_prediction = np.zeros((len(train_dataset), 30))
model.eval()
with torch.no_grad():
    for i, data in enumerate(train_dataset):
        input, label = data
        output = model(input)
        train_prediction[i] = output.detach().cpu().numpy()
        
train_prediction = train_prediction.T
print(f"training score is {calc_score(train_prediction, train_dataset.dataset.get_numpy_label())}")
if len(val_dataset) != 0:
    val_prediction = np.zeros((len(val_dataset), 30))
    model.eval()
    with torch.no_grad():
        for i, data in enumerate(val_dataset):
            input, label = data
            output = model(input)
            val_prediction[i] = output.detach().cpu().numpy()
        
    val_prediction = val_prediction.T
    print(f"validation score is {calc_score(val_prediction, val_dataset.dataset.get_numpy_label())}")

prediction = np.zeros((len(test_dataset), 30))
model.eval()
with torch.no_grad():
    for i, data in enumerate(test_dataset):
        output = model(data)
        prediction[i] = output.detach().cpu().numpy()

sample = pd.read_csv("/kaggle/input/google-quest-challenge/sample_submission.csv")
prediction = prediction.T
prediction = np.clip(prediction, a_min=0.0, a_max=1.0)

for i in range(30):
    sample[target_cols[i]] = prediction[i]


sample.to_csv('submission.csv', index=False)
sample.head()
df = pd.read_csv("/kaggle/input/google-quest-challenge/train.csv")
df[target_cols].head()