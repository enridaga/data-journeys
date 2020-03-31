
### pip install transformers
import re
import string
import torch
import transformers
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
import matplotlib.pyplot as plt
from nltk.corpus import stopwords
from sklearn.manifold import TSNE
from nltk.tokenize import word_tokenize
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, roc_auc_score

df = pd.read_csv("/kaggle/input/sms-spam-collection-dataset/spam.csv", encoding='latin-1')
df.head()
df.drop(["Unnamed: 2", "Unnamed: 3", "Unnamed: 4"], inplace=True, axis=1)

#Adding new features
df["num_words"] = df["v2"].apply(lambda s: len(re.findall(r'\w+', s))) # Count the number of words in the message
df["message_len"] = df["v2"].apply(len) # get the length of the text message

df["v1"].replace({"ham": 0, "spam":1}, inplace=True)

df.rename({"v1": "is_spam", "v2": "message"},axis=1, inplace=True)
def clean_sentence(s):
    """Given a sentence remove its punctuation and stop words"""
    
    stop_words = set(stopwords.words('english'))
    s = s.translate(str.maketrans('','',string.punctuation)) # remove punctuation
    tokens = word_tokenize(s)
    cleaned_s = [w for w in tokens if w not in stop_words] # removing stop-words
    return " ".join(cleaned_s[:10]) # using the first 10 tokens only

df["message"] = df["message"].apply(clean_sentence)
# Loading pretrained model/tokenizer
tokenizer = transformers.DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
model = transformers.DistilBertModel.from_pretrained("distilbert-base-uncased")
tokenized = df["message"].apply(lambda x: tokenizer.encode(x, add_special_tokens=True))
tokenized
max_len = tokenized.apply(len).max() # get the length of the longest tokenized sentence

padded = np.array([i + [0]*(max_len-len(i)) for i in tokenized.values]) # padd the rest of the sentence with zeros if the sentence is smaller than the longest sentence
padded
attention_mask = np.where(padded != 0, 1, 0)
attention_mask
input_ids = torch.tensor(padded)  # create a torch tensor for the padded sentences
attention_mask = torch.tensor(attention_mask) # create a torch tensor for the attention matrix

with torch.no_grad():
    encoder_hidden_state = model(input_ids, attention_mask=attention_mask)
X = encoder_hidden_state[0][:,0,:].numpy()
X = np.hstack((X, df[["num_words", "message_len"]].to_numpy().reshape(-1, 2))) # addind the the engineered features from the beginning
y = df["is_spam"]
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, random_state=42)
X_embedded = TSNE(n_components=2, random_state=42).fit_transform(X_train)
X_embedded.shape
# creating the dataframe for plotting
def creat_plotting_data(data, labels=y_train):
    """Creates a dataframe from the given data, used for plotting"""
    
    df = pd.DataFrame(data)
    df["is_spam"] = labels.to_numpy()
    df.rename({0:"v1", 1:"v2", 768:"num_words", 769: "message_len"}, axis=1, inplace=True)
    return df

# creating the dataframes for plotting
plotting_data = creat_plotting_data(X_train)
plotting_data_embedded = creat_plotting_data(X_embedded)
plt.figure(figsize=(16, 10))
ax = sns.scatterplot(x="v1", y="v2", hue="is_spam", data=plotting_data_embedded)
ax.set(title = "Spam messages are generally closer together due to the BERT embeddings")
plt.show()
f,ax = plt.subplots(figsize=(16,10))
sns.kdeplot(plotting_data.loc[plotting_data.is_spam == 1, "num_words"], shade=True, label="Spam")
sns.kdeplot(plotting_data.loc[plotting_data.is_spam == 0, "num_words"], shade=True, label="Ham", clip=(0, 35)) # removing observations with message length above 35 because there is an outlier
ax.set(xlabel = "Number of words", ylabel = "Density",title = "Spam messages have more words than ham messages")
plt.show()
f,ax = plt.subplots(figsize=(16,10))
sns.kdeplot(plotting_data.loc[plotting_data.is_spam == 1, "message_len"], shade=True, label="Spam")
sns.kdeplot(plotting_data.loc[plotting_data.is_spam == 0, "message_len"], shade=True, label="Ham", clip=(0, 250)) # removing observations with message length above 250 because there is an outlier
ax.set(xlabel = "Message length", ylabel = "Density",title = "Spam messages are longer than ham messages, concentrated on 150 characters")
plt.show()
rf_classifier = RandomForestClassifier(n_estimators=1500, class_weight="balanced", n_jobs=-1, random_state=42) # Create a baseline random forest (no cross-validation, no hyperparameter tuning)
rf_classifier.fit(X_train, y_train)
preds = rf_classifier.predict(X_test)
fig = plt.figure(figsize=(10,4))
heatmap = sns.heatmap(data = pd.DataFrame(confusion_matrix(y_test, preds)), annot = True, fmt = "d", cmap=sns.color_palette("Reds", 50))
heatmap.yaxis.set_ticklabels(heatmap.yaxis.get_ticklabels(), rotation=0, ha='right', fontsize=14)
heatmap.xaxis.set_ticklabels(heatmap.xaxis.get_ticklabels(), rotation=45, ha='right', fontsize=14)
plt.ylabel('Ground Truth')
plt.xlabel('Prediction')
plt.show()
print(f"""Accuray: {round(accuracy_score(y_test, preds), 5) * 100}%
ROC-AUC: {round(roc_auc_score(y_test, preds), 5) * 100}%""")
print(classification_report(y_test, preds))