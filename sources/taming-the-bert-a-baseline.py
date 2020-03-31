
# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))
import zipfile
import sys
import time

# Any results you write to the current directory are saved as output.
#downloading weights and cofiguration file for the model
### wget https://storage.googleapis.com/bert_models/2018_10_18/uncased_L-12_H-768_A-12.zip
with zipfile.ZipFile("uncased_L-12_H-768_A-12.zip","r") as zip_ref:
    zip_ref.extractall()
### ls 'uncased_L-12_H-768_A-12'
### wget https://raw.githubusercontent.com/google-research/bert/master/modeling.py 
### wget https://raw.githubusercontent.com/google-research/bert/master/extract_features.py 
### wget https://raw.githubusercontent.com/google-research/bert/master/tokenization.py
import modeling
import extract_features
import tokenization
import tensorflow as tf
### wget https://raw.githubusercontent.com/google-research-datasets/gap-coreference/master/gap-development.tsv
### wget https://raw.githubusercontent.com/google-research-datasets/gap-coreference/master/gap-validation.tsv
### wget https://raw.githubusercontent.com/google-research-datasets/gap-coreference/master/gap-test.tsv
### ls
def compute_offset_no_spaces(text, offset):
	count = 0
	for pos in range(offset):
		if text[pos] != " ": count +=1
	return count

def count_chars_no_special(text):
	count = 0
	special_char_list = ["#"]
	for pos in range(len(text)):
		if text[pos] not in special_char_list: count +=1
	return count

def count_length_no_special(text):
	count = 0
	special_char_list = ["#", " "]
	for pos in range(len(text)):
		if text[pos] not in special_char_list: count +=1
	return count
def run_bert(data):
	'''
	Runs a forward propagation of BERT on input text, extracting contextual word embeddings
	Input: data, a pandas DataFrame containing the information in one of the GAP files

	Output: emb, a pandas DataFrame containing contextual embeddings for the words A, B and Pronoun. Each embedding is a numpy array of shape (768)
	columns: "emb_A": the embedding for word A
	         "emb_B": the embedding for word B
	         "emb_P": the embedding for the pronoun
	         "label": the answer to the coreference problem: "A", "B" or "NEITHER"
	'''
    # From the current file, take the text only, and write it in a file which will be passed to BERT
	text = data["Text"]
	text.to_csv("input.txt", index = False, header = False)

    # The script extract_features.py runs forward propagation through BERT, and writes the output in the file output.jsonl
    # I'm lazy, so I'm only saving the output of the last layer. Feel free to change --layers = -1 to save the output of other layers.
	os.system("python3 extract_features.py \
	  --input_file=input.txt \
	  --output_file=output.jsonl \
	  --vocab_file=uncased_L-12_H-768_A-12/vocab.txt \
	  --bert_config_file=uncased_L-12_H-768_A-12/bert_config.json \
	  --init_checkpoint=uncased_L-12_H-768_A-12/bert_model.ckpt \
	  --layers=-1 \
	  --max_seq_length=256 \
	  --batch_size=8")

	bert_output = pd.read_json("output.jsonl", lines = True)

	os.system("rm output.jsonl")
	os.system("rm input.txt")

	index = data.index
	columns = ["emb_A", "emb_B", "emb_P", "label"]
	emb = pd.DataFrame(index = index, columns = columns)
	emb.index.name = "ID"

	for i in range(len(data)): # For each line in the data file
		# get the words A, B, Pronoun. Convert them to lower case, since we're using the uncased version of BERT
		P = data.loc[i,"Pronoun"].lower()
		A = data.loc[i,"A"].lower()
		B = data.loc[i,"B"].lower()

		# For each word, find the offset not counting spaces. This is necessary for comparison with the output of BERT
		P_offset = compute_offset_no_spaces(data.loc[i,"Text"], data.loc[i,"Pronoun-offset"])
		A_offset = compute_offset_no_spaces(data.loc[i,"Text"], data.loc[i,"A-offset"])
		B_offset = compute_offset_no_spaces(data.loc[i,"Text"], data.loc[i,"B-offset"])
		# Figure out the length of A, B, not counting spaces or special characters
		A_length = count_length_no_special(A)
		B_length = count_length_no_special(B)

		# Initialize embeddings with zeros
		emb_A = np.zeros(768)
		emb_B = np.zeros(768)
		emb_P = np.zeros(768)

		# Initialize counts
		count_chars = 0
		cnt_A, cnt_B, cnt_P = 0, 0, 0

		features = pd.DataFrame(bert_output.loc[i,"features"]) # Get the BERT embeddings for the current line in the data file
		for j in range(2,len(features)):  # Iterate over the BERT tokens for the current line; we skip over the first 2 tokens, which don't correspond to words
			token = features.loc[j,"token"]

			# See if the character count until the current token matches the offset of any of the 3 target words
			if count_chars  == P_offset: 
				# print(token)
				emb_P += np.array(features.loc[j,"layers"][0]['values'])
				cnt_P += 1
			if count_chars in range(A_offset, A_offset + A_length): 
				# print(token)
				emb_A += np.array(features.loc[j,"layers"][0]['values'])
				cnt_A +=1
			if count_chars in range(B_offset, B_offset + B_length): 
				# print(token)
				emb_B += np.array(features.loc[j,"layers"][0]['values'])
				cnt_B +=1								
			# Update the character count
			count_chars += count_length_no_special(token)
		# Taking the average between tokens in the span of A or B, so divide the current value by the count	
		emb_A /= cnt_A
		emb_B /= cnt_B

		# Work out the label of the current piece of text
		label = "Neither"
		if (data.loc[i,"A-coref"] == True):
			label = "A"
		if (data.loc[i,"B-coref"] == True):
			label = "B"

		# Put everything together in emb
		emb.iloc[i] = [emb_A, emb_B, emb_P, label]

	return emb
print("Started at ", time.ctime())
test_data = pd.read_csv("gap-test.tsv", sep = '\t')
test_emb = run_bert(test_data)
test_emb.to_json("contextual_embeddings_gap_test.json", orient = 'columns')

validation_data = pd.read_csv("gap-validation.tsv", sep = '\t')
validation_emb = run_bert(validation_data)
validation_emb.to_json("contextual_embeddings_gap_validation.json", orient = 'columns')

development_data = pd.read_csv("gap-development.tsv", sep = '\t')
development_emb = run_bert(development_data)
development_emb.to_json("contextual_embeddings_gap_development.json", orient = 'columns')
print("Finished at ", time.ctime())
from keras import backend, models, layers, initializers, regularizers, constraints, optimizers
from keras import callbacks as kc
from keras import optimizers as ko

from sklearn.model_selection import cross_val_score, KFold, train_test_split
from sklearn.metrics import log_loss
import time


dense_layer_sizes = [37]
dropout_rate = 0.6
learning_rate = 0.001
n_fold = 5
batch_size = 32
epochs = 1000
patience = 100
# n_test = 100
lambd = 0.1 # L2 regularization
def build_mlp_model(input_shape):
	X_input = layers.Input(input_shape)

	# First dense layer
	X = layers.Dense(dense_layer_sizes[0], name = 'dense0')(X_input)
	X = layers.BatchNormalization(name = 'bn0')(X)
	X = layers.Activation('relu')(X)
	X = layers.Dropout(dropout_rate, seed = 7)(X)

	# Second dense layer
# 	X = layers.Dense(dense_layer_sizes[0], name = 'dense1')(X)
# 	X = layers.BatchNormalization(name = 'bn1')(X)
# 	X = layers.Activation('relu')(X)
# 	X = layers.Dropout(dropout_rate, seed = 9)(X)

	# Output layer
	X = layers.Dense(3, name = 'output', kernel_regularizer = regularizers.l2(lambd))(X)
	X = layers.Activation('softmax')(X)

	# Create model
	model = models.Model(input = X_input, output = X, name = "classif_model")
	return model
def parse_json(embeddings):
	'''
	Parses the embeddigns given by BERT, and suitably formats them to be passed to the MLP model

	Input: embeddings, a DataFrame containing contextual embeddings from BERT, as well as the labels for the classification problem
	columns: "emb_A": contextual embedding for the word A
	         "emb_B": contextual embedding for the word B
	         "emb_P": contextual embedding for the pronoun
	         "label": the answer to the coreference problem: "A", "B" or "NEITHER"

	Output: X, a numpy array containing, for each line in the GAP file, the concatenation of the embeddings of the target words
	        Y, a numpy array containing, for each line in the GAP file, the one-hot encoded answer to the coreference problem
	'''
	embeddings.sort_index(inplace = True) # Sorting the DataFrame, because reading from the json file messed with the order
	X = np.zeros((len(embeddings),3*768))
	Y = np.zeros((len(embeddings), 3))

	# Concatenate features
	for i in range(len(embeddings)):
		A = np.array(embeddings.loc[i,"emb_A"])
		B = np.array(embeddings.loc[i,"emb_B"])
		P = np.array(embeddings.loc[i,"emb_P"])
		X[i] = np.concatenate((A,B,P))

	# One-hot encoding for labels
	for i in range(len(embeddings)):
		label = embeddings.loc[i,"label"]
		if label == "A":
			Y[i,0] = 1
		elif label == "B":
			Y[i,1] = 1
		else:
			Y[i,2] = 1

	return X, Y
# Read development embeddigns from json file - this is the output of Bert
development = pd.read_json("contextual_embeddings_gap_development.json")
X_development, Y_development = parse_json(development)

validation = pd.read_json("contextual_embeddings_gap_validation.json")
X_validation, Y_validation = parse_json(validation)

test = pd.read_json("contextual_embeddings_gap_test.json")
X_test, Y_test = parse_json(test)
# There may be a few NaN values, where the offset of a target word is greater than the max_seq_length of BERT.
# They are very few, so I'm just dropping the rows.
remove_test = [row for row in range(len(X_test)) if np.sum(np.isnan(X_test[row]))]
X_test = np.delete(X_test, remove_test, 0)
Y_test = np.delete(Y_test, remove_test, 0)

remove_validation = [row for row in range(len(X_validation)) if np.sum(np.isnan(X_validation[row]))]
X_validation = np.delete(X_validation, remove_validation, 0)
Y_validation = np.delete(Y_validation, remove_validation, 0)

# We want predictions for all development rows. So instead of removing rows, make them 0
remove_development = [row for row in range(len(X_development)) if np.sum(np.isnan(X_development[row]))]
X_development[remove_development] = np.zeros(3*768)
# Will train on data from the gap-test and gap-validation files, in total 2454 rows
X_train = np.concatenate((X_test, X_validation), axis = 0)
Y_train = np.concatenate((Y_test, Y_validation), axis = 0)

# Will predict probabilities for data from the gap-development file; initializing the predictions
prediction = np.zeros((len(X_development),3)) # testing predictions
# Training and cross-validation
folds = KFold(n_splits=n_fold, shuffle=True, random_state=3)
scores = []
for fold_n, (train_index, valid_index) in enumerate(folds.split(X_train)):
	# split training and validation data
	print('Fold', fold_n, 'started at', time.ctime())
	X_tr, X_val = X_train[train_index], X_train[valid_index]
	Y_tr, Y_val = Y_train[train_index], Y_train[valid_index]

	# Define the model, re-initializing for each fold
	classif_model = build_mlp_model([X_train.shape[1]])
	classif_model.compile(optimizer = optimizers.Adam(lr = learning_rate), loss = "categorical_crossentropy")
	callbacks = [kc.EarlyStopping(monitor='val_loss', patience=patience, restore_best_weights = True)]

	# train the model
	classif_model.fit(x = X_tr, y = Y_tr, epochs = epochs, batch_size = batch_size, callbacks = callbacks, validation_data = (X_val, Y_val), verbose = 0)

	# make predictions on validation and test data
	pred_valid = classif_model.predict(x = X_val, verbose = 0)
	pred = classif_model.predict(x = X_development, verbose = 0)

	# oof[valid_index] = pred_valid.reshape(-1,)
	scores.append(log_loss(Y_val, pred_valid))
	prediction += pred
prediction /= n_fold

# Print CV scores, as well as score on the test data
print('CV mean score: {0:.4f}, std: {1:.4f}.'.format(np.mean(scores), np.std(scores)))
print(scores)
print("Test score:", log_loss(Y_development,prediction))
# Write the prediction to file for submission
submission = pd.read_csv("../input/sample_submission_stage_1.csv", index_col = "ID")
submission["A"] = prediction[:,0]
submission["B"] = prediction[:,1]
submission["NEITHER"] = prediction[:,2]
submission.to_csv("submission_bert.csv")