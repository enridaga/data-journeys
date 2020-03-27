
import pandas as pd
import numpy as np
from sklearn.model_selection import GroupKFold
import matplotlib.pyplot as plt
from tqdm.notebook import tqdm
import re
import tensorflow_hub as hub
import tensorflow as tf
import gc
import os
from scipy.stats import spearmanr
from math import floor, ceil

np.set_printoptions(suppress=True)
PATH = '../input/google-quest-challenge/'

df_train = pd.read_csv(PATH+'train.csv')
df_test = pd.read_csv(PATH+'test.csv')
df_sub = pd.read_csv(PATH+'sample_submission.csv')
print('train shape =', df_train.shape)
print('test shape =', df_test.shape)

output_categories = list(df_train.columns[11:])
input_categories = list(df_train.columns[[1,2,5,9,10]])
print('\noutput categories:\n\t', output_categories)
print('\ninput categories:\n\t', input_categories)
def convert_to_use_vectors(df, embed):
    t = np.empty((len(df), 512))
    q = np.empty((len(df), 512))
    a = np.empty((len(df), 512))
    for i, instance in tqdm(df.iterrows()):
        t[i, :] = embed([str(instance.question_title)])["outputs"].numpy().flatten()
        q[i, :] = embed([str(instance.question_body)])["outputs"].numpy().flatten()
        a[i, :] = embed([str(instance.answer)])["outputs"].numpy().flatten()
    return t, q, a

embed = hub.load('../input/universalsentenceencoderlarge4/')

train_t, train_q, train_a = convert_to_use_vectors(df_train, embed)
test_t, test_q, test_a = convert_to_use_vectors(df_test, embed)
def onehot_features(train, test, column):
    from pandas.api.types import CategoricalDtype
    categories = train[column].dropna().unique()
    train[column] = train[column].astype(CategoricalDtype(categories))
    test[column] = test[column].astype(CategoricalDtype(categories))

    train = pd.get_dummies(train[column])
    test = pd.get_dummies(test[column])
    return train, test


train_category, test_category = onehot_features(df_train, df_test, 'category')
assert all(train_category.columns == test_category.columns), 'Mismatch between train and test set'
train_category = np.asarray(train_category, dtype=np.float32)
test_category = np.asarray(test_category, dtype=np.float32)
print('train category one hot shape = {}'.format(train_category.shape))
print('test category one hot shape = {}'.format(test_category.shape))

train_host, test_host = onehot_features(df_train, df_test, 'host')
assert all(train_host.columns == test_host.columns), 'Mismatch between train and test set'
train_host = np.asarray(train_host, dtype=np.float32)
test_host = np.asarray(test_host, dtype=np.float32)
print('train host one hot shape = {}'.format(train_host.shape))
print('test host one hot shape = {}'.format(test_host.shape))
def compute_spearmanr(trues, preds):
    rhos = []
    for tcol, pcol in zip(np.transpose(trues), np.transpose(preds)):
        rho = spearmanr(tcol, pcol + np.random.normal(0, 1e-7, pcol.shape[0]))
        rhos.append(rho.correlation)
    return np.mean(rhos)

def initialize_weights(init, shape):
    W = tf.Variable(init(shape))
    b = tf.Variable(0.)
    return W, b

def dense_hidden(x, w, b, dropout_rate, training):
    x = tf.matmul(x, w) + b
    x = tf.nn.relu(x)
    if training:
        return tf.nn.dropout(x, dropout_rate)
    return x

def dense_output(x, w, b):
    x = tf.matmul(x, w) + b
    x = tf.nn.sigmoid(x)
    return x


class NeuralNet(tf.keras.Model):

    def __init__(self, inputs_units, dense_units, dropout_rates, aux_units, name="NeuralNet"):
        """
        Parameters
        ----------
        input_units : list (3 elements)
            length of (1) use vector, (2) one-hot vec cateogry & (3) one-hot vec host
            for an input example, i.e. input_units=[512, 5, 63]
        dense_units : list
            containing the number of units that will be used for each layer
        dropout_rates : list 
            the rates of dropout for each of the dense_units/layers (except output layer)
        aux_units: int
            number of units for the two auxillary layers
        
        """
        
        super(NeuralNet, self).__init__(name)
        
        glorot = tf.initializers.glorot_uniform()
        self.dropout_rates = dropout_rates
        self.W1, self.b1 = initialize_weights(glorot, (input_units[0], dense_units[0]))
        self.W2, self.b2 = initialize_weights(glorot, (dense_units[0], dense_units[1]))
        self.W3, self.b3 = initialize_weights(glorot, (dense_units[0], dense_units[2]))
        self.W4, self.b4 = initialize_weights(glorot, (dense_units[0], dense_units[3]))
        self.W5, self.b5 = initialize_weights(
            glorot, (dense_units[1]+dense_units[2]+dense_units[3]+(aux_units*2), dense_units[4]))
        
        self.W_aux1, self.b_aux1 = initialize_weights(glorot, (inputs_units[1], aux_units))
        self.W_aux2, self.b_aux2 = initialize_weights(glorot, (inputs_units[2], aux_units))
    
    @tf.function
    def call(self, inputs, training=False):
        
        # sharing first layer ("text" input)
        x0 = dense_hidden(inputs[0], self.W1, self.b1, self.dropout_rates[0], training)
        x1 = dense_hidden(inputs[1], self.W1, self.b1, self.dropout_rates[0], training)
        x2 = dense_hidden(inputs[2], self.W1, self.b1, self.dropout_rates[0], training)
        # no more sharing
        x0 = dense_hidden(x0, self.W2, self.b2, self.dropout_rates[1], training)
        x1 = dense_hidden(x1, self.W3, self.b3, self.dropout_rates[2], training)
        x2 = dense_hidden(x2, self.W4, self.b4, self.dropout_rates[3], training)
        
        # aux layers
        # category input -> hidden_layer
        x3 = dense_hidden(inputs[3], self.W_aux1, self.b_aux1, self.dropout_rates[-1], training)
        # host input -> hidden layer
        x4 = dense_hidden(inputs[4], self.W_aux2, self.b_aux2, self.dropout_rates[-1], training)
        
        # concat and output
        x = tf.concat([x0, x1, x2, x3, x4], axis=1)
        return dense_output(x, self.W5, self.b5)
def create_dataset(inputs, outputs=None, batch_size=8):
    dataset_inputs = tf.data.Dataset.from_tensor_slices(inputs)
    if outputs is not None:
        dataset_outputs = tf.data.Dataset.from_tensor_slices(outputs)
        return tf.data.Dataset.zip((dataset_inputs, dataset_outputs)).batch(batch_size)
    return dataset_inputs.batch(batch_size)

def get_train_step_fn():
    """This is a workaround so that the tf.function decorator
    works for the cross-validation. i.e. when it's called 
    a second and third time etc.."""
    @tf.function
    def train_step(model, loss_function, optimizer, metric, x, y):
        with tf.GradientTape() as tape:
            pred = model(x, training=True)
            loss = loss_function(y, pred)
        # compute gradients of all trainable variables with respect to the loss
        grad = tape.gradient(loss, model.trainable_variables)
        # apply gradients to the variables/updating them
        optimizer.apply_gradients(zip(grad, model.trainable_variables))
        # batch metric calculation
        metric(y, pred)

        return loss, pred
    return train_step

def train_and_predict(model, loss_function, optimizer, metric, 
                      train_dataset, valid_dataset, test_dataset, num_epochs):
    
    train_step = get_train_step_fn()
    valid_snapshot_preds, test_snapshot_preds = [], []
    train_scores, valid_scores = [0.], [0.]
    for epoch in range(num_epochs):
        
        print("\nepoch %03d" % (epoch+1))
        
        # training loop
        epoch_loss = 0.
        train_preds, train_trues = np.empty((0, 30), np.float32), np.empty((0, 30), np.float32)
        for batch, (x_train, y_train) in enumerate(train_dataset):
            loss, pred = train_step(model, loss_function, optimizer, metric, x_train, y_train)
            epoch_loss += loss
            train_trues = np.append(train_trues, y_train.numpy(), axis=0)
            train_preds = np.append(train_preds, pred.numpy(), axis=0)
            
            print("batch {:03d} : train loss {:.3f} : train cosine {:.3f} : train spearman {:.3f} : valid spearman {:.3f}"
                  .format(batch+1, epoch_loss/(batch+1), metric.result().numpy(), train_scores[-1], valid_scores[-1]), end="\r")
            
        train_scores.append(compute_spearmanr(train_trues, train_preds))
        
        # validation loop
        dropout_preds = []
        for _ in range(30):
            valid_preds, valid_trues = np.empty((0, 30), np.float32), np.empty((0, 30), np.float32)
            for (x_val, y_val) in valid_dataset:
                valid_preds = np.append(valid_preds, model(x_val, training=True).numpy(), axis=0) # note training = True for dropout
                valid_trues = np.append(valid_trues, y_val.numpy(), axis=0)
            dropout_preds.append(valid_preds)
        valid_snapshot_preds.append(np.average(dropout_preds, axis=0))
        valid_scores.append(compute_spearmanr(valid_trues, np.average(valid_snapshot_preds, axis=0)))
        
        # just to update current print before moving to next epoch
        print("batch {:03d} : train loss {:.3f} : train cosine {:.3f} : train spearman {:.3f} : valid spearman {:.3f}"
              .format(batch+1, epoch_loss/(batch+1), metric.result().numpy(), train_scores[-1], valid_scores[-1]), end="\r")
        
        # manually resetting metric
        metric.reset_states()
        
        # test loop 
        dropout_preds = []
        for _ in range(30):
            test_preds = np.empty((0, 30), np.float32)
            for x_test in test_dataset:
                test_preds = np.append(test_preds, model(x_test, training=True).numpy(), axis=0) # note training = True for dropout
            dropout_preds.append(test_preds)
        test_snapshot_preds.append(np.average(dropout_preds, axis=0))

        
    return valid_snapshot_preds, test_snapshot_preds, train_scores, valid_scores

input_units = [512, 5, 63] # use, cat, host input size
dense_units = [512, 256, 256, 256, 30] # for shared layer, non-shared layers, output layer
aux_units = 256 # for aux layers
dropout_rates = [0.3, 0.2, 0.2, 0.2] # for shared layer, non-shared layers
num_folds = 5
num_epochs = 12
learning_rate = 5e-4
batch_size = 64

model = None
optimizer = tf.keras.optimizers.Adam(learning_rate)
loss_function = tf.keras.losses.BinaryCrossentropy()
metric = tf.keras.metrics.CosineSimilarity()

gkf = GroupKFold(n_splits=num_folds).split(X=df_train.question_body, groups=df_train.question_body)

valid_fold_predictions = []
test_fold_predictions = []
train_fold_scores = []
valid_fold_scores = []
for fold, (train_idx, valid_idx) in enumerate(gkf):
    
    print("\n\nfold {:01d}".format(fold+1))
    model = NeuralNet(input_units, dense_units, dropout_rates, aux_units)

    train_dataset = create_dataset(
        (train_t[train_idx], train_q[train_idx], train_a[train_idx], 
         train_category[train_idx], train_host[train_idx]), 
         np.asarray(df_train[output_categories].iloc[train_idx]),
         batch_size)

    valid_dataset = create_dataset(
        (train_t[valid_idx], train_q[valid_idx], train_a[valid_idx], 
         train_category[valid_idx], train_host[valid_idx]), 
         np.asarray(df_train[output_categories].iloc[valid_idx]),
         batch_size)

    test_dataset = create_dataset(
        (test_t, test_q, test_a, 
         test_category, test_host),
         None,
         batch_size)
    
    valid_preds, test_preds, train_scores, valid_scores = train_and_predict(
        model, loss_function, optimizer, metric,
        train_dataset, valid_dataset, test_dataset, num_epochs)

    valid_fold_predictions.append(valid_preds)
    test_fold_predictions.append(test_preds)
    train_fold_scores.append(train_scores)
    valid_fold_scores.append(valid_scores)

fig, axes = plt.subplots(1, num_folds, figsize=(num_folds*5, num_folds))

for i, ax in enumerate(axes.reshape(-1)):
    ax.plot(np.linspace(0., len(train_fold_scores[i])-1, len(train_fold_scores[i])), 
            train_fold_scores[i], label='training')
    ax.plot(np.linspace(0., len(valid_fold_scores[i])-1, len(valid_fold_scores[i])), 
            valid_fold_scores[i], label='validation')
    ax.set_title("Fold {}".format(i+1))
    if i == 0:
        ax.set_ylabel("spearman rho")
    ax.set_xlabel("epochs")
    ax.legend()
# if snapshot preds:
def compute_final_predictions(test_fold_predictions):
    snapshot_averages = [np.average(test_fold_predictions[i], axis=0) for i in range(len(test_fold_predictions))]
    return np.mean(snapshot_averages, axis=0)

df_sub.iloc[:, 1:] = compute_final_predictions(test_fold_predictions)
df_sub.head()
# if not snapshot preds:
# df_sub.iloc[:, 1:] = np.mean(test_fold_predictions, axis=0)
# df_sub.head()
df_sub.to_csv("submission.csv", index=False)
