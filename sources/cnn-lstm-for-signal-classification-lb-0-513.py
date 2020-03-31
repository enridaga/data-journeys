
import keras
import keras.backend as K
from keras.layers import LSTM,Dropout,Dense,TimeDistributed,Conv1D,MaxPooling1D,Flatten
from keras.models import Sequential
import tensorflow as tf
import gc
from numba import jit
from IPython.display import display, clear_output
from tqdm import tqdm
import matplotlib.pyplot as plt
### matplotlib inline
import seaborn as sns
import sys
sns.set_style("whitegrid")
import pyarrow.parquet as pq
import pandas as pd
import numpy as np
### %time 
train_set = pq.read_pandas('../input/train.parquet').to_pandas()
### %time
meta_train = pd.read_csv('../input/metadata_train.csv')
@jit('float32(float32[:,:], int32)')
def feature_extractor(x, n_part=1000):
    lenght = len(x)
    pool = np.int32(np.ceil(lenght/n_part))
    output = np.zeros((n_part,))
    for j, i in enumerate(range(0,lenght, pool)):
        if i+pool < lenght:
            k = x[i:i+pool]
        else:
            k = x[i:]
        output[j] = np.max(k, axis=0) - np.min(k, axis=0)
    return output
x_train = []
y_train = []
for i in tqdm(meta_train.signal_id):
    idx = meta_train.loc[meta_train.signal_id==i, 'signal_id'].values.tolist()
    y_train.append(meta_train.loc[meta_train.signal_id==i, 'target'].values)
    x_train.append(abs(feature_extractor(train_set.iloc[:, idx].values, n_part=400)))
del train_set; gc.collect()
y_train = np.array(y_train).reshape(-1,)
X_train = np.array(x_train).reshape(-1,x_train[0].shape[0])
def keras_auc(y_true, y_pred):
    auc = tf.metrics.auc(y_true, y_pred)[1]
    K.get_session().run(tf.local_variables_initializer())
    return auc
n_signals = 1 #So far each instance is one signal. We will diversify them in next step
n_outputs = 1 #Binary Classification
#Build the model
verbose, epochs, batch_size = True, 15, 16
n_steps, n_length = 40, 10
X_train = X_train.reshape((X_train.shape[0], n_steps, n_length, n_signals))
# define model
model = Sequential()
model.add(TimeDistributed(Conv1D(filters=64, kernel_size=3, activation='relu'), input_shape=(None,n_length,n_signals)))
model.add(TimeDistributed(Conv1D(filters=64, kernel_size=3, activation='relu')))
model.add(TimeDistributed(Dropout(0.5)))
model.add(TimeDistributed(MaxPooling1D(pool_size=2)))
model.add(TimeDistributed(Flatten()))
model.add(LSTM(100))
model.add(Dropout(0.5))
model.add(Dense(100, activation='relu'))
model.add(Dense(n_outputs, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=[keras_auc])
model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, verbose=verbose)
model.save_weights('model1.hdf5')
#%%time
#test_set = pq.read_pandas('../input/test.parquet').to_pandas()
#%%time
#meta_test = pd.read_csv('../input/metadata_test.csv')
#x_test = []
#for i in tqdm(meta_test.signal_id.values):
#    idx=i-8712
#    clear_output(wait=True)
#    x_test.append(abs(feature_extractor(test_set.iloc[:, idx].values, n_part=400)))
#del test_set; gc.collect()
#X_test = x_test.reshape((x_test.shape[0], n_steps, n_length, n_signals))
#preds = model.predict(X_test)
#threshpreds = (preds>0.5)*1
#sub = pd.read_csv('../input/sample_submission.csv')
#sub.target = threshpreds
#sub.to_csv('first_sub.csv',index=False)
#Gave me an LB score of 0.450
#Both numpy and scipy has utilities for FFT which is an endlessly useful algorithm
from numpy.fft import *
from scipy import fftpack
### %time 
train_set = pq.read_pandas('../input/train.parquet').to_pandas()
#FFT to filter out HF components and get main signal profile
def low_pass(s, threshold=1e4):
    fourier = rfft(s)
    frequencies = rfftfreq(s.size, d=2e-2/s.size)
    fourier[frequencies > threshold] = 0
    return irfft(fourier)
def phase_indices(signal_num):
    phase1 = 3*signal_num
    phase2 = 3*signal_num + 1
    phase3 = 3*signal_num + 2
    return phase1,phase2,phase3
s_id = 14
p1,p2,p3 = phase_indices(s_id)
plt.figure(figsize=(10,5))
plt.title('Signal %d / Target:%d'%(s_id,meta_train[meta_train.id_measurement==s_id].target.unique()[0]))
plt.plot(train_set.iloc[:,p1])
plt.plot(train_set.iloc[:,p2])
plt.plot(train_set.iloc[:,p3])
lf_signal_1 = low_pass(train_set.iloc[:,p1])
lf_signal_2 = low_pass(train_set.iloc[:,p2])
lf_signal_3 = low_pass(train_set.iloc[:,p3])
plt.figure(figsize=(10,5))
plt.title('De-noised Signal %d / Target:%d'%(s_id,meta_train[meta_train.id_measurement==s_id].target.unique()[0]))
plt.plot(lf_signal_1)
plt.plot(lf_signal_2)
plt.plot(lf_signal_3)
plt.figure(figsize=(10,5))
plt.title('Signal %d AbsVal / Target: %d'%(s_id,meta_train[meta_train.id_measurement==s_id].target.unique()[0]))
plt.plot(np.abs(lf_signal_1))
plt.plot(np.abs(lf_signal_2))
plt.plot(np.abs(lf_signal_3))
plt.figure(figsize=(10,5))
plt.title('Signal %d / Target: %d'%(s_id,meta_train[meta_train.id_measurement==s_id].target.unique()[0]))
plt.plot(lf_signal_1)
plt.plot(lf_signal_2)
plt.plot(lf_signal_3)
plt.plot((np.abs(lf_signal_1)+np.abs(lf_signal_2)+np.abs(lf_signal_3)))
plt.legend(['phase 1','phase 2','phase 3','DC Component'],loc=1)
###Filter out low frequencies from the signal to get HF characteristics
def high_pass(s, threshold=1e7):
    fourier = rfft(s)
    frequencies = rfftfreq(s.size, d=2e-2/s.size)
    fourier[frequencies < threshold] = 0
    return irfft(fourier)
hf_signal_1 = high_pass(train_set.iloc[:,p1])
hf_signal_2 = high_pass(train_set.iloc[:,p2])
hf_signal_3 = high_pass(train_set.iloc[:,p3])
plt.figure(figsize=(10,5))
plt.title('Signal %d / Target:%d'%(s_id,meta_train[meta_train.id_measurement==s_id].target.unique()[0]))
plt.plot(hf_signal_1)
#plt.plot(hf_signal_2)
#plt.plot(hf_signal_3)
signal = train_set.iloc[:,p1]
### %time
x = signal
X = fftpack.fft(x,n=400)
freqs = fftpack.fftfreq(n=400,d=2e-2/x.size) 
plt.plot(x)
fig, ax = plt.subplots()
ax.set_title('Full Spectrum with Scipy')
ax.set_xlabel('Frequency in Hertz [Hz]')
ax.set_ylabel('Frequency Domain (Spectrum) Magnitude')
ax.stem(freqs[1:], np.abs(X)[1:])
### %time
x = high_pass(train_set.iloc[:,p1])
X = fftpack.fft(x,n=400)
freqs = fftpack.fftfreq(n=400,d=2e-2/x.size) 
plt.plot(x)
fig, ax = plt.subplots()
ax.set_title('High Frequency Spectrum')
ax.set_xlabel('Frequency in Hertz [Hz]')
ax.set_ylabel('Frequency Domain (Spectrum) Magnitude')
ax.stem(freqs[1:], np.abs(X)[1:])
### %time
x = low_pass(signal)
X = fftpack.fft(x,n=400)
freqs = fftpack.fftfreq(n=400,d=2e-2/x.size) 
plt.plot(x)
fig, ax = plt.subplots()
ax.set_title('Low Frequency Spectrum')
ax.set_xlabel('Frequency in Hertz [Hz]')
ax.set_ylabel('Frequency Domain (Spectrum) Magnitude')
ax.stem(freqs[1:], np.abs(X)[1:])
p1,p2,p3 = phase_indices(100)
signal = train_set.iloc[:,p1]
from scipy import signal as sgn
M = 1024
rate = 1/(2e-2/signal.size)

freqs, times, Sx = sgn.spectrogram(signal.values, fs=rate, window='hanning',
                                      nperseg=1024, noverlap=M - 100,
                                      detrend='constant', scaling='spectrum')

f, ax = plt.subplots(figsize=(10, 5))
ax.set_ylabel('Frequency [Hz]')
ax.set_xlabel('Time(s)')
ax.set_title('Spectogram')
ax.pcolormesh(times, freqs, np.log10(Sx), cmap='viridis')
x_train_lp = []
x_train_hp = []
x_train_dc = []
for i in meta_train.signal_id:
    idx = meta_train.loc[meta_train.signal_id==i, 'signal_id'].values.tolist()
    clear_output(wait=True)
    display(idx)
    hp = high_pass(train_set.iloc[:, idx[0]])
    lp = low_pass(train_set.iloc[:, idx[0]])
    meas_id = meta_train.id_measurement[meta_train.signal_id==idx].values[0]
    p1,p2,p3=phase_indices(meas_id)
    lf_signal_1,lf_signal_2,lf_signal_3 = low_pass(train_set.iloc[:,p1]), low_pass(train_set.iloc[:,p2]), low_pass(train_set.iloc[:,p3])
    dc = np.abs(lf_signal_1)+np.abs(lf_signal_2)+np.abs(lf_signal_3)
    x_train_lp.append(abs(feature_extractor(lp, n_part=400)))
    x_train_hp.append(abs(feature_extractor(hp, n_part=400)))
    x_train_dc.append(abs(feature_extractor(dc, n_part=400)))
del train_set; gc.collect()
#x_test_lp = []
#x_test_hp = []
#x_test_dc = []
#for i in tqdm(meta_test.signal_id):
#    idx = idx=i-8712
#    clear_output(wait=True)
#    #display(idx)
#    hp = high_pass(test_set.iloc[:, idx])
#    lp = low_pass(test_set.iloc[:, idx])
#    meas_id = meta_test.id_measurement[meta_test.signal_id==i].values[0]
#    p1,p2,p3=phase_indices(meas_id)
#    lf_signal_1,lf_signal_2,lf_signal_3 = low_pass(test_set.iloc[:,p1-8712]), low_pass(test_set.iloc[:,p2-8712]), low_pass(test_set.iloc[:,p3-8712])
#    dc = np.abs(lf_signal_1)+np.abs(lf_signal_2)+np.abs(lf_signal_3)
#    x_test_lp.append(abs(feature_extractor(lp, n_part=400)))
#    x_test_hp.append(abs(feature_extractor(hp, n_part=400)))
#    x_test_dc.append(abs(feature_extractor(dc, n_part=400)))
x_train = np.array(x_train).reshape(-1,x_train[0].shape[0])
x_train_lp = np.array(x_train).reshape(-1,x_train_lp[0].shape[0])
x_train_hp = np.array(x_train).reshape(-1,x_train_hp[0].shape[0])
x_train_dc = np.array(x_train).reshape(-1,x_train_dc[0].shape[0])
#x_test = np.array(x_test).reshape(-1,x_test[0].shape[0])
#x_test_lp = np.array(x_test).reshape(-1,x_test_lp[0].shape[0])
#x_test_hp = np.array(x_test).reshape(-1,x_test_hp[0].shape[0])
#x_test_dc = np.array(x_test).reshape(-1,x_test_dc[0].shape[0])
train = np.dstack((x_train,x_train_lp,x_train_hp,x_train_dc))
#test = np.dstack((x_test,x_test_lp,x_test_hp,x_test_dc))
y_train = np.array(y_train).reshape(-1,)
verbose, epochs, batch_size = True, 15, 16
n_signals,n_steps, n_length = 4,40, 10
train = train.reshape((train.shape[0], n_steps, n_length, n_signals))
# define model
model = Sequential()
model.add(TimeDistributed(Conv1D(filters=64, kernel_size=3, activation='relu'), input_shape=(None,n_length,n_signals)))
model.add(TimeDistributed(Conv1D(filters=64, kernel_size=3, activation='relu')))
model.add(TimeDistributed(Dropout(0.5)))
model.add(TimeDistributed(MaxPooling1D(pool_size=2)))
model.add(TimeDistributed(Flatten()))
model.add(LSTM(100))
model.add(Dropout(0.5))
model.add(Dense(100, activation='relu'))
model.add(Dense(n_outputs, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=[keras_auc])
# fit network
model.fit(train, y_train, epochs=epochs, batch_size=batch_size, verbose=verbose)
model.save_weights('model2.hdf5')
#X_test = test.reshape((test.shape[0], n_steps, n_length, n_signals))
#preds = model.predict(X_test)
#threshpreds = (preds>0.5)*1
#sub = pd.read_csv('data/sample_submission.csv')
#sub.target = threshpreds
#sub.to_csv('submissions/second_sub.csv',index=False)