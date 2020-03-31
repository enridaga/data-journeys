
import sys
import scipy
import matplotlib
import numpy as np
import pandas as pd
print(sys.version)
print('pandas:', pd.__version__)
print('numpy:', np.__version__)
print('scipy:', scipy.__version__)
print('matplotlib:', matplotlib.__version__)
print('ok')
### matplotlib inline

import os
import time
import numpy as np
import pandas as pd
import scipy.signal as sg
from tqdm import tqdm_notebook
import matplotlib.pyplot as plt
print('ok')
DATA_DIR = r'../input'  # set for local environment!
TEST_DIR = r'../input/test'  # set for local environment!
print('ok')
# note '<ctrl> /'' to block uncomment
train_df = pd.read_csv(os.path.join(DATA_DIR, 'train.csv'), dtype={'acoustic_data': np.int16, 'time_to_failure': np.float32})
print(train_df.shape)
print('ok')
ld = os.listdir(TEST_DIR)
sizes = np.zeros(len(ld))

for i, f in enumerate(ld):
    df = pd.read_csv(os.path.join(TEST_DIR, f))
    sizes[i] = df.shape[0]

print(np.mean(sizes))  # all were 150,000
print(np.min(sizes))
print(np.max(sizes))
print('ok')
train_ad_sample_df = train_df['acoustic_data'].values[::100]
train_ttf_sample_df = train_df['time_to_failure'].values[::100]

def plot_acc_ttf_data(train_ad_sample_df, train_ttf_sample_df, title="Acoustic data and time to failure: 1% sampled data"):
    fig, ax1 = plt.subplots(figsize=(12, 8))
    plt.title(title)
    plt.plot(train_ad_sample_df, color='tab:orange')
    ax1.set_ylabel('acoustic data', color='tab:orange')
    plt.legend(['acoustic data'], loc=(0.01, 0.95))
    ax2 = ax1.twinx()
    plt.plot(train_ttf_sample_df, color='tab:blue')
    ax2.set_ylabel('time to failure', color='tab:blue')
    plt.legend(['time to failure'], loc=(0.01, 0.9))
    plt.grid(True)

plot_acc_ttf_data(train_ad_sample_df, train_ttf_sample_df)
del train_ad_sample_df
del train_ttf_sample_df
# plot 150k sample slices of the training data, matches size of test data (~0.375 seconds long)
# plots signal and decreasing time to the next quake
np.random.seed(2018)
rand_idxs = np.random.randint(0, 629145480-150000, size=16, dtype=np.int32)
f, axes = plt.subplots(4, 4, figsize=(14, 8))
i = 0
j = 0

for st_idx in rand_idxs:
    ad = train_df['acoustic_data'].values[st_idx: st_idx + 150000]
    ttf = train_df['time_to_failure'].values[st_idx: st_idx + 150000]
    
    axes[j][i].plot(ad, color='tab:orange')
    
    s = axes[j][i].twinx()
    s.plot(ttf, color='tab:blue')
    
    i += 1
    if i >= 4:
        i = 0
        j += 1

plt.tight_layout()
plt.show()
del ad, ttf
print('ok')
ttf_diff = train_df['time_to_failure'].diff()
ttf_diff = ttf_diff.loc[ttf_diff > 0]
print(ttf_diff.index)
# plot -150,000 to +30000 samples right around the earthquake
failure_idxs = [5656574,  50085878, 104677356, 138772453, 187641820, 218652630,
                245829585, 307838917, 338276287, 375377848, 419368880, 461811623,
                495800225, 528777115, 585568144, 621985673]
f, axes = plt.subplots(4, 4, figsize=(14, 8))
i = 0
j = 0

for idx in failure_idxs:
    ad = train_df['acoustic_data'].values[idx - 150000: idx + 30000]
    ttf = train_df['time_to_failure'].values[idx - 150000: idx + 30000]
    
    axes[j][i].plot(ad, color='tab:orange')
    
    s = axes[j][i].twinx()
    s.plot(ttf, color='tab:blue')
    
    i += 1
    if i >= 4:
        i = 0
        j += 1

plt.tight_layout()
plt.show()
del ad, ttf
f, axes = plt.subplots(4, 4, figsize=(14, 8))
i = 0
j = 0

for idx in failure_idxs:
    ad = train_df['acoustic_data'].values[idx - 2000000: idx + 30000]
    ttf = train_df['time_to_failure'].values[idx - 2000000: idx + 30000]
    
    axes[j][i].plot(ad, color='tab:orange')
    axes[j][i].set_xticklabels([])
    
    s = axes[j][i].twinx()
    s.plot(ttf, color='tab:blue')
    
    i += 1
    if i >= 4:
        i = 0
        j += 1

plt.tight_layout()
plt.show()
del ad, ttf
# plot test signals
ld = os.listdir(TEST_DIR)
ld = ld[32:48]
f, axes = plt.subplots(4, 4, figsize=(14, 8))
i = 0
j = 0
    
for sig_file in ld:
    sig = pd.read_csv(os.path.join(TEST_DIR, sig_file))['acoustic_data']
    axes[j][i].plot(sig, color='tab:orange')

    i += 1
    if i >= 4:
        i = 0
        j += 1

plt.tight_layout()
plt.show()
del sig
# plot frequency components of the signal
MAX_FREQ_IDX = 75000

signals = ld[0:12]
fig = plt.figure(figsize=(12, 5))

for i, signal in enumerate(signals):
    df = pd.read_csv(os.path.join(TEST_DIR, signal))
    ad = df['acoustic_data'].values
    ad = ad - np.mean(ad)  # remove DC component, otherwise it dominates the plot

    b, a = sg.butter(6, Wn=20000 / 75000)
    ad = sg.lfilter(b, a, ad)

    zc = np.fft.fft(ad)
    zc = zc[:75000]  # eliminate aliased portion of signal per Nyquist criteria

    realFFT = np.real(zc)
    imagFFT = np.imag(zc)
    magFFT = np.sqrt(realFFT ** 2 + imagFFT ** 2)

    plt.subplot(4, 3, i+1)
    plt.plot(magFFT, color='tab:blue')

plt.tight_layout()
plt.show()
# plot frequency components of the signal with a gentle window
import warnings
from scipy.signal import hann
warnings.filterwarnings("ignore")
MAX_FREQ_IDX = 75000
ld = os.listdir(TEST_DIR)
signals = ld[0:12]
fig = plt.figure(figsize=(12, 5))

for i, signal in enumerate(signals):
    df = pd.read_csv(os.path.join(TEST_DIR, signal))
    ad = df['acoustic_data'].values
    ad = ad - np.mean(ad)  # remove DC component, otherwise it dominates the plot

    hann_win = sg.hanning(M=24)
    ad_beg = ad[0: 12] * hann_win[0: 12]
    ad_end = ad[-12:] * hann_win[-12:]
    ad = np.concatenate((ad_beg, ad[12: -12], ad_end), axis=0)

    zc = np.fft.fft(ad)
    zc = zc[:75000]  # eliminate aliased portion of signal per Nyquist criteria

    realFFT = np.real(zc)
    imagFFT = np.imag(zc)
    magFFT = np.sqrt(realFFT ** 2 + imagFFT ** 2)

    plt.subplot(4, 3, i+1)
    plt.plot(magFFT, color='tab:blue')

plt.tight_layout()
plt.show()
import warnings
from scipy.signal import hann
warnings.filterwarnings("ignore")
MAX_FREQ_IDX = 75000
ld = os.listdir(TEST_DIR)
signals = ld[0:12]
fig = plt.figure(figsize=(12, 5))

for i, signal in enumerate(signals):
    df = pd.read_csv(os.path.join(TEST_DIR, signal))
    ad = df['acoustic_data'].values
    ad = ad - np.mean(ad)  # remove DC component, otherwise it dominates the plot

    hann_win = sg.hanning(M=24)
    ad_beg = ad[0: 12] * hann_win[0: 12]
    ad_end = ad[-12:] * hann_win[-12:]
    ad = np.concatenate((ad_beg, ad[12: -12], ad_end), axis=0)

    zc = np.fft.fft(ad)
    zc = zc[:75000]  # eliminate aliased portion of signal per Nyquist criteria

    realFFT = np.real(zc)
    imagFFT = np.imag(zc)
    phzFFT = np.arctan(imagFFT / realFFT)
    phzFFT[phzFFT == -np.inf] = -np.pi / 2.0
    phzFFT[phzFFT == np.inf] = np.pi / 2.0
    phzFFT = np.nan_to_num(phzFFT)

    plt.subplot(4, 3, i+1)
    plt.plot(phzFFT, color='tab:blue')

plt.tight_layout()
plt.show()
# determine is a signal contains a 'big peak' as defined by an absolute value more than 2000
ld = os.listdir(TEST_DIR)
peaks = np.zeros(len(ld))

for i, f in enumerate(ld):
    df = pd.read_csv(os.path.join(TEST_DIR, f))
    peaks[i] = df['acoustic_data'].abs().max()
    
peaks_lg = peaks[peaks >= 2000.0] 
print(peaks_lg.shape[0])
print(np.float32(peaks_lg.shape[0]) / np.float32(peaks.shape[0]) * 100.0)
print(np.float32(2624) / np.float32(4194) * 16)