
import os
from random import shuffle
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats
import numba
import warnings
sns.set()
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=RuntimeWarning)

@numba.jit
def get_stats(arr):
    """Memory efficient stats (min, max and mean). """
    size  = len(arr)
    min_value = max_value = arr[0]
    mean_value = 0
    for i in numba.prange(size):
        if arr[i] < min_value:
            min_value = arr[i]
        if arr[i] > max_value:
            max_value = arr[i]
        mean_value += arr[i]
    return min_value, max_value, mean_value/size

@numba.jit
def get_diff(arr, threshold):
    """Find uniques ttf differences between rows. """
    diff_list = []
    size  = len(arr)
    uniques = 0
    for i in numba.prange(size - 1):
        diff = abs(arr[i+1] - arr[i])
        if uniques == 0:
            diff_list.append(diff)
            uniques += 1
        else:
            for j in numba.prange(uniques):
                if abs(diff - diff_list[j]) < threshold or abs(diff - diff_list[j]) > 1:
                    break
            else:
                diff_list.append(diff)
                uniques += 1
    return diff_list
print(os.listdir("../input/"))
test_folder_files = os.listdir("../input/test")
print(test_folder_files[:10])  # print first 10
print("\nNumber of files in the test folder", len(test_folder_files))
sample_sub = pd.read_csv('../input/sample_submission.csv')
print("Submission shape", sample_sub.shape)
sample_sub.head()
train = pd.read_csv('../input/train.csv', dtype={'acoustic_data': np.int16, 'time_to_failure': np.float64})
print("train shape", train.shape)
pd.set_option("display.precision", 15)  # show more decimals
train.head()
pd.set_option("display.precision", 8)
train.acoustic_data.describe()
train_sample = train.sample(frac=0.01)
plt.figure(figsize=(10,5))
plt.title("Acoustic data distribution")
ax = sns.distplot(train_sample.acoustic_data, label='Train (1% sample)')
train_sample = train.sample(frac=0.01)
plt.figure(figsize=(10,5))
plt.title("Acoustic data distribution")
tmp = train_sample.acoustic_data[train_sample.acoustic_data.between(-25, 25)]
ax = sns.distplot(tmp, label='Train (1% sample)', kde=False, fit=stats.norm)
tmin, tmax, tmean = get_stats(train.time_to_failure.values)
print("min value: {:.6f}, max value: {:.2f}, mean: {:.4f}".format(tmin, tmax, tmean))
plt.figure(figsize=(10,5))
plt.title("Time to failure distribution")
ax = sns.kdeplot(train_sample.time_to_failure, label='Train (1% sample)')
def single_timeseries(final_idx, init_idx=0, step=1, title="",
                      color1='orange', color2='blue'):
    idx = [i for i in range(init_idx, final_idx, step)]
    fig, ax1 = plt.subplots(figsize=(10, 5))
    fig.suptitle(title, fontsize=14)
    
    ax2 = ax1.twinx()
    ax1.set_xlabel('index')
    ax1.set_ylabel('Acoustic data')
    ax2.set_ylabel('Time to failure')
    p1 = sns.lineplot(data=train.iloc[idx].acoustic_data.values, ax=ax1, color=color1)
    p2 = sns.lineplot(data=train.iloc[idx].time_to_failure.values, ax=ax2, color=color2)


def double_timeseries(final_idx1, final_idx2, init_idx1=0, init_idx2=0, step=1, title=""):
    idx1 = [i for i in range(init_idx1, final_idx1, step)]
    idx2 = [i for i in range(init_idx2, final_idx2, step)]
    
    fig, (ax1a, ax2a) = plt.subplots(1,2, figsize=(12,5))
    fig.subplots_adjust(wspace=0.4)
    ax1b = ax1a.twinx()
    ax2b = ax2a.twinx()
    
    ax1a.set_xlabel('index')
    ax1a.set_ylabel('Acoustic data')
    ax2a.set_ylabel('Time to failure')
    p1 = sns.lineplot(data=train.iloc[idx1].acoustic_data.values, ax=ax1a, color='orange')
    p2 = sns.lineplot(data=train.iloc[idx1].time_to_failure.values, ax=ax1b, color='blue')
    
    p3 = sns.lineplot(data=train.iloc[idx2].acoustic_data.values, ax=ax2a, color='orange')
    p4 = sns.lineplot(data=train.iloc[idx2].time_to_failure.values, ax=ax2b, color='blue')
    
single_timeseries(1000, title="First thousand rows")
single_timeseries(10000, title="Ten thousand rows")
single_timeseries(10000000, step=10, title="Ten million rows")
single_timeseries(629145000, step=1000, title="All training data")
single_timeseries(629145000, step=1000, title="All training data", color2='white')
peaks = train[train.acoustic_data.abs() > 500]
peaks.time_to_failure.describe()
plt.figure(figsize=(10,5))
plt.title("Cumulative distribution - time to failure with high signal")
ax = sns.distplot(peaks.time_to_failure, hist_kws=dict(cumulative=True), kde_kws=dict(cumulative=True))
unique_diff = get_diff(train.time_to_failure.values, threshold=1e-10)
plt.figure(figsize=(10,5))
plt.title("Time to failure - unique differences")
ax = sns.scatterplot(x=range(len(unique_diff)), y=unique_diff, color='red')
pd.set_option("display.precision", 4)
test1 = pd.read_csv('../input/test/seg_37669c.csv', dtype='int16')
print(test1.describe())
plt.figure(figsize=(10,5))
plt.title("Acoustic data distribution")
ax = sns.distplot(test1.acoustic_data, label='seg_37669c', kde=False)
fig, axis = plt.subplots(5, 2, figsize=(12,14))
shuffle(test_folder_files)
xrow = xcol = 0
for f in test_folder_files[:10]:
    tmp = pd.read_csv('../input/test/{}'.format(f), dtype='int16')
    ax = sns.distplot(tmp.acoustic_data, label=f.replace('.csv',''), ax=axis[xrow][xcol], kde=False)
    if xcol == 0:
        xcol += 1
    else:
        xcol = 0
        xrow += 1
fig, axis = plt.subplots(5, 2, figsize=(12,14))
xrow = xcol = 0
for f in test_folder_files[:10]:
    tmp = pd.read_csv('../input/test/{}'.format(f), dtype='int16')
    ax = sns.lineplot(data=tmp.acoustic_data.values,
                      label=f.replace('.csv',''),
                      ax=axis[xrow][xcol],
                      color='orange')
    if xcol == 0:
        xcol += 1
    else:
        xcol = 0
        xrow += 1
rolling_mean = []
rolling_std = []
last_time = []
init_idx = 0
for _ in range(4194):  # 629M / 150k = 4194
    x = train.iloc[init_idx:init_idx + 150000]
    last_time.append(x.time_to_failure.values[-1])
    rolling_mean.append(x.acoustic_data.abs().mean())
    rolling_std.append(x.acoustic_data.abs().std())
    init_idx += 150000
    
rolling_mean = np.array(rolling_mean)
last_time = np.array(last_time)

# plot rolling mean
fig, ax1 = plt.subplots(figsize=(10, 5))
fig.suptitle('Mean for chunks with 150k samples of training data', fontsize=14)

ax2 = ax1.twinx()
ax1.set_xlabel('index')
ax1.set_ylabel('Acoustic data')
ax2.set_ylabel('Time to failure')
p1 = sns.lineplot(data=rolling_mean, ax=ax1, color='orange')
p2 = sns.lineplot(data=last_time, ax=ax2, color='gray')
# plot rolling mean
fig, ax1 = plt.subplots(figsize=(10, 5))
fig.suptitle('Mean (< 8) for chunks of 150k samples', fontsize=14)

ax2 = ax1.twinx()
ax1.set_xlabel('index')
ax1.set_ylabel('Acoustic data')
ax2.set_ylabel('Time to failure')
p1 = sns.lineplot(data=rolling_mean[rolling_mean < 8], ax=ax1, color='orange')
p2 = sns.lineplot(data=last_time, ax=ax2, color='gray')
frame = pd.DataFrame({'rolling_std': rolling_std, 'time': np.around(last_time, 1)})
s = frame.groupby('time').rolling_std.mean()
s = s[s < 20]  # remove one outlier
plt.figure(figsize=(10, 5))
plt.title("Std for chunks with 150k samples of training data")
plt.xlabel("Time to failure")
plt.ylabel("Acoustic data")
ax = sns.lineplot(x=s.index, y=s.values)