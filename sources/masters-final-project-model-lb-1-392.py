
import os
import time
import warnings
import traceback
import numpy as np
import pandas as pd
from scipy import stats
import scipy.signal as sg
import multiprocessing as mp
from scipy.signal import hann
from scipy.signal import hilbert
from scipy.signal import convolve
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler

import xgboost as xgb
import lightgbm as lgb
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error

from tqdm import tqdm
warnings.filterwarnings("ignore")
OUTPUT_DIR = r'd:\#earthquake\final_model'  # set for local environment
DATA_DIR = r'd:\#earthquake\data'  # set for local environment

SIG_LEN = 150000
NUM_SEG_PER_PROC = 4000
NUM_THREADS = 6

NY_FREQ_IDX = 75000  # the test signals are 150k samples long, Nyquist is thus 75k.
CUTOFF = 18000
MAX_FREQ_IDX = 20000
FREQ_STEP = 2500
def split_raw_data():
    df = pd.read_csv(os.path.join(DATA_DIR, 'train.csv'))

    max_start_index = len(df.index) - SIG_LEN
    slice_len = int(max_start_index / 6)

    for i in range(NUM_THREADS):
        print('working', i)
        df0 = df.iloc[slice_len * i: (slice_len * (i + 1)) + SIG_LEN]
        df0.to_csv(os.path.join(DATA_DIR, 'raw_data_%d.csv' % i), index=False)
        del df0

    del df
def build_rnd_idxs():
    rnd_idxs = np.zeros(shape=(NUM_THREADS, NUM_SEG_PER_PROC), dtype=np.int32)
    max_start_idx = 100000000

    for i in range(NUM_THREADS):
        np.random.seed(5591 + i)
        start_indices = np.random.randint(0, max_start_idx, size=NUM_SEG_PER_PROC, dtype=np.int32)
        rnd_idxs[i, :] = start_indices

    for i in range(NUM_THREADS):
        print(rnd_idxs[i, :8])
        print(rnd_idxs[i, -8:])
        print(min(rnd_idxs[i,:]), max(rnd_idxs[i,:]))

    np.savetxt(fname=os.path.join(OUTPUT_DIR, 'start_indices_4k.csv'), X=np.transpose(rnd_idxs), fmt='%d', delimiter=',')
def add_trend_feature(arr, abs_values=False):
    idx = np.array(range(len(arr)))
    if abs_values:
        arr = np.abs(arr)
    lr = LinearRegression()
    lr.fit(idx.reshape(-1, 1), arr)
    return lr.coef_[0]

def classic_sta_lta(x, length_sta, length_lta):
    sta = np.cumsum(x ** 2)
    # Convert to float
    sta = np.require(sta, dtype=np.float)
    # Copy for LTA
    lta = sta.copy()
    # Compute the STA and the LTA
    sta[length_sta:] = sta[length_sta:] - sta[:-length_sta]
    sta /= length_sta
    lta[length_lta:] = lta[length_lta:] - lta[:-length_lta]
    lta /= length_lta
    # Pad zeros
    sta[:length_lta - 1] = 0
    # Avoid division by zero by setting zero values to tiny float
    dtiny = np.finfo(0.0).tiny
    idx = lta < dtiny
    lta[idx] = dtiny
    return sta / lta
def des_bw_filter_lp(cutoff=CUTOFF):  # low pass filter
    b, a = sg.butter(4, Wn=cutoff/NY_FREQ_IDX)
    return b, a

def des_bw_filter_hp(cutoff=CUTOFF):  # high pass filter
    b, a = sg.butter(4, Wn=cutoff/NY_FREQ_IDX, btype='highpass')
    return b, a

def des_bw_filter_bp(low, high):  # band pass filter
    b, a = sg.butter(4, Wn=(low/NY_FREQ_IDX, high/NY_FREQ_IDX), btype='bandpass')
    return b, a
def create_features(seg_id, seg, X, st, end):
    try:
        X.loc[seg_id, 'seg_id'] = np.int32(seg_id)
        X.loc[seg_id, 'seg_start'] = np.int32(st)
        X.loc[seg_id, 'seg_end'] = np.int32(end)
    except:
        pass

    xc = pd.Series(seg['acoustic_data'].values)
    xcdm = xc - np.mean(xc)

    b, a = des_bw_filter_lp(cutoff=18000)
    xcz = sg.lfilter(b, a, xcdm)

    zc = np.fft.fft(xcz)
    zc = zc[:MAX_FREQ_IDX]

    # FFT transform values
    realFFT = np.real(zc)
    imagFFT = np.imag(zc)

    freq_bands = [x for x in range(0, MAX_FREQ_IDX, FREQ_STEP)]
    magFFT = np.sqrt(realFFT ** 2 + imagFFT ** 2)
    phzFFT = np.arctan(imagFFT / realFFT)
    phzFFT[phzFFT == -np.inf] = -np.pi / 2.0
    phzFFT[phzFFT == np.inf] = np.pi / 2.0
    phzFFT = np.nan_to_num(phzFFT)

    for freq in freq_bands:
        X.loc[seg_id, 'FFT_Mag_01q%d' % freq] = np.quantile(magFFT[freq: freq + FREQ_STEP], 0.01)
        X.loc[seg_id, 'FFT_Mag_10q%d' % freq] = np.quantile(magFFT[freq: freq + FREQ_STEP], 0.1)
        X.loc[seg_id, 'FFT_Mag_90q%d' % freq] = np.quantile(magFFT[freq: freq + FREQ_STEP], 0.9)
        X.loc[seg_id, 'FFT_Mag_99q%d' % freq] = np.quantile(magFFT[freq: freq + FREQ_STEP], 0.99)
        X.loc[seg_id, 'FFT_Mag_mean%d' % freq] = np.mean(magFFT[freq: freq + FREQ_STEP])
        X.loc[seg_id, 'FFT_Mag_std%d' % freq] = np.std(magFFT[freq: freq + FREQ_STEP])
        X.loc[seg_id, 'FFT_Mag_max%d' % freq] = np.max(magFFT[freq: freq + FREQ_STEP])

        X.loc[seg_id, 'FFT_Phz_mean%d' % freq] = np.mean(phzFFT[freq: freq + FREQ_STEP])
        X.loc[seg_id, 'FFT_Phz_std%d' % freq] = np.std(phzFFT[freq: freq + FREQ_STEP])

    X.loc[seg_id, 'FFT_Rmean'] = realFFT.mean()
    X.loc[seg_id, 'FFT_Rstd'] = realFFT.std()
    X.loc[seg_id, 'FFT_Rmax'] = realFFT.max()
    X.loc[seg_id, 'FFT_Rmin'] = realFFT.min()
    X.loc[seg_id, 'FFT_Imean'] = imagFFT.mean()
    X.loc[seg_id, 'FFT_Istd'] = imagFFT.std()
    X.loc[seg_id, 'FFT_Imax'] = imagFFT.max()
    X.loc[seg_id, 'FFT_Imin'] = imagFFT.min()

    X.loc[seg_id, 'FFT_Rmean_first_6000'] = realFFT[:6000].mean()
    X.loc[seg_id, 'FFT_Rstd__first_6000'] = realFFT[:6000].std()
    X.loc[seg_id, 'FFT_Rmax_first_6000'] = realFFT[:6000].max()
    X.loc[seg_id, 'FFT_Rmin_first_6000'] = realFFT[:6000].min()
    X.loc[seg_id, 'FFT_Rmean_first_18000'] = realFFT[:18000].mean()
    X.loc[seg_id, 'FFT_Rstd_first_18000'] = realFFT[:18000].std()
    X.loc[seg_id, 'FFT_Rmax_first_18000'] = realFFT[:18000].max()
    X.loc[seg_id, 'FFT_Rmin_first_18000'] = realFFT[:18000].min()

    del xcz
    del zc

    b, a = des_bw_filter_lp(cutoff=2500)
    xc0 = sg.lfilter(b, a, xcdm)

    b, a = des_bw_filter_bp(low=2500, high=5000)
    xc1 = sg.lfilter(b, a, xcdm)

    b, a = des_bw_filter_bp(low=5000, high=7500)
    xc2 = sg.lfilter(b, a, xcdm)

    b, a = des_bw_filter_bp(low=7500, high=10000)
    xc3 = sg.lfilter(b, a, xcdm)

    b, a = des_bw_filter_bp(low=10000, high=12500)
    xc4 = sg.lfilter(b, a, xcdm)

    b, a = des_bw_filter_bp(low=12500, high=15000)
    xc5 = sg.lfilter(b, a, xcdm)

    b, a = des_bw_filter_bp(low=15000, high=17500)
    xc6 = sg.lfilter(b, a, xcdm)

    b, a = des_bw_filter_bp(low=17500, high=20000)
    xc7 = sg.lfilter(b, a, xcdm)

    b, a = des_bw_filter_hp(cutoff=20000)
    xc8 = sg.lfilter(b, a, xcdm)

    sigs = [xc, pd.Series(xc0), pd.Series(xc1), pd.Series(xc2), pd.Series(xc3),
            pd.Series(xc4), pd.Series(xc5), pd.Series(xc6), pd.Series(xc7), pd.Series(xc8)]

    for i, sig in enumerate(sigs):
        X.loc[seg_id, 'mean_%d' % i] = sig.mean()
        X.loc[seg_id, 'std_%d' % i] = sig.std()
        X.loc[seg_id, 'max_%d' % i] = sig.max()
        X.loc[seg_id, 'min_%d' % i] = sig.min()

        X.loc[seg_id, 'mean_change_abs_%d' % i] = np.mean(np.diff(sig))
        X.loc[seg_id, 'mean_change_rate_%d' % i] = np.mean(np.nonzero((np.diff(sig) / sig[:-1]))[0])
        X.loc[seg_id, 'abs_max_%d' % i] = np.abs(sig).max()
        X.loc[seg_id, 'abs_min_%d' % i] = np.abs(sig).min()

        X.loc[seg_id, 'std_first_50000_%d' % i] = sig[:50000].std()
        X.loc[seg_id, 'std_last_50000_%d' % i] = sig[-50000:].std()
        X.loc[seg_id, 'std_first_10000_%d' % i] = sig[:10000].std()
        X.loc[seg_id, 'std_last_10000_%d' % i] = sig[-10000:].std()

        X.loc[seg_id, 'avg_first_50000_%d' % i] = sig[:50000].mean()
        X.loc[seg_id, 'avg_last_50000_%d' % i] = sig[-50000:].mean()
        X.loc[seg_id, 'avg_first_10000_%d' % i] = sig[:10000].mean()
        X.loc[seg_id, 'avg_last_10000_%d' % i] = sig[-10000:].mean()

        X.loc[seg_id, 'min_first_50000_%d' % i] = sig[:50000].min()
        X.loc[seg_id, 'min_last_50000_%d' % i] = sig[-50000:].min()
        X.loc[seg_id, 'min_first_10000_%d' % i] = sig[:10000].min()
        X.loc[seg_id, 'min_last_10000_%d' % i] = sig[-10000:].min()

        X.loc[seg_id, 'max_first_50000_%d' % i] = sig[:50000].max()
        X.loc[seg_id, 'max_last_50000_%d' % i] = sig[-50000:].max()
        X.loc[seg_id, 'max_first_10000_%d' % i] = sig[:10000].max()
        X.loc[seg_id, 'max_last_10000_%d' % i] = sig[-10000:].max()

        X.loc[seg_id, 'max_to_min_%d' % i] = sig.max() / np.abs(sig.min())
        X.loc[seg_id, 'max_to_min_diff_%d' % i] = sig.max() - np.abs(sig.min())
        X.loc[seg_id, 'count_big_%d' % i] = len(sig[np.abs(sig) > 500])
        X.loc[seg_id, 'sum_%d' % i] = sig.sum()

        X.loc[seg_id, 'mean_change_rate_first_50000_%d' % i] = np.mean(np.nonzero((np.diff(sig[:50000]) / sig[:50000][:-1]))[0])
        X.loc[seg_id, 'mean_change_rate_last_50000_%d' % i] = np.mean(np.nonzero((np.diff(sig[-50000:]) / sig[-50000:][:-1]))[0])
        X.loc[seg_id, 'mean_change_rate_first_10000_%d' % i] = np.mean(np.nonzero((np.diff(sig[:10000]) / sig[:10000][:-1]))[0])
        X.loc[seg_id, 'mean_change_rate_last_10000_%d' % i] = np.mean(np.nonzero((np.diff(sig[-10000:]) / sig[-10000:][:-1]))[0])

        X.loc[seg_id, 'q95_%d' % i] = np.quantile(sig, 0.95)
        X.loc[seg_id, 'q99_%d' % i] = np.quantile(sig, 0.99)
        X.loc[seg_id, 'q05_%d' % i] = np.quantile(sig, 0.05)
        X.loc[seg_id, 'q01_%d' % i] = np.quantile(sig, 0.01)

        X.loc[seg_id, 'abs_q95_%d' % i] = np.quantile(np.abs(sig), 0.95)
        X.loc[seg_id, 'abs_q99_%d' % i] = np.quantile(np.abs(sig), 0.99)
        X.loc[seg_id, 'abs_q05_%d' % i] = np.quantile(np.abs(sig), 0.05)
        X.loc[seg_id, 'abs_q01_%d' % i] = np.quantile(np.abs(sig), 0.01)

        X.loc[seg_id, 'trend_%d' % i] = add_trend_feature(sig)
        X.loc[seg_id, 'abs_trend_%d' % i] = add_trend_feature(sig, abs_values=True)
        X.loc[seg_id, 'abs_mean_%d' % i] = np.abs(sig).mean()
        X.loc[seg_id, 'abs_std_%d' % i] = np.abs(sig).std()

        X.loc[seg_id, 'mad_%d' % i] = sig.mad()
        X.loc[seg_id, 'kurt_%d' % i] = sig.kurtosis()
        X.loc[seg_id, 'skew_%d' % i] = sig.skew()
        X.loc[seg_id, 'med_%d' % i] = sig.median()

        X.loc[seg_id, 'Hilbert_mean_%d' % i] = np.abs(hilbert(sig)).mean()
        X.loc[seg_id, 'Hann_window_mean'] = (convolve(xc, hann(150), mode='same') / sum(hann(150))).mean()

        X.loc[seg_id, 'classic_sta_lta1_mean_%d' % i] = classic_sta_lta(sig, 500, 10000).mean()
        X.loc[seg_id, 'classic_sta_lta2_mean_%d' % i] = classic_sta_lta(sig, 5000, 100000).mean()
        X.loc[seg_id, 'classic_sta_lta3_mean_%d' % i] = classic_sta_lta(sig, 3333, 6666).mean()
        X.loc[seg_id, 'classic_sta_lta4_mean_%d' % i] = classic_sta_lta(sig, 10000, 25000).mean()

        X.loc[seg_id, 'Moving_average_700_mean_%d' % i] = sig.rolling(window=700).mean().mean(skipna=True)
        X.loc[seg_id, 'Moving_average_1500_mean_%d' % i] = sig.rolling(window=1500).mean().mean(skipna=True)
        X.loc[seg_id, 'Moving_average_3000_mean_%d' % i] = sig.rolling(window=3000).mean().mean(skipna=True)
        X.loc[seg_id, 'Moving_average_6000_mean_%d' % i] = sig.rolling(window=6000).mean().mean(skipna=True)

        ewma = pd.Series.ewm
        X.loc[seg_id, 'exp_Moving_average_300_mean_%d' % i] = ewma(sig, span=300).mean().mean(skipna=True)
        X.loc[seg_id, 'exp_Moving_average_3000_mean_%d' % i] = ewma(sig, span=3000).mean().mean(skipna=True)
        X.loc[seg_id, 'exp_Moving_average_30000_mean_%d' % i] = ewma(sig, span=6000).mean().mean(skipna=True)

        no_of_std = 2
        X.loc[seg_id, 'MA_700MA_std_mean_%d' % i] = sig.rolling(window=700).std().mean()
        X.loc[seg_id, 'MA_700MA_BB_high_mean_%d' % i] = (
                    X.loc[seg_id, 'Moving_average_700_mean_%d' % i] + no_of_std * X.loc[seg_id, 'MA_700MA_std_mean_%d' % i]).mean()
        X.loc[seg_id, 'MA_700MA_BB_low_mean_%d' % i] = (
                    X.loc[seg_id, 'Moving_average_700_mean_%d' % i] - no_of_std * X.loc[seg_id, 'MA_700MA_std_mean_%d' % i]).mean()
        X.loc[seg_id, 'MA_400MA_std_mean_%d' % i] = sig.rolling(window=400).std().mean()
        X.loc[seg_id, 'MA_400MA_BB_high_mean_%d' % i] = (
                    X.loc[seg_id, 'Moving_average_700_mean_%d' % i] + no_of_std * X.loc[seg_id, 'MA_400MA_std_mean_%d' % i]).mean()
        X.loc[seg_id, 'MA_400MA_BB_low_mean_%d' % i] = (
                    X.loc[seg_id, 'Moving_average_700_mean_%d' % i] - no_of_std * X.loc[seg_id, 'MA_400MA_std_mean_%d' % i]).mean()
        X.loc[seg_id, 'MA_1000MA_std_mean_%d' % i] = sig.rolling(window=1000).std().mean()

        X.loc[seg_id, 'iqr_%d' % i] = np.subtract(*np.percentile(sig, [75, 25]))
        X.loc[seg_id, 'q999_%d' % i] = np.quantile(sig, 0.999)
        X.loc[seg_id, 'q001_%d' % i] = np.quantile(sig, 0.001)
        X.loc[seg_id, 'ave10_%d' % i] = stats.trim_mean(sig, 0.1)

    for windows in [10, 100, 1000]:
        x_roll_std = xc.rolling(windows).std().dropna().values
        x_roll_mean = xc.rolling(windows).mean().dropna().values

        X.loc[seg_id, 'ave_roll_std_' + str(windows)] = x_roll_std.mean()
        X.loc[seg_id, 'std_roll_std_' + str(windows)] = x_roll_std.std()
        X.loc[seg_id, 'max_roll_std_' + str(windows)] = x_roll_std.max()
        X.loc[seg_id, 'min_roll_std_' + str(windows)] = x_roll_std.min()
        X.loc[seg_id, 'q01_roll_std_' + str(windows)] = np.quantile(x_roll_std, 0.01)
        X.loc[seg_id, 'q05_roll_std_' + str(windows)] = np.quantile(x_roll_std, 0.05)
        X.loc[seg_id, 'q95_roll_std_' + str(windows)] = np.quantile(x_roll_std, 0.95)
        X.loc[seg_id, 'q99_roll_std_' + str(windows)] = np.quantile(x_roll_std, 0.99)
        X.loc[seg_id, 'av_change_abs_roll_std_' + str(windows)] = np.mean(np.diff(x_roll_std))
        X.loc[seg_id, 'av_change_rate_roll_std_' + str(windows)] = np.mean(
            np.nonzero((np.diff(x_roll_std) / x_roll_std[:-1]))[0])
        X.loc[seg_id, 'abs_max_roll_std_' + str(windows)] = np.abs(x_roll_std).max()

        X.loc[seg_id, 'ave_roll_mean_' + str(windows)] = x_roll_mean.mean()
        X.loc[seg_id, 'std_roll_mean_' + str(windows)] = x_roll_mean.std()
        X.loc[seg_id, 'max_roll_mean_' + str(windows)] = x_roll_mean.max()
        X.loc[seg_id, 'min_roll_mean_' + str(windows)] = x_roll_mean.min()
        X.loc[seg_id, 'q01_roll_mean_' + str(windows)] = np.quantile(x_roll_mean, 0.01)
        X.loc[seg_id, 'q05_roll_mean_' + str(windows)] = np.quantile(x_roll_mean, 0.05)
        X.loc[seg_id, 'q95_roll_mean_' + str(windows)] = np.quantile(x_roll_mean, 0.95)
        X.loc[seg_id, 'q99_roll_mean_' + str(windows)] = np.quantile(x_roll_mean, 0.99)
        X.loc[seg_id, 'av_change_abs_roll_mean_' + str(windows)] = np.mean(np.diff(x_roll_mean))
        X.loc[seg_id, 'av_change_rate_roll_mean_' + str(windows)] = np.mean(
            np.nonzero((np.diff(x_roll_mean) / x_roll_mean[:-1]))[0])
        X.loc[seg_id, 'abs_max_roll_mean_' + str(windows)] = np.abs(x_roll_mean).max()

    return X
def build_fields(proc_id):
    success = 1
    count = 0
    try:
        seg_st = int(NUM_SEG_PER_PROC * proc_id)
        train_df = pd.read_csv(os.path.join(DATA_DIR, 'raw_data_%d.csv' % proc_id), dtype={'acoustic_data': np.int16, 'time_to_failure': np.float32})
        len_df = len(train_df.index)
        start_indices = (np.loadtxt(fname=os.path.join(OUTPUT_DIR, 'start_indices_4k.csv'), dtype=np.int32, delimiter=','))[:, proc_id]
        train_X = pd.DataFrame(dtype=np.float64)
        train_y = pd.DataFrame(dtype=np.float64, columns=['time_to_failure'])
        t0 = time.time()

        for seg_id, start_idx in zip(range(seg_st, seg_st + NUM_SEG_PER_PROC), start_indices):
            end_idx = np.int32(start_idx + 150000)
            print('working: %d, %d, %d to %d of %d' % (proc_id, seg_id, start_idx, end_idx, len_df))
            seg = train_df.iloc[start_idx: end_idx]
            # train_X = create_features_pk_det(seg_id, seg, train_X, start_idx, end_idx)
            train_X = create_features(seg_id, seg, train_X, start_idx, end_idx)
            train_y.loc[seg_id, 'time_to_failure'] = seg['time_to_failure'].values[-1]

            if count == 10: 
                print('saving: %d, %d to %d' % (seg_id, start_idx, end_idx))
                train_X.to_csv('train_x_%d.csv' % proc_id, index=False)
                train_y.to_csv('train_y_%d.csv' % proc_id, index=False)

            count += 1

        print('final_save, process id: %d, loop time: %.2f for %d iterations' % (proc_id, time.time() - t0, count))
        train_X.to_csv(os.path.join(OUTPUT_DIR, 'train_x_%d.csv' % proc_id), index=False)
        train_y.to_csv(os.path.join(OUTPUT_DIR, 'train_y_%d.csv' % proc_id), index=False)

    except:
        print(traceback.format_exc())
        success = 0

    return success  # 1 on success, 0 if fail
def run_mp_build():
    t0 = time.time()
    num_proc = NUM_THREADS
    pool = mp.Pool(processes=num_proc)
    results = [pool.apply_async(build_fields, args=(pid, )) for pid in range(NUM_THREADS)]
    output = [p.get() for p in results]
    num_built = sum(output)
    pool.close()
    pool.join()
    print(num_built)
    print('Run time: %.2f' % (time.time() - t0))
def join_mp_build():
    df0 = pd.read_csv(os.path.join(OUTPUT_DIR, 'train_x_%d.csv' % 0))
    df1 = pd.read_csv(os.path.join(OUTPUT_DIR, 'train_y_%d.csv' % 0))

    for i in range(1, NUM_THREADS):
        print('working %d' % i)
        temp = pd.read_csv(os.path.join(OUTPUT_DIR, 'train_x_%d.csv' % i))
        df0 = df0.append(temp)

        temp = pd.read_csv(os.path.join(OUTPUT_DIR, 'train_y_%d.csv' % i))
        df1 = df1.append(temp)

    df0.to_csv(os.path.join(OUTPUT_DIR, 'train_x.csv'), index=False)
    df1.to_csv(os.path.join(OUTPUT_DIR, 'train_y.csv'), index=False)
def build_test_fields():
    train_X = pd.read_csv(os.path.join(OUTPUT_DIR, 'train_x.csv'))
    try:
        train_X.drop(labels=['seg_id', 'seg_start', 'seg_end'], axis=1, inplace=True)
    except:
        pass

    submission = pd.read_csv(os.path.join(DATA_DIR, 'sample_submission.csv'), index_col='seg_id')
    test_X = pd.DataFrame(columns=train_X.columns, dtype=np.float64, index=submission.index)

    print('start for loop')
    count = 0
    for seg_id in tqdm_notebook(test_X.index):  # just tqdm in IDE
        seg = pd.read_csv(os.path.join(DATA_DIR, 'test', str(seg_id) + '.csv'))
        # train_X = create_features_pk_det(seg_id, seg, train_X, start_idx, end_idx)
        test_X = create_features(seg_id, seg, test_X, 0, 0)

        if count % 100 == 0:
            print('working', seg_id)
        count += 1

    test_X.to_csv(os.path.join(OUTPUT_DIR, 'test_x.csv'), index=False)
def scale_fields(fn_train='train_x.csv', fn_test='test_x.csv', 
                 fn_out_train='scaled_train_X.csv' , fn_out_test='scaled_test_X.csv'):
    train_X = pd.read_csv(os.path.join(OUTPUT_DIR, fn_train))
    try:
        train_X.drop(labels=['seg_id', 'seg_start', 'seg_end'], axis=1, inplace=True)
    except:
        pass
    test_X = pd.read_csv(os.path.join(OUTPUT_DIR, fn_test))

    print('start scaler')
    scaler = StandardScaler()
    scaler.fit(train_X)
    scaled_train_X = pd.DataFrame(scaler.transform(train_X), columns=train_X.columns)
    scaled_test_X = pd.DataFrame(scaler.transform(test_X), columns=test_X.columns)

    scaled_train_X.to_csv(os.path.join(OUTPUT_DIR, fn_out_train), index=False)
    scaled_test_X.to_csv(os.path.join(OUTPUT_DIR, fn_out_test), index=False)
split_raw_data()
build_rnd_idxs()
run_mp_build()
join_mp_build()
build_test_fields()
scale_fields()

# do something like this in the IDE, call the functions above in order
# if __name__ == "__main__":
#     function name()
    
def create_features_pk_det(seg_id, seg, X, st, end):
    X.loc[seg_id, 'seg_id'] = np.int32(seg_id)
    X.loc[seg_id, 'seg_start'] = np.int32(st)
    X.loc[seg_id, 'seg_end'] = np.int32(end)

    sig = pd.Series(seg['acoustic_data'].values)
    b, a = des_bw_filter_lp(cutoff=18000)
    sig = sg.lfilter(b, a, sig)

    peakind = []
    noise_pct = .001
    count = 0

    while len(peakind) < 12 and count < 24:
        peakind = sg.find_peaks_cwt(sig, np.arange(1, 16), noise_perc=noise_pct, min_snr=4.0)
        noise_pct *= 2.0
        count += 1

    if len(peakind) < 12:
        print('Warning: Failed to find 12 peaks for %d' % seg_id)

    while len(peakind) < 12:
        peakind.append(149999)

    df_pk = pd.DataFrame(data={'pk': sig[peakind], 'idx': peakind}, columns=['pk', 'idx'])
    df_pk.sort_values(by='pk', ascending=False, inplace=True)

    for i in range(0, 12):
        X.loc[seg_id, 'pk_idx_%d' % i] = df_pk['idx'].iloc[i]
        X.loc[seg_id, 'pk_val_%d' % i] = df_pk['pk'].iloc[i]

    return X
import pandas as pd

df = pd.read_csv('test_x_8pk.csv')
df_out = None

for pks in df.itertuples():
    data = {'pk_idxs': [pks.pk_idx_0, pks.pk_idx_1, pks.pk_idx_2, pks.pk_idx_3, pks.pk_idx_4, pks.pk_idx_5, pks.pk_idx_6, pks.pk_idx_7, pks.pk_idx_8, pks.pk_idx_9, pks.pk_idx_10, pks.pk_idx_11],
            'pk_vals': [pks.pk_val_0, pks.pk_val_1, pks.pk_val_2, pks.pk_val_3, pks.pk_val_4, pks.pk_val_5, pks.pk_val_6, pks.pk_val_7, pks.pk_val_8, pks.pk_val_9, pks.pk_val_10, pks.pk_val_11]}
    pdf = pd.DataFrame(data=data)
    pdf.sort_values(by='pk_idxs', axis=0, inplace=True)

    data = {'pk_idx_0': pdf['pk_idxs'].iloc[0], 'pk_val_0': pdf['pk_vals'].iloc[0],
            'pk_idx_1': pdf['pk_idxs'].iloc[1], 'pk_val_1': pdf['pk_vals'].iloc[1],
            'pk_idx_2': pdf['pk_idxs'].iloc[2], 'pk_val_2': pdf['pk_vals'].iloc[2],
            'pk_idx_3': pdf['pk_idxs'].iloc[3], 'pk_val_3': pdf['pk_vals'].iloc[3],
            'pk_idx_4': pdf['pk_idxs'].iloc[4], 'pk_val_4': pdf['pk_vals'].iloc[4],
            'pk_idx_5': pdf['pk_idxs'].iloc[5], 'pk_val_5': pdf['pk_vals'].iloc[5],
            'pk_idx_6': pdf['pk_idxs'].iloc[6], 'pk_val_6': pdf['pk_vals'].iloc[6],
            'pk_idx_7': pdf['pk_idxs'].iloc[7], 'pk_val_7': pdf['pk_vals'].iloc[7],
            'pk_idx_8': pdf['pk_idxs'].iloc[8], 'pk_val_8': pdf['pk_vals'].iloc[8],
            'pk_idx_9': pdf['pk_idxs'].iloc[9], 'pk_val_9': pdf['pk_vals'].iloc[9],
            'pk_idx_10': pdf['pk_idxs'].iloc[10], 'pk_val_10': pdf['pk_vals'].iloc[10],
            'pk_idx_11': pdf['pk_idxs'].iloc[11], 'pk_val_11': pdf['pk_vals'].iloc[11]}

    if df_out is None:
        df_out = pd.DataFrame(data=data, index=[0])
    else:
        temp = pd.DataFrame(data=data, index=[0])
        df_out = df_out.append(temp, ignore_index=True)

df_out = df_out[['pk_idx_0', 'pk_val_0',
                   'pk_idx_1', 'pk_val_1',
                   'pk_idx_2', 'pk_val_2',
                   'pk_idx_3', 'pk_val_3',
                   'pk_idx_4', 'pk_val_4',
                   'pk_idx_5', 'pk_val_5',
                   'pk_idx_6', 'pk_val_6',
                   'pk_idx_7', 'pk_val_7',
                   'pk_idx_8', 'pk_val_8',
                   'pk_idx_9', 'pk_val_9',
                   'pk_idx_10', 'pk_val_10',
                   'pk_idx_11', 'pk_val_11']]
print(df_out.head())
print(df_out.tail())
df_out.to_csv('test_x_8pk_by_idx.csv')
import numpy as np
import pandas as pd

pk_idx_base = 'pk_idx_'
pk_val_base = 'pk_val_'

print('do train')
df = pd.read_csv(r'pk8/train_x_8pk.csv')
slopes = np.zeros((len(df.index), 12))

for i in df.index:
    for j in range(12):
        pk_idx = pk_idx_base + str(j)
        pk_val = pk_val_base + str(j)
        slopes[i, j] = df[pk_val].iloc[i] / (150000 - df[pk_idx].iloc[i])

for j in range(12):
    df['slope_' + str(j)] = slopes[:, j]

print(df.head())
df.to_csv(r'pk8/train_x_8_slope.csv', index=False)

df = pd.read_csv(r'pk8/test_x_8pk.csv')
slopes = np.zeros((len(df.index), 12))

print('do test')
for i in df.index:
    for j in range(12):
        pk_idx = pk_idx_base + str(j)
        pk_val = pk_val_base + str(j)
        slopes[i, j] = df[pk_val].iloc[i] / (150000 - df[pk_idx].iloc[i])

for j in range(12):
    df['slope_' + str(j)] = slopes[:, j]

print(df.head())
df.to_csv(r'pk8/test_x_8_slope.csv', index=False)

print('!DONE!')
params = {'num_leaves': 21,
         'min_data_in_leaf': 20,
         'objective':'regression',
         'learning_rate': 0.001,
         'max_depth': 108,
         "boosting": "gbdt",
         "feature_fraction": 0.91,
         "bagging_freq": 1,
         "bagging_fraction": 0.91,
         "bagging_seed": 42,
         "metric": 'mae',
         "lambda_l1": 0.1,
         "verbosity": -1,
         "random_state": 42}


def lgb_base_model():
    maes = []
    rmses = []
    submission = pd.read_csv(os.path.join(DATA_DIR, 'sample_submission.csv'), index_col='seg_id')
    scaled_train_X = pd.read_csv(r'train_8_and_9\scaled_train_X_8.csv')
    scaled_test_X = pd.read_csv(r'train_8_and_9\scaled_test_X_8.csv')
    train_y = pd.read_csv(r'train_8_and_9\train_y_8.csv')
    predictions = np.zeros(len(scaled_test_X))

    n_fold = 8
    folds = KFold(n_splits=n_fold, shuffle=True, random_state=42)

    fold_importance_df = pd.DataFrame()
    fold_importance_df["Feature"] = scaled_train_X.columns

    for fold_, (trn_idx, val_idx) in enumerate(folds.split(scaled_train_X, train_y.values)):
        print('working fold %d' % fold_)
        strLog = "fold {}".format(fold_)
        print(strLog)

        X_tr, X_val = scaled_train_X.iloc[trn_idx], scaled_train_X.iloc[val_idx]
        y_tr, y_val = train_y.iloc[trn_idx], train_y.iloc[val_idx]

        model = lgb.LGBMRegressor(**params, n_estimators=80000, n_jobs=-1)
        model.fit(X_tr, y_tr,
                  eval_set=[(X_tr, y_tr), (X_val, y_val)], eval_metric='mae',
                  verbose=1000, early_stopping_rounds=200)

        # predictions
        preds = model.predict(scaled_test_X, num_iteration=model.best_iteration_)
        predictions += preds / folds.n_splits
        preds = model.predict(X_val, num_iteration=model.best_iteration_)

        # mean absolute error
        mae = mean_absolute_error(y_val, preds)
        print('MAE: %.6f' % mae)
        maes.append(mae)

        # root mean squared error
        rmse = mean_squared_error(y_val, preds)
        print('RMSE: %.6f' % rmse)
        rmses.append(rmse)

        fold_importance_df['importance_%d' % fold_] = model.feature_importances_[:len(scaled_train_X.columns)]

    print('MAEs', maes)
    print('MAE mean: %.6f' % np.mean(maes))
    print('RMSEs', rmses)
    print('RMSE mean: %.6f' % np.mean(rmses))

    submission.time_to_failure = predictions
    submission.to_csv('submission_lgb_8_80k_108dp.csv', index=False)
    fold_importance_df.to_csv('fold_imp_lgb_8_80k_108dp.csv')  # index needed, it is seg id

# do this in the IDE, call the function
# if __name__ == "__main__":
#     lgb_base_model()
params = {'num_leaves': 21,
         'min_data_in_leaf': 20,
         'objective':'regression',
         'max_depth': 108,
         'learning_rate': 0.001,
         "boosting": "gbdt",
         "feature_fraction": 0.91,
         "bagging_freq": 1,
         "bagging_fraction": 0.91,
         "bagging_seed": 42,
         "metric": 'mae',
         "lambda_l1": 0.1,
         "verbosity": -1,
         "random_state": 42}


def lgb_trimmed_model():
    maes = []
    rmses = []
    tr_maes = []
    tr_rmses = []
    submission = pd.read_csv(os.path.join(DATA_DIR, 'sample_submission.csv'), index_col='seg_id')

    scaled_train_X = pd.read_csv(r'pk8/scaled_train_X_8.csv')
    df = pd.read_csv(r'pk8/scaled_train_X_8_slope.csv')
    scaled_train_X = scaled_train_X.join(df)

    scaled_test_X = pd.read_csv(r'pk8/scaled_test_X_8.csv')
    df = pd.read_csv(r'pk8/scaled_test_X_8_slope.csv')
    scaled_test_X = scaled_test_X.join(df)

    pcol = []
    pcor = []
    pval = []
    y = pd.read_csv(r'pk8/train_y_8.csv')['time_to_failure'].values

    for col in scaled_train_X.columns:
        pcol.append(col)
        pcor.append(abs(pearsonr(scaled_train_X[col], y)[0]))
        pval.append(abs(pearsonr(scaled_train_X[col], y)[1]))

    df = pd.DataFrame(data={'col': pcol, 'cor': pcor, 'pval': pval}, index=range(len(pcol)))
    df.sort_values(by=['cor', 'pval'], inplace=True)
    df.dropna(inplace=True)
    df = df.loc[df['pval'] <= 0.05]

    drop_cols = []

    for col in scaled_train_X.columns:
        if col not in df['col'].tolist():
            drop_cols.append(col)

    scaled_train_X.drop(labels=drop_cols, axis=1, inplace=True)
    scaled_test_X.drop(labels=drop_cols, axis=1, inplace=True)

    train_y = pd.read_csv(r'pk8/train_y_8.csv')
    predictions = np.zeros(len(scaled_test_X))
    preds_train = np.zeros(len(scaled_train_X))

    print('shapes of train and test:', scaled_train_X.shape, scaled_test_X.shape)

    n_fold = 6
    folds = KFold(n_splits=n_fold, shuffle=False, random_state=42)

    for fold_, (trn_idx, val_idx) in enumerate(folds.split(scaled_train_X, train_y.values)):
        print('working fold %d' % fold_)
        strLog = "fold {}".format(fold_)
        print(strLog)

        X_tr, X_val = scaled_train_X.iloc[trn_idx], scaled_train_X.iloc[val_idx]
        y_tr, y_val = train_y.iloc[trn_idx], train_y.iloc[val_idx]

        model = lgb.LGBMRegressor(**params, n_estimators=60000, n_jobs=-1)
        model.fit(X_tr, y_tr,
                  eval_set=[(X_tr, y_tr), (X_val, y_val)], eval_metric='mae',
                  verbose=1000, early_stopping_rounds=200)

        # model = xgb.XGBRegressor(n_estimators=1000,
        #                                learning_rate=0.1,
        #                                max_depth=6,
        #                                subsample=0.9,
        #                                colsample_bytree=0.67,
        #                                reg_lambda=1.0, # seems best within 0.5 of 2.0
        #                                # gamma=1,
        #                                random_state=777+fold_,
        #                                n_jobs=12,
        #                                verbosity=2)
        # model.fit(X_tr, y_tr)

        # predictions
        preds = model.predict(scaled_test_X)  #, num_iteration=model.best_iteration_)
        predictions += preds / folds.n_splits
        preds = model.predict(scaled_train_X)  #, num_iteration=model.best_iteration_)
        preds_train += preds / folds.n_splits

        preds = model.predict(X_val)  #, num_iteration=model.best_iteration_)

        # mean absolute error
        mae = mean_absolute_error(y_val, preds)
        print('MAE: %.6f' % mae)
        maes.append(mae)

        # root mean squared error
        rmse = mean_squared_error(y_val, preds)
        print('RMSE: %.6f' % rmse)
        rmses.append(rmse)

        # training for over fit
        preds = model.predict(X_tr)  #, num_iteration=model.best_iteration_)

        mae = mean_absolute_error(y_tr, preds)
        print('Tr MAE: %.6f' % mae)
        tr_maes.append(mae)

        rmse = mean_squared_error(y_tr, preds)
        print('Tr RMSE: %.6f' % rmse)
        tr_rmses.append(rmse)

    print('MAEs', maes)
    print('MAE mean: %.6f' % np.mean(maes))
    print('RMSEs', rmses)
    print('RMSE mean: %.6f' % np.mean(rmses))

    print('Tr MAEs', tr_maes)
    print('Tr MAE mean: %.6f' % np.mean(tr_maes))
    print('Tr RMSEs', rmses)
    print('Tr RMSE mean: %.6f' % np.mean(tr_rmses))

    submission.time_to_failure = predictions
    submission.to_csv('submission_xgb_slope_pearson_6fold.csv')  # index needed, it is seg id

    pr_tr = pd.DataFrame(data=preds_train, columns=['time_to_failure'], index=range(0, preds_train.shape[0]))
    pr_tr.to_csv(r'preds_tr_xgb_slope_pearson_6fold.csv', index=False)
    print('Train shape: {}, Test shape: {}, Y shape: {}'.format(scaled_train_X.shape, scaled_test_X.shape, train_y.shape))
 
# do this in the IDE, call the function above
# if __name__ == "__main__":
#     lgb_trimmed_model()
import sys
import scipy
import sklearn
print(sys.version)
print('pandas:', pd.__version__)
print('numpy:', np.__version__)
print('scipy:', scipy.__version__)
print('sklearn:', sklearn.__version__)
print('light gbm:', lgb.__version__)
print('xgboost:', xgb.__version__)