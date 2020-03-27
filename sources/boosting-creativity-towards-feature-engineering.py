
#import libs
import numpy as np
import pandas as pd 
import lightgbm as lgb
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
import os
import sys

import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import norm, skewnorm
from collections import defaultdict

#import data
train_df = pd.read_csv('../input/train.csv')
#test_df = pd.read_csv('../input/test.csv')
train_df.head()
def running_mean(x, N):
   cumsum = np.cumsum(np.insert(x, 0, 0)) 
   return np.concatenate([x[0:N-1], (cumsum[N:] - cumsum[:-N]) / N])

var = 'var_0' #feat we are going to analyse
mydf = train_df.copy() #df we are going to analyze

def df_to_bin(var, mydf):
    IQR = mydf[var].quantile([0.75]).values - mydf[var].quantile([0.25]).values #inter-quartile range
    n = 200000 #size of our dataframe
    bin_size = 2.5*IQR/n**(1/3) #Freedman-Diaconis rule to define the bin size
    bin_number = int((mydf[var].max() - mydf[var].min())/bin_size)

    #creat new feature based on the bins
    mydf['new' + var] = pd.cut(mydf[var], bins = bin_number, labels = range(bin_number)).astype('float')

    #separate dfs betweens target ==0 and target == 1
    df_no_transfer = mydf['new' + var][mydf['target'] == 0].value_counts().sort_index()
    #get the bin frequency height (H) and the index of that bin (X)
    H0,X0 = list(df_no_transfer.values), list(df_no_transfer.index) 

    df_with_transfer = mydf['new' + var][mydf['target'] == 1].value_counts().sort_index()
    H1,X1 = list(df_with_transfer.values), list(df_with_transfer.index) 

    #If there are any bins with 0 frequency between 0 and the total number of bins, lets put it in our list
    for i in range(bin_number):
        if i not in X0:
            H0.append(0)
            X0.append(i)

        if i not in X1:
            H1.append(0)
            X1.append(i)

    #sort the bins so the list index == bin number
    H0 = [h for _,h in sorted(zip(X0,H0))]
    X0 = [x for x,_ in sorted(zip(X0,H0))]
    H1 = [h for _,h in sorted(zip(X1,H1))]
    X1 = [x for x,_ in sorted(zip(X1,H1))]

    H0 = running_mean(H0, 2)  
    H1 = running_mean(H1, 2)  
    Hsum = H0 + H1
    
    return bin_size, bin_number, mydf, H0, H1, Hsum, X0, X1

var = 'var_0'
bin_size, bin_number, mydf, H0, H1, Hsum, X0, X1 = df_to_bin(var, train_df)

#plot the result
fig, ax1 = plt.subplots()
ax1.plot(H0, label='target = 0 (doesnt make a transfer)')
ax1.plot(H1, label='target = 1 (makes a transfer)')
ax1.plot(Hsum, label='total data')
plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
fig.suptitle(var, fontsize=20)
ax1.set_ylabel('Frequency', color='g', fontsize=12)
ax1.tick_params(axis='y', colors='green')
plt.xlabel('Bin Number', fontsize=12)
plt.show()
Hmin = 80
def get_probs(bin_number, H1, Hsum, Hmin):
    probs = []
    #finds min_i
    for i in range(bin_number):
        if Hsum[i] > Hmin:
            min_i = i
            break

    #finds max_i
    for i in reversed(range(bin_number)):
        if Hsum[i] > Hmin:
            max_i = i
            break

    #get probabilities to plot
    for i in range(0, bin_number):
        if i < min_i or i > max_i:
            probs.append(0)
        else:
            probs.append(H1[i]/Hsum[i]) 
    
    return probs, min_i, max_i

probs, min_i, max_i = get_probs(bin_number, H1, Hsum, Hmin)

#plot the result
fig, ax1 = plt.subplots()
ax2 = ax1.twinx()
ax1.plot(H0, label='target = 0')
ax1.plot(H1, label='target = 1')
ax1.plot(Hsum, label='total data')
ax2.plot(probs, 'r', label='Probability target = 1')
ax1.legend(bbox_to_anchor=(1.10, 1), loc=3)
ax2.legend(bbox_to_anchor=(1.10, 0.90), loc=3)
fig.suptitle(var, fontsize=20)
ax1.set_ylabel('Frequency', color='g', fontsize=12)
ax1.tick_params(axis='y', colors='green')
ax2.set_ylabel('Probability', color='r', fontsize=12)
ax2.tick_params(axis='y', colors='red')
ax1.set_xlabel('Bin Number', fontsize=12)
plt.show()
def get_central_point(bin_number, Hsum, H0, H1):

    #finds the slop at each point       
    slope = []
    for i in range(bin_number - 1):

        init = Hsum[i]
        end = Hsum[i+1]
        x = 1
        y = end - init
        slope.append(np.arctan(y/x) * 180/3.1416)

    slope.append(slope[-1]) #last slope is equal to its previous
    slope = np.array(slope)

    #finds the slop difference at each point   
    slopediff = [0]
    for i in range(1, bin_number):
        slopediff.append(np.abs(slope[i] - slope[i-1]))

    #finds two "straight lines" of length 15 on the left and right of the distribution
    min_len = 15
    max_slop_diff = 0.05
    len_max_l = 0
    len_max_r = 0
    count = 0
    #at each iteration, if we don't have the 15 points if max_slop_diff between them,
    #we will lower this difference threshold
    while len_max_l < min_len or len_max_r < min_len:
        count += 1 #if after 5 iterations we dont get those 15 points, we will lower the minimum amount of points
        slope_dict_left = defaultdict(list)
        slope_dict_right = defaultdict(list)
        l, r = 0, 0
        maxl = 0
        maxr = 0

        for i in range(bin_number):
            s = slope[i]
            sd = slopediff[i]
            if s > 0 and sd < max_slop_diff and Hsum[i] > 1000:
                slope_dict_left[l].append(i)
            else:
                l_len = len(slope_dict_left[l])
                if l_len > len(slope_dict_left[maxl]):
                    maxl = l
                l += 1
            if s < 0 and sd < max_slop_diff and Hsum[i] > 1000:
                slope_dict_right[r].append(i)
            else:
                r_len = len(slope_dict_right[r])
                if r_len > len(slope_dict_right[maxr]):
                    maxr = r
                r += 1

        len_max_l = len(slope_dict_left[maxl])
        len_max_r = len(slope_dict_right[maxr])

        #only used if we dont get the 15 points on left and right side
        if count > 5:
            if min_len > 12:
                min_len = min_len*0.9
            max_slop_diff = max_slop_diff*1.1

            H0 = running_mean(H0, 2)
            H1 = running_mean(H1, 2)
            Hsum = H0 + H1

            #finds the slop at each point       
            slope = []
            for i in range(bin_number - 1):

                init = Hsum[i]
                end = Hsum[i+1]
                x = 1
                y = end - init
                slope.append(np.arctan(y/x) * 180/3.1416)
            slope.append(slope[-1])

            slope = np.array(slope)

            slopediff = [0]
            for i in range(1, bin_number):
                slopediff.append(np.abs(slope[i] - slope[i-1]))

        else:
            max_slop_diff = max_slop_diff*1.1

    #select the lines with the highest length
    slope_left = slope_dict_left[maxl]#[int(len_max_l*0.30): int(len_max_l*0.90)]
    slope_right = slope_dict_right[maxr]#[int(len_max_r*0.10): int(len_max_r*0.70)]

    #now lets align their maximum and minimum height H, with a maxium difference of 100
    max_diff = 100

    while Hsum[slope_left[-1]] - Hsum[slope_right[0]] > max_diff:
        slope_left = slope_left[:-2]

    while Hsum[slope_right[0]] - Hsum[slope_left[-1]] > max_diff:
        slope_right = slope_right[1:]

    while Hsum[slope_right[-1]] - Hsum[slope_left[0]] > max_diff:
        slope_left = slope_left[1:]

    while Hsum[slope_left[0]] - Hsum[slope_right[-1]] > +max_diff:
        slope_right = slope_right[:-2]

    #Find the median of the lines
    i_median_left = int(len(slope_left)/2)
    i_median_right = int(len(slope_right)/2)

    #Finds the central point of the distribution based on the this two lines
    l = 0
    r = 0
    H_left = Hsum[slope_left]
    H_right = Hsum[slope_right]
    indexes = []
    if len(slope_left) < len(slope_right):

        for h in H_left:
            indexes.append(np.argmin(np.abs(H_right-h)))

        for index_left, index_right in zip(range(len(slope_left)), indexes):
            l += slope_left[index_left]
            r += slope_right[index_right]

        central_point = np.round((l+r)/(2*(index_left+1)))
    else:
        for h in H_right:
            indexes.append(np.argmin(np.abs(H_left-h)))

        for index_right, index_left in zip(range(len(slope_right)), indexes):
            l += slope_left[index_left]
            r += slope_right[index_right]

        central_point = np.round((l+r)/(2*(index_right+1)))

    central_point = np.round(central_point)
    
    return central_point, Hsum, slope_left, slope_right, i_median_left, i_median_right

central_point, Hsum, slope_left, slope_right, i_median_left, i_median_right = get_central_point(bin_number, Hsum, H0, H1)

#plot the result
fig, ax1 = plt.subplots()
ax2 = ax1.twinx()
ax1.plot(H0, label='target = 0')
ax1.plot(H1, label='target = 1')
ax1.plot(Hsum, label='total data')
ax1.plot(slope_left, Hsum[slope_left], 'k')
ax1.plot(slope_right, Hsum[slope_right], 'k')
ax1.plot([central_point, central_point], [0,3000], 'grey')
ax2.plot(probs, 'r', label='Probability target = 1')
ax1.legend(bbox_to_anchor=(1.10, 1), loc=3)
ax2.legend(bbox_to_anchor=(1.10, 0.90), loc=3)

ax1.set_ylabel('Frequency', color='g', fontsize=12)
ax1.tick_params(axis='y', colors='green')
ax1.set_xlabel('Bin Number', fontsize=12)

ax2.set_ylabel('Probability', color='r', fontsize=12)
ax2.tick_params(axis='y', colors='red')

fig.suptitle(var, fontsize=20)
plt.xlabel('bin_number', fontsize=12)
plt.show()
def get_ratio(central_point, bin_number, Hsum, Hmin):

    ratio = []
    len_1st_half = central_point
    len_2nd_half = bin_number - central_point
    i = 0
    if len_1st_half > len_2nd_half:
        index = central_point - len_2nd_half

        while i < index:
            ratio.append(0)
            i += 1

        for i in range(i, bin_number):
            i_end = int(central_point + (central_point - i) - 1)
            if Hsum[i_end] < Hmin or Hsum[i] < Hmin:
                ratio.append(0)
            else:
                ratio.append((Hsum[i])/(Hsum[i_end]))


    else:
        index = central_point - len_2nd_half

        for i in range(i, bin_number):
            i_end = int(central_point + (central_point - i) - 1)
            if i > 2*central_point:
                ratio.append(0)
                continue
            if Hsum[i_end] < Hmin or Hsum[i] < Hmin:
                ratio.append(0)
            else:
                ratio.append((Hsum[i])/(Hsum[i_end]))
            

    return ratio

ratio = get_ratio(central_point, bin_number, Hsum, Hmin)

#plot the result
fig, ax1 = plt.subplots()
ax2 = ax1.twinx()
ax3 = ax1.twinx()
ax1.plot(H0, label='target = 0')
ax1.plot(H1, label='target = 1')
ax1.plot(Hsum, label='total data')
ax1.plot(slope_left, Hsum[slope_left], 'k')
ax1.plot(slope_right, Hsum[slope_right], 'k')
ax1.plot([central_point, central_point], [0,3000], 'grey')
ax2.plot(probs, 'r', label='Probability target = 1')
ax3.plot(ratio, 'b', label='height ratio')
ax1.legend(bbox_to_anchor=(1.1, 1), loc=3)
ax2.legend(bbox_to_anchor=(1.1, 0.90), loc=3)
ax3.legend(bbox_to_anchor=(1.1, 0.80), loc=3)
fig.suptitle(var, fontsize=20)
ax1.set_ylabel('Frequency', color='g', fontsize=12)
ax1.tick_params(axis='y', colors='green')
ax1.set_xlabel('Bin Number', fontsize=12)

ax2.set_ylabel('Probability', color='r', fontsize=12)
ax2.tick_params(axis='y', colors='red')

ax3.set_ylabel('Frequency ratio', color='b', fontsize=12)
ax3.tick_params(axis='y', colors='blue')
ax3.yaxis.set_label_coords(1.18, 0.5)
plt.show()
#function we will use to plot
def plot_graph(var, df, Hmin):
    bin_size, bin_number, df, H0, H1, Hsum, X0, X1 = df_to_bin(var, train_df)
    probs, min_i, max_i = get_probs(bin_number, H1, Hsum, Hmin)
    central_point, Hsum, slope_left, slope_right, i_median_left, i_median_right = get_central_point(bin_number, Hsum, H0, H1)
    ratio = get_ratio(central_point, bin_number, Hsum, Hmin)
    
    fig, ax1 = plt.subplots()
    ax2 = ax1.twinx()
    ax3 = ax1.twinx()
    ax1.plot(H0, label='target = 0')
    ax1.plot(H1, label='target = 1')
    ax1.plot(Hsum, label='total data')
    ax1.plot(slope_left, Hsum[slope_left], 'k')
    ax1.plot(slope_right, Hsum[slope_right], 'k')
    ax1.plot([central_point, central_point], [0,3000], 'grey')
    ax2.plot(probs, 'r', label='Probability target = 1')
    ax3.plot(ratio, 'b', label='height ratio')
    ax1.legend(bbox_to_anchor=(1.1, 1), loc=3)
    ax2.legend(bbox_to_anchor=(1.1, 0.90), loc=3)
    ax3.legend(bbox_to_anchor=(1.1, 0.80), loc=3)
    
    ax1.set_ylabel('Frequency', color='g', fontsize=12)
    ax1.tick_params(axis='y', colors='green')
    ax1.set_xlabel('Bin Number', fontsize=12)
    
    ax2.set_ylabel('Probability', color='r', fontsize=12)
    ax2.tick_params(axis='y', colors='red')

    ax3.set_ylabel('Frequency ratio', color='b', fontsize=12)
    ax3.tick_params(axis='y', colors='blue')
    ax3.yaxis.set_label_coords(1.18, 0.5)
    
    fig.suptitle(var, fontsize=20)
    plt.xlabel('bin_number', fontsize=12)
    plt.show()
    



regular_features = ['var_0', 'var_1', 'var_101', 'var_102', 'var_105', 'var_106', 'var_108', 'var_109', 'var_11', 'var_110', 'var_111', 'var_113', 'var_114', 'var_115', 'var_116', 'var_118', 'var_119', 'var_12', 'var_122', 'var_123', 'var_125', 'var_127', 'var_128', 'var_129', 'var_13', 'var_130', 'var_131', 'var_132', 'var_133', 'var_134', 'var_135', 'var_137', 'var_138', 'var_139', 'var_141', 'var_143', 'var_144', 'var_145', 'var_146', 'var_147', 'var_148', 'var_15', 'var_150', 'var_151', 'var_152', 'var_153', 'var_154', 'var_155', 'var_157', 'var_159', 'var_16', 'var_162', 'var_163', 'var_164', 'var_165', 'var_166', 'var_167', 'var_168', 'var_169', 'var_170', 'var_171', 'var_173', 'var_174', 'var_175', 'var_176', 'var_179', 'var_18', 'var_180', 'var_181', 'var_182', 'var_184', 'var_187', 'var_188', 'var_189', 'var_19', 'var_190', 'var_191', 'var_193', 'var_194', 'var_195', 'var_196', 'var_197', 'var_198', 'var_2', 'var_22', 'var_24', 'var_25', 'var_26', 'var_28', 'var_3', 'var_32', 'var_34', 'var_35', 'var_36', 'var_37', 'var_39', 'var_4', 'var_40', 'var_42', 'var_43', 'var_44', 'var_48', 'var_49', 'var_5', 'var_50', 'var_51', 'var_52', 'var_53', 'var_55', 'var_56', 'var_59', 'var_6', 'var_60', 'var_61', 'var_62', 'var_63', 'var_64', 'var_66', 'var_67', 'var_68', 'var_69', 'var_70', 'var_71', 'var_72', 'var_73', 'var_74', 'var_75', 'var_76', 'var_78', 'var_79', 'var_80', 'var_81', 'var_82', 'var_83', 'var_84', 'var_85', 'var_86', 'var_88', 'var_89', 'var_9', 'var_90', 'var_91', 'var_92', 'var_93', 'var_94', 'var_95', 'var_96', 'var_97', 'var_99']
for var in regular_features[:3]:
    plot_graph(var, train_df, Hmin = 80)
reversed_features = ['var_104', 'var_107', 'var_112', 'var_121', 'var_14', 'var_140', 'var_142', 'var_149', 'var_156', 'var_160', 'var_172', 'var_177', 'var_178', 'var_186', 'var_192', 'var_199', 'var_20', 'var_21', 'var_23', 'var_31', 'var_33', 'var_45', 'var_57', 'var_65', 'var_77', 'var_8', 'var_87']
for var in reversed_features[:3]:
    plot_graph(var, train_df, Hmin = 80)
flat_features = ['var_10', 'var_100', 'var_103', 'var_117', 'var_124', 'var_126', 'var_136', 'var_158', 'var_161', 'var_17', 'var_183', 'var_185', 'var_27', 'var_29', 'var_30', 'var_38', 'var_41', 'var_47', 'var_7', 'var_98']
for var in flat_features[:3]:
    plot_graph(var, train_df, Hmin = 80)
extreme_features = ['var_120', 'var_46', 'var_54', 'var_58']
for var in extreme_features[:3]:
    plot_graph(var, train_df, Hmin = 80)
for var in ['var_4', 'var_5', 'var_26']:
    plot_graph(var, train_df, Hmin = 80)
from itertools import starmap
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib import mlab
from scipy.stats import truncnorm
from sklearn.mixture import GaussianMixture
#EXAMPLE VARIABLE 4
mydf = train_df
var = 'var_4'
print(var)

n_components = 2
mydf.sort_values(var, inplace = True)
samples = np.array(mydf[var])
samples_max = max(samples)
samples_min = min(samples)


mixture = GaussianMixture(n_components=n_components, weights_init = [0.80, 0.20], means_init = [[11], [14.5]]).fit(samples.reshape(-1, 1))
means_hat = mixture.means_.flatten()
weights_hat = mixture.weights_.flatten()
sds_hat = np.sqrt(mixture.covariances_).flatten()


means_hat = [m for m, _ in sorted(zip(means_hat, sds_hat))]
sds_hat = [s for _, s in sorted(zip(means_hat, sds_hat))]

#print('mixture converged', mixture.converged_)
#print('means', means_hat)
#print('standard deviations', sds_hat)
#print('weights', weights_hat)
fig, ax1 = plt.subplots()
ax2 = ax1.twinx()
for mu1_h, sd1_h, weight, i in zip(means_hat, sds_hat, weights_hat, range(n_components)):

    x_axis = np.linspace(samples_min, samples_max, int(np.round(weight*len(samples))))

    Y = np.random.normal(mu1_h, sd1_h, int(np.round(weight*len(samples))))
    Y = np.exp(mixture.score_samples(x_axis.reshape(-1,1)))
    ax2.plot(x_axis, Y)
    ax1.hist(np.random.normal(mu1_h, sd1_h, int(np.round(weight*len(samples)))))

    mydf[var+'_' + str(i)] = np.nan

mydf.sort_values('ID_code', inplace = True)
plt.show

plot_graph('var_4', train_df, Hmin = 80)