
import pandas as pd

import numpy as np

#%matplotlib inline

import matplotlib.pyplot as plt

import matplotlib.lines as mlines

from mpl_toolkits.mplot3d import Axes3D

import seaborn as sns

from sklearn.model_selection import train_test_split, learning_curve

from sklearn.metrics import average_precision_score

from xgboost.sklearn import XGBClassifier

from xgboost import plot_importance, to_graphviz
import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning)
df = pd.read_csv('../input/PS_20174392719_1491204439457_log.csv')

df = df.rename(columns={'oldbalanceOrg':'oldBalanceOrig', 'newbalanceOrig':'newBalanceOrig', \

                        'oldbalanceDest':'oldBalanceDest', 'newbalanceDest':'newBalanceDest'})

print(df.head())
df.isnull().values.any()
print('\n The types of fraudulent transactions are {}'.format(\

list(df.loc[df.isFraud == 1].type.drop_duplicates().values))) # only 'CASH_OUT' 

                                                             # & 'TRANSFER'



dfFraudTransfer = df.loc[(df.isFraud == 1) & (df.type == 'TRANSFER')]

dfFraudCashout = df.loc[(df.isFraud == 1) & (df.type == 'CASH_OUT')]



print ('\n The number of fraudulent TRANSFERs = {}'.\

       format(len(dfFraudTransfer))) # 4097



print ('\n The number of fraudulent CASH_OUTs = {}'.\

       format(len(dfFraudCashout))) # 4116
print('\nThe type of transactions in which isFlaggedFraud is set: \

{}'.format(list(df.loc[df.isFlaggedFraud == 1].type.drop_duplicates()))) 

                                                            # only 'TRANSFER'



dfTransfer = df.loc[df.type == 'TRANSFER']

dfFlagged = df.loc[df.isFlaggedFraud == 1]

dfNotFlagged = df.loc[df.isFlaggedFraud == 0]



print('\nMin amount transacted when isFlaggedFraud is set= {}'\

                                  .format(dfFlagged.amount.min())) # 353874.22



print('\nMax amount transacted in a TRANSFER where isFlaggedFraud is not set=\

 {}'.format(dfTransfer.loc[dfTransfer.isFlaggedFraud == 0].amount.max())) # 92445516.64
print('\nThe number of TRANSFERs where isFlaggedFraud = 0, yet oldBalanceDest = 0 and\

 newBalanceDest = 0: {}'.\

format(len(dfTransfer.loc[(dfTransfer.isFlaggedFraud == 0) & \

(dfTransfer.oldBalanceDest == 0) & (dfTransfer.newBalanceDest == 0)]))) # 4158
print('\nMin, Max of oldBalanceOrig for isFlaggedFraud = 1 TRANSFERs: {}'.\

format([round(dfFlagged.oldBalanceOrig.min()), round(dfFlagged.oldBalanceOrig.max())]))



print('\nMin, Max of oldBalanceOrig for isFlaggedFraud = 0 TRANSFERs where \

oldBalanceOrig = \

newBalanceOrig: {}'.format(\

[dfTransfer.loc[(dfTransfer.isFlaggedFraud == 0) & (dfTransfer.oldBalanceOrig \

== dfTransfer.newBalanceOrig)].oldBalanceOrig.min(), \

round(dfTransfer.loc[(dfTransfer.isFlaggedFraud == 0) & (dfTransfer.oldBalanceOrig \

               == dfTransfer.newBalanceOrig)].oldBalanceOrig.max())]))
print('\nHave originators of transactions flagged as fraud transacted more than \

once? {}'\

.format((dfFlagged.nameOrig.isin(pd.concat([dfNotFlagged.nameOrig, \

                                    dfNotFlagged.nameDest]))).any())) # False



print('\nHave destinations for transactions flagged as fraud initiated\

 other transactions? \

{}'.format((dfFlagged.nameDest.isin(dfNotFlagged.nameOrig)).any())) # False



# Since only 2 destination accounts of 16 that have 'isFlaggedFraud' set have been

# destination accounts more than once,

# clearly 'isFlaggedFraud' being set is independent of whether a 

# destination account has been used before or not



print('\nHow many destination accounts of transactions flagged as fraud have been \

destination accounts more than once?: {}'\

.format(sum(dfFlagged.nameDest.isin(dfNotFlagged.nameDest)))) # 2
print('\nAre there any merchants among originator accounts for CASH_IN \

transactions? {}'.format(\

(df.loc[df.type == 'CASH_IN'].nameOrig.str.contains('M')).any())) # False
print('\nAre there any merchants among destination accounts for CASH_OUT \

transactions? {}'.format(\

(df.loc[df.type == 'CASH_OUT'].nameDest.str.contains('M')).any())) # False
print('\nAre there merchants among any originator accounts? {}'.format(\

      df.nameOrig.str.contains('M').any())) # False



print('\nAre there any transactions having merchants among destination accounts\

 other than the PAYMENT type? {}'.format(\

(df.loc[df.nameDest.str.contains('M')].type != 'PAYMENT').any())) # False
print('\nWithin fraudulent transactions, are there destinations for TRANSFERS \

that are also originators for CASH_OUTs? {}'.format(\

(dfFraudTransfer.nameDest.isin(dfFraudCashout.nameOrig)).any())) # False

dfNotFraud = df.loc[df.isFraud == 0]
print('\nFraudulent TRANSFERs whose destination accounts are originators of \

genuine CASH_OUTs: \n\n{}'.format(dfFraudTransfer.loc[dfFraudTransfer.nameDest.\

isin(dfNotFraud.loc[dfNotFraud.type == 'CASH_OUT'].nameOrig.drop_duplicates())]))
print('\nFraudulent TRANSFER to C423543548 occured at step = 486 whereas \

genuine CASH_OUT from this account occured earlier at step = {}'.format(\

dfNotFraud.loc[(dfNotFraud.type == 'CASH_OUT') & (dfNotFraud.nameOrig == \

                           'C423543548')].step.values)) # 185
X = df.loc[(df.type == 'TRANSFER') | (df.type == 'CASH_OUT')]



randomState = 5

np.random.seed(randomState)



#X = X.loc[np.random.choice(X.index, 100000, replace = False)]



Y = X['isFraud']

del X['isFraud']



# Eliminate columns shown to be irrelevant for analysis in the EDA

X = X.drop(['nameOrig', 'nameDest', 'isFlaggedFraud'], axis = 1)



# Binary-encoding of labelled data in 'type'

X.loc[X.type == 'TRANSFER', 'type'] = 0

X.loc[X.type == 'CASH_OUT', 'type'] = 1

X.type = X.type.astype(int) # convert dtype('O') to dtype(int)
Xfraud = X.loc[Y == 1]

XnonFraud = X.loc[Y == 0]

print('\nThe fraction of fraudulent transactions with \'oldBalanceDest\' = \

\'newBalanceDest\' = 0 although the transacted \'amount\' is non-zero is: {}'.\

format(len(Xfraud.loc[(Xfraud.oldBalanceDest == 0) & \

(Xfraud.newBalanceDest == 0) & (Xfraud.amount)]) / (1.0 * len(Xfraud))))



print('\nThe fraction of genuine transactions with \'oldBalanceDest\' = \

newBalanceDest\' = 0 although the transacted \'amount\' is non-zero is: {}'.\

format(len(XnonFraud.loc[(XnonFraud.oldBalanceDest == 0) & \

(XnonFraud.newBalanceDest == 0) & (XnonFraud.amount)]) / (1.0 * len(XnonFraud))))
X.loc[(X.oldBalanceDest == 0) & (X.newBalanceDest == 0) & (X.amount != 0), \

      ['oldBalanceDest', 'newBalanceDest']] = - 1
X.loc[(X.oldBalanceOrig == 0) & (X.newBalanceOrig == 0) & (X.amount != 0), \

      ['oldBalanceOrig', 'newBalanceOrig']] = np.nan
X['errorBalanceOrig'] = X.newBalanceOrig + X.amount - X.oldBalanceOrig

X['errorBalanceDest'] = X.oldBalanceDest + X.amount - X.newBalanceDest
limit = len(X)



def plotStrip(x, y, hue, figsize = (14, 9)):

    

    fig = plt.figure(figsize = figsize)

    colours = plt.cm.tab10(np.linspace(0, 1, 9))

    with sns.axes_style('ticks'):

        ax = sns.stripplot(x, y, \

             hue = hue, jitter = 0.4, marker = '.', \

             size = 4, palette = colours)

        ax.set_xlabel('')

        ax.set_xticklabels(['genuine', 'fraudulent'], size = 16)

        for axis in ['top','bottom','left','right']:

            ax.spines[axis].set_linewidth(2)



        handles, labels = ax.get_legend_handles_labels()

        plt.legend(handles, ['Transfer', 'Cash out'], bbox_to_anchor=(1, 1), \

               loc=2, borderaxespad=0, fontsize = 16);

    return ax
ax = plotStrip(Y[:limit], X.step[:limit], X.type[:limit])

ax.set_ylabel('time [hour]', size = 16)

ax.set_title('Striped vs. homogenous fingerprints of genuine and fraudulent \

transactions over time', size = 20);
limit = len(X)

ax = plotStrip(Y[:limit], X.amount[:limit], X.type[:limit], figsize = (14, 9))

ax.set_ylabel('amount', size = 16)

ax.set_title('Same-signed fingerprints of genuine \

and fraudulent transactions over amount', size = 18);
limit = len(X)

ax = plotStrip(Y[:limit], - X.errorBalanceDest[:limit], X.type[:limit], \

              figsize = (14, 9))

ax.set_ylabel('- errorBalanceDest', size = 16)

ax.set_title('Opposite polarity fingerprints over the error in \

destination account balances', size = 18);
# Long computation in this cell (~2.5 minutes)

x = 'errorBalanceDest'

y = 'step'

z = 'errorBalanceOrig'

zOffset = 0.02

limit = len(X)



sns.reset_orig() # prevent seaborn from over-riding mplot3d defaults



fig = plt.figure(figsize = (10, 12))

ax = fig.add_subplot(111, projection='3d')



ax.scatter(X.loc[Y == 0, x][:limit], X.loc[Y == 0, y][:limit], \

  -np.log10(X.loc[Y == 0, z][:limit] + zOffset), c = 'g', marker = '.', \

  s = 1, label = 'genuine')

    

ax.scatter(X.loc[Y == 1, x][:limit], X.loc[Y == 1, y][:limit], \

  -np.log10(X.loc[Y == 1, z][:limit] + zOffset), c = 'r', marker = '.', \

  s = 1, label = 'fraudulent')



ax.set_xlabel(x, size = 16); 

ax.set_ylabel(y + ' [hour]', size = 16); 

ax.set_zlabel('- log$_{10}$ (' + z + ')', size = 16)

ax.set_title('Error-based features separate out genuine and fraudulent \

transactions', size = 20)



plt.axis('tight')

ax.grid(1)



noFraudMarker = mlines.Line2D([], [], linewidth = 0, color='g', marker='.',

                          markersize = 10, label='genuine')

fraudMarker = mlines.Line2D([], [], linewidth = 0, color='r', marker='.',

                          markersize = 10, label='fraudulent')



plt.legend(handles = [noFraudMarker, fraudMarker], \

           bbox_to_anchor = (1.20, 0.38 ), frameon = False, prop={'size': 16});
Xfraud = X.loc[Y == 1] # update Xfraud & XnonFraud with cleaned data

XnonFraud = X.loc[Y == 0]

                  

correlationNonFraud = XnonFraud.loc[:, X.columns != 'step'].corr()

mask = np.zeros_like(correlationNonFraud)

indices = np.triu_indices_from(correlationNonFraud)

mask[indices] = True



grid_kws = {"width_ratios": (.9, .9, .05), "wspace": 0.2}

f, (ax1, ax2, cbar_ax) = plt.subplots(1, 3, gridspec_kw=grid_kws, \

                                     figsize = (14, 9))



cmap = sns.diverging_palette(220, 8, as_cmap=True)

ax1 =sns.heatmap(correlationNonFraud, ax = ax1, vmin = -1, vmax = 1, \

    cmap = cmap, square = False, linewidths = 0.5, mask = mask, cbar = False)

ax1.set_xticklabels(ax1.get_xticklabels(), size = 16); 

ax1.set_yticklabels(ax1.get_yticklabels(), size = 16); 

ax1.set_title('Genuine \n transactions', size = 20)



correlationFraud = Xfraud.loc[:, X.columns != 'step'].corr()

ax2 = sns.heatmap(correlationFraud, vmin = -1, vmax = 1, cmap = cmap, \

 ax = ax2, square = False, linewidths = 0.5, mask = mask, yticklabels = False, \

    cbar_ax = cbar_ax, cbar_kws={'orientation': 'vertical', \

                                 'ticks': [-1, -0.5, 0, 0.5, 1]})

ax2.set_xticklabels(ax2.get_xticklabels(), size = 16); 

ax2.set_title('Fraudulent \n transactions', size = 20);



cbar_ax.set_yticklabels(cbar_ax.get_yticklabels(), size = 14);
print('skew = {}'.format( len(Xfraud) / float(len(X)) ))
trainX, testX, trainY, testY = train_test_split(X, Y, test_size = 0.2, \

                                                random_state = randomState)
# Long computation in this cell (~1.8 minutes)

weights = (Y == 0).sum() / (1.0 * (Y == 1).sum())

clf = XGBClassifier(max_depth = 3, scale_pos_weight = weights, \

                n_jobs = 4)

probabilities = clf.fit(trainX, trainY).predict_proba(testX)

print('AUPRC = {}'.format(average_precision_score(testY, \

                                              probabilities[:, 1])))
fig = plt.figure(figsize = (14, 9))

ax = fig.add_subplot(111)



colours = plt.cm.Set1(np.linspace(0, 1, 9))



ax = plot_importance(clf, height = 1, color = colours, grid = False, \

                     show_values = False, importance_type = 'cover', ax = ax);

for axis in ['top','bottom','left','right']:

            ax.spines[axis].set_linewidth(2)

        

ax.set_xlabel('importance score', size = 16);

ax.set_ylabel('features', size = 16);

ax.set_yticklabels(ax.get_yticklabels(), size = 12);

ax.set_title('Ordering of features by importance to the model learnt', size = 20);
to_graphviz(clf)
# Long computation in this cell (~6 minutes)



trainSizes, trainScores, crossValScores = learning_curve(\

XGBClassifier(max_depth = 3, scale_pos_weight = weights, n_jobs = 4), trainX,\

                                         trainY, scoring = 'average_precision')
trainScoresMean = np.mean(trainScores, axis=1)

trainScoresStd = np.std(trainScores, axis=1)

crossValScoresMean = np.mean(crossValScores, axis=1)

crossValScoresStd = np.std(crossValScores, axis=1)



colours = plt.cm.tab10(np.linspace(0, 1, 9))



fig = plt.figure(figsize = (14, 9))

plt.fill_between(trainSizes, trainScoresMean - trainScoresStd,

    trainScoresMean + trainScoresStd, alpha=0.1, color=colours[0])

plt.fill_between(trainSizes, crossValScoresMean - crossValScoresStd,

    crossValScoresMean + crossValScoresStd, alpha=0.1, color=colours[1])

plt.plot(trainSizes, trainScores.mean(axis = 1), 'o-', label = 'train', \

         color = colours[0])

plt.plot(trainSizes, crossValScores.mean(axis = 1), 'o-', label = 'cross-val', \

         color = colours[1])



ax = plt.gca()

for axis in ['top','bottom','left','right']:

    ax.spines[axis].set_linewidth(2)



handles, labels = ax.get_legend_handles_labels()

plt.legend(handles, ['train', 'cross-val'], bbox_to_anchor=(0.8, 0.15), \

               loc=2, borderaxespad=0, fontsize = 16);

plt.xlabel('training set size', size = 16); 

plt.ylabel('AUPRC', size = 16)

plt.title('Learning curves indicate slightly underfit model', size = 20);