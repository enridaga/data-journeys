
# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



# import necessary packages

import matplotlib.pylab as plt

from scipy import interp

from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import roc_curve,auc

from sklearn.model_selection import StratifiedKFold

import matplotlib.patches as patches

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
# import data, see feature names, label count and data info

data = pd .read_csv('../input/voice.csv')

print(data.columns)

label_value_count = data.label.value_counts()

print(label_value_count)

print(data.info())
# Convert string label to float : male = 1, female = 0

dict = {'label':{'male':1,'female':0}}      # label = column name

data.replace(dict,inplace = True)           # replace = str to numerical

x = data.loc[:, data.columns != 'label']

y = data.loc[:,'label']
random_state = np.random.RandomState(0)

clf = RandomForestClassifier(random_state=random_state)

cv = StratifiedKFold(n_splits=5,shuffle=False)

# plot arrows

fig1 = plt.figure(figsize=[12,12])

ax1 = fig1.add_subplot(111,aspect = 'equal')

ax1.add_patch(

    patches.Arrow(0.45,0.5,-0.25,0.25,width=0.3,color='green',alpha = 0.5)

    )

ax1.add_patch(

    patches.Arrow(0.5,0.45,0.25,-0.25,width=0.3,color='red',alpha = 0.5)

    )



tprs = []

aucs = []

mean_fpr = np.linspace(0,1,100)

i = 1

for train,test in cv.split(x,y):

    prediction = clf.fit(x.iloc[train],y.iloc[train]).predict(x.iloc[test])

    fpr, tpr, t = roc_curve(y[test], prediction)

    tprs.append(interp(mean_fpr, fpr, tpr))

    roc_auc = auc(fpr, tpr)

    aucs.append(roc_auc)

    plt.plot(fpr, tpr, lw=2, alpha=0.3, label='ROC fold %d (AUC = %0.2f)' % (i, roc_auc))

    i= i+1



plt.plot([0,1],[0,1],linestyle = '--',lw = 2,color = 'black')

mean_tpr = np.mean(tprs, axis=0)

mean_auc = auc(mean_fpr, mean_tpr)

plt.plot(mean_fpr, mean_tpr, color='blue',

         label=r'Mean ROC (AUC = %0.2f )' % (mean_auc),lw=2, alpha=1)



plt.xlabel('False Positive Rate')

plt.ylabel('True Positive Rate')

plt.title('ROC')

plt.legend(loc="lower right")

plt.text(0.32,0.7,'More accurate area',fontsize = 12)

plt.text(0.63,0.4,'Less accurate area',fontsize = 12)

plt.show()