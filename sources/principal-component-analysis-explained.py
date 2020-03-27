
# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib as mpl

import matplotlib.pyplot as plt

import seaborn as sns

#%matplotlib inline
df = pd.read_csv('../input/HR_comma_sep.csv')
columns_names=df.columns.tolist()

print("Columns names:")

print(columns_names)
df = pd.read_csv('../input/HR_comma_sep.csv')

df.head
df.corr()
correlation = df.corr()

plt.figure(figsize=(10,10))

sns.heatmap(correlation, vmax=1, square=True,annot=True,cmap='cubehelix')



plt.title('Correlation between different fearures')
df['sales'].unique()
sales=df.groupby('sales').sum()

sales
df['sales'].unique()
groupby_sales=df.groupby('sales').mean()

groupby_sales
IT=groupby_sales['satisfaction_level'].IT

RandD=groupby_sales['satisfaction_level'].RandD

accounting=groupby_sales['satisfaction_level'].accounting

hr=groupby_sales['satisfaction_level'].hr

management=groupby_sales['satisfaction_level'].management

marketing=groupby_sales['satisfaction_level'].marketing

product_mng=groupby_sales['satisfaction_level'].product_mng

sales=groupby_sales['satisfaction_level'].sales

support=groupby_sales['satisfaction_level'].support

technical=groupby_sales['satisfaction_level'].technical

technical


department_name=('sales', 'accounting', 'hr', 'technical', 'support', 'management',

       'IT', 'product_mng', 'marketing', 'RandD')

department=(sales, accounting, hr, technical, support, management,

       IT, product_mng, marketing, RandD)

y_pos = np.arange(len(department))

x=np.arange(0,1,0.1)



plt.barh(y_pos, department, align='center', alpha=0.8)

plt.yticks(y_pos,department_name )

plt.xlabel('Satisfaction level')

plt.title('Mean Satisfaction Level of each department')
#Principal Component Analysis
df = pd.read_csv('../input/HR_comma_sep.csv')

df.head(5)
column_names=df.columns.tolist()

print("Column Names:")

print(column_names)
df.info()
df.head()
mid = df['left']

df.drop(labels=['left'], axis=1,inplace = True)

df.insert(0, 'left', left)



df.head()
X = df.iloc[:,1:7].values

y = df.iloc[:,0].values

X,y

df = pd.read_csv('../input/HR_comma_sep.csv')

df.head()
df_drop=df.drop(labels=['Work_accident','sales','salary','promotion_last_5years'],axis=1)

df_drop.head()
X = df_drop.iloc[:0:5].values

y = df_drop.iloc[:,-1].values

X

y

#np.unique(y)

from matplotlib import pyplot as plt

import numpy as np

import math



label_dict = {0: 'Not_Left',

              1: 'Left'

             }



feature_dict = {0: 'satisfaction_level',

                1: 'last_evaluation',

                2: 'number_project',

                3: 'average_montly_hours',

                4: 'time_spend_company'

               }



with plt.style.context('seaborn-whitegrid'):

    plt.figure(figsize=(8, 6))

    for cnt in range(5):

        plt.subplot(3, 2, cnt+1)

        for lab in ('Not_Left', 'Left'):

            plt.hist(X[y==lab, cnt],

                     label=lab,

                     bins=10,

                     alpha=0.3,)

        plt.xlabel(feature_dict[cnt])

    plt.legend(loc='upper right', fancybox=True, fontsize=8)



    plt.tight_layout()

    plt.show()