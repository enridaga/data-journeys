
# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
### matplotlib inline
import warnings
warnings.filterwarnings('ignore')
from sklearn.preprocessing import LabelEncoder
import seaborn as sns
df_ele=pd.read_csv('/kaggle/input/education-in-india/2015_16_Statewise_Elementary.csv')
df_ele.head()
df_ele.shape
df_ele.columns
df_ele.dtypes
df_ele.isnull().sum()

plt.figure(figsize=(19,20))
plt.xticks(rotation='vertical')
plt.xlabel('Population', fontsize=14)
plt.ylabel('States In India', fontsize=14)
plt.title("Population wrt to States in India", fontsize=16)
sns.barplot(df_ele['TOTPOPULAT'],df_ele['STATNAME'])
plt.show()
col=['STATNAME','OVERALL_LI','FEMALE_LIT','MALE_LIT','SEXRATIO','DISTRICTS','TOTPOPULAT','P_URB_POP','P_RUR_POP','AREA_SQKM','SCHTOT','SCHTOTG','SCHTOTGR','SCHTOTPR','ENRTOT','ENRTOTG','ENRTOTGR', 
       'ENRTOTPR', 'TCHTOTG', 'TCHTOTP', 'SCLSTOT', 'STCHTOT', 'ROADTOT', 'SPLAYTOT', 'SWATTOT',  'SELETOT']
#df_ele['ENRTOTPR']
df_ele1=pd.DataFrame(df_ele,columns=col)
df_ele1.head(1)
df_ele.loc[18,'TOTPOPULAT'] = df_ele.loc[18,'TOTPOPULAT']/10
plt.figure(figsize=(10,12))
sns.barplot( df_ele1['TOTPOPULAT'],df_ele1['STATNAME'], alpha=0.8)
plt.xticks(rotation='vertical')
plt.xlabel('Population', fontsize=14)
plt.ylabel('States in India', fontsize=14)
plt.title("Population wrt states in India", fontsize=16)
plt.show()
plt.figure(figsize=(10,12))
sns.barplot( df_ele1['SCHTOT'],df_ele1['STATNAME'], alpha=0.8)
plt.xticks(rotation='vertical')
plt.xlabel('No. of Schools', fontsize=14)
plt.ylabel('States in India', fontsize=14)
plt.title("Schools wrt states in India", fontsize=16)
plt.show()
plt.figure(figsize=(20,12))
for i in range(1,len(df_ele1)):
    plt.subplot(4,9,i)
    plt.title(df_ele1['STATNAME'][i])
    top = ['Gov','pri']
    uttar = df_ele1.loc[df_ele1['STATNAME'] == df_ele1['STATNAME'][i],:]
    value =[float(uttar['SCHTOTG']/uttar['SCHTOT'])*100,float(uttar['SCHTOTPR']/uttar['SCHTOT'])*100]
    plt.pie(value, labels=top, autopct='%1.1f%%',startangle=140)
    plt.axis('equal')
plt.show()
plt.figure(figsize=(10,12))
sns.barplot( df_ele['OVERALL_LI'],df_ele['STATNAME'], alpha=0.8)
plt.xticks(rotation='vertical')
plt.xlabel('Literacy rate', fontsize=14)
plt.ylabel('States in India', fontsize=14)
plt.title("Literacy Rate wrt states in India", fontsize=16)
plt.show()
plt.figure(figsize=(10,12))
sns.barplot( df_ele['FEMALE_LIT'],df_ele['STATNAME'], alpha=0.8)
plt.xticks(rotation='vertical')
plt.xlabel('Female Literacy rate', fontsize=14)
plt.ylabel('States in India', fontsize=14)
plt.title("Literacy Rate wrt states in India", fontsize=16)
plt.show()
#Let us find the difference between the top  3 and bottom 3 statess in female literacy rate.
top_3=df_ele1.sort_values(by='OVERALL_LI', ascending=False).head(3)
top_3
bottom_3=df_ele1.sort_values(by='OVERALL_LI', ascending=True).head(3)
bottom_3
#Let us concatenate top and bottom states
top_bottom=pd.concat([top_3, bottom_3], axis=0, sort=False)
top_bottom
#Let us see how lteracy rate is affected
#1.Let us first check the total population
top_bottom
#Let us see whther thes states are over crowded or not bychechking the population density.
top_bottom['Pop_DEN']=(top_bottom['TOTPOPULAT']/top_bottom['AREA_SQKM'])*1000
top_bottom['Pop_DEN']
#top_bottom.TOTPOPULAT/top_bottom.AREA_SQKM*1000
#population density is not the factor
#2.Let us find th difference in female and male literacy rate
top_bottom['Diff_Lit']=top_bottom['MALE_LIT']-top_bottom['FEMALE_LIT']
top_bottom['Diff_Lit']
#top_bottom['Diff_Lit'].plot(y='Diff_Lit',kind='bar', alpha=0.8)
plt.figure(figsize=(10,8))
sns.barplot( top_bottom['STATNAME'],top_bottom['Diff_Lit'], alpha=0.8)
plt.title("Lteracy rate f Vs M")
#with low literacy states have high differences. 
#Bottom states have high male litracy rates but female literacy rates on these states is low, which makes over all literacy rate down.

#3.Let us check urban and rural population in terms of loteracy rate. 
#we dont have data of rural population. Therefore, rural=100-urban
top_bottom['P_RUR_POP']=100-top_bottom['P_URB_POP']
top_bottom['P_RUR_POP']
plt.figure(figsize=(10,8))
#sns.barplot(top_bottom['STATNAME'], y= top_bottom['P_RUR_POP'])
#top_bottom.plot(y=['P_RUR_POP','P_URB_POP'], kind='bar')
#sns.barplot('P_RUR_POP','STATNAME')
#sns.barplot( top_bottom['STATNAME'])
top_bottom.plot(y = ['P_URB_POP', 'P_RUR_POP'], kind = 'bar')
#We can see that rural population in bottom3 is very high as compared to rural population in top 3
#Its an impt factor.
#It means people living rural areas live dffrent lifestyle than in urban life.
#Therefore, less motivation to go to school in rural areas.

#4. Let us find the sex ratio
top_bottom['SEXRATIO']
plt.figure(figsize=(10,8))
sns.barplot(top_bottom['STATNAME'], top_bottom['SEXRATIO'])

df_ele.loc[:,['SEXRATIO','OVERALL_LI']].corr()

#The graph dont show the sex ratio affecting literacy rate.
#The sex ratio has nothing to do with literacy rate.