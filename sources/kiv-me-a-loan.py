
# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import missingno as msno
from datetime import datetime, timedelta


import matplotlib.pyplot as plt
### matplotlib inline
import seaborn as sns
plt.style.use('seaborn-darkgrid')
palette = plt.get_cmap('Set1')

import plotly.offline as py
py.init_notebook_mode(connected=True)
import plotly.graph_objs as go
import plotly.tools as tls
import plotly.figure_factory as ff


# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os

# Any results you write to the current directory are saved as output.
df_kiva_loans = pd.read_csv("../input/data-science-for-good-kiva-crowdfunding/kiva_loans.csv")
df_loc = pd.read_csv("../input/data-science-for-good-kiva-crowdfunding/loan_themes_by_region.csv")
df_themes = pd.read_csv("../input/data-science-for-good-kiva-crowdfunding/loan_theme_ids.csv")
df_mpi = pd.read_csv("../input/data-science-for-good-kiva-crowdfunding/kiva_mpi_region_locations.csv")

df_kiva_loans.head(5)
msno.matrix(df_kiva_loans)
df_kiva_loans.describe(include = 'all')
countries = df_kiva_loans['country'].value_counts()[df_kiva_loans['country'].value_counts()>3400]
list_countries = list(countries.index) #this is the list of countries that will be most used.
plt.figure(figsize=(12,8))
sns.barplot(y=countries.index, x=countries.values, alpha=0.6)
plt.title("Country distribution of Kiva's users", fontsize=16)
plt.xlabel("Number of borrowers", fontsize=16)
plt.ylabel("Country", fontsize=16)
plt.show();
df_kiva_loans['borrower_genders']=[elem if elem in ['female','male'] else 'group' for elem in df_kiva_loans['borrower_genders'] ]
borrowers = df_kiva_loans['borrower_genders'].value_counts()
labels = (np.array(borrowers.index))

values = (np.array((borrowers / borrowers.sum())*100))

trace = go.Pie(labels=labels, values=values,
              hoverinfo='label+percent',
               textfont=dict(size=20),
                showlegend=True)

layout = go.Layout(
    title="Borrowers' genders"
)

data_trace = [trace]
fig = go.Figure(data=data_trace, layout=layout)
py.iplot(fig, filename="Borrowers_genders")

plt.figure(figsize=(12,8))
sectors = df_kiva_loans['sector'].value_counts()
sns.barplot(y=sectors.index, x=sectors.values, alpha=0.6)
plt.title("Country Distribution of the suervey participants", fontsize=16)
plt.xlabel("Number of participants", fontsize=16)
plt.ylabel("Sector", fontsize=16)
plt.show();
plt.figure(figsize=(15,10))
activities = df_kiva_loans['activity'].value_counts().head(50)
sns.barplot(y=activities.index, x=activities.values, alpha=0.6)
plt.title("Country Distribution of the suervey participants", fontsize=16)
plt.xlabel("Number of participants", fontsize=16)
plt.ylabel("Activity", fontsize=16)
plt.show();
plt.figure(figsize=(10,6))
sns.distplot(df_kiva_loans['loan_amount']);
temp = df_kiva_loans['loan_amount']

plt.figure(figsize=(10,6))
sns.distplot(temp[~((temp-temp.mean()).abs()>3*temp.std())] );
loans_dates = df_kiva_loans.dropna(subset=['disbursed_time', 'funded_time'], how='any', inplace=False)

dates = ['posted_time','disbursed_time','funded_time']
loans_dates[dates] = loans_dates[dates].applymap(lambda x : x.split('+')[0])

loans_dates[dates]=loans_dates[dates].apply(pd.to_datetime)
loans_dates['time_funding']=loans_dates['funded_time']-loans_dates['posted_time']
loans_dates['time_funding'] = loans_dates['time_funding'] / timedelta(days=1) 
#this last line gives us the value for waiting time in days and float format,
# for example: 3 days 12 hours = 3.5
temp = loans_dates['time_funding']

plt.figure(figsize=(10,6))
sns.distplot(temp[~((temp-temp.mean()).abs()>3*temp.std())] );
df_ctime = round(loans_dates.groupby(['country'])['time_funding'].median(),2)
df_camount = round(df_kiva_loans.groupby(['country'])['loan_amount'].median(),2)
df_camount = df_camount[df_camount.index.isin(list_countries)].sort_values()
df_ctime = df_ctime[df_ctime.index.isin(list_countries)].sort_values()

f,ax=plt.subplots(1,2,figsize=(20,10))

sns.barplot(y=df_camount.index, x=df_camount.values, alpha=0.6, ax=ax[0])
ax[0].set_title("Medians of funding amounts per loan country wise ")
ax[0].set_xlabel('Amount in dollars')
ax[0].set_ylabel("Country")

sns.barplot(y=df_ctime.index, x=df_ctime.values, alpha=0.6,ax=ax[1])
ax[1].set_title("Medians of waiting days per loan to be funded country wise  ")
ax[1].set_xlabel('Number of days')
ax[1].set_ylabel("")

plt.tight_layout()
plt.show();
lenders = pd.read_csv('../input/additional-kiva-snapshot/lenders.csv')
lenders.head()
lender_countries = lenders.groupby(['country_code']).count()[['permanent_name']].reset_index()
lender_countries.columns = ['country_code', 'Number of Lenders']
lender_countries.sort_values(by='Number of Lenders', ascending=False,inplace=True)
lender_countries.head()
countries_data = pd.read_csv( '../input/additional-kiva-snapshot/country_stats.csv')
countries_data.head()
countries_data = pd.read_csv( '../input/additional-kiva-snapshot/country_stats.csv')
lender_countries = pd.merge(lender_countries, countries_data[['country_name','country_code']],
                            how='inner', on='country_code')

data = [dict(
        type='choropleth',
        locations=lender_countries['country_name'],
        locationmode='country names',
        z=np.log10(lender_countries['Number of Lenders']+1),
        colorscale='Viridis',
        reversescale=False,
        marker=dict(line=dict(color='rgb(180,180,180)', width=0.5)),
        colorbar=dict(autotick=False, tickprefix='', title='Lenders'),
    )]
layout = dict(
    title = 'Lenders per country in a logarithmic scale ',
    geo = dict(showframe=False, showcoastlines=True, projection=dict(type='Mercator'))
)
fig = dict(data=data, layout=layout)
py.iplot(fig, validate=False, filename='lenders-map')
import matplotlib as mpl 
from wordcloud import WordCloud, STOPWORDS
import imageio

heart_mask = imageio.imread('../input/heartmask2/heart_msk.jpg') #because displaying this wordcloud as a heart seems just about right :)

mpl.rcParams['figure.figsize']=(12.0,8.0)    #(6.0,4.0)
mpl.rcParams['font.size']=10                #10 

more_stopwords = {'org', 'default', 'aspx', 'stratfordrec','nhttp','Hi','also','now','much'}
STOPWORDS = STOPWORDS.union(more_stopwords)

lenders_reason = lenders[~pd.isnull(lenders['loan_because'])][['loan_because']]
lenders_reason_string = " ".join(lenders_reason.loan_because.values)

wordcloud = WordCloud(
                      stopwords=STOPWORDS,
                      background_color='white',
                      width=3200, 
                      height=2000,
                      mask=heart_mask
            ).generate(lenders_reason_string)

plt.imshow(wordcloud)
plt.axis("off")
plt.savefig('./reason_wordcloud.png', dpi=900)
plt.show()
df_mpi.head(10)
mpi_country = df_mpi.groupby('country')['MPI'].mean().reset_index()


data = [dict(
        type='choropleth',
        locations=mpi_country['country'],
        locationmode='country names',
        z=mpi_country['MPI'],
        colorscale='Greens',
        reversescale=True,
        marker=dict(line=dict(color='rgb(180,180,180)', width=0.5)),
        colorbar=dict(autotick=False, tickprefix='', title='MPI'),
    )]
layout = dict(
    title = 'Average MPI per country',
    geo = dict(showframe=False, showcoastlines=True, projection=dict(type='Mercator'))
)
fig = dict(data=data, layout=layout)
py.iplot(fig, validate=False, filename='mpi-map')