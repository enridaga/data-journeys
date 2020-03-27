
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
df = pd.read_csv('../input/forest-fires-in-brazil/amazon.csv', encoding='Latin-1')
df.head(10)
df['year'].describe()
df['year'] = df['year'].astype(int)
months = {'Janeiro':'January', 'Fevereiro':'February', 'Março':'March', 'Abril':'April', 'Maio':'May', 
          'Junho':'June', 'Julho':'July', 'Agosto':'August', 'Setembro':'September', 'Outubro':'October',
         'Novembro':'November', 'Dezembro':'December'}
df['month'] = df['month'].map(months)
print(df['month'].unique())
df['state'].unique()
df['state'] = df['state'].str.replace('á', 'a')
df['state'] = df['state'].str.replace('Piau', 'Piaui')
df['date'].value_counts()
df_year = df[['year', 'number']]
df_year = df_year.groupby('year').sum()
df_year.head()
import matplotlib.ticker as ticker
sns.set(rc = {'axes.facecolor':'#bbbbbb'})
plt.figure(figsize=(15,4))
sns.set_context(rc = {'lines.linewidth':3})
g = sns.lineplot(x=df_year.index, y='number', data=df_year)
g.xaxis.set_major_locator(ticker.MultipleLocator(1))
df_amazonas = df.loc[df['state']=='Amazonas']
df_amazonas = df_amazonas[['year', 'number']].groupby('year', as_index=False).sum()
sns.set(rc = {'axes.facecolor':'#bbbbbb'})
plt.figure(figsize=(15,4))
sns.set_context(rc = {'lines.linewidth':3})
g = sns.lineplot(x='year', y='number', data=df_amazonas)
g.xaxis.set_major_locator(ticker.MultipleLocator(1))
amazon_states = ['Acre','Amapa', 'Amazonas', 'Rondonia', 'Roraiama', 'Para', 'Tocantins']
filter_am = df['state']=='Acre'
for i in range(1, 7):
    filter_am = filter_am|(df['state']==amazon_states[i])
df_amstates = df.loc[filter_am, ['year', 'number','state']]
df_amstates = df_amstates.groupby('year', as_index=False).sum()
sns.set(rc = {'axes.facecolor':'#bbbbbb'})
plt.figure(figsize=(15,4))
sns.set_context(rc = {'lines.linewidth':3})
g = sns.lineplot(x='year', y='number', data=df_amstates)
g.xaxis.set_major_locator(ticker.MultipleLocator(1))
df_st = df[['state', 'number']]
df_st = df_st.groupby('state', as_index=False).sum()
df_st_sorted = pd.DataFrame()
df_st_sorted['number'] = df_st['number'].sort_values(ascending=False)
df_st_sorted['state'] = df_st['state']
df_st_sorted = df_st_sorted.dropna()
sns.set(rc = {'axes.facecolor':'#bbbbbb'})
g = sns.catplot(x='state', y='number', data=df_st_sorted, kind='bar', ci=None, palette='Oranges_r')
g.fig.set_figwidth(18)
g.fig.set_figheight(8)
g.set_xticklabels(rotation=45)
plt.title('Total Number of Fires per State(1998-2017)', size=16)
plt.xlabel('States')
plt.ylabel('Number of Fires')
state_area = pd.Series(index = ['Mato Grosso', 'Paraiba', 'Sao Paulo', 'Rio', 'Bahia', 'Piaui', 'Goias', 
                                'Minas Gerais', 'Tocantins', 'Amazonas', 'Ceara', 'Maranhao', 'Para',
                               'Pernambuco', 'Roraima', 'Santa Catarina', 'Amapa', 'Rondonia', 'Acre', 
                                'Espirito Santo', 'Alagoas', 'Distrito Federal', 'Sergipe'],
                      data = [903366.192, 56585, 248222.362, 43780.172, 564733.177, 251577.738, 340111.783,
                            586522.122, 277720.520, 1559159.148, 148920.472, 331937.450, 1247954.666,
                            98311.616, 224300.506, 95736.165, 142828.521, 237590.547, 164123.040,
                            46095.583, 27778.506, 5779.999, 21915.116])
                
for state in df_st_sorted['state']:
    df_st_sorted.loc[df_st_sorted['state']==state, 'state_area'] = state_area[state]
df_st_sorted.head()
df_st_sorted['fires_per_km2'] = df_st_sorted['number']/df_st_sorted['state_area']
df_km_sorted = pd.DataFrame()
df_km_sorted['fires_per_km2'] = df_st_sorted['fires_per_km2'].sort_values(ascending=False)
df_km_sorted['state'] = df_st['state']
sns.set(rc = {'axes.facecolor':'#bbbbbb'})
g = sns.catplot(x='state', y='fires_per_km2', data=df_km_sorted, kind='bar', ci=None, palette='Oranges_r')
g.fig.set_figwidth(18)
g.fig.set_figheight(8)
g.set_xticklabels(rotation=45)
plt.title('Fires per km2 by State', size=16)
plt.xlabel('States', size=12)
plt.ylabel('Fires per km2 (1998-2017)', size=12)
df_month = df[['month', 'number']].groupby('month', as_index=False).sum()
months = pd.Series(df['month'].unique())
df_month_org = pd.DataFrame()
df_month_org['month'] = months
df_month_org['number'] = df_month['number']
sns.set(rc = {'axes.facecolor':'#bbbbbb'})
g = sns.catplot(x = 'month', y = 'number', data = df_month_org, kind = 'bar', ci = None, palette='hls')
plt.title('Fires per Month', size=16)
plt.ylabel('Number of Fires', size=12)
plt.xlabel('Months', size=12)
g.fig.set_figheight(10)
g.fig.set_figwidth(20)
sns.set(rc = {'axes.facecolor':'#bbbbbb'})
fig = plt.figure(figsize=(16, 80))
states = df['state'].unique()
k = 0
for state in states:
    ax = fig.add_subplot(12, 2, k+1)
    subset = df[(df['state']==state)].groupby('month', as_index = False).sum()
    subset_sorted = pd.DataFrame()
    subset_sorted['month'] = months
    subset_sorted['number'] = subset['number']
    ax.plot(subset_sorted['month'], subset_sorted['number'], c='red')
    plt.title('Number of Fires per Month in {} (1998-2017)'.format(str(state)), size =13, fontweight='demi')
    plt.xticks(rotation=45, size=11, y=0.02)
    plt.ylabel('Number of Fires')
    plt.tick_params(direction="in")
    k+=1
plt.show()
df_mt = df.loc[df['state'] == 'Mato Grosso']
sns.set(rc = {'axes.facecolor':'#bbbbbb'})
fig = plt.figure(figsize=(18,12))
years = [2012, 2013, 2014, 2015, 2016, 2017]
colors = ['red', 'blue', 'green', 'yellow', 'black', 'purple']
#only a few years were chosen in oreder to make the graph with a cleaner look
#color are the color that will be used
k = 0
for year in years:
    subset = df_mt.loc[df_mt['year']==year]
    subset = subset.groupby('month', as_index = False).sum()
    #subset is the dataframe of the year that will be used grouped by month
    subset_sorted = pd.DataFrame()
    subset_sorted['month'] = months
    subset_sorted['number'] = subset['number']
    #subset_sorted puts the months in the correct order
    plt.plot(subset_sorted['month'], subset_sorted['number'], label=year, c=colors[k], linewidth=3)
    k += 1
plt.legend(prop={'size':'17'})
plt.xticks(size=13)
plt.title('Fires per Month in Mato Gross (2012-2017)', size=16)
plt.show()
