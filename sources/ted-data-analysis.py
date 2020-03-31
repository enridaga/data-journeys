
### matplotlib inline

import pandas as pd

import numpy as np

from scipy import stats

import matplotlib.pyplot as plt

import seaborn as sns

import json

from pandas.io.json import json_normalize

from wordcloud import WordCloud, STOPWORDS
month_order = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']

day_order = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
df = pd.read_csv('../input/ted_main.csv')

df.columns
df = df[['name', 'title', 'description', 'main_speaker', 'speaker_occupation', 'num_speaker', 'duration', 'event', 'film_date', 'published_date', 'comments', 'tags', 'languages', 'ratings', 'related_talks', 'url', 'views']]
import datetime

df['film_date'] = df['film_date'].apply(lambda x: datetime.datetime.fromtimestamp( int(x)).strftime('%d-%m-%Y'))

df['published_date'] = df['published_date'].apply(lambda x: datetime.datetime.fromtimestamp( int(x)).strftime('%d-%m-%Y'))
df.head()
len(df)
pop_talks = df[['title', 'main_speaker', 'views', 'film_date']].sort_values('views', ascending=False)[:15]

pop_talks
pop_talks['abbr'] = pop_talks['main_speaker'].apply(lambda x: x[:3])

sns.set_style("whitegrid")

plt.figure(figsize=(10,6))

sns.barplot(x='abbr', y='views', data=pop_talks)
sns.distplot(df['views'])
sns.distplot(df[df['views'] < 0.4e7]['views'])
df['views'].describe()
df['comments'].describe()
sns.distplot(df['comments'])
sns.distplot(df[df['comments'] < 500]['comments'])
sns.jointplot(x='views', y='comments', data=df)
df[['views', 'comments']].corr()
df[['title', 'main_speaker','views', 'comments']].sort_values('comments', ascending=False).head(10)
df['dis_quo'] = df['comments']/df['views']
df[['title', 'main_speaker','views', 'comments', 'dis_quo', 'film_date']].sort_values('dis_quo', ascending=False).head(10)
df['month'] = df['film_date'].apply(lambda x: month_order[int(x.split('-')[1]) - 1])



month_df = pd.DataFrame(df['month'].value_counts()).reset_index()

month_df.columns = ['month', 'talks']
sns.barplot(x='month', y='talks', data=month_df, order=month_order)
df_x = df[df['event'].str.contains('TEDx')]

x_month_df = pd.DataFrame(df_x['month'].value_counts().reset_index())

x_month_df.columns = ['month', 'talks']
sns.barplot(x='month', y='talks', data=x_month_df, order=month_order)
def getday(x):

    day, month, year = (int(i) for i in x.split('-'))    

    answer = datetime.date(year, month, day).weekday()

    return day_order[answer]
df['day'] = df['film_date'].apply(getday)
day_df = pd.DataFrame(df['day'].value_counts()).reset_index()

day_df.columns = ['day', 'talks']
sns.barplot(x='day', y='talks', data=day_df, order=day_order)
df['year'] = df['film_date'].apply(lambda x: x.split('-')[2])

year_df = pd.DataFrame(df['year'].value_counts().reset_index())

year_df.columns = ['year', 'talks']
plt.figure(figsize=(18,5))

sns.pointplot(x='year', y='talks', data=year_df)
months = {'Jan': 1, 'Feb': 2, 'Mar': 3, 'Apr': 4, 'May': 5, 'Jun': 6, 'Jul': 7, 'Aug': 8, 'Sep': 9, 'Oct': 10, 'Nov': 11, 'Dec': 12}
hmap_df = df.copy()

hmap_df['film_date'] = hmap_df['film_date'].apply(lambda x: month_order[int(x.split('-')[1]) - 1] + " " + str(x.split('-')[2]))

hmap_df = pd.pivot_table(hmap_df[['film_date', 'title']], index='film_date', aggfunc='count').reset_index()

hmap_df['month_num'] = hmap_df['film_date'].apply(lambda x: months[x.split()[0]])

hmap_df['year'] = hmap_df['film_date'].apply(lambda x: x.split()[1])

hmap_df = hmap_df.sort_values(['year', 'month_num'])

hmap_df = hmap_df[['month_num', 'year', 'title']]

hmap_df = hmap_df.pivot('month_num', 'year', 'title')

hmap_df = hmap_df.fillna(0)
f, ax = plt.subplots(figsize=(12, 8))

sns.heatmap(hmap_df, annot=True, linewidths=.5, ax=ax, fmt='n', yticklabels=month_order)

speaker_df = df.groupby('main_speaker').count().reset_index()[['main_speaker', 'comments']]

speaker_df.columns = ['main_speaker', 'appearances']

speaker_df = speaker_df.sort_values('appearances', ascending=False)

speaker_df.head(10)
occupation_df = df.groupby('speaker_occupation').count().reset_index()[['speaker_occupation', 'comments']]

occupation_df.columns = ['occupation', 'appearances']

occupation_df = occupation_df.sort_values('appearances', ascending=False)
plt.figure(figsize=(15,5))

sns.barplot(x='occupation', y='appearances', data=occupation_df.head(10))

plt.show()
fig, ax = plt.subplots(nrows=1, ncols=1,figsize=(15, 8))

sns.boxplot(x='speaker_occupation', y='views', data=df[df['speaker_occupation'].isin(occupation_df.head(10)['occupation'])], palette="muted", ax =ax)

ax.set_ylim([0, 0.4e7])

plt.show()
df['num_speaker'].value_counts()
df[df['num_speaker'] == 5][['title', 'description', 'main_speaker', 'event']]
events_df = df[['title', 'event']].groupby('event').count().reset_index()

events_df.columns = ['event', 'talks']

events_df = events_df.sort_values('talks', ascending=False)

events_df.head(10)
df['languages'].describe()
df[df['languages'] == 72]
sns.jointplot(x='languages', y='views', data=df)

plt.show()
import ast

df['tags'] = df['tags'].apply(lambda x: ast.literal_eval(x))
s = df.apply(lambda x: pd.Series(x['tags']),axis=1).stack().reset_index(level=1, drop=True)

s.name = 'theme'
theme_df = df.drop('tags', axis=1).join(s)

theme_df.head()
len(theme_df['theme'].value_counts())
pop_themes = pd.DataFrame(theme_df['theme'].value_counts()).reset_index()

pop_themes.columns = ['theme', 'talks']

pop_themes.head(10)
plt.figure(figsize=(15,5))

sns.barplot(x='theme', y='talks', data=pop_themes.head(10))

plt.show()
pop_theme_talks = theme_df[(theme_df['theme'].isin(pop_themes.head(8)['theme'])) & (theme_df['theme'] != 'TEDx')]

pop_theme_talks['year'] = pop_theme_talks['year'].astype('int')

pop_theme_talks = pop_theme_talks[pop_theme_talks['year'] > 2008]
themes = list(pop_themes.head(8)['theme'])

themes.remove('TEDx')

ctab = pd.crosstab([pop_theme_talks['year']], pop_theme_talks['theme']).apply(lambda x: x/x.sum(), axis=1)

ctab[themes].plot(kind='bar', stacked=True, colormap='rainbow', figsize=(12,8)).legend(loc='center left', bbox_to_anchor=(1, 0.5))

plt.show()
ctab[themes].plot(kind='line', stacked=False, colormap='rainbow', figsize=(12,8)).legend(loc='center left', bbox_to_anchor=(1, 0.5))

plt.show()
pop_theme_talks = theme_df[theme_df['theme'].isin(pop_themes.head(10)['theme'])]

fig, ax = plt.subplots(nrows=1, ncols=1,figsize=(15, 8))

sns.boxplot(x='theme', y='views', data=pop_theme_talks, palette="muted", ax =ax)

ax.set_ylim([0, 0.4e7])
#Convert to minutes

df['duration'] = df['duration']/60

df['duration'].describe()
df[df['duration'] == 2.25]
df[df['duration'] == 87.6]
sns.jointplot(x='duration', y='views', data=df[df['duration'] < 25])

plt.xlabel('Duration')

plt.ylabel('Views')

plt.show()
df2 = pd.read_csv('../input/transcripts.csv')

df2.head()
len(df2)
df3 = pd.merge(left=df,right=df2, how='left', left_on='url', right_on='url')

df3.head()
df3['transcript'] = df3['transcript'].fillna('')

df3['wc'] = df3['transcript'].apply(lambda x: len(x.split()))
df3['wc'].describe()
df3['wpm'] = df3['wc']/df3['duration']

df3['wpm'].describe()
df3[df3['wpm'] > 245]
sns.jointplot(x='wpm', y='views', data=df3[df3['duration'] < 25])

plt.show()
df.iloc[1]['ratings']
df['ratings'] = df['ratings'].apply(lambda x: ast.literal_eval(x))
df['funny'] = df['ratings'].apply(lambda x: x[0]['count'])

df['jawdrop'] = df['ratings'].apply(lambda x: x[-3]['count'])

df['beautiful'] = df['ratings'].apply(lambda x: x[3]['count'])

df['confusing'] = df['ratings'].apply(lambda x: x[2]['count'])

df.head()
df[['title', 'main_speaker', 'views', 'published_date', 'funny']].sort_values('funny', ascending=False)[:10]
df[['title', 'main_speaker', 'views', 'published_date', 'beautiful']].sort_values('beautiful', ascending=False)[:10]
df[['title', 'main_speaker', 'views', 'published_date', 'jawdrop']].sort_values('jawdrop', ascending=False)[:10]
df[['title', 'main_speaker', 'views', 'published_date', 'confusing']].sort_values('confusing', ascending=False)[:10]
df['related_talks'] = df['related_talks'].apply(lambda x: ast.literal_eval(x))
s = df.apply(lambda x: pd.Series(x['related_talks']),axis=1).stack().reset_index(level=1, drop=True)

s.name = 'related'
related_df = df.drop('related_talks', axis=1).join(s)

related_df['related'] = related_df['related'].apply(lambda x: x['title'])
d = dict(related_df['title'].drop_duplicates())

d = {v: k for k, v in d.items()}
related_df['title'] = related_df['title'].apply(lambda x: d[x])

related_df['related'] = related_df['related'].apply(lambda x: d[x])
related_df = related_df[['title', 'related']]

related_df.head()
edges = list(zip(related_df['title'], related_df['related']))
import networkx as nx

G = nx.Graph()

G.add_edges_from(edges)
plt.figure(figsize=(25, 25))

nx.draw(G, with_labels=False)
corpus = ' '.join(df2['transcript'])

corpus = corpus.replace('.', '. ')
wordcloud = WordCloud(stopwords=STOPWORDS, background_color='white',width=2400,height=2000).generate(corpus)

plt.figure(figsize=(12,15))

plt.imshow(wordcloud)

plt.axis('off')

plt.show()