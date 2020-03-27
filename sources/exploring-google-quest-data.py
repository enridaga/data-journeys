
# Loading packages

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
import missingno as msno
from wordcloud import WordCloud

from plotly.offline import init_notebook_mode, iplot
init_notebook_mode(connected=True)
import plotly.offline as py
py.init_notebook_mode(connected=True)

import warnings
warnings.filterwarnings('ignore')
# Reading data

train = pd.read_csv("../input/google-quest-challenge/train.csv")
Sample = pd.read_csv("../input/google-quest-challenge/sample_submission.csv")
# Print first few rows of train data

train.head()
# Shape of train data

train.shape
# Some basic info of train data

train.info()
# Describe train data

train.describe()
# Let's see the list of column names

list(train.columns[1:])
msno.matrix(train)
train.select_dtypes(include = ['object']).columns.values
train.select_dtypes(include = ['float64', 'int64']).columns.values
train['question_title'].value_counts().head(30)
len(train['question_title'].unique())
# Question Title

wordcloud = WordCloud(width = 1000, height = 600, max_font_size = 200, max_words = 150, 
                      background_color='white').generate(" ".join(train.question_title))

plt.figure(figsize=[10,10])
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")
plt.show()
# Question Body

wordcloud = WordCloud(width = 1000, height = 600, max_font_size = 200, max_words = 150, 
                      background_color='white').generate(" ".join(train.question_body))

plt.figure(figsize=[10,10])
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")
plt.show()
# Answer

wordcloud = WordCloud(width = 1000, height = 600, max_font_size = 200, max_words = 150,
                      background_color='white').generate(" ".join(train.answer))

plt.figure(figsize=[10,10])
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")
plt.show()
Cat = train['category'].value_counts()

fig = go.Figure([go.Bar(x=Cat.index, y=Cat)])
fig.update_layout(title = "Count of categories")
py.iplot(fig, filename='test')
Host = train['host'].value_counts()

fig = go.Figure(data = [go.Scatter(x = Host.index, y = Host.values)])
fig.update_layout(title = "Distribution of Host")
py.iplot(fig, filename='test')
targetCol = list(Sample.columns[1:])
train[targetCol].values
corr = train[targetCol].corr()
mask = np.zeros_like(corr, dtype=np.bool)
mask[np.triu_indices_from(mask)] = True
f, ax = plt.subplots(figsize=(15, 14))
cmap = sns.diverging_palette(220, 10, as_cmap=True)
sns.heatmap(corr, mask=mask, cmap=cmap, vmax=.3, center=0,square=True, linewidths=.5, cbar_kws={"shrink": .5})
sns.relplot(x="question_type_instructions", y="answer_type_instructions", data=train)
sns.relplot(y="question_opinion_seeking", x="question_fact_seeking", data=train)
