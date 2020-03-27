
'''Import basic modules.'''
import pandas as pd
import numpy as np
from scipy import stats

'''Customize visualization
Seaborn and matplotlib visualization.'''
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style("whitegrid")
#%matplotlib inline

'''Special Visualization'''
from wordcloud import WordCloud 
import missingno as msno

'''Plotly visualization .'''
import plotly.offline as py
from plotly.offline import iplot, init_notebook_mode
import plotly.graph_objs as go
py.init_notebook_mode(connected = True) # Required to use plotly offline in jupyter notebook

import cufflinks as cf #importing plotly and cufflinks in offline mode  
import plotly.offline  
cf.go_offline()  
cf.set_config_file(offline=False, world_readable=True)

'''Display markdown formatted output like bold, italic bold etc.'''
from IPython.display import Markdown
def bold(string):
    display(Markdown(string))
df = pd.read_csv('../input/netflix-shows/netflix_titles_nov_2019.csv')
df["date_added"] = pd.to_datetime(df['date_added'])
df['year_added'] = df['date_added'].dt.year
df['month_added'] = df['date_added'].dt.month

df['season_count'] = df.apply(lambda x : x['duration'].split(" ")[0] if "Season" in x['duration'] else "", axis = 1)
df['duration'] = df.apply(lambda x : x['duration'].split(" ")[0] if "Season" not in x['duration'] else "", axis = 1)
df.head()
'''A Function To Plot Pie Plot using Plotly'''

def pie_plot(cnt_srs, colors, title):
    labels=cnt_srs.index
    values=cnt_srs.values
    trace = go.Pie(labels=labels, 
                   values=values, 
                   title=title, 
                   hoverinfo='percent+value', 
                   textinfo='percent',
                   textposition='inside',
                   hole=0.7,
                   showlegend=True,
                   marker=dict(colors=colors,
                               line=dict(color='#000000',
                                         width=2),
                              )
                  )
    return trace

bold("**NETFLIX HAVE MORE MOVIES THAN TV SHOWS**")
py.iplot([pie_plot(df['type'].value_counts(), ['cyan', 'gold'], 'Content Type')])
d1 = df[df["type"] == "TV Show"]
d2 = df[df["type"] == "Movie"]

col = "year_added"

vc1 = d1[col].value_counts().reset_index()
vc1 = vc1.rename(columns = {col : "count", "index" : col})
vc1['percent'] = vc1['count'].apply(lambda x : 100*x/sum(vc1['count']))
vc1 = vc1.sort_values(col)

vc2 = d2[col].value_counts().reset_index()
vc2 = vc2.rename(columns = {col : "count", "index" : col})
vc2['percent'] = vc2['count'].apply(lambda x : 100*x/sum(vc2['count']))
vc2 = vc2.sort_values(col)

trace1 = go.Scatter(
                    x=vc1[col], 
                    y=vc1["count"], 
                    name="TV Shows", 
                    marker=dict(color = 'rgb(249, 6, 6)',
                             line=dict(color='rgb(0,0,0)',width=1.5)))

trace2 = go.Scatter(
                    x=vc2[col], 
                    y=vc2["count"], 
                    name="Movies", 
                    marker= dict(color = 'rgb(26, 118, 255)',
                              line=dict(color='rgb(0,0,0)',width=1.5)))
layout = go.Layout(hovermode= 'closest', title = 'Content added over the years' , xaxis = dict(title = 'Year'), yaxis = dict(title = 'Count'),template= "plotly_dark")
fig = go.Figure(data = [trace1, trace2], layout=layout)
fig.show()
temp_df = df['rating'].value_counts().reset_index()


# create trace1
trace1 = go.Bar(
                x = temp_df['index'],
                y = temp_df['rating'],
                marker = dict(color = 'rgb(255,165,0)',
                              line=dict(color='rgb(0,0,0)',width=1.5)))
layout = go.Layout(template= "plotly_dark",title = 'MOST OF PROGRAMME ON NEYFLIX IS TV-14 & TV-MA RATED' , xaxis = dict(title = 'Rating'), yaxis = dict(title = 'Count'))
fig = go.Figure(data = [trace1], layout = layout)
fig.show()

def pie_plot(cnt_srs, title):
    labels=cnt_srs.index
    values=cnt_srs.values
    trace = go.Pie(labels=labels, 
                   values=values, 
                   title=title, 
                   hoverinfo='percent+value', 
                   textinfo='percent',
                   textposition='inside',
                   hole=0.7,
                   showlegend=True,
                   marker=dict(colors=plt.cm.viridis_r(np.linspace(0, 1, 14)),
                               line=dict(color='#000000',
                                         width=2),
                              )
                  )
    return trace

py.iplot([pie_plot(df['rating'].value_counts(), 'Content Type')])
df1 = df[df["type"] == "TV Show"]
df2 = df[df["type"] == "Movie"]

temp_df1 = df1['rating'].value_counts().reset_index()
temp_df2 = df2['rating'].value_counts().reset_index()


# create trace1
trace1 = go.Bar(
                x = temp_df1['index'],
                y = temp_df1['rating'],
                name="TV Shows",
                marker = dict(color = 'rgb(249, 6, 6)',
                             line=dict(color='rgb(0,0,0)',width=1.5)))
# create trace2 
trace2 = go.Bar(
                x = temp_df2['index'],
                y = temp_df2['rating'],
                name = "Movies",
                marker = dict(color = 'rgb(26, 118, 255)',
                              line=dict(color='rgb(0,0,0)',width=1.5)))


layout = go.Layout(template= "plotly_dark",title = 'RATING BY CONTENT TYPE' , xaxis = dict(title = 'Rating'), yaxis = dict(title = 'Count'))
fig = go.Figure(data = [trace1, trace2], layout = layout)
fig.show()
bold('**MOST POPULAR GENRES ON NETFILX ARE:**')
bold('**DOCUMENTARIES,COMEDIES, DRAMAS, INTERNATIONAL, ACTION**')
import squarify
df['Genres'] = df['listed_in'].str.extract('([A-Z]\w{2,})', expand=True)
temp_df = df['Genres'].value_counts().reset_index()

sizes=np.array(temp_df['Genres'])
labels=temp_df['index']
colors = [plt.cm.Paired(i/float(len(labels))) for i in range(len(labels))]
plt.figure(figsize=(12,8), dpi= 100)
squarify.plot(sizes=sizes, label=labels, color = colors, alpha=.5, edgecolor="black", linewidth=3, text_kwargs={'fontsize':15})
plt.title('Treemap of Genres of Netflix Show', fontsize = 15)
plt.axis('off')
plt.show()
bold('**HEATMAP(Correlation)**')
from sklearn.preprocessing import MultiLabelBinarizer # Similar to One-Hot Encoding

data= df['listed_in'].astype(str).apply(lambda s : s.replace('&',' ').replace(',', ' ').split()) 

test = data
mlb = MultiLabelBinarizer()
res = pd.DataFrame(mlb.fit_transform(test), columns=mlb.classes_)
corr = res.corr()
mask = np.zeros_like(corr, dtype=np.bool)
mask[np.triu_indices_from(mask)] = True
f, ax = plt.subplots(figsize=(35, 34))
cmap = sns.diverging_palette(220, 10, as_cmap=True)
sns.heatmap(corr, mask=mask, cmap=cmap, vmax=.3, center=0,square=True, linewidths=.5, cbar_kws={"shrink": .5})
plt.show()
bold('**NETWORKX (Correlation)**')
import networkx as nx

stocks = corr.index.values
cor_matrix = np.asmatrix(corr)
G = nx.from_numpy_matrix(cor_matrix)
G = nx.relabel_nodes(G,lambda x: stocks[x])
G.edges(data=True)

def create_corr_network(G, corr_direction, min_correlation):
    H = G.copy()
    for stock1, stock2, weight in G.edges(data=True):
        if corr_direction == "positive":
            if weight["weight"] <0 or weight["weight"] < min_correlation:
                H.remove_edge(stock1, stock2)
        else:
            if weight["weight"] >=0 or weight["weight"] > min_correlation:
                H.remove_edge(stock1, stock2)
                
    edges,weights = zip(*nx.get_edge_attributes(H,'weight').items())
    weights = tuple([(1+abs(x))**2 for x in weights])
    d = nx.degree(H)
    nodelist, node_sizes = zip(*d)
    positions=nx.circular_layout(H)
    
    plt.figure(figsize=(10,10), dpi=72)
    
    nx.draw_networkx_nodes(H,positions,node_color='#DA70D6',nodelist=nodelist,
                           node_size=tuple([x**2 for x in node_sizes]),alpha=0.8)
    
    nx.draw_networkx_labels(H, positions, font_size=8, 
                            font_family='sans-serif')
    
    if corr_direction == "positive": edge_colour = plt.cm.GnBu 
    else: edge_colour = plt.cm.PuRd
        
    nx.draw_networkx_edges(H, positions, edge_list=edges,style='solid',
                          width=weights, edge_color = weights, edge_cmap = edge_colour,
                          edge_vmin = min(weights), edge_vmax=max(weights))
    plt.axis('off')
    plt.show() 
    
create_corr_network(G, 'positive', 0.3)
create_corr_network(G, 'positive', -0.3)
temp_df1 = df['release_year'].value_counts().reset_index()


# create trace1
trace1 = go.Bar(
                x = temp_df1['index'],
                y = temp_df1['release_year'],
                marker = dict(color = 'rgb(255,165,0)',
                             line=dict(color='rgb(0,0,0)',width=1.5)))
layout = go.Layout(template= "plotly_dark",title = 'CONTENT RELEASE OVER THE YEAR' , xaxis = dict(title = 'Rating'), yaxis = dict(title = 'Count'))
fig = go.Figure(data = [trace1], layout = layout)
fig.show()
df1 = df[df["type"] == "TV Show"]
df2 = df[df["type"] == "Movie"]

temp_df1 = df1['release_year'].value_counts().reset_index()
temp_df2 = df2['release_year'].value_counts().reset_index()


# create trace1
trace1 = go.Bar(
                x = temp_df1['index'],
                y = temp_df1['release_year'],
                name="TV Shows",
                marker = dict(color = 'rgb(249, 6, 6)'))
# create trace2 
trace2 = go.Bar(
                x = temp_df2['index'],
                y = temp_df2['release_year'],
                name = "Movies",
                marker = dict(color = 'rgb(26, 118, 255)'))


layout = go.Layout(template= "plotly_dark",title = 'CONTENT RELEASE OVER THE YEAR BY CONTENT TYPE' , xaxis = dict(title = 'Year'), yaxis = dict(title = 'Count'))
fig = go.Figure(data = [trace1, trace2], layout = layout)
fig.show()
trace = go.Histogram(
                     x = df['duration'],
                     xbins=dict(size=0.5),
                     marker = dict(color = 'rgb(26, 118, 255)'))
layout = go.Layout(template= "plotly_dark", title = 'Distribution of Movies Duration', xaxis = dict(title = 'Minutes'))
fig = go.Figure(data = [trace], layout = layout)
fig.show()
bold("**GREY'S ANATOMY AND NCIS HAVE HIGHEST NO. OF SEASON ON NETFLIX**")
display(df[df['season_count'] == '15'][['title','director', 'cast','country','release_year']])

# image
import urllib.request
from PIL import Image

plt.subplots(figsize=(30,60))
plt.subplot(121)
image = Image.open(urllib.request.urlopen(url='https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcQ0A6upRBYwNBw68tempa18gIxAliLNWkv60-X-fbgQ6rgQOGwC&s'))
plt.imshow(image)
plt.axis('off')

plt.subplot(122)
image = Image.open(urllib.request.urlopen(url='https://i.pinimg.com/originals/58/dc/ba/58dcba558659a13a843c489fa29146f2.jpg'))
plt.imshow(image)
plt.axis('off')
plt.show()

# plot
trace = go.Histogram(
                     x = df['season_count'],
                     marker = dict(color = 'rgb(249, 6, 6)'))
layout = go.Layout(template= "plotly_dark", title = 'Seasons of TV Shows', xaxis = dict(title = 'No. of Seasons'))
fig = go.Figure(data = [trace], layout = layout)
fig.show()
bold('**OLDEST MOVIES ON NETFLIX**')
oldest = df.sort_values("release_year", ascending = True)
oldest = oldest[oldest['duration'] != ""]
display(oldest[['title', "release_year", 'listed_in','country']][:10])

plt.subplots(figsize=(30,60))
plt.subplot(121)
image = Image.open(urllib.request.urlopen(url='https://m.media-amazon.com/images/M/MV5BMTY3NTMyMDQ4NF5BMl5BanBnXkFtZTgwMjkzODgwMzE@._V1_QL50_.jpg'))
plt.imshow(image)
plt.axis('off')

plt.subplot(122)
image = Image.open(urllib.request.urlopen(url='https://m.media-amazon.com/images/M/MV5BMTY3Njg3MDUxMl5BMl5BanBnXkFtZTcwMzE4MTU1MQ@@._V1_QL50_.jpg'))
plt.imshow(image)
plt.axis('off')
plt.show()
bold('**OLDEST TV SHOW ON NETFLIX**')
oldest = df.sort_values("release_year", ascending = True)
oldest = oldest[oldest['season_count'] != ""]
display(oldest[['title', "release_year", 'listed_in','country']][:10])

plt.subplots(figsize=(30,60))
plt.subplot(121)
image = Image.open(urllib.request.urlopen(url='https://images-na.ssl-images-amazon.com/images/I/71ddmI5x94L._SL1500_.jpg'))
plt.imshow(image)
plt.axis('off')

plt.subplot(122)
image = Image.open(urllib.request.urlopen(url='https://images-na.ssl-images-amazon.com/images/I/51KqZA%2B42OL._SY445_.jpg'))
plt.imshow(image)
plt.axis('off')
plt.show()
temp_df = df['country'].value_counts().reset_index()[:20]


# create trace1
trace1 = go.Bar(
                x = temp_df['index'],
                y = temp_df['country'],
                marker = dict(color = 'rgb(153,255,153)',
                              line=dict(color='rgb(0,0,0)',width=1.5)))
layout = go.Layout(template= "plotly_dark",title = 'TOP 20 COUNTIES WITH MOST CONTENT' , xaxis = dict(title = 'Countries'), yaxis = dict(title = 'Count'))
fig = go.Figure(data = [trace1], layout = layout)
fig.show()
from collections import Counter

temp_df = df[df['type']=='Movie']
temp_df = temp_df[temp_df['country']=='India']

categories = ", ".join(temp_df['director'].fillna("")).split(", ")
counter_list = Counter(categories).most_common(11)
counter_list = [_ for _ in counter_list if _[0] != ""]
labels = [_[0] for _ in counter_list][::-1]
values = [_[1] for _ in counter_list][::-1]
trace1 = go.Bar(
                x = labels,
                y = values,
                marker = dict(color = 'rgb(51,255,255)',
                              line=dict(color='rgb(0,0,0)',width=1.5)))
layout = go.Layout(template= "plotly_dark",title = 'TOP 10 MOVIES DIRECTORS FROM INDIA WITH MOST CONTENT' , xaxis = dict(title = 'Directors'), yaxis = dict(title = 'Count'))
fig = go.Figure(data = [trace1], layout = layout)
fig.show()

plt.subplots(figsize=(30,60))
plt.subplot(121)
image = Image.open(urllib.request.urlopen(url='https://upload.wikimedia.org/wikipedia/commons/7/7f/S._S._Rajamouli_at_the_trailer_launch_of_Baahubali.jpg'))
plt.title('S.S.Rajamouli', fontsize=35)
plt.imshow(image)
plt.axis('off')

plt.subplot(122)
image = Image.open(urllib.request.urlopen(url='https://i2.cinestaan.com/image-bank/640-360/45001-46000/45877.jpg'))
plt.title('Dibakar Banerji', fontsize=35)
plt.imshow(image)
plt.axis('off')
plt.show()
temp_df = df[df['type']=='Movie']
temp_df = temp_df[temp_df['country']=='United States']

categories = ", ".join(temp_df['director'].fillna("")).split(", ")
counter_list = Counter(categories).most_common(11)
counter_list = [_ for _ in counter_list if _[0] != ""]
labels = [_[0] for _ in counter_list][::-1]
values = [_[1] for _ in counter_list][::-1]
trace1 = go.Bar(
                x = labels,
                y = values,
                marker = dict(color = 'rgb(255,51,153)',
                              line=dict(color='rgb(0,0,0)',width=1.5)))
layout = go.Layout(template= "plotly_dark",title = 'TOP 10 MOVIES DIRECTORS FROM U.S. WITH MOST CONTENT' , xaxis = dict(title = 'Directors'), yaxis = dict(title = 'Count'))
fig = go.Figure(data = [trace1], layout = layout)
fig.show()

plt.subplots(figsize=(10,20))
plt.subplot(121)
image = Image.open(urllib.request.urlopen(url='http://www.gstatic.com/tv/thumb/persons/530202/530202_v9_ba.jpg'))
plt.title('Jay Karas', fontsize=15)
plt.imshow(image)
plt.axis('off')
plt.show()
temp_df = df[df['type']=='Movie']
temp_df = temp_df[temp_df['country']=='India']

categories = ", ".join(temp_df['cast'].fillna("")).split(", ")
counter_list = Counter(categories).most_common(11)
counter_list = [_ for _ in counter_list if _[0] != ""]
labels = [_[0] for _ in counter_list][::-1]
values = [_[1] for _ in counter_list][::-1]
trace1 = go.Bar(
                x = labels,
                y = values,
                marker = dict(color = 'rgb(51,255,255)',
                              line=dict(color='rgb(0,0,0)',width=1.5)))
layout = go.Layout(template= "plotly_dark",title = 'TOP 10 MOVIES ACTORS FROM INDIA WITH MOST CONTENT' , xaxis = dict(title = 'Actors'), yaxis = dict(title = 'Count'))
fig = go.Figure(data = [trace1], layout = layout)
fig.show()

plt.subplots(figsize=(30,60))
plt.subplot(121)
image = Image.open(urllib.request.urlopen(url='https://timesofindia.indiatimes.com/thumb/msid-67834646,imgsize-23733,width-800,height-600,resizemode-4/67834646.jpg'))
plt.title('Anupam Kher', fontsize=35)
plt.imshow(image)
plt.axis('off')

plt.subplot(122)
image = Image.open(urllib.request.urlopen(url='https://timesofindia.indiatimes.com/thumb/msid-69340156,width-800,height-600,resizemode-4/69340156.jpg'))
plt.title('Shah Rukh Khan', fontsize=35)
plt.imshow(image)
plt.axis('off')
plt.show()
temp_df = df[df['type']=='Movie']
temp_df = temp_df[temp_df['country']=='United States']

categories = ", ".join(temp_df['cast'].fillna("")).split(", ")
counter_list = Counter(categories).most_common(11)
counter_list = [_ for _ in counter_list if _[0] != ""]
labels = [_[0] for _ in counter_list][::-1]
values = [_[1] for _ in counter_list][::-1]
trace1 = go.Bar(
                x = labels,
                y = values,
                marker = dict(color = 'rgb(255,51,153)',
                              line=dict(color='rgb(0,0,0)',width=1.5)))
layout = go.Layout(template= "plotly_dark",title = 'TOP 10 MOVIES ACTORS FROM U.S. WITH MOST CONTENT' , xaxis = dict(title = 'Actors'), yaxis = dict(title = 'Count'))
fig = go.Figure(data = [trace1], layout = layout)
fig.show()

plt.subplots(figsize=(20,40))
plt.subplot(121)
image = Image.open(urllib.request.urlopen(url='https://m.media-amazon.com/images/M/MV5BNTg0YTkxNGEtNjM2YS00NzYwLWIwMDktYmMzMzE0NTRiZDQwXkEyXkFqcGdeQXVyMjQwMDg0Ng@@._V1_.jpg'))
plt.title('Luara Bailey', fontsize=35)
plt.imshow(image)
plt.axis('off')

plt.subplot(122)
image = Image.open(urllib.request.urlopen(url='https://cdn.britannica.com/24/157824-050-D8E9E191/Adam-Sandler-2011.jpg'))
plt.title('Adam Sandler', fontsize=35)
plt.imshow(image)
plt.axis('off')
plt.show()
temp_df1 = df[df['type']=='TV Show']
temp_df1 = temp_df1[temp_df1['country']=='United States']
categories1 = ", ".join(temp_df1['director'].fillna("")).split(", ")
counter_list1 = Counter(categories1).most_common(11)
counter_list1 = [_ for _ in counter_list1 if _[0] != ""]
labels1 = [_[0] for _ in counter_list1][::-1]
values1 = [_[1] for _ in counter_list1][::-1]

temp_df2 = df[df['type']=='TV Show']
temp_df2 = temp_df2[temp_df2['country']=='India']
categories2 = ", ".join(temp_df2['director'].fillna("")).split(", ")
counter_list2 = Counter(categories2).most_common(11)
counter_list2 = [_ for _ in counter_list2 if _[0] != ""]
labels2 = [_[0] for _ in counter_list2][::-1]
values2 = [_[1] for _ in counter_list2][::-1]

from plotly.subplots import make_subplots

fig = make_subplots(rows=1, cols=2,subplot_titles=("United States", "India"))

trace1 = go.Bar(
                x = labels1,
                y = values1,
                marker = dict(color = 'rgb(255,51,153)',
                              line=dict(color='rgb(0,0,0)',width=1.5))
               )

trace2 = go.Bar(
                x = labels2,
                y = values2,
                marker = dict(color = 'rgb(51,255,255)',
                              line=dict(color='rgb(0,0,0)',width=1.5))
               
                )

fig.append_trace(trace1, 1, 1)
fig.append_trace(trace2, 1, 2)
fig.update_layout(height=600, width=600,template= "plotly_dark",title_text = 'TOP 10 TV SHOW DIRECTORS MOST CONTENT')
fig.show()
temp_df1 = df[df['type']=='TV Show']
temp_df1 = temp_df1[temp_df1['country']=='United States']
categories1 = ", ".join(temp_df1['cast'].fillna("")).split(", ")
counter_list1 = Counter(categories1).most_common(11)
counter_list1 = [_ for _ in counter_list1 if _[0] != ""]
labels1 = [_[0] for _ in counter_list1][::-1]
values1 = [_[1] for _ in counter_list1][::-1]

temp_df2 = df[df['type']=='TV Show']
temp_df2 = temp_df2[temp_df2['country']=='India']
categories2 = ", ".join(temp_df2['cast'].fillna("")).split(", ")
counter_list2 = Counter(categories2).most_common(11)
counter_list2 = [_ for _ in counter_list2 if _[0] != ""]
labels2 = [_[0] for _ in counter_list2][::-1]
values2 = [_[1] for _ in counter_list2][::-1]

from plotly.subplots import make_subplots

fig = make_subplots(rows=1, cols=2,subplot_titles=("United States", "India"))

trace1 = go.Bar(
                x = labels1,
                y = values1,
                marker = dict(color = 'rgb(255,51,153)',
                              line=dict(color='rgb(0,0,0)',width=1.5))
               )
trace2 = go.Bar(
                x = labels2,
                y = values2,
                marker = dict(color = 'rgb(51,255,255)',
                              line=dict(color='rgb(0,0,0)',width=1.5))
               
                )

fig.append_trace(trace1, 1, 1)
fig.append_trace(trace2, 1, 2)
fig.update_layout(height=600, width=600,template= "plotly_dark",title_text = 'TOP 10 TV SHOW ACTORS MOST CONTENT')
fig.show()

