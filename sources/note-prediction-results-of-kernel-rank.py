
# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
import plotly.offline as py
py.init_notebook_mode(connected=True)
import plotly.graph_objs as go
import plotly.tools as tls
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))
import warnings
warnings.filterwarnings('ignore')
# Any results you write to the current directory are saved as output.
# Read Data
data = pd.read_excel("../input/kaggle_user_data.xlsx")
# data_last is the data at 2017-12-17 (y/m/d)
data_last = data.loc[data["date"] == "2017-12-17"]
# data_first is the data at 2017-11-20 (y/m/d)
data_first = data.loc[data["date"] == "2017-11-20"]
# We will use them later
# Lets look at what is inside in data
data.head(24)
# Concat kernel_level and discussion_level columns and use WordCloud
lis = pd.concat([data["kernel_level"],data["discussion_level"]])
plt.subplots(figsize=(12,12))
wordcloud = WordCloud(
                          background_color='white',
                          width=512,
                          height=384
                         ).generate(" ".join(lis))
plt.imshow(wordcloud)
plt.axis('off')
plt.show()
# Although WordCloud is cool. It does not give any numeric values therefore lets use count plot
# Kernel level of top 23
sns.countplot(data_last.kernel_level,palette=sns.light_palette("navy", reverse=True))
plt.show()
# Discussion level of top 23
sns.countplot(data_last.discussion_level,palette=sns.dark_palette("purple"))
plt.show()

# Number of gold medal (kernel)
new_index = (data_last['kernel_gold'].sort_values(ascending=False)).index.values
sorted_data = data_last.reindex(new_index)
plt.figure(figsize=(15,10))
ax= sns.barplot(x=sorted_data.name, y=sorted_data['kernel_gold'],
                palette=sns.color_palette("YlOrBr", len(sorted_data)))
plt.xticks(rotation= 90)
plt.yticks(np.arange(0,14,1))
plt.ylabel("Number of Gold Medal")
plt.title('Kernel Gold Medal',color = "gold")
plt.show()
# Number of silve medal (kernel)
new_index = (data_last['kernel_silver'].sort_values(ascending=False)).index.values
sorted_data = data_last.reindex(new_index)
plt.figure(figsize=(15,10))
silver = ["#cacaca", "#cacaca", "#cacaca", "#c0c0c0", "#c0c0c0", "#c0c0c0","#b6b6b6", "#b6b6b6", "#b6b6b6", "#acacac", "#acacac", "#acacac","#a3a3a3", "#a3a3a3", "#a3a3a3", "#999999", "#999999", "#999999","#8f8f8f", "#8f8f8f", "#8f8f8f", "#858585", "#858585", "#858585"]
ax= sns.barplot(x=sorted_data.name, y=sorted_data['kernel_silver'],
                palette=sns.color_palette(silver))
plt.xticks(rotation= 90)
plt.yticks(np.arange(0,12,1))
plt.ylabel("Number of Silver Medal")
plt.title('Kernel Silver Medal',color = "silver")
plt.show()
# Number of Bronze medal (kernel)
new_index = (data_last['kernel_bronze'].sort_values(ascending=False)).index.values
sorted_data = data_last.reindex(new_index)
plt.figure(figsize=(15,10))
ax= sns.barplot(x=sorted_data.name, y=sorted_data['kernel_bronze'],
                palette=sns.color_palette("OrRd", len(sorted_data)))
plt.xticks(rotation= 90)
plt.yticks(np.arange(0,56,2))
plt.ylabel("Number of Bronze Medal")
plt.title('Kernel Bronze Medal',color = "firebrick")
plt.show()
# Number of total Medal (kernel)
data_last["kernal_total_medal"] = data_last["kernel_gold"]+data_last["kernel_silver"]+data_last["kernel_bronze"]
new_index = (data_last['kernal_total_medal'].sort_values(ascending=False)).index.values
sorted_data = data_last.reindex(new_index)
sorted_data["kernel_rank_point"] = "kernel rank: " + sorted_data["kernel_rank"].astype("str")+" ,kernel point: "+sorted_data["kernel_points"].astype("str")
trace0 = go.Bar(
    x=sorted_data.name,
    y=sorted_data['kernal_total_medal'],
    text=sorted_data["kernel_rank_point"] ,
    marker=dict(
        color='rgb(34,34,87)',
        line=dict(
            color='rgb(0,0,255)',
            width=1.5,
        )
    ),
    opacity=0.6
)

data = [trace0]
layout = go.Layout(
    title='Kernel Total Medal',
)

fig = go.Figure(data=data, layout=layout)
py.iplot(fig, filename='text-hover-bar')
# Number of gold medal (discussion)
new_index = (data_last['discussion_gold'].sort_values(ascending=False)).index.values
sorted_data = data_last.reindex(new_index)
plt.figure(figsize=(15,10))
ax= sns.barplot(x=sorted_data.name, y=sorted_data['discussion_gold'],
                palette=sns.color_palette("YlOrBr", len(sorted_data)))
plt.xticks(rotation= 90)
plt.yticks(np.arange(0,sorted_data["discussion_gold"].max()+1,2))
plt.ylabel("Number of Gold Medal")
plt.title('Discussion Gold Medal',color = "gold")
plt.show()
# Number of silve medal (Discussion)
new_index = (data_last['discussion_silver'].sort_values(ascending=False)).index.values
sorted_data = data_last.reindex(new_index)
plt.figure(figsize=(15,10))
silver = ["#cacaca", "#cacaca", "#cacaca", "#c0c0c0", "#c0c0c0", "#c0c0c0","#b6b6b6", "#b6b6b6", "#b6b6b6", "#acacac", "#acacac", "#acacac","#a3a3a3", "#a3a3a3", "#a3a3a3", "#999999", "#999999", "#999999","#8f8f8f", "#8f8f8f", "#8f8f8f", "#858585", "#858585", "#858585"]
ax= sns.barplot(x=sorted_data.name, y=sorted_data['discussion_silver'],
                palette=sns.color_palette(silver))
plt.xticks(rotation= 90)
plt.yticks(np.arange(0,sorted_data['discussion_silver'].max()+1,2))
plt.ylabel("Number of Silver Medal")
plt.title('Discussion Silver Medal',color = "silver")
plt.show()
# Number of Bronze medal (Discussion)
new_index = (data_last['discussion_bronze'].sort_values(ascending=False)).index.values
sorted_data = data_last.reindex(new_index)
plt.figure(figsize=(15,10))
ax= sns.barplot(x=sorted_data.name, y=sorted_data['discussion_bronze'],
                palette=sns.color_palette("OrRd", len(sorted_data)))
plt.xticks(rotation= 90)
plt.yticks(np.arange(0,data_last['discussion_bronze'].max()+20,20))
plt.ylabel("Number of Bronze Medal")
plt.title('Discussion Bronze Medal',color = "firebrick")
plt.show()
# Number of total Medal (Discussion)
data_last["dis_total_medal"] = data_last["discussion_gold"]+data_last["discussion_silver"]+data_last["discussion_bronze"]
new_index = (data_last['dis_total_medal'].sort_values(ascending=False)).index.values
sorted_data = data_last.reindex(new_index)
sorted_data["disc_rank"] = "discussion rank: " + sorted_data["discussion_rank"].astype("str")
trace0 = go.Bar(
    x=sorted_data.name,
    y=sorted_data['dis_total_medal'],
    text=sorted_data["disc_rank"] ,
    marker=dict(
        color='rgb(238,205,69)',
        line=dict(
            color='rgb(255,0,0)',
            width=1.5,
        )
    ),
    opacity=0.6
)

data = [trace0]
layout = go.Layout(
    title='Discussion Total Medal',
)

fig = go.Figure(data=data, layout=layout)
py.iplot(fig, filename='text-hover-bar')
## Joint plot  Anisotropic
data = pd.read_excel("../input/kaggle_user_data.xlsx")
data_Anisotropic = data.loc[data["name"] == "Anisotropic"]
sns.jointplot("kernel_points", "followers", data=data_Anisotropic,kind="kde", space=0, color="g")
plt.show()

# # Pair plot CPMP
data_CPMP = data.loc[data["name"] == "CPMP"]
data_CPMP["total_discussion_medal"] = data_CPMP["discussion_gold"]+data_CPMP["discussion_silver"]+data_CPMP["discussion_bronze"]
g = sns.pairplot(data_CPMP.loc[:,["discussion_number",  "total_discussion_medal","followers"]])
g = g.map_upper(plt.scatter,color ="purple" )
g = g.map_lower(sns.kdeplot, cmap=sns.cubehelix_palette(light=1, as_cmap=True))
g = g.map_diag(sns.kdeplot, lw=3, legend=True)

## kernel_rank change of each people
f,ax1 = plt.subplots(figsize =(15,12))
data["date_str"] = data["date"].astype("str")
c = 0
color = 0
color_list =['white','blue','blueviolet','brown','burlywood','cadetblue','chartreuse','chocolate','coral','cornflowerblue','hotpink','crimson','cyan','darkblue','darkcyan','darkgoldenrod','darkgray','darkgreen','darkkhaki','darkmagenta','darkolivegreen','darkorange','darkorchid','steelblue','turquoise','yellowgreen'
]
for each in data["name"].unique():
    sns.pointplot(x='date_str',y='kernel_rank',data=data.loc[data["name"] == each],alpha=0.8, color= color_list[color],scale=0.5)
    plt.text(27.5,49-c,each,fontsize = 10,fontweight = "bold",style = 'normal',color = color_list[color])
    c = c+1.2
    color = color + 1
plt.xticks(rotation= 90)
plt.yticks(np.arange(1,49,1))
plt.xlabel('Date',fontsize = 10,color='black')
plt.ylabel('Kernel Rank',fontsize = 15,color='black')
plt.title('Kernel Rank Change of Each People',fontsize = 20,color='red')
plt.show()
## number of follower change of each people
plt.style.use('dark_background')
f,ax1 = plt.subplots(figsize =(15,12))
data["date_str"] = data["date"].astype("str")
c = 0
color = 0
color_list =['white','blue','blueviolet','brown','burlywood','cadetblue','chartreuse','chocolate','coral','cornflowerblue','hotpink','crimson','cyan','darkblue','darkcyan','darkgoldenrod','darkgray','darkgreen','darkkhaki','darkmagenta','darkolivegreen','darkorange','darkorchid', 'steelblue','turquoise','yellowgreen'
]
for each in data["name"].unique():
    sns.pointplot(x='date_str',y='followers',data=data.loc[data["name"] == each],alpha=0.8, color= color_list[color],scale=0.5)
    plt.text(27.5,1600-c,each,fontsize = 10,style = 'italic',color = color_list[color])
    c = c+40
    color = color + 1
plt.xticks(rotation= 90)
plt.yticks(np.arange(25,1700,125))
plt.xlabel('Date',fontsize = 10,color='white')
plt.ylabel('Number of Followers',fontsize = 15,color='white')
plt.title('Number of Followers Change of Each User',fontsize = 20,color='white')
plt.grid(False)
plt.savefig('graph.png')
plt.show()
# followers vs following
plt.subplots(figsize =(15,12))
melted = pd.melt(frame=data,id_vars = 'name', value_vars= ['followers','following'])
plt.xticks(rotation= 90)
ax = sns.barplot(x="name", y="value", hue="variable", data=melted,errwidth=None )
plt.xlabel('Users')
plt.ylabel('Number of Followers and Following')
plt.title('Followers vs Following')
plt.show()
# monthly point change
data_last["montly_kernel_point"] = data_last["kernel_points"].values-data_first["kernel_points"].values

new_index = (data_last['montly_kernel_point'].sort_values(ascending=False)).index.values
sorted_data = data_last.reindex(new_index)
plt.figure(figsize=(15,10))
ax= sns.barplot(x=sorted_data.name, y=sorted_data['montly_kernel_point'],
                palette=sns.color_palette("Set2", len(sorted_data)))
plt.xticks(rotation= 90)
plt.yticks(np.arange(-30,250,10))
plt.ylabel("Monthly Kernel Point Change")
plt.xlabel("Kernelers")
plt.title('Change in Kernel Point in One Month',color = "white")
plt.show()
# current rank
data_last["name_with_rank"]= data_last.name+" "+data_last.kernel_rank.values.astype("str")
new_index = (data_last['kernel_points'].sort_values(ascending=False)).index.values
sorted_data = data_last.reindex(new_index)
plt.figure(figsize=(15,10))
sns.barplot(x=sorted_data.name_with_rank, y=sorted_data['kernel_points'],
                palette=sns.color_palette("Set2", len(sorted_data)))
plt.xticks(rotation= 90)
plt.yticks(np.arange(0,1300,50))
plt.ylabel("Kernel Point")
plt.xlabel("Kernelers with Rank")
plt.title('Current Kernel Rank (17.12.2017) ',color = "white")
plt.show()


# 6 months later
data_last["6_months_later_kernel_points"] = data_last["kernel_points"] + data_last["montly_kernel_point"]*6
new_index = (data_last['6_months_later_kernel_points'].sort_values(ascending=False)).index.values
sorted_data = data_last.reindex(new_index)
f,ax = plt.subplots(figsize = (12,15))
sns.barplot(x=sorted_data['kernel_points'],y=sorted_data.name_with_rank,color='blue',alpha = 0.7,label='Actual kernel points')
sns.barplot(x=sorted_data['6_months_later_kernel_points'],y=sorted_data.name_with_rank,color='green',alpha = 0.5,label='6 months later kernel points' )

ax.legend(loc='lower right',frameon = True)
ax.set(xlabel='Kernel Points 6 month later', ylabel='Kernelers',title = "Predicted Kernel Points and Ranks 6 Months Later ")
plt.show()
sorted_data['6_months_later_kernel_points']