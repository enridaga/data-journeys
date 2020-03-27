
# import the usual frameworks
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import collections
import warnings

from IPython.core.display import display, HTML
from sklearn.preprocessing import MinMaxScaler
    
# import plotly 
import plotly
import plotly.figure_factory as ff
import plotly.graph_objs as go
import plotly.offline as py
import plotly.tools as tls

# for color scales in plotly
import colorlover as cl 

# configure things
warnings.filterwarnings('ignore')

pd.options.display.float_format = '{:,.2f}'.format  
pd.options.display.max_columns = 999

py.init_notebook_mode(connected=True)

%load_ext autoreload
%autoreload 2
#%matplotlib inline
#!pip list
# !pwd
# !ls
df = pd.read_csv(
    '../input/ds-job-listing-technology/ds_job_listing_software.csv',
    usecols=['Keyword','LinkedIn', 'Indeed', 'SimplyHired', 'Monster'],
    skiprows=0,
    nrows=37, 
    thousands=',',
    index_col=0,
   
)
df
df.info()
df.describe()
scale = MinMaxScaler()
scaled_df = pd.DataFrame(
    scale.fit_transform(df), 
    columns = df.columns,
    index = df.index)    
scaled_df.head()
scaled_df['combined'] = scaled_df[["LinkedIn", "Indeed", "SimplyHired", "Monster"]].mean(axis = 1)
scaled_df.head()
num_sites = 8

y_linkedin = scaled_df.iloc[:num_sites, 0]
y_indeed = scaled_df.iloc[:num_sites, 1]
y_monster = scaled_df.iloc[:num_sites, 2]
y_simply = scaled_df.iloc[:num_sites, 3]

y_linkedin
indeed = go.Bar(x=scaled_df.index, y=y_indeed, name = "Indeed")
simply = go.Bar(x=scaled_df.index, y=y_simply, name="SimplyHired")
monster = go.Bar(x=scaled_df.index, y=y_monster, name="Monster")
linked = go.Bar(x=scaled_df.index, y=y_linkedin, name="LinkedIn")

data = [linked, indeed, simply, monster]

layout = go.Layout(
    barmode='group',
    title="Top Software in Job Listings for Data Scientists",
    xaxis={'title': 'Software'},
    yaxis={'title': "Scaled Listings", 'separatethousands': True,
    }
)

fig = go.Figure(data=data, layout=layout)
py.iplot(fig)
p_s_df = scaled_df * 100
p_s_df = p_s_df.round(2)
p_s_df.columns = ['LinkedIn', 'Indeed', 'SimplyHired', 'Monster', 'Score']
p_s_df = p_s_df.sort_values(by=['Score'], ascending = False)
p_s_df.head()
p_s_df.rename(index = {'Microsoft Office': 'MS Office'}, inplace = True)
p_s_df
p_s_df_20 = p_s_df.iloc[:20,:]
p_s_df_20
cmax=200
cmin=50
color_s = np.linspace(cmin, cmax, 20)

data = [
    go.Bar(
        x=p_s_df_20.index,          
        y=p_s_df_20['Score'],
        marker=dict(
            colorscale='Jet',
            color=color_s,
        ),
    )
]

layout = {
    'title': 'Top 20 Technology Skills in Data Scientist Job Listings',
    #'xaxis': {'title': 'Technology'},
    'yaxis': {'title': "Score"}
}

fig = go.Figure(data=data, layout=layout)
py.iplot(fig)
cmax=200
cmin=50
color_s = np.linspace(cmin, cmax, 44)

data = [
    go.Bar(
        x=p_s_df.index,          
        y=p_s_df['Score'],
        marker=dict(
            colorscale='Jet',
            #cauto=True,
            color=color_s,
        ),
        # text=p_s_df['Score'],
        # textposition='outside',
        # textfont=dict(size=10)
    )
]

layout = {
    'title': 'Technology Skills in Data Scientist Job Listings',
    #'xaxis': {'tickmode': 'linear'},
    'yaxis': {'title': "Score"}
}

fig = go.Figure(data=data, layout=layout)
py.iplot(fig)
total_ds = {
    'LinkedIn': 8610,
    'Indeed': 5138,
    'SimplyHired': 3829,
    'Monster': 3746,
}
for key, value in total_ds.items():
    df[key + " %"] = df[key] / value * 100
df
df['avg_percent'] = df.iloc[:,-4:].mean(axis=1)
df
df = df.sort_values(by="avg_percent", ascending = False)
df
cmax=200
cmin=50
color_s = np.linspace(cmin, cmax, 44)

data = [
    go.Bar(
        x=df.index,          
        y=df['avg_percent'],
        marker=dict(
            colorscale='Jet',
            color=color_s,
        ),
    )
]

layout = dict(
    title="Technology Skills in Data Scientist Job Listings",
    yaxis=dict(title="% of Listings",)
)

fig = go.Figure(data=data, layout=layout)
py.iplot(fig)
cmax=200
cmin=50
color_s = np.linspace(cmin, cmax, 20)

data = [
    go.Bar(
        x=df.index,          
        y=df.iloc[:20, -1],
        marker=dict(
            colorscale='Jet',
            color=color_s,
        ),
    )
]

layout = dict(
    title="Top 20 Technology Skills in Data Scientist Job Listings",
    yaxis=dict(title="% of Listings",)
)

fig = go.Figure(data=data, layout=layout)
py.iplot(fig)
# percentage of data scientist jobs on glassdoor early 2017
glassdoor = {
    'Python': 72,
    'R': 64,
    'SQL': 51,
    'Spark': 27,
    'Hadoop': 39,
    'Java': 33,
    'SAS': 30,
    'Tableau': 14,
    'Hive': 17,
    'Matlab': 20
}
# make a data frame of just these
# could just merge the series and df 

series_gd = pd.Series(glassdoor)
series_gd
df_gd = pd.merge(
    df, 
    pd.DataFrame(
        series_gd, 
        columns=['gd_percent']), 
    right_index=True, 
    left_index=True, 
    how="inner"
)

# df_gd = df_gd[""]
# df_gd.columns = 
df_gd
avg = go.Bar(x=df_gd.index, y=df_gd['avg_percent'], name="Oct. 2018 Averaged")
gd = go.Bar(x=df_gd.index, y=df_gd['gd_percent'], name="Early 2017 Glassdoor")


data = [gd, avg]

layout = go.Layout(
    barmode='group',
    title="Technology Skills 2017 Glassdoor Compared to 2018 Average",
    xaxis={'title': 'Technology'},
    yaxis={'title': '% of Listings'},
)

fig = go.Figure(data=data, layout=layout)
py.iplot(fig)
df_skills = pd.read_csv(
    '../input/data-scientist-general-skills-2018-revised/ds_general_skills_revised.csv',
    nrows=15,
    thousands=',',
    index_col=0,  
    )
df_skills

# this dataset was updated with "data engineering included 10/15/18"
df_skills.rename(index={'AI composite': 'AI', 'NLP composite': 'NLP'}, inplace = True)
df_skills
scale = MinMaxScaler()
scaled_df = pd.DataFrame(
    scale.fit_transform(df_skills), 
    columns = df_skills.columns,
    index = df_skills.index)    
scaled_df
scaled_df['big_sites'] = scaled_df[["LinkedIn", "Indeed", "SimplyHired", "Monster"]].mean(axis = 1)
scaled_df.sort_values(by = 'big_sites', ascending = False)
y_linkedin = scaled_df.iloc[:, 0]
y_indeed = scaled_df.iloc[:, 1]
y_simply = scaled_df.iloc[:, 2]
y_monster = scaled_df.iloc[:, 3]

y_linkedin
indeed = go.Bar(x=scaled_df.index, y=y_indeed, name = "Indeed")
simply = go.Bar(x=scaled_df.index, y=y_simply, name="SimplyHired")
linked = go.Bar(x=scaled_df.index, y=y_linkedin, name="LinkedIn")
monster = go.Bar(x=scaled_df.index, y=y_monster, name="Monster")

data = [linked, indeed, simply, monster]

layout = go.Layout(
    barmode='group',
    title="Data Science Skills in Online Job Listings",
    xaxis={'title': 'Skills'},
    yaxis={'title': "Scaled Listings", 'separatethousands': True,
    }
)

fig = go.Figure(data=data, layout=layout)
py.iplot(fig)
p_s_df = scaled_df * 100
p_s_df = p_s_df.round(2)
p_s_df.columns = ['LinkedIn', 'Indeed', 'SimplyHired', 'Monster', 'Score']
p_s_df = p_s_df.sort_values(by=['Score'], ascending = False)
p_s_df
cmax=200
cmin=50
color_s = np.linspace(cmin, cmax, 14)

data = [
    go.Bar(
        x=p_s_df.index,          
        y=p_s_df['Score'],
        marker=dict(
            colorscale='Jet',
            #cauto=True,
            color=color_s,
        ),
        # text=p_s_df['Score'],
        # textposition='outside',
        # textfont=dict(size=10)
    )
]

layout = {
    'title': 'General Skills in Data Scientist Job Listings',
    'xaxis': {'tickmode': 'linear'},
    'yaxis': {'title': "Score"}
}

fig = go.Figure(data=data, layout=layout)
py.iplot(fig)
total_ds = {
    'LinkedIn': 8610,
    'Indeed': 5138,
    'SimplyHired': 3829,
    'Monster': 3746,
}
for key, value in total_ds.items():
    df_skills[key + " %"] = df_skills[key] / value * 100
df_skills
df_skills['avg_percent'] = df_skills.iloc[:,-4:].mean(axis=1)
df_skills
df_skills = df_skills.sort_values(by="avg_percent", ascending = False)
df_skills
cmax=200
cmin=50
color_s = np.linspace(cmin, cmax, 16)

data = [
    go.Bar(
        x=df_skills.index,          
        y=df_skills['avg_percent'],
        marker=dict(
            colorscale='Jet',
            color=color_s,
        ),
    )
]

layout = dict(
    title="General Skills in Data Scientist Job Listings",
    yaxis=dict(title="Average % of Listings",)
)

fig = go.Figure(data=data, layout=layout)
py.iplot(fig)
ds_results = pd.Series({
    'LinkedIn': 8610,
    'Indeed': 5138, 
    'SimplyHired': 3829,
    'Monster': 3746,
    'AngelList': 658
})
ds_results
cmax=200
cmin=50
color_s = np.linspace(cmin, cmax, 5)

data = [
    go.Bar(
        x=ds_results.index,          
        y=ds_results.values,
        marker=dict(
            colorscale='Jet',
            #cauto=True,
            color=color_s,
        ),
        # text=p_s_df['Score'],
        # textposition='outside',
        # textfont=dict(size=10)
    )
]

layout = {
    'title': "Data Scientist Job Listings",
    'xaxis': {'title': 'Website'},
    'yaxis': {'title': 'Listings', 'separatethousands': True,}
}

fig = go.Figure(data=data, layout=layout)
py.iplot(fig)



