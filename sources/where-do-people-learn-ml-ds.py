
import numpy as np 
import pandas as pd 
import os
import math
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import preprocessing

from plotly import tools
from IPython.core import display as ICD
import plotly.graph_objs as go
from plotly.offline import init_notebook_mode, iplot
import warnings
warnings.filterwarnings("ignore")
init_notebook_mode(connected=True)
pd.set_option('display.max_columns', 5000)
base_dir = '../input/kaggle-survey-2018/'
fileName = 'multipleChoiceResponses.csv'
filePath = os.path.join(base_dir,fileName)
survey_df = pd.read_csv(filePath) 
responses_df = survey_df[1:]
responses_df_orig = responses_df.copy()
responses_df = responses_df[~pd.isnull(responses_df['Q35_Part_1'])]
count_dict = {
    'Self-taught' : (responses_df['Q35_Part_1'].astype(float)>0).sum(),
    'Online courses (Coursera, Udemy, edX, etc.)' : (responses_df['Q35_Part_2'].astype(float)>0).sum(),
    'Work' : (responses_df['Q35_Part_3'].astype(float)>0).sum(),
    'University' : (responses_df['Q35_Part_4'].astype(float)>0).sum(),
    'Kaggle competitions' : (responses_df['Q35_Part_5'].astype(float)>0).sum(),
    'Other' : (responses_df['Q35_Part_6'].astype(float)>0).sum()
}

cnt_srs = pd.Series(count_dict)

trace = go.Bar(
    x=cnt_srs.index,
    y=cnt_srs.values,
    marker=dict(
        color='blue',
    #    colorscale = 'Picnic',
    #    reversescale = True
    ),
)

layout = go.Layout(
    title='Number of respondents for each learning category'
)

data = [trace]
fig = go.Figure(data=data, layout=layout)
iplot(fig, filename="learningcategory")
responses_df['Q35_Part_5'].loc[responses_df['Q35_Part_5'].astype(float)<0] = 0   

ys = [responses_df['Q35_Part_1'].values, 
      responses_df['Q35_Part_2'].values,
      responses_df['Q35_Part_3'].values,
      responses_df['Q35_Part_4'].values,
      responses_df['Q35_Part_5'].values,
      responses_df['Q35_Part_6'].values
     ]
names = ["Self-taught",
         'Online courses (Coursera, Udemy, edX, etc.)',
         'Work',
         'University',
         'Kaggle competitions',
         'Other'
        ]
#colors = ['rgba(93, 164, 214, 0.5)', 'rgba(255, 144, 14, 0.5)', 'rgba(44, 160, 101, 0.5)', 'rgba(255, 65, 54, 0.5)', 'rgba(207, 114, 255, 0.5)', 'rgba(127, 96, 0, 0.5)']
colors = ["#3498db", "#95a5a6", "#e74c3c", "#34495e", "#2ecc71","#df6a84"]

trace = []

for i in range(6):
    trace.append ( 
        go.Box(
            y=ys[i],
            name=names[i],
            marker = dict(
                color=colors[i],
            )
        )
    )
layout = go.Layout(
    title='Box plots on % contribution of each ML / DS training category'
)

fig = go.Figure(data=trace, layout=layout)
iplot(fig, filename="TimeSpent")
def get_trace(country_name):
    responses_df = responses_df_orig.copy()
    responses_df = responses_df[responses_df["Q3"]==country_name]
    responses_df['Q35_Part_5'].loc[responses_df['Q35_Part_5'].astype(float)<0] = 0   
    ys = [responses_df['Q35_Part_1'].values, 
      responses_df['Q35_Part_2'].values,
      responses_df['Q35_Part_3'].values,
      responses_df['Q35_Part_4'].values,
      responses_df['Q35_Part_5'].values,
      responses_df['Q35_Part_6'].values
     ]
    names = ["Self-taught",
         'Online courses (Coursera, Udemy, edX, etc.)',
         'Work',
         'University',
         'Kaggle competitions',
         'Other'
        ]
    colors = ["#3498db", "#95a5a6", "#e74c3c", "#34495e", "#2ecc71","#df6a84"]

    trace = []

    for i in range(6):
        trace.append ( 
            go.Box(
                y=ys[i],
                name=names[i],
                marker = dict(
                    color=colors[i],
                )
            )
        )
    return trace
    
traces_list = []
country_names = ["United States of America", "India", "China", "Russia", "Brazil", "Germany",
                "United Kingdom of Great Britain and Northern Ireland", "Canada", "France", "Japan"]
for country_name in country_names:
    traces_list.append(get_trace(country_name))
    
# Creating two subplots
fig = tools.make_subplots(rows=int(len(country_names)/2), cols=2, vertical_spacing=0.1, 
                          subplot_titles=country_names)

for ind, traces in enumerate(traces_list):
    for trace in traces:
        fig.append_trace(trace, int(np.floor(ind/2)+1), int((ind%2) + 1))

fig['layout'].update(height=2000, width=800, paper_bgcolor='rgb(233,233,233)', title="% Contribution of DS / ML training by Country", showlegend=False)
iplot(fig, filename='mldscountry')

def get_trace(country_name):
    responses_df = responses_df_orig.copy()
    responses_df = responses_df[responses_df["Q6"]==country_name]
    responses_df['Q35_Part_5'].loc[responses_df['Q35_Part_5'].astype(float)<0] = 0   
    ys = [responses_df['Q35_Part_1'].values, 
      responses_df['Q35_Part_2'].values,
      responses_df['Q35_Part_3'].values,
      responses_df['Q35_Part_4'].values,
      responses_df['Q35_Part_5'].values,
      responses_df['Q35_Part_6'].values
     ]
    names = ["Self-taught",
         'Online courses (Coursera, Udemy, edX, etc.)',
         'Work',
         'University',
         'Kaggle competitions',
         'Other'
        ]
    colors = ["#3498db", "#95a5a6", "#e74c3c", "#34495e", "#2ecc71","#df6a84"]

    trace = []

    for i in range(6):
        trace.append ( 
            go.Box(
                y=ys[i],
                name=names[i],
                marker = dict(
                    color=colors[i],
                )
            )
        )
    return trace
    
traces_list = []
country_names = ["Student", "Data Scientist", "Software Engineer", "Data Analyst"]
for country_name in country_names:
    traces_list.append(get_trace(country_name))
    
# Creating two subplots
fig = tools.make_subplots(rows=int(len(country_names)/2), cols=2, vertical_spacing=0.2, 
                          subplot_titles=country_names)

for ind, traces in enumerate(traces_list):
    for trace in traces:
        fig.append_trace(trace, int(np.floor(ind/2)+1), int((ind%2) + 1))

fig['layout'].update(height=800, width=800, paper_bgcolor='rgb(233,233,233)', title="% Contribution of DS / ML training by Profession", showlegend=False)
iplot(fig, filename='mldscountry')

def get_trace(country_name):
    responses_df = responses_df_orig.copy()
    responses_df = responses_df[responses_df["Q4"]==country_name]
    responses_df['Q35_Part_5'].loc[responses_df['Q35_Part_5'].astype(float)<0] = 0   
    ys = [responses_df['Q35_Part_1'].values, 
      responses_df['Q35_Part_2'].values,
      responses_df['Q35_Part_3'].values,
      responses_df['Q35_Part_4'].values,
      responses_df['Q35_Part_5'].values,
      responses_df['Q35_Part_6'].values
     ]
    names = ["Self-taught",
         'Online courses (Coursera, Udemy, edX, etc.)',
         'Work',
         'University',
         'Kaggle competitions',
         'Other'
        ]
    colors = ["#3498db", "#95a5a6", "#e74c3c", "#34495e", "#2ecc71","#df6a84"]

    trace = []

    for i in range(6):
        trace.append ( 
            go.Box(
                y=ys[i],
                name=names[i],
                marker = dict(
                    color=colors[i],
                )
            )
        )
    return trace
    
traces_list = []
country_names = ["Master’s degree", "Bachelor’s degree", "Doctoral degree", "Professional degree"]
for country_name in country_names:
    traces_list.append(get_trace(country_name))
    
# Creating two subplots
fig = tools.make_subplots(rows=int(len(country_names)/2), cols=2, vertical_spacing=0.2, 
                          subplot_titles=country_names)

for ind, traces in enumerate(traces_list):
    for trace in traces:
        fig.append_trace(trace, int(np.floor(ind/2)+1), int((ind%2) + 1))

fig['layout'].update(height=800, width=800, paper_bgcolor='rgb(233,233,233)', title="% Contribution of DS / ML training by Degree", showlegend=False)
iplot(fig, filename='mldscountry')

def get_trace(country_name):
    responses_df = responses_df_orig.copy()
    responses_df = responses_df[responses_df["Q1"]==country_name]
    responses_df['Q35_Part_5'].loc[responses_df['Q35_Part_5'].astype(float)<0] = 0   
    ys = [responses_df['Q35_Part_1'].values, 
      responses_df['Q35_Part_2'].values,
      responses_df['Q35_Part_3'].values,
      responses_df['Q35_Part_4'].values,
      responses_df['Q35_Part_5'].values,
      responses_df['Q35_Part_6'].values
     ]
    names = ["Self-taught",
         'Online courses (Coursera, Udemy, edX, etc.)',
         'Work',
         'University',
         'Kaggle competitions',
         'Other'
        ]
    colors = ["#3498db", "#95a5a6", "#e74c3c", "#34495e", "#2ecc71","#df6a84"]

    trace = []

    for i in range(6):
        trace.append ( 
            go.Box(
                y=ys[i],
                name=names[i],
                marker = dict(
                    color=colors[i],
                )
            )
        )
    return trace
    
traces_list = []
country_names = ["Male", "Female"]
for country_name in country_names:
    traces_list.append(get_trace(country_name))
    
# Creating two subplots
fig = tools.make_subplots(rows=int(len(country_names)/2), cols=2, vertical_spacing=0.2, 
                          subplot_titles=country_names)

for ind, traces in enumerate(traces_list):
    for trace in traces:
        fig.append_trace(trace, int(np.floor(ind/2)+1), int((ind%2) + 1))

fig['layout'].update(height=400, width=800, paper_bgcolor='rgb(233,233,233)', title="% Contribution of DS / ML training by Gender", showlegend=False)
iplot(fig, filename='mldscountry')
name_dict = {
    'Q35_Part_1' : "Self-taught",
    'Q35_Part_2' : "Online Courses",
    'Q35_Part_3' : "Work",
    'Q35_Part_4' : "University",
    'Q35_Part_5' : "Kaggle competitions"
}
colors = ["#3498db", "#95a5a6", "#e74c3c", "#34495e", "#2ecc71","#df6a84"]
def get_trace(country_name, color):
    responses_df = responses_df_orig.copy()
    #responses_df = responses_df[responses_df["Q4"]==country_name]
    responses_df['Q35_Part_5'].loc[responses_df['Q35_Part_5'].astype(float)<0] = 0  
    responses_df = responses_df.sort_values(by="Q2")
    trace = go.Box(
        y = responses_df[country_name].values,
        x = responses_df['Q2'].values,
        name = name_dict[country_name],
        marker = dict(
                    color=color,
                )
    )
    return trace

traces_list = []
country_names = ["Q35_Part_1", "Q35_Part_2", "Q35_Part_3", "Q35_Part_4", "Q35_Part_5"]
for ind, country_name in enumerate(country_names):
    traces_list.append(get_trace(country_name, colors[ind]))
    
# Creating two subplots
fig = tools.make_subplots(rows=int(len(country_names)), cols=1, vertical_spacing=0.05, 
                          subplot_titles=[name_dict[cn] for cn in country_names])

for ind, trace in enumerate(traces_list):
        fig.append_trace(trace, ind+1, 1)

fig['layout'].update(height=1600, width=800, paper_bgcolor='rgb(233,233,233)', title="% Contribution of DS / ML training category by Age", showlegend=True)
iplot(fig, filename='mldscountry')

survey_freeform_df = pd.read_csv(base_dir + "freeFormResponses.csv").loc[1:,:]
col_name = "Q35_OTHER_TEXT"

def preprocess(x):
    if str(x) != "nan":
        x = str(x).lower()
        if x[-1] == "s":
            x = x[:-1]
    return x

survey_freeform_df[col_name] = survey_freeform_df[col_name].apply(lambda x: preprocess(x))

cnt_srs = survey_freeform_df[col_name].value_counts().head(20)

trace = go.Bar(
    x=cnt_srs.index,
    y=cnt_srs.values,
    marker=dict(
        color='blue',
    ),
)

layout = go.Layout(
    title='Count of other DS / ML learning category - free form text'
)

data = [trace]
fig = go.Figure(data=data, layout=layout)
iplot(fig, filename="platform")
responses_df = responses_df_orig.copy()

count_dict = {
    'Coursera' : (responses_df['Q36_Part_2'].count()),
    'Udemy' : (responses_df['Q36_Part_9'].count()),
    'DataCamp' : (responses_df['Q36_Part_4'].count()),
    'Kaggle Learn' : (responses_df['Q36_Part_6'].count()),
    'Udacity' : (responses_df['Q36_Part_1'].count()),
    'edX' : (responses_df['Q36_Part_3'].count()),
    'Online University Courses' : (responses_df['Q36_Part_11'].count()),
    'Fast.AI' : (responses_df['Q36_Part_7'].count()),
    'Developers.google.com' : (responses_df['Q36_Part_8'].count()),
    'DataQuest' : (responses_df['Q36_Part_5'].count()),
    'The School of AI' : (responses_df['Q36_Part_10'].count())
}


cnt_srs = pd.Series(count_dict)

trace = go.Bar(
    x=cnt_srs.index,
    y=cnt_srs.values,
    marker=dict(
        color='green',
    ),
)

layout = go.Layout(
    title='Number of Respondents for each online platform'
)

data = [trace]
fig = go.Figure(data=data, layout=layout)
iplot(fig, filename="platform")
col_name = "Q36_OTHER_TEXT"

def preprocess(x):
    if str(x) != "nan":
        x = str(x).lower()
    return x

survey_freeform_df[col_name] = survey_freeform_df[col_name].apply(lambda x: preprocess(x))

cnt_srs = survey_freeform_df[col_name].value_counts().head(20)

trace = go.Bar(
    x=cnt_srs.index,
    y=cnt_srs.values,
    marker=dict(
        color='red',
    ),
)

layout = go.Layout(
    title='Number of respondents for other online platforms - free form text'
)

data = [trace]
fig = go.Figure(data=data, layout=layout)
iplot(fig, filename="platform")
responses_df = responses_df_orig.copy()
temp_series = responses_df['Q37'].value_counts()
labels = (np.array(temp_series.index))
sizes = (np.array((temp_series / temp_series.sum())*100))

trace = go.Pie(labels=labels, values=sizes)
layout = go.Layout(
    title='Most Used Online Platform distribution'
)
data = [trace]
fig = go.Figure(data=data, layout=layout)
iplot(fig, filename="onlinecourse")
responses_df = responses_df_orig.copy()

from sklearn import preprocessing
lbl = preprocessing.LabelEncoder()

gdf = responses_df.groupby(['Q3', 'Q37']).size().reset_index()
gdf.columns = ['country', 'platform', 'count']
gdf = gdf.sort_values(by=['country','count'])
gdf = gdf.drop_duplicates(subset='country', keep='last')
gdf['count'] = lbl.fit_transform(gdf['platform'].values)

colorscale = [[0, 'rgb(102,194,165)'], [0.33, 'rgb(253,174,97)'], [0.66, 'rgb(213,62,79)'], [1.0, 'rgb(158,1,66)']]
data = [ dict(
        type = 'choropleth',
        autocolorscale = False,
        colorscale = colorscale,
        showscale = False,
        locations = gdf.country,
        z = gdf['count'].values,
        locationmode = 'country names',
        text = gdf.platform,
        marker = dict(
            line = dict(color = '#fff', width = 2)) )           ]

layout = dict(
    height=800,
    title = 'Most Used Online Platform by Country',
    geo = dict(
        showframe = True,
        showocean = True,
        oceancolor = '#222',
        projection = dict(
        type = 'orthographic',
            rotation = dict(
                    lon = 60,
                    lat = 10),
        ),
        lonaxis =  dict(
                showgrid = False,
                gridcolor = 'rgb(102, 102, 102)'
            ),
        lataxis = dict(
                showgrid = False,
                gridcolor = 'rgb(102, 102, 102)'
                )
            ),
        )
fig = dict(data=data, layout=layout)
iplot(fig)
responses_df = responses_df_orig.copy()
lbl = preprocessing.LabelEncoder()

gdf = responses_df.groupby(['Q3', 'Q37']).size().reset_index()
gdf.columns = ['country', 'platform', 'count']
gdf = gdf.sort_values(by=['country','count'], ascending=False)
gdf = gdf.groupby(['country']).nth(2).reset_index()
gdf['count'] = lbl.fit_transform(gdf['platform'].values)

colorscale = [[0, 'rgb(102,194,165)'], [0.33, 'rgb(253,174,97)'], [0.66, 'rgb(213,62,79)'], [1.0, 'rgb(158,1,66)']]
data = [ dict(
        type = 'choropleth',
        autocolorscale = False,
        colorscale = 'picnic',
        showscale = False,
        locations = gdf.country,
        z = gdf['count'].values,
        locationmode = 'country names',
        text = gdf.platform,
        marker = dict(
            line = dict(color = '#fff', width = 2)) )           ]

layout = dict(
    height=800,
    title = 'Second Most Used Online Platform by Country',
    geo = dict(
        showframe = True,
        showocean = True,
        oceancolor = '#222',
        projection = dict(
        type = 'orthographic',
            rotation = dict(
                    lon = 60,
                    lat = 10),
        ),
        lonaxis =  dict(
                showgrid = False,
                gridcolor = 'rgb(102, 102, 102)'
            ),
        lataxis = dict(
                showgrid = False,
                gridcolor = 'rgb(102, 102, 102)'
                )
            ),
        )
fig = dict(data=data, layout=layout)
iplot(fig)
responses_df = responses_df_orig.copy()

temp_series = responses_df['Q39_Part_1'].value_counts()
labels = (np.array(temp_series.index))
sizes = (np.array((temp_series / temp_series.sum())*100))

trace = go.Pie(labels=labels, values=sizes, hole=0.4)
layout = go.Layout(
    height = 700,
    width = 700,
    title='How are Online Courses compared to Brick & Mortar Courses',
)
data = [trace]
fig = go.Figure(data=data, layout=layout)
iplot(fig, filename="onlinecourse")
coursera_df = pd.read_csv("../input/100k-courseras-course-reviews-dataset/reviews_by_course.csv")
courses = ["machine-learning", "python-data", "r-programming", "data-scientists-tools", "ml-foundations", "python-data-analysis"]
coursera_df = coursera_df[coursera_df["CourseId"].isin(courses)]

cnt_srs = coursera_df["CourseId"].value_counts()

trace = go.Bar(
    x=cnt_srs.index,
    y=cnt_srs.values,
    marker=dict(
        color='yellow',
    ),
)

layout = go.Layout(
    title='Number of Reviews for each coursera course'
)

data = [trace]
fig = go.Figure(data=data, layout=layout)
iplot(fig, filename="platform")

names = []
ones = []
twos = []
threes = []
fours = []
fives = []
for col in courses[::-1]:
    tmp_df = coursera_df[coursera_df["CourseId"]==col]
    cnt_srs= tmp_df["Label"].value_counts()
    cnt_srs_sum = float(cnt_srs.values.sum()) 
    names.append(col)
    ones.append(cnt_srs[1] / cnt_srs_sum * 100)
    twos.append(cnt_srs[2] / cnt_srs_sum * 100)
    threes.append(cnt_srs[3] / cnt_srs_sum * 100)
    fours.append(cnt_srs[4] / cnt_srs_sum * 100)
    fives.append(cnt_srs[5] / cnt_srs_sum * 100)

trace1 = go.Bar(
    y=names,
    x=ones,
    orientation = 'h',
    name = "Very Negative"
)
trace2 = go.Bar(
    y=names,
    x=twos,
    orientation = 'h',
    name = "Negative"
)
trace3 = go.Bar(
    y=names,
    x=threes,
    orientation = 'h',
    name = "Neutral"
)
trace4 = go.Bar(
    y=names,
    x=fours,
    orientation = 'h',
    name = "Positive"
)
trace5 = go.Bar(
    y=names,
    x=fives,
    orientation = 'h',
    name = "Very Positive"
)

layout = go.Layout(
    title='Coursera Course Reviews in Percentage',
    barmode='stack',
    width = 800,
    height = 800,
    #yaxis=dict(tickangle=-45),
)

data = [trace5, trace4, trace3, trace2, trace1]
fig = go.Figure(data=data, layout=layout)
iplot(fig, filename="CourseraReviews")
responses_df = responses_df_orig.copy()

temp_series = responses_df['Q39_Part_2'].value_counts()
labels = (np.array(temp_series.index))
sizes = (np.array((temp_series / temp_series.sum())*100))

trace = go.Pie(labels=labels, values=sizes, hole=0.4)
layout = go.Layout(
    height = 700,
    width = 700,
    title='How are In-person bootcamps compared to Tradational Brick & Mortar Institutions',
)
data = [trace]
fig = go.Figure(data=data, layout=layout)
iplot(fig, filename="bootcamps")
responses_df = responses_df_orig.copy()

count_dict = {
    'Kaggle Forums' : (responses_df['Q38_Part_4'].count()),
    'Medium Blog Posts' : (responses_df['Q38_Part_18'].count()),
    'ArXiv & Preprints' : (responses_df['Q38_Part_11'].count()),
    'Twitter' : (responses_df['Q38_Part_1'].count()),
    'None / I do not know' : (responses_df['Q38_Part_21'].count()),
    'r/machinelearning' : (responses_df['Q38_Part_3'].count()),
    'KDNuggets' : (responses_df['Q38_Part_14'].count()),
    'Journal Publications' : (responses_df['Q38_Part_12'].count()),
    'Siraj Raval Youtube' : (responses_df['Q38_Part_6'].count()),
    'HackerNews' : (responses_df['Q38_Part_2'].count()),
    'FiveThirtyEight.com' : (responses_df['Q38_Part_10'].count())
}


cnt_srs = pd.Series(count_dict)

trace = go.Bar(
    x=cnt_srs.index,
    y=cnt_srs.values,
    marker=dict(
        color='red',
    ),
)

layout = go.Layout(
    title='Favorite Media Source on DS topics'
)

data = [trace]
fig = go.Figure(data=data, layout=layout)
iplot(fig, filename="platform")
responses_df = responses_df_orig.copy()

map_dict = {
    'Q38_Part_4' : 'Kaggle Forums',
    'Q38_Part_18' : 'Medium Blog Posts',
    'Q38_Part_11' : 'ArXiv & Preprints',
    'Q38_Part_1' : 'Twitter', 
    'Q38_Part_21' : 'None / I do not know',
    'Q38_Part_3' : 'r/machinelearning',
    'Q38_Part_14' : 'KDNuggets',
    'Q38_Part_12' : 'Journal Publications',
    'Q38_Part_6' : 'Siraj Raval Youtube',
    'Q38_Part_2' : 'HackerNews',
    'Q38_Part_10' : 'FiveThirtyEight.com'
}

fdf = pd.DataFrame()
for key, item in map_dict.items():
    tdf = responses_df.groupby('Q3')[key].count().reset_index()
    tdf.columns = ['country', 'cnt']
    tdf['source'] = item
    fdf = pd.concat([fdf, tdf])

fdf = fdf.sort_values(by=['country','cnt'])
fdf = fdf.drop_duplicates(subset='country', keep='last')
lbl = preprocessing.LabelEncoder()
fdf['source_lbl'] = lbl.fit_transform(fdf['source'].values)
    
data = [ dict(
        type = 'choropleth',
        autocolorscale = False,
        colorscale = 'Jet',
        showscale = False,
        locations = fdf.country,
        z = fdf['source_lbl'].values,
        locationmode = 'country names',
        text = fdf.source,
        marker = dict(
            line = dict(color = '#fff', width = 2)) )           ]

layout = dict(
    height=800,
    title = 'Favorite media sources by Country',
    geo = dict(
        showframe = True,
        showocean = True,
        oceancolor = '#222',
        projection = dict(
        type = 'orthographic',
            rotation = dict(
                    lon = 270,
                    lat = 10),
        ),
        lonaxis =  dict(
                showgrid = False,
                gridcolor = 'rgb(102, 102, 102)'
            ),
        lataxis = dict(
                showgrid = False,
                gridcolor = 'rgb(102, 102, 102)'
                )
            ),
        )
fig = dict(data=data, layout=layout)
iplot(fig)
from wordcloud import WordCloud, STOPWORDS

survey_freeform_df = pd.read_csv(base_dir + "freeFormResponses.csv").loc[1:,:]
col_name = "Q38_OTHER_TEXT"

text = ''
text_array = survey_freeform_df[col_name].values
for ind, t in enumerate(text_array):
    if str(t) != "nan":
        text = " ".join([text, "".join(t.lower())])
text = text.strip()
    
plt.figure(figsize=(24.0,16.0))
wordcloud = WordCloud(background_color='black', width=800, height=400, max_font_size=100, max_words=100).generate(text)
wordcloud.recolor(random_state=ind*312)
plt.imshow(wordcloud)
plt.title("Other online resources for ML / DS", fontsize=40)
plt.axis("off")
#plt.show()
plt.tight_layout() 
