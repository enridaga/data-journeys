
import numpy as np
import pandas as pd
import plotly.offline as py
from plotly.offline import init_notebook_mode, iplot
import plotly.graph_objs as go
from plotly import tools
init_notebook_mode(connected=True)  
from plotly.tools import FigureFactory as ff
import pycountry
import random
import squarify
from collections import Counter
import warnings
warnings.filterwarnings('ignore')
df = pd.read_csv('../input/survey_results_public.csv')
schema = pd.read_csv('../input/survey_results_schema.csv')
pd.options.display.max_colwidth = 300
schema
df.head()
df.columns.values
# Auxilary functions
def remove_coma(val):
    value = val.replace(",","")
    return value

def random_colors(number_of_colors):
    color = ["#"+''.join([random.choice('0123456789ABCDEF') for j in range(6)])
                 for i in range(number_of_colors)]
    return color

def remove_coma(val):
    value = val.replace(",","")
    return value

def get_list(col_name):
    full_list = ";".join(col_name)
    each_word = full_list.split(";")
    each_word = Counter(each_word).most_common()
    return pd.DataFrame(each_word)
def calculate_percent(val):
    return val/ len(df) *100


def simple_graph(dataframe,type_of_graph, top = 0):
    data_frame = df[dataframe].value_counts()
    layout = go.Layout()
    
    if type_of_graph == 'barh':
        top_category = get_list(df[dataframe].dropna())
        if top !=None:
            data = [go.Bar(
                x=top_category[1].head(top),
                y=top_category[0].head(top),
                orientation = 'h',
                marker=dict(color=random_colors(10), line=dict(color='rgb(8,48,107)',width=1.5,)),
                opacity = 0.6
            )]
        else:
            data = [go.Bar(
            x=top_category[1],
            y=top_category[0],
            orientation = 'h',
            marker=dict(color=random_colors(10), line=dict(color='rgb(8,48,107)',width=1.5,)),
            opacity = 0.6
        )]

    elif type_of_graph == 'barv':
        top_category = get_list(df[dataframe].dropna())
        if top !=None:
            data = [go.Bar(
                x=top_category[0].head(top),
                y=top_category[1].head(top),
                marker=dict(color=random_colors(10), line=dict(color='rgb(8,48,107)',width=1.5,)),
                opacity = 0.6
        )]
        else:
            data = [go.Bar(
                x=top_category[0],
                y=top_category[1],
                marker=dict(color=random_colors(10), line=dict(color='rgb(8,48,107)',width=1.5,)),
                opacity = 0.6
            )]      

    elif type_of_graph == 'pie':
        data = [go.Pie(
            labels = data_frame.index,
            values = data_frame.values,
            marker = dict(colors = random_colors(20)),
            textfont = dict(size = 20)
        )]
    
    elif type_of_graph == 'pie_':
        data = [go.Pie(
            labels = data_frame.index,
            values = data_frame.values,
            marker = dict(colors = random_colors(20)),
            textfont = dict(size = 20)
        )]
        layout = go.Layout(legend=dict(orientation="h"), autosize=False,width=700,height=700)
        pass
    
    fig = go.Figure(data = data, layout = layout)
    py.iplot(fig)
    
    
    
def funnel_chart(index, values):
    values = values
    phases = index
    colors = random_colors(10)
    n_phase = len(phases)
    plot_width = 400
    section_h = 100
    section_d = 10
    unit_width = plot_width / max(values)
    phase_w = [int(value * unit_width) for value in values]
    height = section_h * n_phase + section_d * (n_phase - 1)
    shapes = []
    label_y = []
    for i in range(n_phase):
            if (i == n_phase-1):
                    points = [phase_w[i] / 2, height, phase_w[i] / 2, height - section_h]
            else:
                    points = [phase_w[i] / 2, height, phase_w[i+1] / 2, height - section_h]

            path = 'M {0} {1} L {2} {3} L -{2} {3} L -{0} {1} Z'.format(*points)

            shape = {'type': 'path','path': path,'fillcolor': colors[i],
                    'line': {
                        'width': 1,
                        'color': colors[i]
                    }
            }
            shapes.append(shape)
            label_y.append(height - (section_h / 2))
            height = height - (section_h + section_d)
    label_trace = go.Scatter(
        x=[-350]*n_phase,
        y=label_y,
        mode='text',
        text=phases,
        textfont=dict(
            color='rgb(200,200,200)',
            size=15
        )
    )

    value_trace = go.Scatter(
        x=[350]*n_phase,
        y=label_y,
        mode='text',
        text=values,
        textfont=dict(
            color='rgb(200,200,200)',
            size=15
        )
    )

    data = [label_trace, value_trace]
    layout = go.Layout(title="<b>Funnel Chart</b>",titlefont=dict(size=20,color='rgb(203,203,203)'),
        shapes=shapes,
        height=560,
        width=800,
        showlegend=False,
        paper_bgcolor='rgba(44,58,71,1)',
        plot_bgcolor='rgba(44,58,71,1)',
        xaxis=dict(
            showticklabels=False,
            zeroline=False,
        ),
        yaxis=dict(
            showticklabels=False,
            zeroline=False
        )
    )
    fig = go.Figure(data=data, layout=layout)
    py.iplot(fig)

def bubble_chart(col):
    data = get_list(df[col].dropna())
    data = data[:10]
    data = data.reindex(index=data.index[::-1])

    size = np.array(data[1]*0.001)
    size
    trace0 = go.Scatter(
        x=data[0],
        y=data[1],
        mode='markers',
        marker=dict(color = random_colors(10),size= size)
    )

    data = [trace0]
    py.iplot(data)
percent = df.isnull().sum()/ len(df) *100
percent.sort_values(ascending = False)
countries = df['Country'].value_counts()

countries = countries.to_frame().reset_index()
countries.loc[2]['code'] = 'test'
for i,country in enumerate(countries['index']):
    user_input = country
    mapping = {country.name: country.alpha_3 for country in pycountry.countries}
    countries.set_value(i, 'code', mapping.get(user_input))
data = [ dict(
        type = 'choropleth',
        locations = countries['code'],
        z = countries['Country'],
        text = countries['index'],
        colorscale = [[0,"rgb(5, 10, 172)"],[0.35,"rgb(40, 60, 190)"],[0.5,"rgb(70, 100, 245)"],\
            [0.6,"rgb(90, 120, 245)"],[0.7,"rgb(106, 137, 247)"],[1,"rgb(220, 220, 220)"]],
        autocolorscale = False,
        reversescale = True,
        marker = dict(
            line = dict (
                color = 'rgb(180,180,180)',
                width = 0.5
            ) ),
        colorbar = dict(
            autotick = False,
            tickprefix = '',
            title = 'Total Count'),
      ) ]

layout = dict(
    title = 'countries which responded to the survey',
    geo = dict(
        showframe = False,
        showcoastlines = False,
        projection = dict(
            type = 'Mercator'
        )
    )
)

fig = dict( data=data, layout=layout )
py.iplot( fig, validate=False)
simple_graph('Hobby','pie')
data = df[['Hobby','Gender']].dropna()

trace1 = go.Bar(
    x=['Female', 'Male'],
    y=[data[(data['Gender'] == 'Female') & (data['Hobby'] == 'Yes')].count()[0], data[(data['Gender'] == 'Male') & (data['Hobby'] == 'Yes')].count()[0]],
    name='Yes',
    opacity=0.6
)
trace2 = go.Bar(
    x=['Female', 'Male'],
    y=[data[(data['Gender'] == 'Female') & (data['Hobby'] == 'No')].count()[0], data[(data['Gender'] == 'Male') & (data['Hobby'] == 'No')].count()[0]],
    name='No',
    opacity=0.6
)

data = [trace1, trace2]
layout = go.Layout(
    barmode='group'
)

fig = go.Figure(data=data, layout=layout)
py.iplot(fig)
data = df[ (df['Hobby'] == 'Yes')]
country = data["Country"].dropna()

for i in country.unique():
    if country[country == i].count() < 600:
        country[country == i] = 'Others'
x = 0
y = 0
width = 50
height = 50
type_list = country.value_counts().index
values = country.value_counts().values

normed = squarify.normalize_sizes(values, width, height)
rects = squarify.squarify(normed, x, y, width, height)

color_brewer = random_colors(20)
shapes = []
annotations = []
counter = 0

for r in rects:
    shapes.append( 
        dict(
            type = 'rect', 
            x0 = r['x'], 
            y0 = r['y'], 
            x1 = r['x']+r['dx'], 
            y1 = r['y']+r['dy'],
            line = dict( width = 1 ),
            fillcolor = color_brewer[counter]
        ) 
    )
    annotations.append(
        dict(
            x = r['x']+(r['dx']/2),
            y = r['y']+(r['dy']/2),
            text = "{}".format(type_list[counter]),
            showarrow = False
        )
    )
    counter = counter + 1
    if counter >= len(color_brewer):
        counter = 0

trace0 = go.Scatter(
    x = [ r['x']+(r['dx']/2) for r in rects ], 
    y = [ r['y']+(r['dy']/2) for r in rects ],
    text = [ str(v) for v in values ], 
    mode = 'text',
)

layout = dict(
    height=600, 
    width=850,
    xaxis=dict(showgrid=False,zeroline=False),
    yaxis=dict(showgrid=False,zeroline=False),
    shapes=shapes,
    annotations=annotations,
    hovermode='closest',
    font=dict(color="#FFFFFF"),
    margin = go.Margin(
            l=0,
            r=0,
            pad=0
        )
)

figure = dict(data=[trace0], layout=layout)
iplot(figure)
data = df[['Hobby','Age']]
age = data['Age'].dropna().unique()

label = np.concatenate((np.array(data['Age'].dropna().unique()), np.array(['Yes','No'])), axis = 0)
data = dict(
    type='sankey',
    node = dict(
      pad = 15,
      thickness = 20,
      line = dict(
        color = random_colors(1),
        width = 0.5
      ),
      label = label,
      color = random_colors(20)
    ),
    link = dict(
      source = [0,1,2,3,4,5,6,0,1,2,3,4,5,6],
      target = [7,7,7,7,7,7,7,8,8,8,8,8,8,8],
      value = [len(data[(data['Age'] =='25 - 34 years old') & (data['Hobby'] == 'Yes')])
               ,len(data[(data['Age'] =='35 - 44 years old') & (data['Hobby'] == 'Yes')])
               ,len(data[(data['Age'] =='18 - 24 years old') & (data['Hobby'] == 'Yes')])
               ,len(data[(data['Age'] =='45 - 54 years old') & (data['Hobby'] == 'Yes')])
               ,len(data[(data['Age'] =='55 - 64 years old') & (data['Hobby'] == 'Yes')])
               ,len(data[(data['Age'] =='Under 18 years old') & (data['Hobby'] == 'Yes')])
               ,len(data[(data['Age'] =='65 years or older') & (data['Hobby'] == 'Yes')])
               ,len(data[(data['Age'] =='25 - 34 years old') & (data['Hobby'] == 'No')])
               ,len(data[(data['Age'] =='35 - 44 years old') & (data['Hobby'] == 'No')])
               ,len(data[(data['Age'] =='18 - 24 years old') & (data['Hobby'] == 'No')])
               ,len(data[(data['Age'] =='45 - 54 years old') & (data['Hobby'] == 'No')])
               ,len(data[(data['Age'] =='55 - 64 years old') & (data['Hobby'] == 'No')])
               ,len(data[(data['Age'] =='Under 18 years old') & (data['Hobby'] == 'No')])
               ,len(data[(data['Age'] =='65 years or older') & (data['Hobby'] == 'No')])
              ]
  ))


fig = dict(data=[data])
py.iplot(fig, validate=False)
data = df[['Hobby','OpenSource']].dropna()

trace1 = go.Bar(
    x=['Yes', 'No'],
    y=[data[(data['OpenSource'] == 'Yes') & (data['Hobby'] == 'Yes')].count()[0], data[(data['OpenSource'] == 'Yes') & (data['Hobby'] == 'No')].count()[0]],
    name='Yes',
    opacity=0.6
)

data = [trace1]
layout = go.Layout(
    barmode='group'
)

fig = go.Figure(data=data, layout=layout)
py.iplot(fig)
simple_graph('OpenSource','pie')
data = df[['OpenSource','Gender']].dropna()

trace1 = go.Bar(
    x=['Female', 'Male'],
    y=[data[(data['Gender'] == 'Female') & (data['OpenSource'] == 'Yes')].count()[0], data[(data['Gender'] == 'Male') & (data['OpenSource'] == 'Yes')].count()[0]],
    name='Yes',
    opacity=0.6
)
trace2 = go.Bar(
    x=['Female', 'Male'],
    y=[data[(data['Gender'] == 'Female') & (data['OpenSource'] == 'No')].count()[0], data[(data['Gender'] == 'Male') & (data['OpenSource'] == 'No')].count()[0]],
    name='No',
    opacity=0.6
)

data = [trace1, trace2]
layout = go.Layout(
    barmode='group',
    orientation = 'v'
)

fig = go.Figure(data=data, layout=layout)
py.iplot(fig)
data = df[df['OpenSource'] == 'Yes']
countries = data['Country'].value_counts()

countries = countries.to_frame().reset_index()
countries.loc[2]['code'] = 'test'
for i,country in enumerate(countries['index']):
    user_input = country
    mapping = {country.name: country.alpha_3 for country in pycountry.countries}
    countries.set_value(i, 'code', mapping.get(user_input))
data = [ dict(
        type = 'choropleth',
        locations = countries['code'],
        z = countries['Country'],
        text = countries['index'],
        colorscale = [[0,"rgb(5, 10, 172)"],[0.35,"rgb(40, 60, 190)"],[0.5,"rgb(70, 100, 245)"],\
            [0.6,"rgb(90, 120, 245)"],[0.7,"rgb(106, 137, 247)"],[1,"rgb(220, 220, 220)"]],
        autocolorscale = False,
        reversescale = True,
        marker = dict(
            line = dict (
                color = 'rgb(180,180,180)',
                width = 0.5
            ) ),
        colorbar = dict(
            autotick = False,
            tickprefix = '$',
            title = 'Total Count'),
      ) ]

layout = dict(
    title = 'countries which responded to the survey',
    geo = dict(
        showframe = False,
        showcoastlines = False,
        projection = dict(
            type = 'Mercator'
        )
    )
)

fig = dict( data=data, layout=layout )
py.iplot( fig, validate=False)
data = df[['OpenSource','Age']]
age = data['Age'].dropna().unique()

label = np.concatenate((np.array(data['Age'].dropna().unique()), np.array(['Yes','No'])), axis = 0)
data = dict(
    type='sankey',
    node = dict(
      pad = 15,
      thickness = 20,
      line = dict(
        color = random_colors(1),
        width = 0.5
      ),
      label = label,
      color = random_colors(20)
    ),
    link = dict(
      source = [0,1,2,3,4,5,6,0,1,2,3,4,5,6],
      target = [8,8,8,8,8,8,8,7,7,7,7,7,7,7],
      value = [len(data[(data['Age'] =='25 - 34 years old') & (data['OpenSource'] == 'Yes')])
               ,len(data[(data['Age'] =='35 - 44 years old') & (data['OpenSource'] == 'Yes')])
               ,len(data[(data['Age'] =='18 - 24 years old') & (data['OpenSource'] == 'Yes')])
               ,len(data[(data['Age'] =='45 - 54 years old') & (data['OpenSource'] == 'Yes')])
               ,len(data[(data['Age'] =='55 - 64 years old') & (data['OpenSource'] == 'Yes')])
               ,len(data[(data['Age'] =='Under 18 years old') & (data['OpenSource'] == 'Yes')])
               ,len(data[(data['Age'] =='65 years or older') & (data['OpenSource'] == 'Yes')])
               ,len(data[(data['Age'] =='25 - 34 years old') & (data['OpenSource'] == 'No')])
               ,len(data[(data['Age'] =='35 - 44 years old') & (data['OpenSource'] == 'No')])
               ,len(data[(data['Age'] =='18 - 24 years old') & (data['OpenSource'] == 'No')])
               ,len(data[(data['Age'] =='45 - 54 years old') & (data['OpenSource'] == 'No')])
               ,len(data[(data['Age'] =='55 - 64 years old') & (data['OpenSource'] == 'No')])
               ,len(data[(data['Age'] =='Under 18 years old') & (data['OpenSource'] == 'No')])
               ,len(data[(data['Age'] =='65 years or older') & (data['OpenSource'] == 'No')])
              ]
  ))


fig = dict(data=[data])
py.iplot(fig, validate=False)
data = df[df['OpenSource'] == 'Yes']

data = get_list(df['LanguageWorkedWith'].dropna())

x = 0
y = 0
width = 50
height = 50
type_list = data[0]
values = data[1]

normed = squarify.normalize_sizes(values, width, height)
rects = squarify.squarify(normed, x, y, width, height)

color_brewer = random_colors(20)
shapes = []
annotations = []
counter = 0

for r in rects:
    shapes.append( 
        dict(
            type = 'rect', 
            x0 = r['x'], 
            y0 = r['y'], 
            x1 = r['x']+r['dx'], 
            y1 = r['y']+r['dy'],
            line = dict( width = 1 ),
            fillcolor = color_brewer[counter]
        ) 
    )
    annotations.append(
        dict(
            x = r['x']+(r['dx']/2),
            y = r['y']+(r['dy']/2),
            text = "{}".format(type_list[counter]),
            showarrow = False
        )
    )
    counter = counter + 1
    if counter >= len(color_brewer):
        counter = 0

trace0 = go.Scatter(
    x = [ r['x']+(r['dx']/2) for r in rects ], 
    y = [ r['y']+(r['dy']/2) for r in rects ],
    text = [ str(v) for v in values ], 
    mode = 'text',
)

layout = dict(
    height=600, 
    width=850,
    xaxis=dict(showgrid=False,zeroline=False),
    yaxis=dict(showgrid=False,zeroline=False),
    shapes=shapes,
    annotations=annotations,
    hovermode='closest',
    font=dict(color="#FFFFFF"),
    margin = go.Margin(
            l=0,
            r=0,
            pad=0
        )
)

figure = dict(data=[trace0], layout=layout)
iplot(figure)
data = df[['Hobby','OpenSource']].dropna()

trace1 = go.Bar(
    x=['Yes', 'No'],
    y=[data[(data['OpenSource'] == 'No') & (data['Hobby'] == 'No')].count()[0]],
    name='Yes',
    opacity=0.6
)

data = [trace1]
layout = go.Layout(
)

fig = go.Figure(data=data, layout=layout)
py.iplot(fig)
bubble_chart('LanguageWorkedWith')
bubble_chart('LanguageDesireNextYear')
top_languages = get_list(df['LanguageWorkedWith'].dropna())
top_desire_languages = get_list(df['LanguageDesireNextYear'].dropna())
top_languages = top_languages.sort_values(by=[0])
top_desire_languages = top_desire_languages.sort_values(by=[0])

raise_fall_ratio = pd.DataFrame()
raise_fall_ratio['2018'] = top_languages[0]
raise_fall_ratio['2018_percent'] = top_languages[1].apply(calculate_percent)
raise_fall_ratio['2019_percent'] = top_desire_languages[1].apply(calculate_percent)

trace1 = go.Bar(
    x=raise_fall_ratio['2018'],
    y=raise_fall_ratio['2018_percent'],
    name='2018'
)
trace2 = go.Bar(
    x=raise_fall_ratio['2018'],
    y=raise_fall_ratio['2019_percent'],
    name='2019'
)

data = [trace1, trace2]
layout = go.Layout(
    barmode='stack'
)

fig = go.Figure(data=data, layout=layout)
py.iplot(fig)

data = df[['LanguageWorkedWith','Age']].dropna().reset_index()
d = []
for i, val in enumerate(data['LanguageWorkedWith']):
    lang = val.split(';')
    for j in lang:
        temp = np.array([j, data['Age'][i]])
        d.append(temp)
new_df = pd.DataFrame(d, columns=('Lang','Age'))

lang_list = ['C++','C#','PHP','Python','Bash/Shell','Java','SQL','CSS','HTML','JavaScript']
scatter = new_df.groupby(['Lang','Age']).size().reset_index()
scatter = scatter[scatter['Lang'].isin(lang_list)]


data = [go.Scatter3d(
    x=scatter['Lang'],
    y=scatter['Age'],
    z=scatter[0],
    mode='markers',
    marker=dict(line=dict(color=random_colors(1),width=0.5),opacity=0.8)
)]
layout = go.Layout(
    autosize=False,
    width=700,
    height=700,
    title = 'Please click and drag for the graph to begin'
)
fig = go.Figure(data=data, layout=layout)
py.iplot(fig)
data = df[['LanguageWorkedWith','CompanySize']].dropna().reset_index()
d = []
for i, val in enumerate(data['LanguageWorkedWith']):
    lang = val.split(';')
    for j in lang:
        temp = np.array([j, data['CompanySize'][i]])
        d.append(temp)
new_df = pd.DataFrame(d, columns=('Lang','CompanySize'))

company_list = ['20 to 99 employees', '10,000 or more employees','100 to 499 employees', '10 to 19 employees',
       '500 to 999 employees', '1,000 to 4,999 employees','5,000 to 9,999 employees', 'Fewer than 10 employees']


scatter = new_df.groupby(['Lang','CompanySize']).size().reset_index()
scatter = scatter[scatter['Lang'].isin(lang_list)]


data = [go.Scatter3d(
    x=scatter['Lang'],
    y=scatter['CompanySize'],
    z=scatter[0],
    mode='markers',
    marker=dict(line=dict(color=random_colors(1),width=0.9),opacity=0.8)
)]
layout = go.Layout(
    autosize=False,
    width=700,
    height=700,
    title = 'Please click and drag for the graph to begin'
)
fig = go.Figure(data=data, layout=layout)
py.iplot(fig)
bubble_chart('DatabaseWorkedWith')
bubble_chart('DatabaseDesireNextYear')
top_database = get_list(df['DatabaseWorkedWith'].dropna())
top_desire_database = get_list(df['DatabaseDesireNextYear'].dropna())

top_database_ = top_database.sort_values(by=[0])
top_desire_database_ = top_desire_database.sort_values(by=[0])

raise_fall_ratio = pd.DataFrame()
raise_fall_ratio['2018'] = top_database_[0]
raise_fall_ratio['2018_percent'] = top_database_[1].apply(calculate_percent)
raise_fall_ratio['2019_percent'] = top_desire_database_[1].apply(calculate_percent)

trace1 = go.Bar(
    x=raise_fall_ratio['2018'],
    y=raise_fall_ratio['2018_percent'],
    name='2018'
)
trace2 = go.Bar(
    x=raise_fall_ratio['2018'],
    y=raise_fall_ratio['2019_percent'],
    name='2019'
)

data = [trace1, trace2]
layout = go.Layout(
    barmode='stack'
)

fig = go.Figure(data=data, layout=layout)
py.iplot(fig)
data = df[['DatabaseWorkedWith','Age']].dropna().reset_index()
d = []
for i, val in enumerate(data['DatabaseWorkedWith']):
    lang = val.split(';')
    for j in lang:
        temp = np.array([j, data['Age'][i]])
        d.append(temp)
new_df = pd.DataFrame(d, columns=('Database','Age'))

database_list = ['Microsoft Azure (Tables, CosmosDB, SQL, etc)', 'Oracle','MariaDB', 'Elasticsearch',
       'Redis', 'SQLite','MongoDB', 'MySQL']


scatter = new_df.groupby(['Database','Age']).size().reset_index()
scatter = scatter[scatter['Database'].isin(database_list)]


data = [go.Scatter3d(
    x=scatter['Database'],
    y=scatter['Age'],
    z=scatter[0],
    mode='markers',
    marker=dict(line=dict(color=random_colors(1),width=0.9),opacity=0.8)
)]
layout = go.Layout(
    autosize=False,
    width=700,
    height=700,
    title = 'Please click and drag for the graph to begin'
)
fig = go.Figure(data=data, layout=layout)
py.iplot(fig)
data = df[['DatabaseWorkedWith','CompanySize']].dropna().reset_index()
d = []
for i, val in enumerate(data['DatabaseWorkedWith']):
    lang = val.split(';')
    for j in lang:
        temp = np.array([j, data['CompanySize'][i]])
        d.append(temp)
new_df = pd.DataFrame(d, columns=('Database','CompanySize'))

database_list = ['Microsoft Azure (Tables, CosmosDB, SQL, etc)', 'Oracle','MariaDB', 'Elasticsearch',
       'Redis', 'SQLite','MongoDB', 'MySQL']


scatter = new_df.groupby(['Database','CompanySize']).size().reset_index()
scatter = scatter[scatter['Database'].isin(database_list)]


data = [go.Scatter3d(
    x=scatter['Database'],
    y=scatter['CompanySize'],
    z=scatter[0],
    mode='markers',
    marker=dict(line=dict(color=random_colors(1),width=0.9),opacity=0.8)
)]
layout = go.Layout(
    autosize=False,
    width=700,
    height=700,
    title = 'Please click and drag for the graph to begin'
)
fig = go.Figure(data=data, layout=layout)
py.iplot(fig)
bubble_chart('PlatformWorkedWith')
bubble_chart('PlatformDesireNextYear')
top_platform = get_list(df['PlatformWorkedWith'].dropna())
top_desire_platform = get_list(df['PlatformDesireNextYear'].dropna())

top_platform_ = top_platform.sort_values(by=[0])
top_desire_platform_ = top_desire_platform.sort_values(by=[0])

raise_fall_ratio = pd.DataFrame()
raise_fall_ratio['2018'] = top_platform_[0]
raise_fall_ratio['2018_percent'] = top_platform_[1].apply(calculate_percent)
raise_fall_ratio['2019_percent'] = top_desire_platform_[1].apply(calculate_percent)

trace1 = go.Bar(
    x=raise_fall_ratio['2018'],
    y=raise_fall_ratio['2018_percent'],
    name='2018'
)
trace2 = go.Bar(
    x=raise_fall_ratio['2018'],
    y=raise_fall_ratio['2019_percent'],
    name='2019'
)

data = [trace1, trace2]
layout = go.Layout(
    barmode='stack'
)

fig = go.Figure(data=data, layout=layout)
py.iplot(fig)

data = df[['PlatformWorkedWith','Age']].dropna().reset_index()
d = []
for i, val in enumerate(data['PlatformWorkedWith']):
    lang = val.split(';')
    for j in lang:
        temp = np.array([j, data['Age'][i]])
        d.append(temp)
new_df = pd.DataFrame(d, columns=('Platform','Age'))

platform_list = ['Azure', 'Firebase','iOS', 'WordPress',
       'Android', 'Raspberry Pi','Mac OS', 'AWS','Linux']


scatter = new_df.groupby(['Platform','Age']).size().reset_index()
scatter = scatter[scatter['Platform'].isin(platform_list)]


data = [go.Scatter3d(
    x=scatter['Platform'],
    y=scatter['Age'],
    z=scatter[0],
    mode='markers',
    marker=dict(line=dict(color=random_colors(1),width=0.9),opacity=0.8)
)]
layout = go.Layout(
    autosize=False,
    width=700,
    height=700,
    title = 'Please click and drag for the graph to begin'
)
fig = go.Figure(data=data, layout=layout)
py.iplot(fig)
data = df[['PlatformWorkedWith','CompanySize']].dropna().reset_index()
d = []
for i, val in enumerate(data['PlatformWorkedWith']):
    lang = val.split(';')
    for j in lang:
        temp = np.array([j, data['CompanySize'][i]])
        d.append(temp)
new_df = pd.DataFrame(d, columns=('Platform','CompanySize'))

platform_list = ['Azure', 'Firebase','iOS', 'WordPress',
       'Android', 'Raspberry Pi','Mac OS', 'AWS','Linux']


scatter = new_df.groupby(['Platform','CompanySize']).size().reset_index()
scatter = scatter[scatter['Platform'].isin(platform_list)]


data = [go.Scatter3d(
    x=scatter['Platform'],
    y=scatter['CompanySize'],
    z=scatter[0],
    mode='markers',
    marker=dict(line=dict(color=random_colors(1),width=0.9),opacity=0.8)
)]
layout = go.Layout(
    autosize=False,
    width=700,
    height=700,
    title = 'Please click and drag for the graph to begin'
)
fig = go.Figure(data=data, layout=layout)
py.iplot(fig)
data = get_list(df['FrameworkWorkedWith'].dropna())
funnel_chart(data[0][:5], data[1][:5])
data = get_list(df['OperatingSystem'].dropna())
funnel_chart(data[0][:5], data[1][:5])
data = get_list(df['Methodology'].dropna())
funnel_chart(data[0][:5], data[1][:5])
simple_graph('VersionControl','barh',10)
simple_graph('AIDangerous','barv',10)
simple_graph('AIInteresting','barv',10)
simple_graph('AIResponsible','barv',10)
simple_graph('AIFuture','barv',10)
simple_graph('NumberMonitors','barv',10)
simple_graph('HopeFiveYears','barv',10)
simple_graph('WakeTime','barv',10)
simple_graph('JobSatisfaction', 'pie')
simple_graph('Exercise','barv',10)
simple_graph('TimeFullyProductive','barv',10)
simple_graph('TimeAfterBootcamp','barv',10)
simple_graph('AgreeDisagree1','barv',10)
simple_graph('AgreeDisagree2','barv',10)
simple_graph('AgreeDisagree3','barv',10)
simple_graph('IDE','barh',10)
simple_graph('CheckInCode','barv',10)
simple_graph('DevType','barv',10)
simple_graph('FormalEducation', 'pie_')
simple_graph('JobSatisfaction','pie')
simple_graph('UndergradMajor', 'pie')
simple_graph('Student', 'pie')
simple_graph('CompanySize','barv',10)
simple_graph('CommunicationTools','barv',10)
data = pd.DataFrame(columns = df['CompanySize'].dropna().unique(),index = df['TimeFullyProductive'].dropna().unique())
for col in data.columns:
    for index in data.index:
        data[col][index] = len(df[(df['CompanySize'] == col) & (df['TimeFullyProductive'] == index)])
        
graph = [
    go.Scatter(
        x=data.index,
        y=data['20 to 99 employees'],
        name = '20 to 99 employees',
    ),
    go.Scatter(
        x=data.index,
        y=data['10,000 or more employees'],
        name = '10,000 or more employees',
    ),
    go.Scatter(
        x=data.index,
        y=data['100 to 499 employees'],
        name = '100 to 499 employees',
    ),
    go.Scatter(
        x=data.index,
        y=data['10 to 19 employees'],
        name = '10 to 19 employees',
    ),
    go.Scatter(
        x=data.index,
        y=data['500 to 999 employees'],
        name = '500 to 999 employees',
    ),
    go.Scatter(
        x=data.index,
        y=data['1,000 to 4,999 employees'],
        name = '1,000 to 4,999 employees',
    ),
    go.Scatter(
        x=data.index,
        y=data['5,000 to 9,999 employees'],
        name = '5,000 to 9,999 employees',
    ),
    go.Scatter(
        x=data.index,
        y=data['Fewer than 10 employees'],
        name = 'Fewer than 10 employees',
    )
]

layout = go.Layout(barmode='stack')

fig = go.Figure(data=graph, layout=layout)

py.iplot(fig)
data = pd.DataFrame(columns = df['CompanySize'].dropna().unique(),index = df['JobSatisfaction'].dropna().unique())
for col in data.columns:
    for index in data.index:
        data[col][index] = len(df[(df['CompanySize'] == col) & (df['JobSatisfaction'] == index)])
        
graph = [
    go.Scatter(
        x=data.index,
        y=data['20 to 99 employees'],
        name = '20 to 99 employees',
    ),
    go.Scatter(
        x=data.index,
        y=data['10,000 or more employees'],
        name = '10,000 or more employees',
    ),
    go.Scatter(
        x=data.index,
        y=data['100 to 499 employees'],
        name = '100 to 499 employees',
    ),
    go.Scatter(
        x=data.index,
        y=data['10 to 19 employees'],
        name = '10 to 19 employees',
    ),
    go.Scatter(
        x=data.index,
        y=data['500 to 999 employees'],
        name = '500 to 999 employees',
    ),
    go.Scatter(
        x=data.index,
        y=data['1,000 to 4,999 employees'],
        name = '1,000 to 4,999 employees',
    ),
    go.Scatter(
        x=data.index,
        y=data['5,000 to 9,999 employees'],
        name = '5,000 to 9,999 employees',
    ),
    go.Scatter(
        x=data.index,
        y=data['Fewer than 10 employees'],
        name = 'Fewer than 10 employees',
    )
]

layout = go.Layout(barmode='stack')

fig = go.Figure(data=graph, layout=layout)

py.iplot(fig)
data = pd.DataFrame(columns = df['CompanySize'].dropna().unique(),index = df['Hobby'].dropna().unique())
for col in data.columns:
    for index in data.index:
        data[col][index] = len(df[(df['CompanySize'] == col) & (df['Hobby'] == index)])
        
graph = [
    go.Bar(
        x=data.index,
        y=data['20 to 99 employees'],
        name = '20 to 99 employees',
    ),
    go.Bar(
        x=data.index,
        y=data['10,000 or more employees'],
        name = '10,000 or more employees',
    ),
    go.Bar(
        x=data.index,
        y=data['100 to 499 employees'],
        name = '100 to 499 employees',
    ),
    go.Bar(
        x=data.index,
        y=data['10 to 19 employees'],
        name = '10 to 19 employees',
    ),
    go.Bar(
        x=data.index,
        y=data['500 to 999 employees'],
        name = '500 to 999 employees',
    ),
    go.Bar(
        x=data.index,
        y=data['1,000 to 4,999 employees'],
        name = '1,000 to 4,999 employees',
    ),
    go.Bar(
        x=data.index,
        y=data['5,000 to 9,999 employees'],
        name = '5,000 to 9,999 employees',
    ),
    go.Bar(
        x=data.index,
        y=data['Fewer than 10 employees'],
        name = 'Fewer than 10 employees',
    )
]

layout = go.Layout(barmode='stack')

fig = go.Figure(data=graph, layout=layout)

py.iplot(fig)
countries = df[['Country','Gender']]
countries = countries[countries['Gender'] == 'Female']
countries = countries['Country'].value_counts()

countries = countries.to_frame().reset_index()
countries.loc[2]['code'] = ''
for i,country in enumerate(countries['index']):
    user_input = country
    mapping = {country.name: country.alpha_3 for country in pycountry.countries}
    countries.set_value(i, 'code', mapping.get(user_input))
data = [ dict(
        type = 'choropleth',
        locations = countries['code'],
        z = countries['Country'],
        text = countries['index'],
        colorscale = [[0,"rgb(5, 10, 172)"],[0.35,"rgb(40, 60, 190)"],[0.5,"rgb(70, 100, 245)"],\
            [0.6,"rgb(90, 120, 245)"],[0.7,"rgb(106, 137, 247)"],[1,"rgb(220, 220, 220)"]],
        autocolorscale = False,
        reversescale = True,
        marker = dict(
            line = dict (
                color = 'rgb(180,180,180)',
                width = 0.5
            ) ),
        colorbar = dict(
            autotick = False,
            title = 'Total Female'),
      ) ]

layout = dict(
    title = 'countries with female employees',
    geo = dict(
        showframe = False,
        showcoastlines = False,
        projection = dict(
            type = 'Mercator'
        )
    )
)

fig = dict( data=data, layout=layout )
py.iplot( fig, validate=False)
simple_graph('Gender','barv',10)