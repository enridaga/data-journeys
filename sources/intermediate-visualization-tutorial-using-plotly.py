
# Importing libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from plotly import tools
import plotly.plotly as py
from plotly.offline import init_notebook_mode, iplot
init_notebook_mode(connected=True)
import plotly.graph_objs as go
import plotly.figure_factory as ff
from IPython.display import HTML, Image

# Importing data
df = pd.read_csv('../input/Pokemon.csv')
# To check for null values, let's see the dataset's info
df.info()
# Almost half of the Type 2 attribute is empty but it's because many pokemon have only one type. Still we will fill this with 'Blank'
df = df.fillna(value={'Type 2':'Blank'})
# Let's take a look at the data 
df.head()
fig = ff.create_distplot([df.HP],['HP'],bin_size=5)
iplot(fig, filename='Basic Distplot')
hist_data = [df['HP'],df['Attack'],df['Defense'],df['Sp. Atk'],df['Sp. Def'],df['Speed']]
group_labels = list(df.iloc[:,5:11].columns)

fig = ff.create_distplot(hist_data, group_labels, bin_size=5)
iplot(fig, filename='Distplot of all pokemon stats')
hist_data = [df['Attack'],df['Defense']]
group_labels = ['Attack','Defense']

fig = ff.create_distplot(hist_data, group_labels, bin_size=5)
iplot(fig, filename='Distplot of attack and defense')
trace0 = go.Box(y=df["HP"],name="HP")
trace1 = go.Box(y=df["Attack"],name="Attack")
trace2 = go.Box(y=df["Defense"],name="Defense")
trace3 = go.Box(y=df["Sp. Atk"],name="Sp. Atk")
trace4 = go.Box(y=df["Sp. Def"],name="Sp. Def")
trace5 = go.Box(y=df["Speed"],name="Speed")
data = [trace0, trace1, trace2,trace3, trace4, trace5]
iplot(data)
trace0 = go.Box(
    y=df["HP"],
    boxmean = True,
    name="HP(with Mean)"
)
trace1 = go.Box(
    y=df["Attack"],
    boxmean = 'sd',
    name="Attack(Mean and SD)"
)
trace2 = go.Box(
    y=df["Defense"],
    jitter = 0.5,
    pointpos = -2,
    boxpoints = 'all',
    name = "Defense(All points)"
)
trace3 = go.Box(
    y=df["Sp. Atk"],
    boxpoints = False,
    name = "Sp. Atk(Only Whiskers)"
)
trace4 = go.Box(
    y=df["Sp. Def"],
    boxpoints = 'suspectedoutliers',
    marker = dict(
        outliercolor = 'rgba(219, 64, 82, 0.6)',
        line = dict(
            outliercolor = 'rgba(219, 64, 82, 0.6)',
            outlierwidth = 2)),
    line = dict(
        color = 'rgb(8,81,156)'),
    name = "Sp. Def(Suspected Outliers)"
)
trace5 = go.Box(
    y=df["Speed"],
    boxpoints = 'outliers',
    line = dict(
        color = 'rgb(107,174,214)'),
    name = "Speed(Whiskers and Outliers)"
)

layout = go.Layout(
    title = "Boxplot with customized outliers"
)
data = [trace0, trace1, trace2, trace3, trace4, trace5]
fig = go.Figure(data=data,layout=layout)
iplot(fig, filename = "Customized Boxplot")
x = df[df["Name"] == "Charizard"]
data = [go.Scatterpolar(
  r = [x['HP'].values[0],x['Attack'].values[0],x['Defense'].values[0],x['Sp. Atk'].values[0],x['Sp. Def'].values[0],x['Speed'].values[0],x["HP"].values[0]],
  theta = ['HP','Attack','Defense','Sp. Atk','Sp. Def','Speed','HP'],
  fill = 'toself'
)]

layout = go.Layout(
  polar = dict(
    radialaxis = dict(
      visible = True,
      range = [0, 250]
    )
  ),
  showlegend = False,
  title = "Stats of {}".format(x.Name.values[0])
)
fig = go.Figure(data=data, layout=layout)
iplot(fig, filename = "Single Pokemon stats")
# Creating a method to compare 2 pokemon
def compare2pokemon(x,y):
    x = df[df["Name"] == x]
    y = df[df["Name"] == y]

    trace0 = go.Scatterpolar(
      r = [x['HP'].values[0],x['Attack'].values[0],x['Defense'].values[0],x['Sp. Atk'].values[0],x['Sp. Def'].values[0],x['Speed'].values[0],x["HP"].values[0]],
      theta = ['HP','Attack','Defense','Sp. Atk','Sp. Def','Speed','HP'],
      fill = 'toself',
      name = x.Name.values[0]
    )

    trace1 = go.Scatterpolar(
      r = [y['HP'].values[0],y['Attack'].values[0],y['Defense'].values[0],y['Sp. Atk'].values[0],y['Sp. Def'].values[0],y['Speed'].values[0],y["HP"].values[0]],
      theta = ['HP','Attack','Defense','Sp. Atk','Sp. Def','Speed','HP'],
      fill = 'toself',
      name = y.Name.values[0]
    )

    data = [trace0, trace1]

    layout = go.Layout(
      polar = dict(
        radialaxis = dict(
          visible = True,
          range = [0, 200]
        )
      ),
      showlegend = True,
      title = "{} vs {}".format(x.Name.values[0],y.Name.values[0])
    )
    fig = go.Figure(data=data, layout=layout)
    iplot(fig, filename = "Two Pokemon stats")
# Comparing primeape and muk
compare2pokemon("Primeape","Muk")
# Comparing Groudon and Kyogre
compare2pokemon("Groudon","Kyogre")
trace1 = go.Scatter(
    x = df["Defense"],
    y = df["Attack"],
    mode='markers',
    marker=dict(
        size='16',
        color = df["Speed"],#set color equal to a variable
        colorscale='Electric',
        showscale=True
    ),
    text=df["Name"]
)
data = [trace1]
layout = go.Layout(
  paper_bgcolor='rgba(0,0,0,1)',
  plot_bgcolor='rgba(0,0,0,1)',
  showlegend = False,
  font=dict(family='Courier New, monospace', size=10, color='#ffffff'),
  title="Scatter plot of Defense vs Attack with Speed as colorscale",
)
fig = go.Figure(data=data, layout=layout)
iplot(fig, filename = "Scatterplot")
t = go.Scatter3d(
    x=df["Speed"],
    y=df["Attack"],
    z=df["Defense"],
    mode='markers',
    marker=dict(
        size=4,
        line=dict(
            color='rgba(217, 217, 217, 0.14)',
            width=0.5
        ),
        opacity=1
    )
)
data = [t]
layout = go.Layout(
    margin=dict(
        l=0,
        r=0,
        b=0,
        t=0
    ),
    xaxis=dict(title="Speed"),
    yaxis=dict(title="Attack"),
    title = "Speed vs Attack vs Defense"
)
fig = go.Figure(data=data, layout=layout)
iplot(fig, filename='3d-scatter')
data = [
    go.Contour(
        x=['HP','Attack','Defense','Sp. Atk','Sp. Def','Speed'],
        z=df[df["Type 1"]=="Bug"].iloc[:,5:11].values,
        colorscale='Jet',
    )
]

layout = go.Layout(
    title = "Distribution of Bug pokemon",
    width = 600,
    height = 800
)

fig = go.Figure(data=data, layout=layout)
iplot(fig, filename='bug-contour')
gen1 = go.Contour(
    x = df[df["Generation"] == 1].iloc[:,5].values,
    y = df[df["Generation"] == 1].iloc[:,6].values,
    z = df[df["Generation"] == 1].iloc[:,7:11].values,
    name = "Generation 1",
    showscale=False,
)
gen2 = go.Contour(
    x = df[df["Generation"] == 2].iloc[:,5].values,
    y = df[df["Generation"] == 2].iloc[:,6].values,
    z = df[df["Generation"] == 2].iloc[:,7:11].values,
    name = "Generation 2",
    showscale=False,
)
gen3 = go.Contour(
    x = df[df["Generation"] == 3].iloc[:,5].values,
    y = df[df["Generation"] == 3].iloc[:,6].values,
    z = df[df["Generation"] == 3].iloc[:,7:11].values,
    name = "Generation 3",
    showscale=False
)
gen4 = go.Contour(
    x = df[df["Generation"] == 4].iloc[:,5].values,
    y = df[df["Generation"] == 4].iloc[:,6].values,
    z = df[df["Generation"] == 4].iloc[:,7:11].values,
    name = "Generation 4",
    showscale=False
)
gen5 = go.Contour(
    x = df[df["Generation"] == 5].iloc[:,5].values,
    y = df[df["Generation"] == 5].iloc[:,6].values,
    z = df[df["Generation"] == 5].iloc[:,7:11].values,
    name = "Generation 5",                     
    showscale=False
)
gen6 = go.Contour(
    x = df[df["Generation"] == 6].iloc[:,5].values,
    y = df[df["Generation"] == 6].iloc[:,6].values,
    z = df[df["Generation"] == 6].iloc[:,7:11].values,
    name = "Generation 6",                     
    showscale=False
)


fig = tools.make_subplots(rows=1, cols=6, subplot_titles=('Generation 1', 'Generation 2', 'Generation 3', 'Generation 4', 'Generation 5', 'Generation 6'), shared_yaxes=True)

fig.append_trace(gen1, 1, 1)
fig.append_trace(gen2, 1, 2)
fig.append_trace(gen3, 1, 3)
fig.append_trace(gen4, 1, 4)
fig.append_trace(gen5, 1, 5)
fig.append_trace(gen6, 1, 6)

fig['layout'].update(height=600, 
                     width=800, 
                     title='Contour subplots for different generations',
                     paper_bgcolor='rgba(0,0,0,1)',
                     plot_bgcolor='rgba(0,0,0,1)',
                     font=dict(size=12, 
                     color='#ffffff'),
                     showlegend=True,
                     margin=go.Margin(
                     l=50,
                     r=50,
                     b=100,
                     t=100,
                     pad=4,
                 ),
                 xaxis=dict(
                        domain=[0, 0.1]
                 ),
                xaxis2=dict(
                        domain=[0.15, 0.30]
                ),
                xaxis3=dict(
                        domain=[0.35, 0.45]
                ),  
                xaxis4=dict(
                        domain=[0.5, 0.6]
                ),            
                xaxis5=dict(
                        domain=[0.65, 0.75]
                ),  
                xaxis6=dict(
                        domain=[0.85, 1]
                )
 )
iplot(fig, filename='contour-subplots')
sizeref = 2.*max(df['HP'])/(3000)

trace0 = go.Scatter(
    x=df["Attack"][df["Type 1"] == "Fire"][df["Generation"] == 1],
    y=df["Defense"][df["Type 1"] == "Fire"][df["Generation"] == 1],
    mode='markers',
    name='Generation 1',
    text=df["Name"][df["Type 1"] == "Fire"][df["Generation"] == 1],
    marker=dict(
        symbol='circle',
        sizemode='area',
        size=df["HP"][df["Type 1"] == "Fire"][df["Generation"] == 1],
        sizeref=sizeref,
        line=dict(
            width=2
        ),
    )
)
trace1 = go.Scatter(
    x=df["Attack"][df["Type 1"] == "Fire"][df["Generation"] == 2],
    y=df["Defense"][df["Type 1"] == "Fire"][df["Generation"] == 2],
    mode='markers',
    name='Generation 2',
    text=df["Name"][df["Type 1"] == "Fire"][df["Generation"] == 2],
    marker=dict(
        sizemode='area',
        size=df["HP"][df["Type 1"] == "Fire"][df["Generation"] == 2],
        sizeref=sizeref,
        line=dict(
            width=2
        ),
    )
)
trace2 = go.Scatter(
    x=df["Attack"][df["Type 1"] == "Fire"][df["Generation"] == 3],
    y=df["Defense"][df["Type 1"] == "Fire"][df["Generation"] == 3],
    mode='markers',
    name='Generation 3',
    text=df["Name"][df["Type 1"] == "Fire"][df["Generation"] == 3],
    marker=dict(
        sizemode='area',
        size=df["HP"][df["Type 1"] == "Fire"][df["Generation"] == 3],
        sizeref=sizeref,
        line=dict(
            width=2
        ),
    )
)
trace3 = go.Scatter(
    x=df["Attack"][df["Type 1"] == "Fire"][df["Generation"] == 4],
    y=df["Defense"][df["Type 1"] == "Fire"][df["Generation"] == 4],
    mode='markers',
    name='Generation 4',
    text=df["Name"][df["Type 1"] == "Fire"][df["Generation"] == 4],
    marker=dict(
        sizemode='area',
        size=df["HP"][df["Type 1"] == "Fire"][df["Generation"] == 4],
        sizeref=sizeref,
        line=dict(
            width=2
        ),
    )
)
trace4 = go.Scatter(
    x=df["Attack"][df["Type 1"] == "Fire"][df["Generation"] == 5],
    y=df["Defense"][df["Type 1"] == "Fire"][df["Generation"] == 5],
    mode='markers',
    name='Generation 5',
    text=df["Name"][df["Type 1"] == "Fire"][df["Generation"] == 5],
    marker=dict(
        sizemode='area',
        size=df["HP"][df["Type 1"] == "Fire"][df["Generation"] == 5],
        sizeref=sizeref,
        line=dict(
            width=2
        ),
    )
)
trace5 = go.Scatter(
    x=df["Attack"][df["Type 1"] == "Fire"][df["Generation"] == 6],
    y=df["Defense"][df["Type 1"] == "Fire"][df["Generation"] == 6],
    mode='markers',
    name='Generation 6',
    text=df["Name"][df["Type 1"] == "Fire"][df["Generation"] == 6],
    marker=dict(
        sizemode='area',
        size=df["HP"][df["Type 1"] == "Fire"][df["Generation"] == 6],
        sizeref=sizeref,
        line=dict(
            width=2
        ),
    )
)

data = [trace0, trace1, trace2, trace3, trace4, trace5]
layout = go.Layout(
    title='Attack vs Defense of Fire Pokemon over generations',
    xaxis=dict(
        title='Attack',
        gridcolor='rgb(255, 255, 255)',
        range=[0,200]
    ),
    yaxis=dict(
        title='Defense',
        gridcolor='rgb(255, 255, 255)',
        range=[0,200]
    ),
    paper_bgcolor='rgb(243, 243, 243)',
    plot_bgcolor='rgb(243, 243, 243)',
)

fig = go.Figure(data=data, layout=layout)
iplot(fig, filename='bubble.png')
import squarify

x = 0.
y = 0.
width = 50.
height = 50.
type_list = list(df["Type 1"].unique())
values = [len(df[df["Type 1"] == i]) for i in type_list]

normed = squarify.normalize_sizes(values, width, height)
rects = squarify.squarify(normed, x, y, width, height)

# Choose colors from http://colorbrewer2.org/ under "Export"
color_brewer = ['#2D3142','#4F5D75','#BFC0C0','#F2D7EE','#EF8354','#839788','#EEE0CB','#BAA898','#BFD7EA','#685044','#E9AFA3','#99B2DD','#F9DEC9','#3A405A','#494949','#FF5D73','#7C7A7A','#CF5C36','#EFC88B']
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
            line = dict( width = 2 ),
            fillcolor = color_brewer[counter]
        ) 
    )
    annotations.append(
        dict(
            x = r['x']+(r['dx']/2),
            y = r['y']+(r['dy']/2),
            text = "{}-{}".format(type_list[counter], values[counter]),
            showarrow = False
        )
    )
    counter = counter + 1
    if counter >= len(color_brewer):
        counter = 0

# For hover text
trace0 = go.Scatter(
    x = [ r['x']+(r['dx']/2) for r in rects ], 
    y = [ r['y']+(r['dy']/2) for r in rects ],
    text = [ str(v) for v in values ], 
    mode = 'text',
)
        
layout = dict(
    height=700, 
    width=700,
    xaxis=dict(showgrid=False,zeroline=False),
    yaxis=dict(showgrid=False,zeroline=False),
    shapes=shapes,
    annotations=annotations,
    hovermode='closest',
    font=dict(color="#FFFFFF")
)

# With hovertext
figure = dict(data=[trace0], layout=layout)
iplot(figure, filename='squarify-treemap')
# Function for bullet chart
def checkpokemonperformance(x):
    x = df[df["Name"] == x]
    data = (
      {"label": "HP", "sublabel": x["HP"].values[0],
       "range": [int(max(df["HP"])*0.5), int(max(df["HP"])*0.75), int(max(df["HP"]))], "performance": [int(max(df["HP"])*0.55),int(max(df["HP"])*0.70)], "point": [x["HP"].values[0]]},
      {"label": "Attack", "sublabel": x["Attack"].values[0],
       "range": [int(max(df["Attack"])*0.5), int(max(df["Attack"])*0.75), int(max(df["Attack"]))], "performance": [int(max(df["Attack"])*0.55),int(max(df["Attack"])*0.70)], "point": [x["Attack"].values[0]]},
      {"label": "Defense", "sublabel": x["Defense"].values[0],
       "range": [int(max(df["Defense"])*0.5), int(max(df["Defense"])*0.75), int(max(df["Defense"]))], "performance": [int(max(df["Defense"])*0.55),int(max(df["Defense"])*0.70)], "point": [x["Defense"].values[0]]},
      {"label": "Sp. Atk", "sublabel": x["Sp. Atk"].values[0],
       "range": [int(max(df["Sp. Atk"])*0.5), int(max(df["Sp. Atk"])*0.75), int(max(df["Sp. Atk"]))], "performance": [int(max(df["Sp. Atk"])*0.55),int(max(df["Sp. Atk"])*0.70)], "point": [x["Sp. Atk"].values[0]]},
      {"label": "Sp. Def", "sublabel": x["Sp. Def"].values[0],
       "range": [int(max(df["Sp. Def"])*0.5), int(max(df["Sp. Def"])*0.75), int(max(df["Sp. Def"]))], "performance": [int(max(df["Sp. Def"])*0.55),int(max(df["Sp. Def"])*0.70)], "point": [x["Sp. Def"].values[0]]},
      {"label": "Speed", "sublabel": x["Speed"].values[0],
       "range": [int(max(df["Speed"])*0.5), int(max(df["Speed"])*0.75), int(max(df["Speed"]))], "performance": [int(max(df["Speed"])*0.55),int(max(df["Speed"])*0.70)], "point": [x["Speed"].values[0]]}
    )
    
    fig = ff.create_bullet(
        data, titles='label', subtitles='sublabel', markers='point',
        measures='performance', ranges='range', orientation='v', width="800", height="800"
    )
    iplot(fig, filename='bullet chart')
checkpokemonperformance("VenusaurMega Venusaur")
checkpokemonperformance("Regigigas")
fig = ff.create_scatterplotmatrix(df.iloc[:,5:12], index='Generation', diag='box', size=2, height=800, width=800)
iplot(fig, filename ='Scatterplotmatrix.png',image='png')
data = []
for i in range(5,11):
    trace = {
            "type": 'violin',
            "x": max(df.iloc[:,i]),
            "y": df.iloc[:,i],
            "name": list(df.columns)[i],
            "box": {
                "visible": True
            },
            "meanline": {
                "visible": True
            }
        }
    data.append(trace)
        
fig = {
    "data": data,
    "layout" : {
        "title": "Violin plot of all stats",
        "yaxis": {
            "zeroline": False,
        }
    }
}

iplot(fig, filename='violin', validate = False)