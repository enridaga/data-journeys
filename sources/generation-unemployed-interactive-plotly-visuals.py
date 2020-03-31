
# Import the relevant libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
### matplotlib inline
import plotly.offline as py
py.init_notebook_mode(connected=True)
import plotly.graph_objs as go
import plotly.tools as tls
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')
country = pd.read_csv('../input/API_ILO_country_YU.csv')
country.head()
print(country.shape)
# Uncomment to view the total list of country names
display(country['Country Name'].unique())
# Create our new list of countries that we want to plot for. This was done manually as I was lazy
# to think of any clever tricks (eg text processing) to filter. 
country_list = ['Afghanistan','Angola','Albania','Argentina','Armenia','Australia'
,'Austria','Azerbaijan','Burundi','Belgium','Benin','Burkina Faso','Bangladesh','Bulgaria'
,'Bahrain','Bosnia and Herzegovina','Belarus','Belize','Bolivia','Brazil','Barbados','Brunei Darussalam'
,'Bhutan','Botswana','Central African Republic','Canada','Switzerland','Chile','China','Cameroon'
,'Congo','Colombia','Comoros','Cabo Verde','Costa Rica','Cuba','Cyprus','Czech Republic','Germany'
,'Denmark','Dominican Republic','Algeria','Ecuador','Egypt','Spain','Estonia','Ethiopia','Finland','Fiji'
,'France','Gabon','United Kingdom','Georgia','Ghana','Guinea','Greece','Guatemala','Guyana','Hong Kong'
,'Honduras','Croatia','Haiti','Hungary','Indonesia','India','Ireland','Iran','Iraq','Iceland','Israel'
,'Italy','Jamaica','Jordan','Japan','Kazakhstan','Kenya','Cambodia','Korea, Rep.','Kuwait','Lebanon','Liberia'
,'Libya','Sri Lanka','Lesotho','Lithuania','Luxembourg','Latvia','Macao','Morocco','Moldova','Madagascar'
,'Maldives','Mexico','Macedonia','Mali','Malta','Myanmar','Montenegro','Mongolia','Mozambique','Mauritania'
,'Mauritius','Malawi','Malaysia','North America','Namibia','Niger','Nigeria','Nicaragua','Netherlands'
,'Norway','Nepal','New Zealand   ','Oman','Pakistan','Panama','Peru','Philippines','Papua New Guinea'
,'Poland','Puerto Rico','Portugal','Paraguay','Qatar','Romania','Russian Federation','Rwanda','Saudi Arabia'
,'Sudan','Senegal','Singapore','Solomon Islands','Sierra Leone','El Salvador','Somalia','Serbia','Slovenia'
,'Sweden','Swaziland','Syrian Arab Republic','Chad','Togo','Thailand','Tajikistan','Turkmenistan','Timor-Leste'
,'Trinidad and Tobago','Tunisia','Turkey','Tanzania','Uganda','Ukraine','Uruguay','United States','Uzbekistan'
,'Venezuela, RB','Vietnam','Yemen, Rep.','South Africa','Congo, Dem. Rep.','Zambia','Zimbabwe'
]
# Create a new dataframe with our cleaned country list
country_clean = country[country['Country Name'].isin(country_list)]
# Plotting 2010 and 2014 visuals
metricscale1=[[0, 'rgb(102,194,165)'], [0.05, 'rgb(102,194,165)'], 
              [0.15, 'rgb(171,221,164)'], [0.2, 'rgb(230,245,152)'], 
              [0.25, 'rgb(255,255,191)'], [0.35, 'rgb(254,224,139)'], 
              [0.45, 'rgb(253,174,97)'], [0.55, 'rgb(213,62,79)'], [1.0, 'rgb(158,1,66)']]
data = [ dict(
        type = 'choropleth',
        autocolorscale = False,
        colorscale = metricscale1,
        showscale = True,
        locations = country_clean['Country Name'].values,
        z = country_clean['2010'].values,
        locationmode = 'country names',
        text = country_clean['Country Name'].values,
        marker = dict(
            line = dict(color = 'rgb(250,250,225)', width = 0.5)),
            colorbar = dict(autotick = True, tickprefix = '', 
            title = 'Unemployment\nRate')
            )
       ]

layout = dict(
    title = 'World Map of Global Youth Unemployment in the Year 2010',
    geo = dict(
        showframe = True,
        showocean = True,
        oceancolor = 'rgb(28,107,160)',
        #oceancolor = 'rgb(222,243,246)',
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
py.iplot(fig, validate=False, filename='worldmap2010')

metricscale2=[[0, 'rgb(102,194,165)'], [0.05, 'rgb(102,194,165)'], 
              [0.15, 'rgb(171,221,164)'], [0.2, 'rgb(230,245,152)'], 
              [0.25, 'rgb(255,255,191)'], [0.35, 'rgb(254,224,139)'], 
              [0.45, 'rgb(253,174,97)'], [0.55, 'rgb(213,62,79)'], [1.0, 'rgb(158,1,66)']]
data = [ dict(
        type = 'choropleth',
        autocolorscale = False,
        colorscale = metricscale2,
        showscale = True,
        locations = country_clean['Country Name'].values,
        z = country_clean['2014'].values,
        locationmode = 'country names',
        text = country_clean['Country Name'].values,
        marker = dict(
            line = dict(color = 'rgb(250,250,200)', width = 0.5)),
            colorbar = dict(autotick = True, tickprefix = '', 
            title = 'Unemployment\nRate')
            )
       ]

layout = dict(
    title = 'World Map of Global Youth Unemployment in the Year 2014',
    geo = dict(
        showframe = True,
        showocean = True,
        oceancolor = 'rgb(28,107,160)',
        projection = dict(
        type = 'orthographic',
            rotation = dict(
                    lon = 60,
                    lat = 10),
        ),
        lonaxis =  dict(
                showgrid = False,
                gridcolor = 'rgb(202, 202, 202)',
                width = '0.05'
            ),
        lataxis = dict(
                showgrid = False,
                gridcolor = 'rgb(102, 102, 102)'
                )
            ),
        )
fig = dict(data=data, layout=layout)
py.iplot(fig, validate=False, filename='worldmap2014')
# Scatter plot of 2010 unemployment rates
trace = go.Scatter(
    y = country_clean['2010'].values,
    mode='markers',
    marker=dict(
        size= country_clean['2010'].values,
        #color = np.random.randn(500), #set color equal to a variable
        color = country_clean['2010'].values,
        colorscale='Portland',
        showscale=True
    ),
    text = country_clean['Country Name'].values
)
data = [trace]

layout= go.Layout(
    autosize= True,
    title= 'Scatter plot of unemployment rates in 2010',
    hovermode= 'closest',
#     xaxis= dict(
#         title= 'Pop',
#         ticklen= 5,
#         zeroline= False,
#         gridwidth= 2,
#     ),
    yaxis=dict(
        title= 'Unemployment Rate',
        ticklen= 5,
        gridwidth= 2,
    ),
    showlegend= False
)
fig = go.Figure(data=data, layout=layout)
py.iplot(fig,filename='scatter2010')

# Scatter plot of 2014 unemployment rates
trace1 = go.Scatter(
    y = country_clean['2014'].values,
    mode='markers',
    marker=dict(
        size=country_clean['2014'].values,
        #color = np.random.randn(500), #set color equal to a variable
        color = country_clean['2014'].values,
        colorscale='Portland',
        showscale=True
    ),
    text = country_clean['Country Name']
)
data = [trace1]

layout= go.Layout(
    title= 'Scatter plot of unemployment rates in 2014',
    hovermode= 'closest',
    xaxis= dict(
        ticklen= 5,
        zeroline= False,
        gridwidth= 2,
    ),
    yaxis=dict(
        title= 'Unemployment Rate',
        ticklen= 5,
        gridwidth= 2,
    ),
    showlegend= False
)

fig = go.Figure(data=data, layout=layout)
py.iplot(fig,filename='scatter2014')
# I first create an array containing the net movement in the unemployment rate.
diff = country_clean['2014'].values - country_clean['2010'].values
x, y = (list(x) for x in zip(*sorted(zip(diff, country_clean['Country Name'].values), 
                                                            reverse = True)))

# Now I want to extract out the top 15 and bottom 15 countries 
Y = np.concatenate([y[0:15], y[-16:-1]])
X = np.concatenate([x[0:15], x[-16:-1]])
# Resize our dataframe first
keys = [c for c in country_clean if c.startswith('20')]
country_resize = pd.melt(country_clean, id_vars='Country Name', value_vars=keys, value_name='key')
country_resize['Year'] = country_resize['variable']

# Use boolean filtering to extract only our top 15 and bottom 15 moving countries
mask = country_resize['Country Name'].isin(Y)
country_final = country_resize[mask]

# Finally plot the seaborn heatmap
plt.figure(figsize=(12,10))
country_pivot = country_final.pivot("Country Name","Year",  "key")
country_pivot = country_pivot.sort_values('2014', ascending=False)
ax = sns.heatmap(country_pivot, cmap='coolwarm', annot=False, linewidths=0, linecolor='white')
plt.title('Movement in Unemployment rate ( Warmer: Higher rate, Cooler: Lower rate )')

# Plot using Seaborn's barplot
sns.set(font_scale=1) 
f, ax = plt.subplots(figsize=(20, 25))
colors_cw = sns.color_palette('coolwarm', len(X))
sns.barplot(X, Y, palette = colors_cw[::-1])
Text = ax.set(xlabel='Decrease in Youth Unemployment Rates', 
              title='Net Increase in Youth Unemployment Rates')

# Plot using Seaborn's barplot
sns.set(font_scale=1) 
f, ax = plt.subplots(figsize=(20, 68))
colors_cw = sns.color_palette('coolwarm', len(x))
sns.barplot(x, y, palette = colors_cw[::-1])
Text = ax.set(xlabel='Decrease in Youth Unemployment Rates', 
              title='Net Increase in Youth Unemployment Rates')