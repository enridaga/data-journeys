
import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

### matplotlib inline

import plotly.offline as py

py.init_notebook_mode(connected=True)

import plotly.graph_objs as go

import plotly.tools as tls

import seaborn as sns

import time

import warnings

warnings.filterwarnings('ignore')



global_temp_country = pd.read_csv('../input/GlobalLandTemperaturesByCountry.csv')
#Let's remove the duplicated countries (in the analysis, we don't consider the presence of 

#colonies at this the countries) and countries for which no information about the temperature



global_temp_country_clear = global_temp_country[~global_temp_country['Country'].isin(

    ['Denmark', 'Antarctica', 'France', 'Europe', 'Netherlands',

     'United Kingdom', 'Africa', 'South America'])]



global_temp_country_clear = global_temp_country_clear.replace(

   ['Denmark (Europe)', 'France (Europe)', 'Netherlands (Europe)', 'United Kingdom (Europe)'],

   ['Denmark', 'France', 'Netherlands', 'United Kingdom'])



#Let's average temperature for each country



countries = np.unique(global_temp_country_clear['Country'])

mean_temp = []

for country in countries:

    mean_temp.append(global_temp_country_clear[global_temp_country_clear['Country'] == 

                                               country]['AverageTemperature'].mean())





    

data = [ dict(

        type = 'choropleth',

        locations = countries,

        z = mean_temp,

        locationmode = 'country names',

        text = countries,

        marker = dict(

            line = dict(color = 'rgb(0,0,0)', width = 1)),

            colorbar = dict(autotick = True, tickprefix = '', 

            title = '# Average\nTemperature,\n°C')

            )

       ]



layout = dict(

    title = 'Average land temperature in countries',

    geo = dict(

        showframe = False,

        showocean = True,

        oceancolor = 'rgb(0,255,255)',

        projection = dict(

        type = 'orthographic',

            rotation = dict(

                    lon = 60,

                    lat = 10),

        ),

        lonaxis =  dict(

                showgrid = True,

                gridcolor = 'rgb(102, 102, 102)'

            ),

        lataxis = dict(

                showgrid = True,

                gridcolor = 'rgb(102, 102, 102)'

                )

            ),

        )



fig = dict(data=data, layout=layout)

py.iplot(fig, validate=False, filename='worldmap')
mean_temp_bar, countries_bar = (list(x) for x in zip(*sorted(zip(mean_temp, countries), 

                                                             reverse = True)))

sns.set(font_scale=0.9) 

f, ax = plt.subplots(figsize=(4.5, 50))

colors_cw = sns.color_palette('coolwarm', len(countries))

sns.barplot(mean_temp_bar, countries_bar, palette = colors_cw[::-1])

Text = ax.set(xlabel='Average temperature', title='Average land temperature in countries')
global_temp = pd.read_csv("../input/GlobalTemperatures.csv")



#Extract the year from a date

years = np.unique(global_temp['dt'].apply(lambda x: x[:4]))

mean_temp_world = []

mean_temp_world_uncertainty = []



for year in years:

    mean_temp_world.append(global_temp[global_temp['dt'].apply(

        lambda x: x[:4]) == year]['LandAverageTemperature'].mean())

    mean_temp_world_uncertainty.append(global_temp[global_temp['dt'].apply(

                lambda x: x[:4]) == year]['LandAverageTemperatureUncertainty'].mean())



trace0 = go.Scatter(

    x = years, 

    y = np.array(mean_temp_world) + np.array(mean_temp_world_uncertainty),

    fill= None,

    mode='lines',

    name='Uncertainty top',

    line=dict(

        color='rgb(0, 255, 255)',

    )

)

trace1 = go.Scatter(

    x = years, 

    y = np.array(mean_temp_world) - np.array(mean_temp_world_uncertainty),

    fill='tonexty',

    mode='lines',

    name='Uncertainty bot',

    line=dict(

        color='rgb(0, 255, 255)',

    )

)



trace2 = go.Scatter(

    x = years, 

    y = mean_temp_world,

    name='Average Temperature',

    line=dict(

        color='rgb(199, 121, 093)',

    )

)

data = [trace0, trace1, trace2]



layout = go.Layout(

    xaxis=dict(title='year'),

    yaxis=dict(title='Average Temperature, °C'),

    title='Average land temperature in world',

    showlegend = False)



fig = go.Figure(data=data, layout=layout)

py.iplot(fig)
continent = ['Russia', 'United States', 'Niger', 'Greenland', 'Australia', 'Bolivia']

mean_temp_year_country = [ [0] * len(years[70:]) for i in range(len(continent))]

j = 0

for country in continent:

    all_temp_country = global_temp_country_clear[global_temp_country_clear['Country'] == country]

    i = 0

    for year in years[70:]:

        mean_temp_year_country[j][i] = all_temp_country[all_temp_country['dt'].apply(

                lambda x: x[:4]) == year]['AverageTemperature'].mean()

        i +=1

    j += 1



traces = []

colors = ['rgb(0, 255, 255)', 'rgb(255, 0, 255)', 'rgb(0, 0, 0)',

          'rgb(255, 0, 0)', 'rgb(0, 255, 0)', 'rgb(0, 0, 255)']

for i in range(len(continent)):

    traces.append(go.Scatter(

        x=years[70:],

        y=mean_temp_year_country[i],

        mode='lines',

        name=continent[i],

        line=dict(color=colors[i]),

    ))



layout = go.Layout(

    xaxis=dict(title='year'),

    yaxis=dict(title='Average Temperature, °C'),

    title='Average land temperature on the continents',)



fig = go.Figure(data=traces, layout=layout)

py.iplot(fig)
#Extract the year from a date

years = np.unique(global_temp_country_clear['dt'].apply(lambda x: x[:4]))



#Let's create an array and add the values of average temperatures in the countries every 10 years

mean_temp_year_country = [ [0] * len(countries) for i in range(len(years[::10]))]



j = 0

for country in countries:

    all_temp_country = global_temp_country_clear[global_temp_country_clear['Country'] == country]

    i = 0

    for year in years[::10]:

        mean_temp_year_country[i][j] = all_temp_country[all_temp_country['dt'].apply(

                lambda x: x[:4]) == year]['AverageTemperature'].mean()

        i +=1

    j += 1
#Let's create a Streaming in Plotly (here, alas, does not work, so commented out)

#stream_tokens = tls.get_credentials_file()['stream_ids']

#token =  stream_tokens[-1]

#stream_id = dict(token=token, maxpoints=60)



data = [ dict(

        type = 'choropleth',

        locations = countries,

        z = mean_temp,

        locationmode = 'country names',

        text = countries,

        marker = dict(

            line = dict(color = 'rgb(0,0,0)', width = 1)),

            colorbar = dict(autotick = True, tickprefix = '',

            title = '# Average\nTemperature,\n°C'),

        #The following line is also needed to create Stream

        #stream = stream_id

            )

       ]



layout = dict(

    title = 'Average land temperature in countries',

    geo = dict(

        showframe = False,

        showocean = True,

        oceancolor = 'rgb(0,255,255)',

        type = 'equirectangular'

    ),

)



fig = dict(data=data, layout=layout)

py.iplot(fig, validate=False, filename='world_temp_map')
#Let's run a Stream



"""s = py.Stream(stream_id=token)

i = 0

s.open()

while True:

    ye = years[::10]

    s.write(dict(z = mean_temp_year_country[i]), dict(

            title = 'Average land temperature in countries.   Year: {0}'.format(ye[i])), validate=False)

    time.sleep(1)

    i += 1

    if i == len(ye):

        i = 0

s.close()"""