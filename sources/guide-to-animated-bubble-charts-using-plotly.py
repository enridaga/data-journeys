
import pandas as pd
import numpy as np
from bubbly.bubbly import bubbleplot 
from __future__ import division
from plotly.offline import init_notebook_mode, iplot
init_notebook_mode()

gapminder_indicators = pd.read_csv('../input/gapminder/gapminder.tsv', delimiter='\t')
honey_production = pd.read_csv('../input/honey-production/honeyproduction.csv')
world_happiness = pd.read_csv('../input/world-happiness/2017.csv')
from bubbly.bubbly import bubbleplot 

figure = bubbleplot(dataset=gapminder_indicators, x_column='gdpPercap', y_column='lifeExp', 
    bubble_column='country', time_column='year', size_column='pop', color_column='continent', 
    x_title="GDP per Capita", y_title="Life Expectancy", title='Gapminder Global Indicators',
    x_logscale=True, scale_bubble=3, height=650)

iplot(figure, config={'scrollzoom': True})
figure = bubbleplot(dataset=honey_production, x_column='totalprod', y_column='prodvalue',
    bubble_column='state', time_column='year', size_column='yieldpercol', color_column='numcol',
    x_title="Total production", y_title="Production value", 
    title='Timeline of honey production for each state in USA', 
    colorbar_title='# colonies', colorscale='Viridis', 
    x_logscale=True, y_logscale=True, scale_bubble=1.8, height=650) 

iplot(figure, config={'scrollzoom': True})
figure = bubbleplot(dataset=honey_production, x_column='prodvalue', y_column='priceperlb',
    bubble_column='state', time_column='year', size_column='yieldpercol', color_column='numcol',
    x_title="Production value", y_title="Price per pound", 
    title='Timeline of honey production and its prices for each state in USA', 
    colorbar_title='Yield per colony (lbs)', colorscale='Blackbody', y_range=[-0.5,0.9],
    x_logscale=True, y_logscale=True, scale_bubble=1.8, height=650) 

iplot(figure, config={'scrollzoom': True})
figure = bubbleplot(dataset=honey_production, x_column='totalprod', y_column='stocks', 
    bubble_column='state', time_column='year', size_column='numcol', color_column='yieldpercol', 
    x_title="Total production", y_title="Stocks", 
    title='Timeline of total honey production and its stock value for each state in USA', 
    colorbar_title='Yield per colony (lbs)', scale_bubble=1.8, colorscale='Earth',
    x_logscale=True, y_logscale=True, height=650) 

iplot(figure, config={'scrollzoom': True})
gapminder_indicators_continents = gapminder_indicators.groupby(['continent', 'year']).mean().reset_index()

figure = bubbleplot(dataset=gapminder_indicators_continents, 
    x_column='year', y_column='gdpPercap', bubble_column='continent',  
    size_column='pop', color_column='continent', 
    x_title="Years", y_title="GDP per capita", 
    title='GDP per capita inequality among geographical regions is increasing over time',
    x_range=[1945, 2015], y_range=[0, 35000],
    scale_bubble=1.5, height=650)

iplot(figure, config={'scrollzoom': True})
dataset = gapminder_indicators[gapminder_indicators['country'].isin(['Hungary', 'Vietnam'])]
figure = bubbleplot(dataset=dataset, x_column='gdpPercap', y_column='lifeExp', 
    bubble_column='country', color_column='year', size_column='pop',
    x_title="GDP per Capita", y_title="Life Expectancy", 
    title='GDP per capita and Life Expectancy of Hungary and Vietnam over time',
    colorbar_title = 'Time in years',
    x_logscale=True, scale_bubble=2, height=650)

iplot(figure, config={'scrollzoom': True})
figure = bubbleplot(dataset=world_happiness, 
    x_column='Economy..GDP.per.Capita.', y_column='Health..Life.Expectancy.', bubble_column='Country',  
    color_column='Happiness.Score', z_column='Trust..Government.Corruption.', size_column='Generosity',
    x_title="GDP per capita", y_title="Life Expectancy", z_title="Corruption",
    title='Impact of Economy, Health and Govt. on Happiness Scores of Nations',
    colorbar_title='Happiness Score', marker_opacity=1, colorscale='Portland',
    scale_bubble=0.8, height=650)

iplot(figure, config={'scrollzoom': True})
figure = bubbleplot(dataset=gapminder_indicators, x_column='pop', y_column='lifeExp', 
    bubble_column='country', time_column='year', z_column='gdpPercap', color_column='continent', 
    x_title="Population", y_title="Life Expectancy", z_title="GDP per Capita",
    title='Gapminder Global Indicators', x_logscale=True, z_logscale=True, 
    scale_bubble=0.8, marker_opacity=0.8, height=700)

iplot(figure, config={'scrollzoom': True})
# Define the dataset and the columns
dataset = gapminder_indicators
x_column = 'gdpPercap'
y_column = 'lifeExp'
bubble_column = 'country'
time_column = 'year'
# Get the years in the dataset
years = dataset[time_column].unique()

# Make the grid
grid = pd.DataFrame()
col_name_template = '{year}+{header}_grid'
for year in years:
    dataset_by_year = dataset[(dataset['year'] == int(year))]
    for col_name in [x_column, y_column, bubble_column]:
        # Each column name is unique
        temp = col_name_template.format(
            year=year, header=col_name
        )
        if dataset_by_year[col_name].size != 0:
            grid = grid.append({'value': list(dataset_by_year[col_name]), 'key': temp}, 
                               ignore_index=True)

grid.sample(10)
# Define figure
figure = {
    'data': [],
    'layout': {},
    'frames': []
}

# Get the earliest year
year = min(years)

# Make the trace
trace = {
    'x': grid.loc[grid['key']==col_name_template.format(
        year=year, header=x_column
    ), 'value'].values[0], 
    'y': grid.loc[grid['key']==col_name_template.format(
        year=year, header=y_column
    ), 'value'].values[0],
    'mode': 'markers',
    'text': grid.loc[grid['key']==col_name_template.format(
        year=year, header=bubble_column
    ), 'value'].values[0]
}
# Append the trace to the figure
figure['data'].append(trace)

# Plot the figure
iplot(figure, config={'scrollzoom': True})
# Get the max and min range of both axes
xmin = min(np.log10(dataset[x_column]))*0.98
xmax = max(np.log10(dataset[x_column]))*1.02
ymin = min(dataset[y_column])*0.75
ymax = max(dataset[y_column])*1.25

# Modify the layout
figure['layout']['xaxis'] = {'title': 'GDP per Capita', 'type': 'log', 
                             'range': [xmin, xmax]}   
figure['layout']['yaxis'] = {'title': 'Life Expectancy', 
                             'range': [ymin, ymax]} 
figure['layout']['title'] = 'Gapminder Global Indicators'
figure['layout']['showlegend'] = False
figure['layout']['hovermode'] = 'closest'
iplot(figure, config={'scrollzoom': True})
for year in years:
    # Make a frame for each year
    frame = {'data': [], 'name': str(year)}
    
    # Make a trace for each frame
    trace = {
        'x': grid.loc[grid['key']==col_name_template.format(
            year=year, header=x_column
        ), 'value'].values[0],
        'y': grid.loc[grid['key']==col_name_template.format(
            year=year, header=y_column
        ), 'value'].values[0],
        'mode': 'markers',
        'text': grid.loc[grid['key']==col_name_template.format(
            year=year, header=bubble_column
        ), 'value'].values[0],
        'type': 'scatter'
    }
    # Add trace to the frame
    frame['data'].append(trace)
    # Add frame to the figure
    figure['frames'].append(frame) 

iplot(figure, config={'scrollzoom': True})
figure['layout']['sliders'] = {
    'args': [
        'slider.value', {
            'duration': 400,
            'ease': 'cubic-in-out'
        }
    ],
    'initialValue': min(years),
    'plotlycommand': 'animate',
    'values': years,
    'visible': True
}
sliders_dict = {
    'active': 0,
    'yanchor': 'top',
    'xanchor': 'left',
    'currentvalue': {
        'font': {'size': 20},
        'prefix': 'Year:',
        'visible': True,
        'xanchor': 'right'
    },
    'transition': {'duration': 300, 'easing': 'cubic-in-out'},
    'pad': {'b': 10, 't': 50},
    'len': 0.9,
    'x': 0.1,
    'y': 0,
    'steps': []
}

from bubbly.bubbly import add_slider_steps

for year in years:
    add_slider_steps(sliders_dict, year)
    
figure['layout']['sliders'] = [sliders_dict]
iplot(figure, config={'scrollzoom': True})
figure['layout']['updatemenus'] = [
    {
        'buttons': [
            {
                'args': [None, {'frame': {'duration': 500, 'redraw': False},
                         'fromcurrent': True, 'transition': {'duration': 300, 
                                                             'easing': 'quadratic-in-out'}}],
                'label': 'Play',
                'method': 'animate'
            },
            {
                'args': [[None], {'frame': {'duration':0, 'redraw': False}, 'mode': 'immediate',
                'transition': {'duration': 0}}],
                'label': 'Pause',
                'method': 'animate'
            }
        ],
        'direction': 'left',
        'pad': {'r': 10, 't': 87},
        'showactive': False,
        'type': 'buttons',
        'x': 0.1,
        'xanchor': 'right',
        'y': 0,
        'yanchor': 'top'
    }
]
iplot(figure, config={'scrollzoom': True})
from bubbly.bubbly import make_grid, set_layout, get_trace

# Define the new variable
size_column = 'pop' 

# Make the grid
years = dataset[time_column].unique()
col_name_template = '{}+{}_grid'
column_names = [x_column, y_column, bubble_column, size_column]
grid = make_grid(dataset, column_names, time_column, years)
    
# Set the layout
figure, sliders_dict = set_layout(x_title='GDP per Capita', y_title='Life Expectancy', 
            title='Gapminder Global Indicators', x_logscale=True, y_logscale=False, 
            show_slider=True, slider_scale=years, show_button=True, show_legend=False, 
            height=650)

# Add the base frame
year = min(years)
col_name_template_year = col_name_template.format(year, {})
trace = get_trace(grid=grid, col_name_template=col_name_template_year, 
                    x_column=x_column, y_column=y_column, 
                    bubble_column=bubble_column, size_column=size_column)
figure['data'].append(trace)


# Add time frames
for year in years:
    frame = {'data': [], 'name': str(year)}
    col_name_template_year = col_name_template.format(year, {})
    trace = get_trace(grid=grid, col_name_template=col_name_template_year, 
                    x_column=x_column, y_column=y_column, 
                    bubble_column=bubble_column, size_column=size_column)
    frame['data'].append(trace)
    figure['frames'].append(frame) 
    add_slider_steps(sliders_dict, year)

# Set the layout once more
figure['layout']['xaxis']['autorange'] = True
figure['layout']['yaxis']['range'] = [20, 100]
figure['layout']['sliders'] = [sliders_dict]

# Plot the animation
iplot(figure, config={'scrollzoom': True})
from bubbly.bubbly import make_grid_with_categories

# Define the new variable
category_column = 'continent'

# Make the grid
years = dataset[time_column].unique()
continents = dataset[category_column].unique()
col_name_template = '{}+{}+{}_grid'
column_names = [x_column, y_column, bubble_column, size_column]
grid = make_grid_with_categories(dataset, column_names, time_column, 
                                 category_column, years, continents)
    
# Set the layout
figure, sliders_dict = set_layout(x_title='GDP per Capita', y_title='Life Expectancy', 
            title='Gapminder Global Indicators', x_logscale=True, y_logscale=False, 
            show_slider=True, slider_scale=years, show_button=True, show_legend=False, height=650)

# Add the base frame
year = min(years)
col_name_template_year = col_name_template.format(year, {}, {})
for continent in continents:
    trace = get_trace(grid=grid, col_name_template=col_name_template_year, 
                    x_column=x_column, y_column=y_column, 
                    bubble_column=bubble_column, size_column=size_column, category=continent)
    figure['data'].append(trace)

# Add time frames
for year in years:
    frame = {'data': [], 'name': str(year)}
    col_name_template_year = col_name_template.format(year, {}, {})
    for continent in continents:
        trace = get_trace(grid=grid, col_name_template=col_name_template_year, 
                    x_column=x_column, y_column=y_column, 
                    bubble_column=bubble_column, size_column=size_column, category=continent)
        frame['data'].append(trace)

    figure['frames'].append(frame) 
    add_slider_steps(sliders_dict, year)

# Set the layout once more
figure['layout']['xaxis']['autorange'] = True
figure['layout']['yaxis']['range'] = [20, 100]
figure['layout']['showlegend'] = True
figure['layout']['sliders'] = [sliders_dict]

# Plot the animation
iplot(figure, config={'scrollzoom': True})
