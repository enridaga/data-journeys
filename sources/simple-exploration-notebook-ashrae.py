
import numpy as np
import pandas as pd
import os
import seaborn as sns

import matplotlib.pyplot as plt
import matplotlib.patches as patches

from plotly import tools, subplots
import plotly.offline as py
py.init_notebook_mode(connected=True)
import plotly.graph_objs as go
import plotly.express as px
pd.set_option('max_columns', 100)

data_path = "/kaggle/input/ashrae-energy-prediction/"

train_df = pd.read_csv(data_path + "train.csv")
train_df["timestamp"] = pd.to_datetime(train_df["timestamp"], format='%Y-%m-%d %H:%M:%S')
train_df.head()
from IPython.core.display import display, HTML

nrows = train_df.shape[0]
nbuildings = train_df["building_id"].nunique()
max_rows_building = train_df["building_id"].value_counts().values[0]
min_rows_building = train_df["building_id"].value_counts().values[-1]
min_date = train_df["timestamp"].min()
max_date = train_df["timestamp"].max()
display(HTML(f"""<br>Number of rows in the dataset: {nrows:,}</br>
             <br>Number of buildings in the dataset: {nbuildings:,}</br>
             <br>Maximum of {max_rows_building:,} rows is available for a building</br>
             <br>Minimum of {min_rows_building:,} rows is available for a building</br>
             <br>Min date value in train data is {min_date}</br>
             <br>Max date value in train data is {max_date}</br>
             """))
cnt_srs = train_df["meter"].value_counts()
cnt_srs = cnt_srs.sort_index()
cnt_srs.index = ["Electricity", "ChilledWater", "Steam", "HotWater"]
trace = go.Bar(
    x=cnt_srs.index,
    y=cnt_srs.values,
    marker=dict(
        color="#1E90FF",
    ),
)

layout = go.Layout(
    title=go.layout.Title(
        text="Number of rows for each meter type",
        x=0.5
    ),
    font=dict(size=14),
    width=800,
    height=500,
)

data = [trace]
fig = go.Figure(data=data, layout=layout)
py.iplot(fig, filename="meter")
from bokeh.layouts import gridplot
from bokeh.plotting import figure, show, output_file
from bokeh.io import output_notebook

output_notebook()
def make_plot(title, hist, edges, xlabel):
    p = figure(title=title, tools='', background_fill_color="#fafafa")
    p.quad(top=hist, bottom=0, left=edges[:-1], right=edges[1:],
           fill_color="#1E90FF", line_color="white", alpha=0.5)

    p.y_range.start = 0
    p.xaxis.axis_label = f'Log of {xlabel} meter reading'
    p.yaxis.axis_label = 'Probability'
    p.grid.grid_line_color="white"
    return p

temp_df = train_df[train_df["meter"]==0]
hist, edges = np.histogram(np.log1p(temp_df["meter_reading"].values), density=True, bins=50)
p1 = make_plot("Meter Reading Distribution for Electricity meter", hist, edges, "electricity")

temp_df = train_df[train_df["meter"]==1]
hist, edges = np.histogram(np.log1p(temp_df["meter_reading"].values), density=True, bins=50)
p2 = make_plot("Meter Reading Distribution for Chilled Water meter", hist, edges, 'chill water')

temp_df = train_df[train_df["meter"]==2]
hist, edges = np.histogram(np.log1p(temp_df["meter_reading"].values), density=True, bins=50)
p3 = make_plot("Meter Reading Distribution for Steam meter", hist, edges, 'steam')

temp_df = train_df[train_df["meter"]==3]
hist, edges = np.histogram(np.log1p(temp_df["meter_reading"].values), density=True, bins=50)
p4 = make_plot("Meter Reading Distribution for Hot Water meter", hist, edges, 'hot water')

show(gridplot([p1,p2,p3,p4], ncols=2, plot_width=400, plot_height=400, toolbar_location=None))
temp_df = train_df[train_df["building_id"]==1298].reset_index(drop=True)

import datetime

def scatter_plot(cnt_srs, color):
    trace = go.Scatter(
        x=cnt_srs.index[::-1],
        y=cnt_srs.values[::-1],
        showlegend=False,
        marker=dict(
            color=color,
        ),
    )
    return trace

tdf = temp_df[temp_df["meter"]==0]
cnt_srs = tdf["meter_reading"]
cnt_srs.index = tdf["timestamp"]
trace1 = scatter_plot(cnt_srs, 'red')

tdf = temp_df[temp_df["meter"]==1]
cnt_srs = tdf["meter_reading"]
cnt_srs.index = tdf["timestamp"]
trace2 = scatter_plot(cnt_srs, 'blue')

tdf = temp_df[temp_df["meter"]==2]
cnt_srs = tdf["meter_reading"]
cnt_srs.index = tdf["timestamp"]
trace3 = scatter_plot(cnt_srs, 'green')

tdf = temp_df[temp_df["meter"]==3]
cnt_srs = tdf["meter_reading"]
cnt_srs.index = tdf["timestamp"]
trace4 = scatter_plot(cnt_srs, 'purple')

subtitles = ["Meter reading over time for electricity meter for building 1298",
             "Meter reading over time for chill water meter for building 1298",
             "Meter reading over time for steam meter for building 1298",
             "Meter reading over time for hot water meter for building 1298"
            ]
fig = subplots.make_subplots(rows=4, cols=1, vertical_spacing=0.08,
                          subplot_titles=subtitles)
fig.append_trace(trace1, 1, 1)
fig.append_trace(trace2, 2, 1)
fig.append_trace(trace3, 3, 1)
fig.append_trace(trace4, 4, 1)
fig['layout'].update(height=1200, width=1000, paper_bgcolor='rgb(233,233,233)')
py.iplot(fig, filename='meter-plots')
temp_df = train_df[train_df["building_id"]==184].reset_index(drop=True)

tdf = temp_df[temp_df["meter"]==0]
cnt_srs = tdf["meter_reading"]
cnt_srs.index = tdf["timestamp"]
trace1 = scatter_plot(cnt_srs, 'red')

tdf = temp_df[temp_df["meter"]==1]
cnt_srs = tdf["meter_reading"]
cnt_srs.index = tdf["timestamp"]
trace2 = scatter_plot(cnt_srs, 'blue')

tdf = temp_df[temp_df["meter"]==2]
cnt_srs = tdf["meter_reading"]
cnt_srs.index = tdf["timestamp"]
trace3 = scatter_plot(cnt_srs, 'green')

tdf = temp_df[temp_df["meter"]==3]
cnt_srs = tdf["meter_reading"]
cnt_srs.index = tdf["timestamp"]
trace4 = scatter_plot(cnt_srs, 'purple')

subtitles = ["Meter reading over time for electricity meter for building 184",
             "Meter reading over time for chill water meter for building 184",
             "Meter reading over time for steam meter for building 184",
             "Meter reading over time for hot water meter for building 184"
            ]
fig = subplots.make_subplots(rows=4, cols=1, vertical_spacing=0.08,
                          subplot_titles=subtitles)
fig.append_trace(trace1, 1, 1)
fig.append_trace(trace2, 2, 1)
fig.append_trace(trace3, 3, 1)
fig.append_trace(trace4, 4, 1)
fig['layout'].update(height=1200, width=1000, paper_bgcolor='rgb(233,233,233)')
py.iplot(fig, filename='meter-plots')
building_df = pd.read_csv(data_path + "building_metadata.csv")
building_df.head()
cnt_srs = building_df["primary_use"].value_counts()
#cnt_srs = cnt_srs.sort_index()
trace = go.Bar(
    x=cnt_srs.index,
    y=cnt_srs.values,
    marker=dict(
        color="#1E90FF",
    ),
)

layout = go.Layout(
    title=go.layout.Title(
        text="Primary use of Buildings - Count",
        x=0.5
    ),
    font=dict(size=14),
    width=1000,
    height=500,
)

data = [trace]
fig = go.Figure(data=data, layout=layout)
py.iplot(fig, filename="meter")
cnt_srs = building_df["floor_count"].value_counts()
#cnt_srs = cnt_srs.sort_index()
trace = go.Bar(
    x=cnt_srs.index,
    y=cnt_srs.values,
    marker=dict(
        color="#1E90FF",
    ),
)

layout = go.Layout(
    title=go.layout.Title(
        text="Floor count in the buildings - Count",
        x=0.5
    ),
    font=dict(size=14),
    width=1000,
    height=500,
)

data = [trace]
fig = go.Figure(data=data, layout=layout)
py.iplot(fig, filename="meter")
cnt_srs = building_df["year_built"].value_counts()
#cnt_srs = cnt_srs.sort_index()
trace = go.Bar(
    x=cnt_srs.index,
    y=cnt_srs.values,
    marker=dict(
        color="#1E90FF",
    ),
)

layout = go.Layout(
    title=go.layout.Title(
        text="Year built of the buildings - Count",
        x=0.5
    ),
    font=dict(size=14),
    width=1000,
    height=500,
)

data = [trace]
fig = go.Figure(data=data, layout=layout)
py.iplot(fig, filename="meter")
fig = px.scatter(building_df, y="square_feet", x="year_built", color="primary_use", size="square_feet")
fig.update_layout(showlegend=True)
fig.show()
weather_df = pd.read_csv(data_path + "weather_train.csv", parse_dates=["timestamp"])
weather_df.head()
fig = px.line(weather_df, x='timestamp', y='air_temperature', color='site_id')
fig.show()
from bokeh.models import Panel, Tabs
from bokeh.io import output_notebook, show
from bokeh.plotting import figure

output_notebook()

def get_plots(df):
    p = figure(plot_width=1000, plot_height=350, x_axis_type="datetime", title="Air temperature distribution over time")
    p.line(df['timestamp'], df['air_temperature'], color='navy', alpha=0.5)
    return p

tab_list = []
for site in range(16):
    temp_df = weather_df[weather_df["site_id"]==site]
    p = get_plots(temp_df)
    tab = Panel(child=p, title=f"Site:{site}")
    tab_list.append(tab)

tabs = Tabs(tabs=tab_list)
show(tabs)
from bokeh.models import Panel, Tabs
from bokeh.io import output_notebook, show
from bokeh.plotting import figure

output_notebook()

def get_plots(df, col, color):
    p = figure(plot_width=1000, plot_height=350, x_axis_type="datetime", title=f"{col} distribution over time")
    p.line(df['timestamp'], df[col], color=color, alpha=0.5)
    return p

col_map = {
    "dew_temperature": "Dew Temperature",
    "sea_level_pressure": "Sea Level Pressure",
    "wind_speed": "Wind Speed",
    "cloud_coverage": "Cloud Coverage",
}

color_map = {
    "dew_temperature": "brown",
    "sea_level_pressure": "green",
    "wind_speed": "red",
    "cloud_coverage": "blue",
}

main_tabs_list = []
cols = ["dew_temperature", "sea_level_pressure", "wind_speed", "cloud_coverage"]
for col in cols:
    tab_list = []
    for site in range(16):
        temp_df = weather_df[weather_df["site_id"]==site]
        p = get_plots(temp_df, col, color_map[col])
        tab = Panel(child=p, title=f"Site:{site}")
        tab_list.append(tab)
    tabs = Tabs(tabs=tab_list)
    panel = Panel(child=tabs, title=col_map[col])
    main_tabs_list.append(panel)

tabs = Tabs(tabs=main_tabs_list)
show(tabs)