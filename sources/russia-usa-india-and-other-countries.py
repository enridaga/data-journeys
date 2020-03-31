
# libraries
import numpy as np 
import pandas as pd 
import os
import math
import matplotlib.pyplot as plt
import seaborn as sns
### matplotlib inline
from plotly import tools
import plotly.graph_objs as go
from plotly.offline import init_notebook_mode, iplot
import warnings
warnings.filterwarnings("ignore")
init_notebook_mode(connected=True)
pd.set_option('display.max_columns', None)
from ipywidgets import interact, interactive, interact_manual
import ipywidgets as widgets
import colorlover as cl

#loading data
DIR = '../input/kaggle-survey-2018/'
df_free = pd.read_csv(DIR + 'freeFormResponses.csv', low_memory=False, header=[0,1])
df_choice = pd.read_csv(DIR + 'multipleChoiceResponses.csv', low_memory=False, header=[0,1])
schema = pd.read_csv(DIR + 'SurveySchema.csv', low_memory=False, header=[0,1])
# Format Dataframes
df_free.columns = ['_'.join(col) for col in df_free.columns]
df_choice.columns = ['_'.join(col) for col in df_choice.columns]
schema.columns = ['_'.join(col) for col in schema.columns]

# For getting all columns
pd.set_option('display.max_columns', None)
country_count = df_choice['Q3_In which country do you currently reside?'].value_counts().reset_index()
country_count.columns = ['country', 'people']
# I use dataset from plotly to get country codes, which are required to plot the data.
country_code = pd.read_csv('../input/plotly-country-code-mapping/2014_world_gdp_with_codes.csv')
country_code.columns = [i.lower() for i in country_code.columns]
country_count.loc[country_count['country'] == 'United States of America', 'country'] = 'United States'
country_count.loc[country_count['country'] == 'United Kingdom of Great Britain and Northern Ireland', 'country'] = 'United Kingdom'
country_count.loc[country_count['country'] == 'South Korea', 'country'] = '"Korea, South"'
country_count.loc[country_count['country'] == 'Viet Nam', 'country'] = 'Vietnam'
country_count.loc[country_count['country'] == 'Iran, Islamic Republic of...', 'country'] = 'Iran'
country_count.loc[country_count['country'] == 'Hong Kong (S.A.R.)', 'country'] = 'Hong Kong'
country_count.loc[country_count['country'] == 'Republic of Korea', 'country'] = '"Korea, North"'
country_count = pd.merge(country_count, country_code, on='country')
data = [ dict(
        type = 'choropleth',
        locations = country_count['code'],
        z = country_count['people'],
        text = country_count['country'],
        colorscale = 'Viridis',
        autocolorscale = False,
        reversescale = True,
        marker = dict(
            line = dict (
                color = 'rgb(180,180,180)',
                width = 0.5
            ) ),
        colorbar = dict(
            autotick = False,
            title = 'Responders'),
      ) ]

layout = dict(
    title = 'Responders by country',
    geo = dict(
        showframe = False,
        showcoastlines = False,
        projection = dict(
            type = 'Mercator'
        )
    )
)

fig = dict( data=data, layout=layout )
iplot( fig, validate=False, filename='d3-world-map' )
df_choice['Q3_orig'] = df_choice['Q3_In which country do you currently reside?']
df_choice.loc[df_choice['Q3_In which country do you currently reside?'].isin(['United States of America', 'Russia', 'India']) == False,
              'Q3_In which country do you currently reside?'] = 'Other countries'
df_choice['Time from Start to Finish (seconds)_Duration (in minutes)'] = df_choice['Time from Start to Finish (seconds)_Duration (in seconds)'] / 60
plt.hist(df_choice['Time from Start to Finish (seconds)_Duration (in minutes)'] / 60, bins=40);
plt.yscale('log');
plt.title('Distribution of time spent on the survey');
plt.xlabel('Time (hours)');
data = []
for i in df_choice['Q3_In which country do you currently reside?'].unique():
    trace = {
            "type": 'violin',
            "x": df_choice.loc[(df_choice['Q3_In which country do you currently reside?'] == i) & (df_choice['Time from Start to Finish (seconds)_Duration (in minutes)'] < 180),
                               'Q3_In which country do you currently reside?'],
            "y": df_choice.loc[(df_choice['Q3_In which country do you currently reside?'] == i) & (df_choice['Time from Start to Finish (seconds)_Duration (in minutes)'] < 180),
                               'Time from Start to Finish (seconds)_Duration (in minutes)'],
            "name": i,
            "meanline": {
                "visible": True
            }
        }
    data.append(trace)

        
fig = {
    "data": data,
    "layout" : {
        "title": "",
        "yaxis": {
            "zeroline": False,
        }
    }
}

fig['layout'].update(title='Distribution of time spent on test by country');
iplot(fig)
data = []
for j, c in enumerate(df_choice['Q3_In which country do you currently reside?'].unique()):
    df_small = df_choice.loc[(df_choice['Q3_In which country do you currently reside?'] == c) & (df_choice['Time from Start to Finish (seconds)_Duration (in minutes)'] < 60),
                            'Time from Start to Finish (seconds)_Duration (in minutes)']
    trace = go.Histogram(
        x=df_small,
        name=c,
        marker=dict(color=j, opacity=0.5),
        showlegend=False
    )  
    data.append(trace)
fig = go.Figure(data=data)
fig['layout'].update(height=400, width=800, barmode='overlay', title='Distribution of time spent on test by country');
iplot(fig);
data = []
for i in df_choice['Q1_What is your gender? - Selected Choice'].unique():
    trace = go.Bar(
        x=df_choice.loc[df_choice['Q1_What is your gender? - Selected Choice'] == i, 'Q2_What is your age (# years)?'].value_counts().sort_index().index,
        y=df_choice.loc[df_choice['Q1_What is your gender? - Selected Choice'] == i, 'Q2_What is your age (# years)?'].value_counts().sort_index().values,
        name=i
    )
    data.append(trace)
layout = go.Layout(
    barmode='stack'
)
fig = go.Figure(data=data, layout=layout)
iplot(fig, filename='stacked-bar')
s = pd.crosstab(df_choice['Q1_What is your gender? - Selected Choice'],
                df_choice['Q2_What is your age (# years)?'], normalize='index').style.background_gradient(cmap='viridis', low=.5, high=0).highlight_null('red')
s
def plot_country_two_vars_dropdown(var1='', var2='', title_name=''):
    df = df_choice.copy()
    df[var2] = df[var2].astype('category')
    df[var1] = df[var1].astype('category')

    data = []
    buttons = []
    n_mult = df[var1].nunique()
    n = df['Q3_orig'].nunique() * n_mult
    for j, c in enumerate(df['Q3_orig'].unique()):
        visibility = [False] * n
        for ind, i in enumerate(df[var1].unique()):
            grouped = df.loc[(df[var1] == i) & (df['Q3_orig'] == c),
                                var2].value_counts().sort_index()
            trace = go.Bar(
                x=grouped.index,
                y=grouped.values,
                name=i,

                showlegend=True if j == 0 else False,
                legendgroup=i,
                visible=True if j == 0 else False
            )

            data.append(trace)
        visibility[j*n_mult:j*n_mult + n_mult] = [True] * n_mult
        buttons.append(dict(label = c,
                            method = 'update',
                            args = [{'visible': visibility},
                                    {'title': f'Responders in {c} by {title_name}'}]))
            
    updatemenus = list([dict(active=-1, buttons=buttons, x=1, y=2)])
    layout = dict(height=400, width=800, title=f"Responders in {df['Q3_orig'].unique()[0]} by {title_name}", updatemenus=updatemenus)
    fig = dict(data=data, layout=layout)
    return fig
def plot_country_two_vars(var1='', var2='', title_name=''):
    colors = cl.scales[str(df_choice[var1].fillna('').nunique())]['qual']['Paired']
    fig = tools.make_subplots(rows=4, cols=1, subplot_titles=('United States of America', 'Other countries', 'India', 'Russia'), print_grid=False)
    for j, c in enumerate(df_choice['Q3_In which country do you currently reside?'].unique()):
        data = []
        for ind, i in enumerate(df_choice[var1].unique()):
            grouped = df_choice.loc[(df_choice[var1] == i) & (df_choice['Q3_In which country do you currently reside?'] == c),
                                var2].value_counts().sort_index()
            trace = go.Bar(
                x=grouped.index,
                y=grouped.values,
                name=i,
                marker=dict(color=colors[ind]),
                showlegend=True if j == 0 else False,
                legendgroup=i
            )
            fig.append_trace(trace, j + 1, 1)    

    fig['layout'].update(height=1000, width=800, title=f'Responders in countries by {title_name}');
    return fig
fig = plot_country_two_vars(var1='Q1_What is your gender? - Selected Choice', var2='Q2_What is your age (# years)?', title_name='age and gender')
iplot(fig);
fig = plot_country_two_vars_dropdown(var1='Q1_What is your gender? - Selected Choice', var2='Q2_What is your age (# years)?', title_name='age and gender')
iplot(fig);
s = pd.crosstab(df_choice['Q2_What is your age (# years)?'],
                df_choice['Q4_What is the highest level of formal education that you have attained or plan to attain within the next 2 years?']).style.background_gradient(cmap='viridis', low=.5, high=0).highlight_null('red')
s
fig = plot_country_two_vars(var1='Q1_What is your gender? - Selected Choice',
                            var2='Q4_What is the highest level of formal education that you have attained or plan to attain within the next 2 years?',
                            title_name='degree')
iplot(fig);
fig = plot_country_two_vars_dropdown(var1='Q1_What is your gender? - Selected Choice',
                                     var2='Q4_What is the highest level of formal education that you have attained or plan to attain within the next 2 years?',
                                     title_name='degree')
iplot(fig);
s = pd.crosstab(df_choice['Q1_What is your gender? - Selected Choice'],
                df_choice['Q4_What is the highest level of formal education that you have attained or plan to attain within the next 2 years?'], normalize='index').style.background_gradient(cmap='viridis', low=.05, high=0).highlight_null('red')
s
def plot_country_one_var_dropdown(var='', title_name=''):
    df_choice[var] = df_choice[var].astype('category')
    data = []
    buttons = []
    n = df_choice['Q3_orig'].nunique()
    for j, c in enumerate(df_choice['Q3_orig'].unique()):
        visibility = [False] * n
        grouped = df_choice.loc[df_choice['Q3_orig'] == c,
                                var].value_counts().sort_index()
        grouped = grouped / grouped.sum()
        if var == 'Q9_What is your current yearly compensation (approximate $USD)?':
            map_dict = {'0-10,000': 0,
                        '10-20,000': 1,
                        '100-125,000': 10,
                        '125-150,000': 11,
                        '150-200,000': 12,
                        '20-30,000': 2,
                        '200-250,000': 13,
                        '250-300,000': 14,
                        '30-40,000': 3,
                        '300-400,000': 15,
                        '40-50,000': 4,
                        '400-500,000': 16,
                        '50-60,000': 5,
                        '60-70,000': 6,
                        '70-80,000': 7,
                        '80-90,000': 8,
                        '90-100,000': 9,
                        '500,000+': 17,
                        'I do not wish to disclose my approximate yearly compensation': 18}
            grouped = grouped.reset_index()
            grouped.columns = ['salary', 'counts']
            grouped['sorting'] = grouped['salary'].apply(lambda x: map_dict[x])
            grouped = grouped.sort_values('sorting', ascending=True)
            trace = go.Bar(
                x=grouped['salary'],
                y=grouped['counts'],
                name=c,
                marker=dict(color=j),
                showlegend=True,
                visible=True if j == 0 else False
            )
            data.append(trace)

        elif var == 'Q24_How long have you been writing code to analyze data?':
            map_dict = {'I have never written code but I want to learn': 8,
                        '5-10 years': 3,
                        '3-5 years': 2,
                        '< 1 year': 0,
                        '1-2 years': 1,
                        '10-20 years': 4,
                        '20-30 years': 5,
                        '30-40 years': 6,
                        'I have never written code and I do not want to learn': 9,
                        '40+ years': 7}
            grouped = grouped.reset_index()
            grouped.columns = ['salary', 'counts']
            grouped['sorting'] = grouped['salary'].apply(lambda x: map_dict[x])
            grouped = grouped.sort_values('sorting', ascending=True)
            trace = go.Bar(
                x=grouped['salary'],
                y=grouped['counts'],
                name=c,
                marker=dict(color=j),
                showlegend=True,
                visible=True if j == 0 else False
            )
            data.append(trace)
        else:
            trace = go.Bar(
                x=grouped.index,
                y=grouped.values,
                name=c,
                marker=dict(color=j),
                showlegend=True,
                visible=True if j == 0 else False
            )
            data.append(trace)

        visibility[j*1:j*1 + 1] = [True]
        buttons.append(dict(label = c,
                            method = 'update',
                            args = [{'visible': visibility},
                                    {'title': f'Responders in {c} by {title_name}'}]))
            
            
    updatemenus = list([dict(active=-1, buttons=buttons, x=1, y=2)])
    layout = dict(height=400, width=800, title=f"Responders in {df_choice['Q3_orig'].unique()[0]} by {title_name}", updatemenus=updatemenus)
    fig = dict(data=data, layout=layout)
    return fig
def plot_country_one_var(var='', title_name=''):
    data = []
    for j, c in enumerate(df_choice['Q3_In which country do you currently reside?'].unique()):
        grouped = df_choice.loc[df_choice['Q3_In which country do you currently reside?'] == c,
                                var].value_counts().sort_index()
        grouped = grouped / grouped.sum()
        if var == 'Q9_What is your current yearly compensation (approximate $USD)?':
            map_dict = {'0-10,000': 0,
                        '10-20,000': 1,
                        '100-125,000': 10,
                        '125-150,000': 11,
                        '150-200,000': 12,
                        '20-30,000': 2,
                        '200-250,000': 13,
                        '250-300,000': 14,
                        '30-40,000': 3,
                        '300-400,000': 15,
                        '40-50,000': 4,
                        '400-500,000': 16,
                        '50-60,000': 5,
                        '60-70,000': 6,
                        '70-80,000': 7,
                        '80-90,000': 8,
                        '90-100,000': 9,
                        '500,000+': 17,
                        'I do not wish to disclose my approximate yearly compensation': 18}
            grouped = grouped.reset_index()
            grouped.columns = ['salary', 'counts']
            grouped['sorting'] = grouped['salary'].apply(lambda x: map_dict[x])
            grouped = grouped.sort_values('sorting', ascending=True)
            trace = go.Bar(
                x=grouped['salary'],
                y=grouped['counts'],
                name=c,
                marker=dict(color=j),
                showlegend=True
            )
            data.append(trace)
        elif var == 'Q24_How long have you been writing code to analyze data?':
            map_dict = {'I have never written code but I want to learn': 8,
                        '5-10 years': 3,
                        '3-5 years': 2,
                        '< 1 year': 0,
                        '1-2 years': 1,
                        '10-20 years': 4,
                        '20-30 years': 5,
                        '30-40 years': 6,
                        'I have never written code and I do not want to learn': 9,
                        '40+ years': 7}
            grouped = grouped.reset_index()
            grouped.columns = ['salary', 'counts']
            grouped['sorting'] = grouped['salary'].apply(lambda x: map_dict[x])
            grouped = grouped.sort_values('sorting', ascending=True)
            trace = go.Bar(
                x=grouped['salary'],
                y=grouped['counts'],
                name=c,
                marker=dict(color=j),
                showlegend=True
            )
            data.append(trace)
        else:
            trace = go.Bar(
                x=grouped.index,
                y=grouped.values,
                name=c,
                marker=dict(color=j),
                showlegend=True
            )
            data.append(trace)    
    fig = go.Figure(data=data)
    fig['layout'].update(height=400, width=800, title=f'Responders in countries by {title_name}', barmode='group');
    return fig

fig = plot_country_one_var(var='Q5_Which best describes your undergraduate major? - Selected Choice', title_name='major')
iplot(fig);
fig = plot_country_one_var_dropdown(var='Q5_Which best describes your undergraduate major? - Selected Choice', title_name='major')
iplot(fig);
s = pd.crosstab(df_choice['Q1_What is your gender? - Selected Choice'],
                df_choice['Q5_Which best describes your undergraduate major? - Selected Choice'], normalize='index').style.background_gradient(cmap='viridis', low=.01, high=0).highlight_null('red')
s
countsDf = df_choice['Q6_Select the title most similar to your current role (or most recent title if retired): - Selected Choice'].value_counts()
trace1 = go.Bar(
                x = countsDf.index,
                y = countsDf.values,
                name = "Kaggle",
                marker = dict(color = 'gold'),
                text = countsDf.index)
data = [trace1]
layout = go.Layout(barmode = "group",title='Title', yaxis= dict(title='Counts'),showlegend=False)
fig = go.Figure(data = data, layout = layout)
iplot(fig)
countsDf = countsDf.reset_index()
countsDf.columns = ['title', 'number']
countsDf.loc[countsDf['title'].isin(['Consultant', 'Business Analyst', 'Marketing Analyst']), 'title'] = 'Data Analyst'
countsDf.loc[countsDf['title'].isin(['Software Engineer', 'DBA/Database Engineer', 'Developer Advocate']), 'title'] = 'Data Engineer'
countsDf.loc[countsDf['title'].isin(['Research Assistant', 'Research Scientist', 'Statistician']), 'title'] = 'Research'
countsDf.loc[countsDf['title'].isin(['Product/Project Manager', 'Chief Officer']), 'title'] = 'Manager'
countsDf.loc[countsDf['title'].isin(['Salesperson', 'Principal Investigator', 'Data Journalist', 'Not employed']), 'title'] = 'Other'
countsDf = countsDf.groupby('title')['number'].sum()
trace1 = go.Bar(
                x = countsDf.index,
                y = countsDf.values,
                name = "Kaggle",
                marker = dict(color = 'brown'),
                text = countsDf.values,
                textposition = 'outside')
data = [trace1]
layout = go.Layout(barmode = "group",title='Grouped titles', yaxis= dict(title='Counts'),showlegend=False)
fig = go.Figure(data = data, layout = layout)
iplot(fig)
s = pd.crosstab(df_choice['Q7_In what industry is your current employer/contract (or your most recent employer if retired)? - Selected Choice'],
                df_choice['Q10_Does your current employer incorporate machine learning methods into their business?']).style.background_gradient(cmap='viridis', low=.1, high=0).highlight_null('red')
s
df_choice.loc[df_choice['Q6_Select the title most similar to your current role (or most recent title if retired): - Selected Choice'].isin(['Consultant', 'Business Analyst', 'Marketing Analyst']),
              'Q6_Select the title most similar to your current role (or most recent title if retired): - Selected Choice'] = 'Data Analyst'
df_choice.loc[df_choice['Q6_Select the title most similar to your current role (or most recent title if retired): - Selected Choice'].isin(['Software Engineer', 'DBA/Database Engineer', 'Developer Advocate']),
              'Q6_Select the title most similar to your current role (or most recent title if retired): - Selected Choice'] = 'Data Engineer'
df_choice.loc[df_choice['Q6_Select the title most similar to your current role (or most recent title if retired): - Selected Choice'].isin(['Research Assistant', 'Research Scientist', 'Statistician']),
              'Q6_Select the title most similar to your current role (or most recent title if retired): - Selected Choice'] = 'Research'
df_choice.loc[df_choice['Q6_Select the title most similar to your current role (or most recent title if retired): - Selected Choice'].isin(['Product/Project Manager', 'Chief Officer']),
              'Q6_Select the title most similar to your current role (or most recent title if retired): - Selected Choice'] = 'Manager'
df_choice.loc[df_choice['Q6_Select the title most similar to your current role (or most recent title if retired): - Selected Choice'].isin(['Salesperson', 'Principal Investigator', 'Data Journalist', 'Not employed']),
              'Q6_Select the title most similar to your current role (or most recent title if retired): - Selected Choice'] = 'Other'
fig = plot_country_one_var(var='Q6_Select the title most similar to your current role (or most recent title if retired): - Selected Choice', title_name='title')
iplot(fig);
fig = plot_country_one_var_dropdown(var='Q6_Select the title most similar to your current role (or most recent title if retired): - Selected Choice',
                                    title_name='title')
iplot(fig);
fig = plot_country_two_vars(var2='Q6_Select the title most similar to your current role (or most recent title if retired): - Selected Choice',
                            var1='Q8_How many years of experience do you have in your current role?', title_name='title and experience')
iplot(fig);
fig = plot_country_two_vars_dropdown(var1='Q8_How many years of experience do you have in your current role?',
                                     var2='Q6_Select the title most similar to your current role (or most recent title if retired): - Selected Choice',
                                     title_name='title and experience')
iplot(fig);
fig = plot_country_one_var(var='Q9_What is your current yearly compensation (approximate $USD)?', title_name='salary')
iplot(fig);
fig = plot_country_one_var_dropdown(var='Q9_What is your current yearly compensation (approximate $USD)?', title_name='salary')
iplot(fig);
fig = plot_country_one_var(var='Q24_How long have you been writing code to analyze data?', title_name='years of coding for data analysis')
iplot(fig);
var1='Q26_Do you consider yourself to be a data scientist?'
var2='Q25_For how many years have you used machine learning methods (at work or in school)?'
title_name='years of learning ML and self-confidence'
colors = cl.scales[str(df_choice[var1].fillna('').nunique())]['qual']['Paired']
fig = tools.make_subplots(rows=4, cols=1, subplot_titles=('United States of America', 'Other countries', 'India', 'Russia'), print_grid=False)
for j, c in enumerate(df_choice['Q3_In which country do you currently reside?'].unique()):
    #data = []
    for ind, i in enumerate(df_choice[var1].unique()):
        grouped = df_choice.loc[(df_choice[var1] == i) & (df_choice['Q3_In which country do you currently reside?'] == c),
                            var2].value_counts().sort_index()
        map_dict = {'I have never studied machine learning but plan to learn in the future': 8,
                    '< 1 year': 0,
                    '4-5 years': 4,
                    '2-3 years': 2,
                    '1-2 years': 1,
                    '5-10 years': 5,
                    '3-4 years': 3,
                    'I have never studied machine learning and I do not plan to': 9,
                    '20+ years': 7,
                    '10-15 years': 6}
        grouped = grouped.reset_index()
        grouped.columns = ['years', 'counts']
        #print(grouped.shape)
        #break
        grouped['sorting'] = grouped['years'].apply(lambda x: map_dict[x])
        grouped = grouped.sort_values('sorting', ascending=True)
        trace = go.Bar(
            x=grouped.years,
            y=grouped.counts,
            name=i,
            marker=dict(color=colors[ind]),
            showlegend=True if j == 0 else False,
            legendgroup=i
        )
        fig.append_trace(trace, j + 1, 1)    

fig['layout'].update(height=1000, width=800, title=f'Responders in countries by {title_name}');

iplot(fig);
def plot_choice_var(var='', title_name=''):
    col_names = [col for col in df_choice.columns if f'{var}_Part' in col]
    data = []
    small_df = df_choice[col_names]
    text_values = [col.split('- ')[2] for col in col_names]
    counts = []
    for m, n in zip(col_names, text_values):
        if small_df[m].nunique() == 0:
            counts.append(0)
        else:
            counts.append(sum(small_df[m] == n))
    trace = go.Bar(
        x=text_values,
        y=counts,
        name=c,
        marker=dict(color='silver'),
        showlegend=False
    )
    data.append(trace)    
    fig = go.Figure(data=data)
    fig['layout'].update(height=400, width=800, title=f'Popular {title_name}');
    return fig

def plot_country_multiple_choice_var(var='', title_name=''):
    col_names = [col for col in df_choice.columns if f'{var}_Part' in col]
    #print(col_names)
    data = []
    for j, c in enumerate(df_choice['Q3_In which country do you currently reside?'].unique()):
        small_df = df_choice.loc[df_choice['Q3_In which country do you currently reside?'] == c, col_names]
        text_values = [col.split('- ')[2] for col in col_names]
        counts = []
        for m, n in zip(col_names, text_values):
            if small_df[m].nunique() == 0:
                counts.append(0)
            else:
                counts.append(sum(small_df[m] == n))
        counts = [i / len(small_df) for i in counts]
        trace = go.Bar(
            x=text_values,
            y=counts,
            name=c,
            marker=dict(color=j),
            showlegend=True
        )
        data.append(trace)    
    fig = go.Figure(data=data)
    fig['layout'].update(height=400, width=800, title=f'Popular {title_name} in different countries', barmode='group');
    return fig

def plot_one_text_var(q='', title=''):
    col_name = [col for col in df_free.columns if q in col][0]
    df_ = df_free[col_name].value_counts().head(7)
    trace = go.Pie(labels=df_.index, 
                   values=df_.values
                  )

    data = [trace]
    layout = go.Layout(
        title=title
    )

    fig = go.Figure(data=data, layout=layout)
    return fig

var_name = dict([('Q38', 'resources'), ('Q13', 'IDE'), ('Q14', 'hosted notebooks'), ('Q15', 'cloud computing services'), ('Q16', 'programming languages'),
         ('Q19', 'ML frameworks'), ('Q21', 'data visualization libraries'), ('Q29', 'data bases'), ('Q31', 'types of data used'), ('Q36', 'online platforms'),
         ('Q47', 'tools for interpretation'), ('Q49', 'tools for reproducibility')])
cols2 = ['Q38', 'Q13', 'Q14', 'Q15','Q16', 'Q19', 'Q21', 'Q29', 'Q31', 'Q36', 'Q47', 'Q49']

def plot_country_multiple_choice_var_dropdown(var=''):
    t = var_name[var]
    #print(t)
    col_names = [col for col in df_choice.columns if f'{var}_Part' in col]
    data = []
    buttons = []
    n = df_choice['Q3_orig'].nunique()
    for j, c in enumerate(sorted(df_choice['Q3_orig'].unique(), reverse=True)[::-1]):
        visibility = [False] * 58
        small_df = df_choice.loc[df_choice['Q3_orig'] == c, col_names]
        text_values = [col.split('- ')[2] for col in col_names]
        counts = []
        for m, n in zip(col_names, text_values):
            if small_df[m].nunique() == 0:
                counts.append(0)
            else:
                counts.append(sum(small_df[m] == n))
        orig_counts = counts.copy()
        counts = [i / len(small_df) for i in counts]
        trace = go.Bar(
            x=text_values,
            y=counts,
            name=c,
            marker=dict(color=j),
            showlegend=True,
            visible=True if j == 0 else False,
            text = orig_counts,
            textposition = 'outside'
        )
        data.append(trace)
        visibility[j:j + 1] = [True]
        buttons.append(dict(label = c,
                                method = 'update',
                                args = [{'visible': visibility},
                                        {'title': f"Responders in {c} by {t}"}]))
    updatemenus = list([dict(active=-1, buttons=buttons, x=1, y=2)])
    layout = dict(height=500, width=800, title=f"Responders in {sorted(df_choice['Q3_orig'].unique())[0]} by {t}", updatemenus=updatemenus)
    fig = dict(data=data, layout=layout)

    iplot(fig);
fig = plot_choice_var(var='Q38', title_name='resources')
iplot(fig);
fig = plot_country_multiple_choice_var(var='Q38', title_name='resources')
iplot(fig);
plot_country_multiple_choice_var_dropdown('Q38')
fig = plot_one_text_var('Q38', title='Other popular resources')
iplot(fig)
fig = plot_choice_var(var='Q13', title_name='IDE')
iplot(fig);
fig = plot_country_multiple_choice_var(var='Q13', title_name='IDE')
iplot(fig);
plot_country_multiple_choice_var_dropdown('Q13')
fig = plot_one_text_var('Q13', title='Other popular IDE')
iplot(fig)
fig = plot_choice_var(var='Q14', title_name='hosted notebooks')
iplot(fig);
fig = plot_country_multiple_choice_var(var='Q14', title_name='hosted notebooks')
iplot(fig);
plot_country_multiple_choice_var_dropdown('Q14')
fig = plot_one_text_var(q='Q14')
iplot(fig)
fig = plot_choice_var(var='Q15', title_name='cloud computing services')
iplot(fig);
fig = plot_country_multiple_choice_var(var='Q15', title_name='cloud computing services')
iplot(fig);
plot_country_multiple_choice_var_dropdown('Q15')
fig = plot_one_text_var('Q15', title='Other popular cloud computing services')
iplot(fig)

fig = plot_choice_var(var='Q16', title_name='programming languages')
iplot(fig);
fig = plot_country_multiple_choice_var(var='Q16', title_name='programming languages')
iplot(fig);
plot_country_multiple_choice_var_dropdown('Q16')
fig = plot_one_text_var('Q16', title='Other popular programming languages')
iplot(fig)
fig = plot_choice_var(var='Q19', title_name='ML frameworks')
iplot(fig);
fig = plot_country_multiple_choice_var(var='Q19', title_name='ML frameworks')
iplot(fig);
plot_country_multiple_choice_var_dropdown('Q19')
fig = plot_one_text_var('Q19', title='Other popular ML frameworks')
iplot(fig)
fig = plot_choice_var(var='Q21', title_name='data visualization libraries')
iplot(fig);
fig = plot_country_multiple_choice_var(var='Q21', title_name='data visualization libraries')
iplot(fig);
plot_country_multiple_choice_var_dropdown('Q21')
fig = plot_one_text_var('Q21', title='Other popular data visualization libraries')
iplot(fig)
fig = plot_choice_var(var='Q29', title_name='data bases')
iplot(fig);
fig = plot_country_multiple_choice_var(var='Q29', title_name='data bases')
iplot(fig);
plot_country_multiple_choice_var_dropdown('Q29')
fig = plot_one_text_var('Q29', title='Other popular data bases')
iplot(fig)
fig = plot_choice_var(var='Q31', title_name='types of data used')
iplot(fig);
fig = plot_country_multiple_choice_var(var='Q31', title_name='types of data used')
iplot(fig);
plot_country_multiple_choice_var_dropdown('Q31')
fig = plot_one_text_var('Q31', title='Other popular types of data used')
iplot(fig)
fig = plot_choice_var(var='Q36', title_name='online platforms')
iplot(fig);
fig = plot_country_multiple_choice_var(var='Q36', title_name='online platforms')
iplot(fig);
plot_country_multiple_choice_var_dropdown('Q36')
fig = plot_one_text_var('Q36', title='Other popular types of online platforms')
iplot(fig)
fig = plot_choice_var(var='Q47', title_name='tools for interpretation')
iplot(fig);
fig = plot_country_multiple_choice_var(var='Q47', title_name='tools for interpretation')
iplot(fig);
plot_country_multiple_choice_var_dropdown('Q47')
fig = plot_choice_var(var='Q49', title_name='tools for reproducibility')
iplot(fig);
fig = plot_country_multiple_choice_var(var='Q49', title_name='tools for reproducibility')
iplot(fig);
plot_country_multiple_choice_var_dropdown('Q49')