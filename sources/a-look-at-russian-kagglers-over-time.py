
### pip install -U vega_datasets notebook vega
# importing libraries
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
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings("ignore")
init_notebook_mode(connected=True)
from ipywidgets import interact, interactive, interact_manual
import ipywidgets as widgets
import colorlover as cl
# loading data from different years
DIR = '/kaggle/input/kaggle-survey-2018/'
df_free_18 = pd.read_csv(DIR + 'freeFormResponses.csv', low_memory=False, header=[0,1])
df_choice_18 = pd.read_csv(DIR + 'multipleChoiceResponses.csv', low_memory=False, header=[0,1])
# Format Dataframes
df_free_18.columns = ['_'.join(col) for col in df_free_18.columns]
df_choice_18.columns = ['_'.join(col) for col in df_choice_18.columns]

DIR = '/kaggle/input/kaggle-survey-2019/'
df_free_19 = pd.read_csv(DIR + 'other_text_responses.csv', low_memory=False)
df_choice_19 = pd.read_csv(DIR + 'multiple_choice_responses.csv', low_memory=False, encoding='latin-1', header=[0,1])
df_choice_19.columns = ['_'.join(col) for col in df_choice_19.columns]

DIR = '/kaggle/input/kaggle-survey-2017/'
df_free_17 = pd.read_csv(DIR + 'freeformResponses.csv', low_memory=False)
df_choice_17 = pd.read_csv(DIR + 'multipleChoiceResponses.csv', low_memory=False, encoding='latin-1')
# processing data for visualizations

top_count = df_choice_19['Q3_In which country do you currently reside?'].value_counts().head(8).reset_index().rename(columns={'Q3_In which country do you currently reside?': 'count', 'index': 'Country'})
# taking only russian responders
df_choice_17 = df_choice_17.loc[df_choice_17['Country'] == 'Russia']
df_choice_18 = df_choice_18.loc[df_choice_18['Q3_In which country do you currently reside?'] == 'Russia']
df_choice_19 = df_choice_19.loc[df_choice_19['Q3_In which country do you currently reside?'] == 'Russia']

def get_age(x: int):
    """
    Convert numerical age to categories.
    """
    if 18 <= x <= 21:
        return '18-21'
    elif 22 <= x <= 24:
        return '22-24'
    elif 25 <= x <= 29:
        return '25-29'
    elif 30 <= x <= 34:
        return '30-34'
    elif 35 <= x <= 39:
        return '35-39'
    elif 40 <= x <= 44:
        return '40-44'
    elif 45 <= x <= 49:
        return '45-49'
    elif 50 <= x <= 54:
        return '50-54'
    elif 55 <= x <= 59:
        return '55-59'
    elif 60 <= x <= 69:
        return '60-69'
    elif x >= 70:
        return '70+'
    
# create a new age column with the same name and unique values in all datasets
df_choice_17['Age_'] = df_choice_17['Age'].apply(lambda x: get_age(x))
df_choice_18['Age_'] = df_choice_18['Q2_What is your age (# years)?']
df_choice_18.loc[df_choice_18['Age_'].isin(['70-79', '80+']), 'Age_'] = '70+'
df_choice_19['Age_'] = df_choice_19['Q1_What is your age (# years)?']

# renaming columns so that it would be easier to work with them
df_choice_17 = df_choice_17.rename(columns={'GenderSelect': 'Gender', 'FormalEducation': 'Degree'})
df_choice_18 = df_choice_18.rename(columns={'Q1_What is your gender? - Selected Choice': 'Gender', 'Q9_What is your current yearly compensation (approximate $USD)?': 'Salary',
                                            'Q4_What is the highest level of formal education that you have attained or plan to attain within the next 2 years?': 'Degree'})
df_choice_19 = df_choice_19.rename(columns={'Q2_What is your gender? - Selected Choice': 'Gender', 'Q10_What is your current yearly compensation (approximate $USD)?': 'Salary',
                                            'Q4_What is the highest level of formal education that you have attained or plan to attain within the next 2 years?': 'Degree'})
df_choice_19['Degree'] = df_choice_19['Degree'].replace({'Masterâs degree': 'Master’s degree', 'Bachelorâs degree': 'Bachelor’s degree',
                                                         'Some college/university study without earning a bachelorâs degree': 'Some college/university study without earning a bachelor’s degree'})
df_choice_17['Degree'] = df_choice_17['Degree'].replace({"Master's degree": 'Master’s degree', "Bachelor's degree": 'Bachelor’s degree',
                                                         "Some college/university study without earning a bachelor's degree": 'Some college/university study without earning a bachelor’s degree',
                                                         "I did not complete any formal education past high school": "No formal education past high school"})

# changing salary values to the same categories
df_choice_19.loc[df_choice_19['Salary'].isin(['$0-999', '1,000-1,999', '2,000-2,999', '3,000-3,999', '4,000-4,999', '5,000-7,499', '7,500-9,999']), 'Salary'] = '0-10,000'
df_choice_19.loc[df_choice_19['Salary'].isin(['10,000-14,999', '15,000-19,999']), 'Salary'] = '10-20,000'
df_choice_19.loc[df_choice_19['Salary'].isin(['20,000-24,999', '25,000-29,999']), 'Salary'] = '20-30,000'
df_choice_19.loc[df_choice_19['Salary'] == '30,000-39,999', 'Salary'] = '30-40,000'
df_choice_19.loc[df_choice_19['Salary'] == '40,000-49,999', 'Salary'] = '40-50,000'
df_choice_19.loc[df_choice_19['Salary'] == '50,000-59,999', 'Salary'] = '50-60,000'
df_choice_19.loc[df_choice_19['Salary'] == '60,000-69,999', 'Salary'] = '60-70,000'
df_choice_19.loc[df_choice_19['Salary'] == '70,000-79,999', 'Salary'] = '70-80,000'
df_choice_19.loc[df_choice_19['Salary'] == '80,000-89,999', 'Salary'] = '80-90,000'
df_choice_19.loc[df_choice_19['Salary'] == '90,000-99,999', 'Salary'] = '90-100,000'
df_choice_19.loc[df_choice_19['Salary'] == '100,000-124,999', 'Salary'] = '100-125,000'
df_choice_19.loc[df_choice_19['Salary'] == '125,000-149,999', 'Salary'] = '125-150,000'
df_choice_19.loc[df_choice_19['Salary'] == '200,000-249,999', 'Salary'] = '200-250,000'
df_choice_18.loc[df_choice_18['Salary'].isin(['400-500,000', '300-400,000']), 'Salary'] = '300,000-500,000'
# Functions

# some of the code is taken from my old kernel: https://www.kaggle.com/artgor/russia-usa-india-and-other-countries

def plot_gender_vars(var1: str = '', title_name: str = ''):
    """
    Make separate count plots for genders over years.
    """
    colors = cl.scales['3']['qual']['Paired']
    names = {0: '2017', 1: '2018', 2: '2019'}
    fig = tools.make_subplots(rows=1, cols=2, subplot_titles=('Male', 'Female'), print_grid=False)
    # there are too little responders, who don't identify as Male/Female, I have decided that I can use the most common genders.
    for j, c in enumerate(['Male', 'Female']):
        data = []
        for i, df in enumerate([df_choice_17, df_choice_18, df_choice_19]):
            grouped = df.loc[(df['Gender'] == c), var1].value_counts().sort_index().reset_index()
            grouped['Age_'] = grouped['Age_'] / np.sum(grouped['Age_'])
            trace = go.Bar(
                x=grouped['index'],
                y=grouped.Age_,
                name=names[i],
                marker=dict(color=colors[i]),
                showlegend=True if j == 0 else False,
                legendgroup=i
            )
            fig.append_trace(trace, 1, j + 1)    

    fig['layout'].update(height=400, width=1000, paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', title=f'Rate of russian kagglers by {title_name} and gender');
    return fig


def plot_var(var1: str = '', title_name: str = ''):
    """
    Plot one variable over years.
    """
    colors = cl.scales['3']['qual']['Paired']
    names = {0: '2017', 1: '2018', 2: '2019'}
    
    data = []
    for i, df in enumerate([df_choice_17, df_choice_18, df_choice_19]):
        grouped = df[var1].value_counts().sort_index().reset_index()
        grouped[var1] = grouped[var1] / np.sum(grouped[var1])
        trace = go.Bar(
            x=grouped['index'],
            y=grouped[var1],
            name=names[i],
            marker=dict(color=colors[i]),
            legendgroup=i
        )
        data.append(trace)
    layout = dict(height=400, width=1000, paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', title=f'Russian kagglers by {title_name}');  
    fig = dict(data=data, layout=layout)
    return fig


def plot_var_salary(var1: str = '', title_name: str = '', normalize: bool = False):
    """
    Plot salary over years. This is a separate function, because
    it is necessary to add code for sorting.
    """
    colors = cl.scales['3']['qual']['Paired']
    names = {0: '2018', 1: '2019'}
    
    data = []
    for i, df in enumerate([df_choice_18, df_choice_19]):
        grouped = df[var1].value_counts().sort_index().reset_index()
        if normalize:
            grouped[var1] = grouped[var1] / np.sum(grouped[var1])
        map_dict = {'0-10,000': 0,
                    '10-20,000': 1,
                    '100-125,000': 10,
                    '125-150,000' : 11,
                    '150-200,000': 12,
                    '20-30,000': 2,
                    '200-250,000': 13,
                    '250-300,000': 14,
                    '30-40,000': 3,
                    '300,000-500,000': 15,
                    '40-50,000': 4,
                    '50-60,000': 5,
                    '60-70,000': 6,
                    '70-80,000': 7,
                    '80-90,000': 8,
                    '90-100,000': 9,
                    '> $500,000': 16,
                    'I do not wish to disclose my approximate yearly compensation': 17}
        grouped['sorting'] = grouped['index'].apply(lambda x: map_dict[x])
        grouped = grouped.loc[grouped['index'] != 'I do not wish to disclose my approximate yearly compensation']
        grouped = grouped.sort_values('sorting', ascending=True)
        trace = go.Bar(
            x=grouped['index'],
            y=grouped[var1],
            name=names[i],
            marker=dict(color=colors[i]),
            legendgroup=i
        )
        data.append(trace)
    layout = dict(height=500, width=1000, paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', title=f'Russian kagglers by {title_name}');  
    fig = dict(data=data, layout=layout)
    return fig


def plot_choice_var(var: str = '', title_name: str = ''):
    """
    Plot a variable, in which responders could select several answers.
    """
    col_names = [col for col in df_choice_19.columns if f'{var}_Part' in col]
    data = []
    small_df = df_choice_19[col_names]
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
        name='c',
        marker=dict(color='silver'),
        showlegend=False
    )
    data.append(trace)    
    fig = go.Figure(data=data)
    fig['layout'].update(height=600, width=1000, paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', title=f'Popular {title_name}');
    return fig
df_count = pd.DataFrame({'Year': [2017, 2018, 2019], 'Count': [df_choice_17.shape[0], df_choice_18.shape[0], df_choice_19.shape[0]]})
top_count = top_count.sort_values('count')

fig = make_subplots(rows=1, cols=2)
fig.add_trace(go.Bar(y=top_count['Country'], x=top_count['count'], orientation='h', name='Number of respondents by country in 2019'), row=1, col=1)
fig.add_trace(go.Bar(x=df_count['Year'], y=df_count['Count'], name='Number of russian responders by year'), row=1, col=2)

fig['layout'].update(height=400, width=1000,paper_bgcolor='rgba(0,0,0,0)',
    plot_bgcolor='rgba(0,0,0,0)');
iplot(fig);
fig = plot_gender_vars(var1='Age_', title_name='age')
iplot(fig);
fig = plot_var(var1='Degree', title_name='degree')
iplot(fig);
fig = plot_var_salary(var1='Salary', title_name='salary')
iplot(fig);
fig = plot_choice_var(var='Q12', title_name='resources')
iplot(fig);
fig = plot_choice_var(var='Q18', title_name='languages')
iplot(fig);
fig = plot_choice_var(var='Q9', title_name='additional activities')
iplot(fig);

fig = plot_choice_var(var='Q27', title_name='nlp tools')
iplot(fig);
fig = plot_choice_var(var='Q28', title_name='libraries')
iplot(fig);
q28_free = df_free_19['Q28_OTHER_TEXT'].str.lower().value_counts().reset_index()
print(f"Catboost users: {q28_free.loc[q28_free['index'] == 'catboost', 'Q28_OTHER_TEXT'].values[0]}")
print(f"Catalyst users: {q28_free.loc[q28_free['index'] == 'catalyst', 'Q28_OTHER_TEXT'].values[0]}")
fig = plot_choice_var(var='Q25', title_name='ml tools')
iplot(fig);
q25_free = df_free_19['Q25_OTHER_TEXT'].str.lower().value_counts().reset_index()
print(f"Catalyst users: {q25_free.loc[q25_free['index'] == 'catalyst', 'Q25_OTHER_TEXT'].values[0]}")
user_ach = pd.read_csv('/kaggle/input/meta-kaggle/UserAchievements.csv')
user_org = pd.read_csv('/kaggle/input/meta-kaggle/UserOrganizations.csv')
ods_users = pd.merge(user_org.loc[user_org['OrganizationId'] == 642], user_ach, how='left', on='UserId')

# renaming tiers and creating dict with values for mapping
tier_dict = {0: 'Novice', 1: 'Contributor', 2: 'Expert', 3: 'Master', 4: 'Grandmaster'}
tier_dict_inv = {v: k for k, v in tier_dict.items()}
ods_users['Tier_names'] = ods_users['Tier'].apply(lambda x: tier_dict[x])

# making plots in a similar style
fig = make_subplots(rows=2, cols=2, subplot_titles=("Number of kagglers with this highest title", "Competition titles", "Script titles", "Discussion titles"))

aggregated = ods_users.groupby('UserId')['Tier'].max().value_counts().reset_index()
aggregated['sorting'] = aggregated['index']
aggregated['Tier_names'] = aggregated['index'].apply(lambda x: tier_dict[x])
aggregated = aggregated.sort_values('sorting', ascending=False)
fig.add_trace(go.Bar(x=aggregated['Tier'],
                     y=aggregated['Tier_names'], orientation='h'), row=1, col=1)

aggregated = ods_users.loc[ods_users['AchievementType'] == 'Competitions', 'Tier_names'].value_counts().reset_index()
aggregated['sorting'] = aggregated['index'].apply(lambda x: tier_dict_inv[x])
aggregated = aggregated.sort_values('sorting', ascending=False)
fig.add_trace(go.Bar(y=aggregated['index'],
                     x=aggregated['Tier_names'], orientation='h'), row=1, col=2)

aggregated = ods_users.loc[ods_users['AchievementType'] == 'Scripts', 'Tier_names'].value_counts().reset_index()
aggregated['sorting'] = aggregated['index'].apply(lambda x: tier_dict_inv[x])
aggregated = aggregated.sort_values('sorting', ascending=False)
fig.add_trace(go.Bar(y=aggregated['index'],
                     x=aggregated['Tier_names'], orientation='h'), row=2, col=1)

aggregated = ods_users.loc[ods_users['AchievementType'] == 'Discussion', 'Tier_names'].value_counts().reset_index()
aggregated['sorting'] = aggregated['index'].apply(lambda x: tier_dict_inv[x])
aggregated = aggregated.sort_values('sorting', ascending=False)
fig.add_trace(go.Bar(y=aggregated['index'],
                     x=aggregated['Tier_names'], orientation='h'), row=2, col=2)

fig['layout'].update(height=800, width=1000, showlegend=False);
iplot(fig);