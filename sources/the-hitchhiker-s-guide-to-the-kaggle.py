
# default
import numpy as np 
import pandas as pd 
import os

# visualization
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import missingno as msno
import networkx as nx
import plotly.graph_objects as go

# read file
question = pd.read_csv('../input/kaggle-survey-2019/questions_only.csv')
schema = pd.read_csv('../input/kaggle-survey-2019/survey_schema.csv')
multiple_choice = pd.read_csv('../input/kaggle-survey-2019/multiple_choice_responses.csv')
other_text =  pd.read_csv('../input/kaggle-survey-2019/other_text_responses.csv')
# check the question list & return the survey answers
# 'QXX' is format of `s`
from IPython.core.display import display, HTML

def q_list(s):
    lst = []
    for i in multiple_choice.columns:
        if i[:3]==s:
            lst.append(i)
    df = multiple_choice[lst]

    df_sub = df.iloc[0].apply(lambda x: ''.join(x.split('-')[2:]))
    q = ''.join([f'<li>{i}</li>' for i in df_sub.values])
    display(HTML(f'<div style="background: #f7f5f5; padding: 10px; border-radius: 10px;"> <h3 style="color:#20639B; padding:10px">{s} : {question.T[0][int(s[1:])]} </h3><ol>{q}</ol> <div>'))
    return df, df_sub
q = ''.join([f'<li>{i}</li>' for i in question.T[0][1:]])
display(HTML(f'<div style="background: #f6f4f4; padding: 10px; border-radius: 10px;">  <h2 style="color:#20639B; padding:10px"> Question List</h2><ol>{q}</ol> </div>'))
lst = []
for i in multiple_choice.columns[1:]:
    lst.append(i.split('_')[0])
lst = pd.Series(lst).apply(lambda x : int(x[1:]))
cnt = lst.value_counts().sort_index()

fig, ax = plt.subplots(1,1,figsize=(17,5))
sns.barplot(cnt.index, cnt, palette="Blues_d", ax=ax)
ax.set_title('# of Sub-Question')
plt.show()
dist = multiple_choice[['Q1', 'Q2', 'Q3']]
dist = dist.rename(columns={"Q1": "Age", "Q2": "Gender", "Q3":"Country"})
dist.drop(0, axis=0, inplace=True)
!pip install pywaffle
from pywaffle import Waffle

gender = dist['Gender'].value_counts()

fig = plt.figure(
    FigureClass=Waffle, 
    rows=5,
    columns=12,
    values=gender,
    colors = ('#20639B', '#ED553B', '#3CAEA3', '#F5D55C'),
    title={'label': 'Gender Distribution', 'loc': 'left'},
    labels=["{}({})".format(a, b) for a, b in zip(gender.index, gender) ],
    legend={'loc': 'lower left', 'bbox_to_anchor': (0, -0.4), 'ncol': len(dist), 'framealpha': 0},
    font_size=30, 
    icons = 'child',
    figsize=(12, 5),  
    icon_legend=True
)
fig, ax = plt.subplots(1, 1, figsize=(20, 5))

sns.set_palette(sns.color_palette(['#20639B', '#ED553B', '#3CAEA3', '#F5D55C']))

sns.countplot(x='Age', hue='Gender', data=dist, 
              order = dist['Age'].value_counts().sort_index().index, 
              ax=ax )

plt.title('Age & Gender Distribution', size=15)
plt.show()
# Age & Gender's distribution
dist_age = dist[['Gender', 'Age']].groupby('Age')['Gender'].value_counts().unstack()

for i in dist_age.columns:
    dist_age[i] = dist_age[i].apply(lambda x : x/gender[i])

top_labels = sorted(dist['Age'].unique())

colors = ['rgba(38, 24, 74, 0.8)', 'rgba(71, 58, 131, 0.8)',
          'rgba(122, 120, 168, 0.8)', 'rgba(164, 163, 204, 0.85)',
          'rgba(190, 192, 213, 1)','rgba(190, 192, 213, 0.95)',
          'rgba(190, 192, 213, 0.90)','rgba(190, 192, 213, 0.85)',
          'rgba(190, 192, 213, 0.80)','rgba(190, 192, 213, 0.75)',
          'rgba(190, 192, 213, 0.70)','rgba(190, 192, 213, 0.65)',
          'rgba(190, 192, 213, 0.60)']

x_data = dist_age.T.values.round(2)

y_data = ['Female',
          'Male',
          'Prefer not to say',
          'Prefer to self-describe']

fig = go.Figure()

for i in range(0, len(x_data[0])):
    for xd, yd in zip(x_data, y_data):
        fig.add_trace(go.Bar(
            x=[xd[i]], y=[yd],
            orientation='h',
            marker=dict(
                color=colors[i],
                line=dict(color='rgb(248, 248, 249)', width=1)
            )
        ))

fig.update_layout(
    xaxis=dict(
        showgrid=False,
        showline=False,
        showticklabels=False,
        zeroline=False,
        domain=[0.15, 1]
    ),
    yaxis=dict(
        showgrid=False,
        showline=False,
        showticklabels=False,
        zeroline=False,
    ),
    barmode='stack',
    paper_bgcolor='rgb(248, 248, 255)',
    plot_bgcolor='rgb(248, 248, 255)',
    margin=dict(l=120, r=10, t=140, b=80),
    showlegend=False,
)

annotations = []

for yd, xd in zip(y_data, x_data):
    # labeling the y-axis
    annotations.append(dict(xref='paper', yref='y',
                            x=0.14, y=yd,
                            xanchor='right',
                            text=str(yd),
                            font=dict(family='Arial', size=14,
                                      color='rgb(67, 67, 67)'),
                            showarrow=False, align='right'))
    # labeling the first percentage of each bar (x_axis)
    annotations.append(dict(xref='x', yref='y',
                            x=xd[0] / 2, y=yd,
                            text=str(xd[0]) + '%',
                            font=dict(family='Arial', size=14,
                                      color='rgb(248, 248, 255)'),
                            showarrow=False))
    # labeling the first Likert scale (on the top)
    if yd == y_data[-1]:
        annotations.append(dict(xref='x', yref='paper',
                                x=xd[0] / 2, y=1.1,
                                text=top_labels[0],
                                font=dict(family='Arial', size=14,
                                          color='rgb(67, 67, 67)'),
                                showarrow=False))
    space = xd[0]
    for i in range(1, len(xd)):
            # labeling the rest of percentages for each bar (x_axis)
            annotations.append(dict(xref='x', yref='y',
                                    x=space + (xd[i]/2), y=yd,
                                    text=str(xd[i]) + '%',
                                    font=dict(family='Arial', size=14,
                                              color='rgb(248, 248, 255)'),
                                    showarrow=False))
            # labeling the Likert scale
            if yd == y_data[-1]:
                annotations.append(dict(xref='x', yref='paper',
                                        x=space + (xd[i]/2), y=1.1,
                                        text=top_labels[i],
                                        font=dict(family='Arial', size=14,
                                                  color='rgb(67, 67, 67)'),
                                        showarrow=False))
            space += xd[i]

fig.update_layout(
    title="Age ratio distribution by gender",
    annotations=annotations)

fig.show()
y = dist.Country.value_counts()

fig = go.Figure(go.Treemap(
    labels = y.index,
    parents = ['World'] * len(y),
    values = y
))

fig.update_layout(title = '# of Survey Participants')
fig.show()
dist_ratio = dist.groupby('Country')['Gender'].value_counts().unstack().fillna(0)
dist_ratio['Female/Male'] = dist_ratio['Female'] / dist_ratio['Male']

print('Average female / male ratio')
print(sum(dist_ratio['Female']) / sum(dist_ratio['Male']))
dist_ratio['Country'] = dist_ratio.index
fig = px.choropleth(dist_ratio, locations='Country',
                    locationmode='country names',
                    color="Female/Male",
                    color_continuous_scale=[[0.0, "rgb(165,0,38)"],
                [0.1111111111111111, "rgb(215,48,39)"],
                [0.2222222222222222, "rgb(244,109,67)"],
                [0.3333333333333333, "rgb(253,174,97)"],
                [0.4444444444444444, "rgb(254,224,144)"],
                [0.5555555555555556, "rgb(224,243,248)"],
                [0.6666666666666666, "rgb(171,217,233)"],
                [0.7777777777777778, "rgb(116,173,209)"],
                [0.8888888888888888, "rgb(69,117,180)"],
                [1.0, "rgb(49,54,149)"]])
fig.update_layout(title="Female to male Ratio")
fig.show()
from plotly.subplots import make_subplots

import numpy as np

x = dist.Country.value_counts().index[:7]

y_saving = dist.groupby('Country').count().loc[x]['Gender']
y_net_worth = dist_ratio.loc[x]['Female/Male']



x = dist.Country.value_counts().index[:7]

# Creating two subplots
fig = make_subplots(rows=1, cols=2, specs=[[{}, {}]], shared_xaxes=True,
                    shared_yaxes=False, vertical_spacing=0.001)

fig.append_trace(go.Bar(
    x=y_saving,
    y=x,
    marker=dict(
        color='rgba(50, 171, 96, 0.6)',
        line=dict(
            color='rgba(50, 171, 96, 1.0)',
            width=1),
    ),
    name='Survey Participants',
    orientation='h',
), 1, 1)

fig.append_trace(go.Scatter(
    x=y_net_worth, y=x,
    mode='lines+markers',
    line_color='rgb(128, 0, 128)',
    name='Female / male ratio',
), 1, 2)

fig.update_layout(
    title='Top 7 Participants by Country',
    yaxis=dict(
        showgrid=False,
        showline=False,
        showticklabels=True,
        domain=[0, 0.85],
    ),
    yaxis2=dict(
        showgrid=False,
        showline=True,
        showticklabels=False,
        linecolor='rgba(102, 102, 102, 0.8)',
        linewidth=2,
        domain=[0, 0.85],
    ),
    xaxis=dict(
        zeroline=False,
        showline=False,
        showticklabels=True,
        showgrid=True,
        domain=[0, 0.42],
        dtick=1000
    ),
    xaxis2=dict(
        zeroline=False,
        showline=False,
        showticklabels=True,
        showgrid=True,
        domain=[0.47, 1],
        side='top',
        dtick=0.05,
    ),
    legend=dict(x=0.029, y=1.038, font_size=10),
    margin=dict(l=100, r=20, t=70, b=70),
    paper_bgcolor='rgb(248, 248, 255)',
    plot_bgcolor='rgb(248, 248, 255)',
)

annotations = []

y_s = np.round(y_saving, decimals=2)
y_nw = np.round(y_net_worth, decimals=4)

# Adding labels
for ydn, yd, xd in zip(y_nw, y_s, x):
    # labeling the scatter savings
    annotations.append(dict(xref='x2', yref='y2',
                            y=xd, x=ydn+0.03,
                            text='{:,}'.format(ydn) + '%',
                            font=dict(family='Arial', size=12,
                                      color='rgb(128, 0, 128)'),
                            showarrow=False))
    # labeling the bar net worth
    annotations.append(dict(xref='x1', yref='y1',
                            y=xd, x=yd + 200,
                            text=str(yd),
                            font=dict(family='Arial', size=12,
                                      color='rgb(50, 171, 96)'),
                            showarrow=False))
# Source
annotations.append(dict(xref='paper', yref='paper',
                        x=-0.2, y=-0.109,
                        text='Most countries with a lot of participants can see that the female / male ratio is below the average.',
                        font=dict(family='Arial', size=10, color='rgb(150,150,150)'),
                        showarrow=False))

fig.update_layout(annotations=annotations)

fig.show()
q24_df, q24_sub = q_list('Q24')
q24_df.drop(0, axis=0, inplace=True)
q24_convert = {b:a for a, b in zip(q24_sub.values, q24_df.columns)}
q24_df.rename(columns=q24_convert, inplace=True)
msno.matrix(q24_df)
msno.matrix(q24_df, sort='descending')
# sns.heatmap(q24_df.corr())
q24_df.drop([' Text'], axis=1, inplace=True)
q24_df = q24_df.fillna(0).apply(lambda x : x!=0)
q24_df.drop([' None',' Other'], axis=1, inplace=True)
corr = q24_df.corr()


mask = np.zeros_like(corr, dtype=np.bool)
mask[np.triu_indices_from(mask)] = True
f, ax = plt.subplots(figsize=(15, 14))
cmap = sns.diverging_palette(150, 275, s=100, l=20, as_cmap=True)
sns.heatmap(corr, mask=mask, cmap=cmap, center=0, square=True, linewidths=.5, cbar_kws={"shrink": .5})
q25_df, q25_sub = q_list('Q25')
q25_convert = {b:a for a, b in zip(q25_sub.values, q25_df.columns)}
q25_df.rename(columns=q25_convert, inplace=True)
msno.matrix(q25_df)
print('There are {} other text answers'.format(len(other_text['Q25_OTHER_TEXT'].dropna())))
q28_df, q28_sub = q_list('Q28')
q28_df.drop(0, axis=0, inplace=True)
q28_convert = {b:a for a, b in zip(q28_sub.values, q28_df.columns)}
q28_df.rename(columns=q28_convert, inplace=True)
msno.matrix(q28_df)
import plotly.graph_objects as go

q28_ans = {}
for i in q28_df.columns[:-1]:
    q28_ans.update(q28_df[i].value_counts())

q28_cnt = pd.Series(q28_ans)

fig = go.Figure([go.Bar(x=q28_cnt.index, y=q28_cnt)])
fig.update_layout(title="ML basis Framworks Ranking")
fig.show()
y = q28_cnt

fig = go.Figure(go.Treemap(
    labels = y.index,
    parents = ['Framework'] * len(y),
    values = y
))

fig.update_layout(title = 'Treemap of Hot Frameworks')
fig.show()
q28_df = q28_df.fillna(0).apply(lambda x : x!=0)
q28_df.drop([' None', ' Other', ' Text'], axis=1, inplace=True)
ML = pd.concat([q24_df, q28_df], axis=1)
fig, ax = plt.subplots(1,1, figsize=(15, 15))
sns.heatmap(ML.corr().iloc[0:10,10:], cmap=sns.color_palette("coolwarm"), center=0,square=True, linewidths=.5, cbar_kws={"shrink": .5})
plt.show()
q19_cnt = multiple_choice['Q19'].value_counts()[:-1]

fig = go.Figure([go.Bar(x=q19_cnt.index, y=q19_cnt, marker_color='crimson')])
fig.update_layout(title="Programming Language Ranking")
fig.show()
q18_df, q18_sub = q_list('Q18')
q18_df.drop(0, axis=0, inplace=True)
q18_convert = {b:a for a, b in zip(q18_sub.values, q18_df.columns)}
q18_df.rename(columns=q18_convert, inplace=True)
msno.matrix(q18_df, color=(0.37, 0.29, 0.48))
from IPython.display import Video

Video("https://thumbs.gfycat.com/UnderstatedAnotherAntelope-mobile.mp4")
data = multiple_choice[['Q19', 'Q23']]
data.drop(0, axis=0, inplace=True)
career = data['Q23'].value_counts()

fig = go.Figure([go.Bar(x=career.index, y=career, marker_color='#F6CD8B')])
fig.update_layout(title="Distribution of ML Career")
fig.show()
data.groupby('Q23')['Q19'].value_counts().unstack().fillna(0).T[[ '< 1 years', '1-2 years', '2-3 years', 
       '3-4 years', '4-5 years', '5-10 years', '10-15 years', '20+ years']].drop(['None','Other'], axis=0)
fig = px.histogram(data.dropna(), x='Q23', y='Q23', color='Q19', template='ggplot2')
fig.update_layout()
fig.show()
data = multiple_choice[['Q19', 'Q15']]
data.drop(0, axis=0, inplace=True)

career = data['Q15'].value_counts()[[ 'I have never written code', '< 1 years', '1-2 years',
       '3-5 years', '5-10 years', '10-20 years', '20+ years' ]]

fig = go.Figure([go.Bar(x=career.index, y=career, marker_color='#A8204E')])
fig.update_layout(title="Distribution of Programming Career")
fig.show()

display(data.groupby('Q15')['Q19'].value_counts().unstack().fillna(0).T[
    [ '< 1 years', '1-2 years', '3-5 years', '5-10 years', '10-20 years', '20+ years']]
        .drop(['None','Other'], axis=0))

fig = px.histogram(data.dropna(), x='Q15', y='Q15', color='Q19', template='ggplot2')
fig.update_layout()
fig.show()
q20_df, q20_sub = q_list('Q20')
q20_df.drop(0, axis=0, inplace=True)
q20_ans = {}
for i in q20_df.columns[:-1]:
    q20_ans.update(q20_df[i].value_counts())

q20_cnt = pd.Series(q20_ans)

fig = go.Figure([go.Bar(x=q20_cnt.index, y=q20_cnt, marker_color='teal')])
fig.update_layout(title="Viz Library")
fig.show()
q34_df, q34_sub = q_list('Q34')
q34_df.drop(0,axis=0,inplace=True)
q34_ans = {}
for i in q34_df.columns[:-1]:
    q34_ans.update(q34_df[i].value_counts())

q34_cnt = pd.Series(q34_ans)
fig = go.Figure([go.Bar(x=q34_cnt.index, y=q34_cnt, marker_color='darkseagreen')])
fig.update_layout(title="Database")
fig.show()
q16_df, q16_sub = q_list('Q16')
q16_df.drop(0, axis=0, inplace=True)
q16_ans = {}
for i in q16_df.columns[:-1]:
    q16_ans.update(q16_df[i].value_counts())

q16_cnt = pd.Series(q16_ans)

fig = go.Figure([go.Bar(x=q16_cnt.index, y=q16_cnt, marker_color='mediumturquoise')])
fig.update_layout(title="IDE")
fig.show()
q17_df, q17_sub = q_list('Q17')
q17_df.drop(0, axis=0, inplace=True)

q17_ans = {}
for i in q17_df.columns[:-1]:
    q17_ans.update(q17_df[i].value_counts())

q17_cnt = pd.Series(q17_ans)

fig = go.Figure([go.Bar(x=q17_cnt.index, y=q17_cnt, marker_color='tomato')])
fig.update_layout(title="Notebook Products")
fig.show()
q5 = multiple_choice[['Q5']]
q5.drop(0, axis=0, inplace=True)
from pywaffle import Waffle

jobs = q5['Q5'].value_counts()

fig = plt.figure(
    FigureClass=Waffle, 
    rows=8,
    columns=13,
    values=jobs,
    title={'label': 'Job Distribution', 'loc': 'left'},
    labels=["{}({})".format(a, b) for a, b in zip(jobs.index, jobs) ],
    legend={'loc': 'lower left', 'bbox_to_anchor': (0, -0.4), 'ncol': len(jobs)//3,  'framealpha': 0},
    font_size=30, 
    icons = 'address-card',
    figsize=(12, 12),  
    icon_legend=True
)
jobs_educational = multiple_choice[['Q4', 'Q5']]
jobs_educational.drop(0, axis=0, inplace=True)
tmp = jobs_educational.groupby('Q5')['Q4'].value_counts()
tmp2 = jobs_educational.groupby('Q5').count()

labels = []
parents = []
values = []

values.append(len(jobs_educational))
parents.append("")
labels.append("Tot")

for a, b in zip(tmp2.values, tmp2.index):
    values.append(a[0])
    parents.append("Tot")
    labels.append(b)

    

for a, b in zip(tmp.values, tmp.index):
    values.append(a)
    parents.append(b[0])
    labels.append(b[1])
    
fig =go.Figure(go.Sunburst(
    labels=labels,
    parents=parents,
    values=values,
    branchvalues="total",
))

fig.update_layout(title="Jobs & Education",
                  margin = dict(t=0, l=0, r=0, b=0))

fig.show()
q18_df, q18_sub = q_list('Q18')
q18_df.drop(0, axis=0, inplace=True)
q18_convert = {b:a for a, b in zip(q18_sub.values, q18_df.columns)}
q18_df.rename(columns=q18_convert, inplace=True)
jobs_lang = multiple_choice[['Q5']]
jobs_lang.drop(0, axis=0, inplace=True)

jobs_lang = pd.concat([jobs_lang, q18_df], axis=1).fillna('')

job_lang_table = pd.DataFrame(index = jobs_lang['Q5'].unique())

labels = list(jobs_lang['Q5'].unique())
parents = [''] * len(jobs_lang['Q5'].unique())
values = [jobs_lang['Q5'].value_counts()[i] for i in jobs_lang['Q5'].unique()]

for i in q18_sub[:-2]:
    tmp = jobs_lang.groupby('Q5')[i].value_counts()
    for val, idx in zip(tmp, tmp.index):
        if idx[0] == '' or idx[1] == '': continue 

        labels.append(idx[1])
        parents.append(idx[0])
        values.append(val)

fig = go.Figure(go.Treemap(
    labels = labels,
    parents = parents,
    values = values,
))

fig.show()
q28_df, q28_sub = q_list('Q28')
q28_df.drop(0, axis=0, inplace=True)
q28_convert = {b:a for a, b in zip(q28_sub.values, q28_df.columns)}
q28_df.rename(columns=q28_convert, inplace=True)

jobs_lang = multiple_choice[['Q5']]
jobs_lang.drop(0, axis=0, inplace=True)

jobs_lang = pd.concat([jobs_lang, q28_df], axis=1).fillna('')

job_lang_table = pd.DataFrame(index = jobs_lang['Q5'].unique())

labels = list(jobs_lang['Q5'].unique())
parents = [''] * len(jobs_lang['Q5'].unique())
values = [jobs_lang['Q5'].value_counts()[i] for i in jobs_lang['Q5'].unique()]

for i in q28_sub[:-2]:
    tmp = jobs_lang.groupby('Q5')[i].value_counts()
    for val, idx in zip(tmp, tmp.index):
        if idx[0] == '' or idx[1] == '': continue 

        labels.append(idx[1])
        parents.append(idx[0])
        values.append(val)

fig = go.Figure(go.Treemap(
    labels = labels,
    parents = parents,
    values = values,
))

fig.show()
salary_career = multiple_choice[['Q4', 'Q10', 'Q15']].drop(0, axis=0)
salary_order = ['$0-999', 
'1,000-1,999',
'2,000-2,999',
'3,000-3,999',
'4,000-4,999',
'5,000-7,499',
'7,500-9,999',
'10,000-14,999', 
'15,000-19,999',
'20,000-24,999',
'25,000-29,999',
'30,000-39,999',
'40,000-49,999', 
'50,000-59,999',
'60,000-69,999',
'70,000-79,999',
'80,000-89,999',
'90,000-99,999',
'100,000-124,999', 
'125,000-149,999', 
'150,000-199,999', 
'200,000-249,999',
'250,000-299,999',
'300,000-500,000',
'> $500,000',]

salary_dist = salary_career['Q10'].value_counts()[salary_order]

fig = go.Figure([go.Bar(x=salary_dist.index, y=salary_dist, marker_color='#5f4b8b')])

fig.update_layout(title="Salary Distribtuion ($USD)")
fig.show()
degree_order = ['No formal education past high school',
'Some college/university study without earning a bachelor’s degree',
'Bachelor’s degree',
'Master’s degree',
'Doctoral degree',
'Professional degree']

salary_degree = salary_career.groupby('Q4')['Q10'].value_counts().unstack().fillna(0)[salary_order]
fig = go.Figure()

for i in degree_order:
    fig.add_trace(go.Line(x=salary_order, y=salary_degree.loc[i], name=i ))

fig.update_layout(title="Salary Distribtuion by Educational ($USD)")
fig.show()
fig = go.Figure()

for i in degree_order:
    fig.add_trace(go.Line(x=salary_order, y=salary_degree.loc[i] / sum(salary_degree.loc[i] ), name=i ))

fig.update_layout(title="Salary Distribtuion by Educational / Ratio ($USD)")
fig.show()
salary_est = [500, 1500, 2500, 3500, 4500, 6250, 8750, 12500, 17500, 22500, 27500, 35000, 45000, 55000, 65000, 75000, 85000, 95000, 112500, 137500, 175000, 225000, 275000, 400000, 500000]

tmp_dict = dict()
for a, b in zip(salary_order, salary_est):
    tmp_dict.update({a:b})
    
display(pd.DataFrame(pd.Series(tmp_dict)).T)
lst = []
tot, length = 0, 0
for i in degree_order:
    lst.append(sum(salary_degree.loc[i].values * salary_est )/ sum(salary_degree.loc[i]))
    tot += sum(salary_degree.loc[i].values * salary_est )
    length += sum(salary_degree.loc[i])

mean_salary = tot/length
    
fig = go.Figure([go.Bar(x=degree_order, y=lst, marker_color='#20639B')])
fig.add_trace(go.Bar(x=['Average'], y=[mean_salary], marker_color='crimson'))
fig.update_layout(title="Educational Degree & Average Salary", showlegend=False)
fig.show()
salary_degree2 = salary_career.groupby('Q15')['Q10'].value_counts().unstack().fillna(0)[salary_order]

career_order = [ 'I have never written code', '< 1 years', '1-2 years',
       '3-5 years', '5-10 years', '10-20 years', '20+ years' ]

fig = go.Figure()

for i in career_order:
    fig.add_trace(go.Line(x=salary_order, y=salary_degree2.loc[i], name=i ))

fig.update_layout(title="Salary Distribtuion by Career ($USD)")
fig.show()
career_order = [ 'I have never written code', '< 1 years', '1-2 years',
       '3-5 years', '5-10 years', '10-20 years', '20+ years' ]

fig = go.Figure()

for i in career_order:
    fig.add_trace(go.Line(x=salary_order, y=salary_degree2.loc[i] / sum(salary_degree2.loc[i] ), name=i ))

fig.update_layout(title="Salary Distribtuion by Career / Ratio ($USD)")
fig.show()
lst = []
for i in career_order:
    lst.append(sum(salary_degree2.loc[i].values * salary_est )/ sum(salary_degree2.loc[i]))
    

fig = go.Figure([go.Bar(x=career_order, y=lst, marker_color='#F3872F')])
fig.add_trace(go.Bar(x=['Average'], y=[mean_salary], marker_color='crimson'))
fig.update_layout(title="Career & Average Salary", showlegend=False)
fig.show()
salary_info = multiple_choice[['Q1', 'Q2', 'Q3', 'Q10']].drop(0, axis=0)
salary_info['age'] = salary_info['Q1'].apply(lambda x : int(x[:2]))
# salary_info['salary'] = salary_info['Q10'].map({a:b for a, b in zip(salary_degree3.columns,salary_est)})
salary_degree3 = salary_info.groupby('age')['Q10'].value_counts().unstack().fillna(0)

salary_degree3.rename(columns = {a:b for a, b in zip(salary_degree3.columns,salary_est)}, inplace=True)

fig = go.Figure()


for i in salary_degree3.index:
    x = []
    y = []
    sz = []

    for j, k in zip(salary_degree3.loc[i], salary_degree3.columns):
        sz.append(j/2.7)
        y.append(k)
        x.append(i)
    fig.add_trace(go.Scatter(
        x=x, y=y,
        mode='markers',
        marker_size=sz,
        name=i)
    )
    
fig.update_layout(title="Salary & Age (Scale)",
                  width=1200,
                  height=700,)

fig.show()
salary_degree3 = salary_info.groupby('Q1')['Q10'].value_counts().unstack().fillna(0)
fig = go.Figure()

for i in salary_info['Q1'].value_counts().sort_index().index:
    fig.add_trace(go.Line(x=salary_order, y=salary_degree3.loc[i] / sum(salary_degree3.loc[i]), name=i ))

fig.update_layout(title="Salary Distribtuion by Age / Ratio ($USD)")
fig.show()
lst = []
for i in salary_info['Q1'].value_counts().sort_index().index:
    lst.append(sum(salary_degree3.loc[i].values * salary_est )/ sum(salary_degree3.loc[i]))
    

fig = go.Figure([go.Bar(x=salary_info['Q1'].value_counts().sort_index().index, y=lst, marker_color='#FF598F')])
fig.add_trace(go.Bar(x=['Average'], y=[mean_salary], marker_color='green'))
fig.update_layout(title="Age & Average Salary", showlegend=False)
fig.show()
salary_degree4 = salary_info.groupby('Q2')['Q10'].value_counts().unstack().fillna(0)[salary_order]

fig = go.Figure()

for i in salary_info['Q2'].value_counts().index:
    fig.add_trace(go.Line(x=salary_order, y=salary_degree4.loc[i], name=i ))

fig.update_layout(title="Salary Distribtuion by Gender ($USD)")
fig.show()
fig = go.Figure()

for i in salary_info['Q2'].value_counts().index:
    fig.add_trace(go.Line(x=salary_order, y=salary_degree4.loc[i] / sum(salary_degree4.loc[i]) , name=i ))

fig.update_layout(title="Salary Distribtuion by Gender / Ratio ($USD)")
fig.show()
lst = []
for i in salary_info['Q2'].value_counts().index:
    lst.append(sum(salary_degree4.loc[i].values * salary_est )/ sum(salary_degree4.loc[i]))
    

fig = go.Figure([go.Bar(x=salary_info['Q2'].value_counts().index, y=lst, marker_color='#6A67CE')])
fig.add_trace(go.Bar(x=['Average'], y=[mean_salary], marker_color='crimson'))
fig.update_layout(title="Gender & Average Salary", showlegend=False)
fig.show()
salary_info['money'] = salary_info['Q10'].map(tmp_dict)
# salary_info['money'] = salary_info['money'].fillna(mean_salary)

salary_info2 = salary_info[:]
salary_info2 = salary_info2.dropna()

salary_gender_country = salary_info2.groupby(['Q3', 'Q2'])['money'].mean().unstack().fillna(0)
salary_gender_country['Female/Male'] = ( salary_gender_country['Female'] - salary_gender_country['Male'])/salary_gender_country['Male']  
display(salary_gender_country)

salary_gender_country['Country'] = salary_gender_country.index
fig = px.choropleth(salary_gender_country, locations='Country',
                    locationmode='country names',
                    color="Female/Male",
                    color_continuous_scale=[[0.0, "rgb(165,0,38)"],
                                            [0.1111111111111111, "rgb(215,48,39)"],
                                            [0.2222222222222222, "rgb(244,109,67)"],
                                            [0.3333333333333333, "rgb(253,174,97)"],
                                            [0.4444444444444444, "rgb(254,224,144)"],
                                            [0.5555555555555556, "rgb(224,243,248)"],
                                            [0.6666666666666666, "rgb(171,217,233)"],
                                            [0.7777777777777778, "rgb(116,173,209)"],
                                            [0.8888888888888888, "rgb(69,117,180)"],
                                            [1.0, "rgb(49,54,149)"]],
                     range_color=[-1,1]
                   )
                    
fig.update_layout(title="Gender wage inequality (difference percentage)")
fig.show()

salary_degree5 = salary_info.groupby('Q3')['Q10'].value_counts().unstack().fillna(0)[salary_order]

lst = []
idx = list(salary_info['Q3'].value_counts().index)
for i in idx:
    lst.append(sum(salary_degree5.loc[i].values * salary_est )/ sum(salary_degree5.loc[i]))

idx.append('AVG')
lst.append(mean_salary)
    
colors = ['lightslategray',] * len(lst)
colors[41] = 'crimson'
    
fig = go.Figure([go.Bar(y=[x for _,x in sorted(zip(lst, idx))], x=sorted(lst), 
                        marker_color=colors,
                        orientation='h'
                       )])

fig.update_layout(title="Country & Average Salary", 
                  width=600,
                  height=1200,
                  showlegend=False)
fig.show()
lst = []
idx = list(salary_info['Q3'].value_counts().index)
for i in idx:
    lst.append(sum(salary_degree5.loc[i].values * salary_est )/ sum(salary_degree5.loc[i]))

country_salary = pd.DataFrame.from_dict( {'Country':idx, 'Salary':lst})

fig = px.choropleth(country_salary, locations='Country',
                    locationmode='country names',
                    color="Salary",
                    color_continuous_scale=[[0.0, "rgb(165,0,38)"],
                [0.1111111111111111, "rgb(215,48,39)"],
                [0.2222222222222222, "rgb(244,109,67)"],
                [0.3333333333333333, "rgb(253,174,97)"],
                [0.4444444444444444, "rgb(254,224,144)"],
                [0.5555555555555556, "rgb(224,243,248)"],
                [0.6666666666666666, "rgb(171,217,233)"],
                [0.7777777777777778, "rgb(116,173,209)"],
                [0.8888888888888888, "rgb(69,117,180)"],
                [1.0, "rgb(49,54,149)"]])
fig.update_layout(title="Average Salary Map")
fig.show()
q12_df, q12_sub = q_list('Q12')
q12_df.drop(0, axis=0, inplace=True)
q12_ans = {}
for i in q12_df.columns[:-1]:
    q12_ans.update(q12_df[i].value_counts())

q12_cnt = pd.Series(q12_ans)

fig = go.Figure([go.Bar(x=q12_cnt.index, y=q12_cnt, marker_color='skyblue')])
fig.update_layout(title="Favorite Source to Study ML")
fig.show()
q13_df, q13_sub = q_list('Q13')
q13_df.drop(0, axis=0, inplace=True)
q13_ans = {}
for i in q13_df.columns[:-1]:
    q13_ans.update(q13_df[i].value_counts())

q13_cnt = pd.Series(q13_ans)

fig = go.Figure([go.Bar(x=q13_cnt.index, y=q13_cnt, marker_color='salmon')])
fig.update_layout(title="Best Place to Study ML")
fig.show()
q11 = multiple_choice[['Q1', 'Q2', 'Q3', 'Q11']].drop(0, axis=0)

paid_order = ['$0 (USD)', '$1-$99', '$100-$999', '$1000-$9,999',  '$10,000-$99,999', '> $100,000 ($USD)']
paid_change = [0, 10, 100, 1000, 10000, 100000]

q11_cnt = q11['Q11'].value_counts()[paid_order]

fig = go.Figure([go.Bar(x=q11_cnt.index, y=q11_cnt, marker_color='#0086FF')])
fig.update_layout(title="Distribution of money spent for five years")
fig.show()
q11_pr = q11.groupby('Q2')['Q11'].value_counts().unstack().fillna(0)[paid_order]

fig = go.Figure()

for i in q11_pr.index:
    fig.add_trace(go.Line(x=paid_order, y=q11_pr.loc[i] / sum(q11_pr.loc[i]), name=i ))

fig.update_layout(title="Spending Money for ML (Gender)")
fig.show()
q11_pr2 = q11.groupby('Q1')['Q11'].value_counts().unstack().fillna(0)[paid_order]

fig = go.Figure()

for i in q11_pr2.index:
    fig.add_trace(go.Line(x=paid_order, y=q11_pr2.loc[i] / sum(q11_pr2.loc[i]), name=i ))

fig.update_layout(title="Spending Money for ML (Age)")
fig.show()