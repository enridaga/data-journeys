
import numpy as np 
import pandas as pd 
import os
from plotly import tools
import matplotlib.pyplot as plt
import seaborn as sns
#%matplotlib inline
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
import plotly as py
import plotly.graph_objs as go
import plotly.express as px
import warnings
warnings.filterwarnings("ignore")
init_notebook_mode(connected=True)
pd.set_option('display.max_columns', None)
print(os.listdir("../input"))
pd.set_option('display.max_columns', None)
import operator
import numpy
mcq=pd.read_csv('/kaggle/input/kaggle-survey-2019/multiple_choice_responses.csv')
question=pd.read_csv('/kaggle/input/kaggle-survey-2019/questions_only.csv')
text_response=pd.read_csv('/kaggle/input/kaggle-survey-2019/other_text_responses.csv')
col=question.columns
for i in range(question.shape[1]):
    
    print(i,question[col[i]])
c= pd.value_counts(mcq.Q5)
d = pd.DataFrame(data=c)
d.columns=['count']
d=d.iloc[:9,:].sort_values('count',ascending=True)
d['count'] = pd.to_numeric(d['count'], errors='coerce')

data=[go.Bar(
            x=d["count"],
            y=d.index,
             marker=dict(
             color=['#8e7bbb','#8e7bbb','#8e7bbb','#8e7bbb','#8e7bbb','#8e7bbb','#8e7bbb','#8e7bbb','#142270'],
             line=dict(
            color=['#8e7bbb','#8e7bbb','#8e7bbb','#8e7bbb','#8e7bbb','#8e7bbb','#8e7bbb','#8e7bbb','#142270'],
            width=1),
                        ),
            orientation='h' )]
layout = go.Layout(
    height=500,
    autosize=True,
    title='Job role wise contribution in the survey',
    hovermode='closest',
    xaxis=dict(title='', ticklen=5, zeroline=False, gridwidth=2, domain=[0.2, 1]),
    yaxis=dict(title='', ticklen=5, gridwidth=10),
    
)

fig = go.Figure(data=data,layout=layout)
py.offline.iplot(fig, filename='horizontal-bar')
df=mcq[mcq['Q5']=='Data Scientist']
country_dist=df['Q3'].value_counts()
fig = px.choropleth(country_dist.values, locations=country_dist.index,
                    locationmode='country names',
                    color=country_dist.values,
                    color_continuous_scale=px.colors.sequential.OrRd)
fig.update_layout(title="Countrywise Distribution of data scientists")
fig.show()
gender=df['Q2'].value_counts()
age=['18-21','22-24','25-29','30-34','35-39','40-44','45-49','50-54','55-59','60-69','70+']
value=[218,616,1134,871,529,292,157,132,72,51,13]
trace0=go.Bar(x=gender.index,
              y=gender.values,
              name='',
             marker=dict(
             color='#03396c '),showlegend=False)
trace1=go.Scatter(x=age,
                 y=value,
                 name=' ',
                   marker=dict(
             color='#2c2f33'),showlegend=False)
colors = ['#1a472a', '#d62d20', '#5d5d5d','#8c564b','#008080','#800000','#808000'] 
counts = df['Q4'].value_counts(sort=True)
labels = counts.index
values = counts.values

trace2= go.Pie(labels=labels, values=values, marker=dict(colors=colors))
fig = tools.make_subplots(rows=2, cols=2, specs=[[{"type": "bar"},{"type": "pie", "rowspan": 2}],
           [ {"type": "scatter"},None]],
                          subplot_titles=('Gender','Highest Education','Age'))

fig.append_trace(trace2, 1, 2)
fig.append_trace(trace0, 1, 1)
fig.append_trace(trace1, 2, 1)

fig['layout'].update( height=700,width=1200,margin=go.layout.Margin(
        l=50,
        r=50,
        b=100,
        t=100,
        pad=4
    ))
py.offline.iplot(fig)



mooc={}
for i in range(12):
    value=df['Q13_Part_'+str(i+1)].value_counts().values[0]
    text=df['Q13_Part_'+str(i+1)].value_counts().index[0]
    mooc[text]=value
mooc=dict( sorted(mooc.items(), key=operator.itemgetter(1)))
    
pub={}
for i in range(12):
    value=df['Q12_Part_'+str(i+1)].value_counts().values[0]
    text=df['Q12_Part_'+str(i+1)].value_counts().index[0]
    pub[text]=value
pub=dict( sorted(pub.items(), key=operator.itemgetter(1)))
    
trace0=go.Bar(x=list(mooc.values()),
              y=list(mooc.keys()),
              name='',
             marker=dict(
             color='#03396c '),showlegend=False,orientation='h')
trace1=go.Bar(x=list(pub.values()),
              y=list(pub.keys()),
              name='',
             marker=dict(
             color='#03396c '),showlegend=False,orientation='h')

fig = tools.make_subplots(rows=2, cols=1, specs=[[{}],[{}]],
                          subplot_titles=('MOOC(Massive Open Online Courses)','Media Source'))
fig.append_trace(trace0, 1, 1)
fig.append_trace(trace1, 2, 1)



fig['layout'].update( height=700,width=900,margin=go.layout.Margin(
        l=50,
        r=50,
        b=50,
        t=50,
        pad=4
    ))
py.offline.iplot(fig)
x=[]
y=[]
for i in range(8):
   
    value=df['Q9_Part_'+str(i+1)].value_counts().values[0]
    text=df['Q9_Part_'+str(i+1)].value_counts().index[0]
    x.append(text)
    y.append(value)
x_new=['Analyze data for business decision','Build/run data infrastructure','Build prototypes to add ML to new areas','Build/run ML service that improves product','Iterartion to improve existing model','Reserach to advance stste-of-the-art','None','Other']   
x_new=numpy.array(x_new)
y=numpy.array(y)
inds = y.argsort()
sorted_xnew = x_new[inds]
sorted_y=y[inds]

trace0 = go.Bar(
    x=sorted_xnew[::-1],
    y=sorted_y[::-1],
    marker=dict(
        color='#420666'
    ),
    opacity=0.6  
)

data = [trace0]
layout = go.Layout(
    title='Activities that make up most of their time at work',
)

fig = go.Figure(data=data, layout=layout)
fig.update_layout(
    margin=dict(l=50, r=50, t=100, b=100),height=500,width=700
)
py.offline.iplot(fig)
years=['<1 years','1-2 years','3-5 years','5-10 years','10-20 years','20+ years']
values=[290,700,1124,741,369,154]

tool=df['Q14'].value_counts().index
value2=df['Q14'].value_counts().values

vis=[]
value3=[]
for i in range(12):
    value=df['Q20_Part_'+str(i+1)].value_counts().values[0]
    text=df['Q20_Part_'+str(i+1)].value_counts().index[0]
    vis.append(text)
    value3.append(value)


trace0 = go.Bar(
    x = years,
    y =values, showlegend = False,
    marker=dict(
        color='#008080'
    )
)
tool=['Local development environments','Cloud-based data software & APIs','Basic statistical software','Advanced statistical software','Business intelligence software']
trace1 = go.Bar(
    x = tool,
    y =value2, showlegend = False,
    marker=dict(
        color='#006666'
    )
)
trace2 = go.Scatter(
    x = vis,
    y =value3,showlegend = False,
    mode = 'lines+markers',
  
    line= dict( color= "#004c4c")
)


fig = tools.make_subplots(rows=2, cols=2, specs=[[{"type": "bar"},{"type": "bar"}],
           [ {"type": "scatter","colspan":2},None]],
                          subplot_titles=('Years spent writing code to Analyze data','Tools used to analyze ','Vis library'))

fig.append_trace(trace0, 1, 1)
fig.append_trace(trace1, 1, 2)
fig.append_trace(trace2, 2, 1)

fig['layout'].update( height=700,width=900,margin=go.layout.Margin(
        l=50,
        r=50,
        b=100,
        t=100,
        pad=4
    ))
py.offline.iplot(fig)



ml_exp=pd.concat([pd.crosstab(df['Q17_Part_1'],df['Q23']),pd.crosstab(df['Q17_Part_2'],df['Q23']),pd.crosstab(df['Q17_Part_3'],df['Q23']),pd.crosstab(df['Q17_Part_4'],df['Q23']),pd.crosstab(df['Q17_Part_5'],df['Q23']),pd.crosstab(df['Q17_Part_6'],df['Q23']),pd.crosstab(df['Q17_Part_7'],df['Q23']),pd.crosstab(df['Q17_Part_8'],df['Q23']),pd.crosstab(df['Q17_Part_9'],df['Q23']),pd.crosstab(df['Q17_Part_10'],df['Q23']),pd.crosstab(df['Q17_Part_11'],df['Q23']),pd.crosstab(df['Q17_Part_12'],df['Q23'])])
ml_exp=ml_exp[['< 1 years','1-2 years','2-3 years','3-4 years','4-5 years','5-10 years','10-15 years']]
ml_exp=ml_exp.fillna(0)
trace1 = go.Bar(
    x=ml_exp.index,
    y=ml_exp['< 1 years'].values,
    name='< 1 years',
    marker=dict(
    color="#66545e")
)
trace2 = go.Bar(
    x=ml_exp.index,
    y=ml_exp['1-2 years'].values,
    name='1-2 years',
    marker=dict(
    color='#15b2b3')
)
trace3 = go.Bar(
    x=ml_exp.index,
    y=ml_exp['2-3 years'].values,
    name='2-3 years',
    marker=dict(
    color='#ff920c')
)
trace4 = go.Bar(
    x=ml_exp.index,
    y=ml_exp['3-4 years'].values,
    name='3-4 years',
    marker=dict(
    color='#f05e68')
)
trace5 = go.Bar(
    x=ml_exp.index,
    y=ml_exp['4-5 years'].values,
    name='4-5 years',
    marker=dict(
    color='#00a3e0')
)
trace6 = go.Bar(
    x=ml_exp.index,
    y=ml_exp['5-10 years'].values,
    name='5-10 years',
    marker=dict(
    color='#4d0e20')
)
trace7 = go.Bar(
    x=ml_exp.index,
    y=ml_exp['10-15 years'].values,
    name='10-15 years',
    marker=dict(
    color='#b35a00')
)


data = [trace1, trace2,trace3,trace4,trace5,trace6,trace7]
layout = go.Layout(
    barmode='group',height=600,width=1200,title=' Hosted notebook product used against years of experience',yaxis_title='Number of people',xaxis_title=''
)

fig = go.Figure(data=data, layout=layout)
py.offline.iplot(fig, filename='grouped-bar')
ml_exp=pd.concat([pd.crosstab(df['Q24_Part_1'],df['Q23']),pd.crosstab(df['Q24_Part_2'],df['Q23']),pd.crosstab(df['Q24_Part_3'],df['Q23']),pd.crosstab(df['Q24_Part_4'],df['Q23']),pd.crosstab(df['Q24_Part_5'],df['Q23']),pd.crosstab(df['Q24_Part_6'],df['Q23']),pd.crosstab(df['Q24_Part_7'],df['Q23']),pd.crosstab(df['Q24_Part_8'],df['Q23']),pd.crosstab(df['Q24_Part_9'],df['Q23']),pd.crosstab(df['Q24_Part_10'],df['Q23']),pd.crosstab(df['Q24_Part_11'],df['Q23']),pd.crosstab(df['Q24_Part_12'],df['Q23'])])
ml_exp=ml_exp[['< 1 years','1-2 years','2-3 years','3-4 years','4-5 years','5-10 years','10-15 years']]
ml_exp=ml_exp.fillna(0)
trace1 = go.Bar(
    x=ml_exp.index,
    y=ml_exp['< 1 years'].values,
    name='< 1 years',
    marker=dict(
    color="#66545e")
)
trace2 = go.Bar(
    x=ml_exp.index,
    y=ml_exp['1-2 years'].values,
    name='1-2 years',
    marker=dict(
    color='#15b2b3')
)
trace3 = go.Bar(
    x=ml_exp.index,
    y=ml_exp['2-3 years'].values,
    name='2-3 years',
    marker=dict(
    color='#ff920c')
)
trace4 = go.Bar(
    x=ml_exp.index,
    y=ml_exp['3-4 years'].values,
    name='3-4 years',
    marker=dict(
    color='#f05e68')
)
trace5 = go.Bar(
    x=ml_exp.index,
    y=ml_exp['4-5 years'].values,
    name='4-5 years',
    marker=dict(
    color='#00a3e0')
)
trace6 = go.Bar(
    x=ml_exp.index,
    y=ml_exp['5-10 years'].values,
    name='5-10 years',
    marker=dict(
    color='#4d0e20')
)
trace7 = go.Bar(
    x=ml_exp.index,
    y=ml_exp['10-15 years'].values,
    name='10-15 years',
    marker=dict(
    color='#b35a00')
)


data = [trace1, trace2,trace3,trace4,trace5,trace6,trace7]
layout = go.Layout(
    barmode='group',height=600,width=1200,title='ML algo used on regular basis against years of experience',yaxis_title='Number of people',xaxis_title=''
)

fig = go.Figure(data=data, layout=layout)
py.offline.iplot(fig, filename='grouped-bar')
text_response['count'] = 1
text_response['ML_algo'] = text_response['Q24_OTHER_TEXT'].str.lower()
text_response.drop(0)[['ML_algo','count']].groupby('ML_algo').sum()[['count']].sort_values('count', ascending=False)

# Create wordcloud
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator
plt.figure(figsize=[15,8])

# Create and generate a word cloud image:
ide_words = ' '.join(text_response['ML_algo'].drop(0).dropna().values)
wordcloud = WordCloud(colormap="tab10",
                      width=1200,
                      height=480,
                      normalize_plurals=False,
                      background_color="white",
                      random_state=5).generate(ide_words)

# Display the generated image:
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")
plt.show()
## tagging practitioner types
df["DS_field"] = "ML"

df.loc[~( df.Q24_Part_7.isna() & df.Q24_Part_8.isna()), "DS_field"] = "Computer Vision"
df.loc[~( df.Q24_Part_9.isna() & df.Q24_Part_10.isna()), "DS_field"] = "NLP"
df_cv=df[df['DS_field']=='Computer Vision']
df_nlp=df[df['DS_field']=='NLP']

cv_name=[]
cv_count=[]
for i in range(7):
    value=df_cv['Q26_Part_'+str(i+1)].value_counts().values[0]
    text=df_cv['Q26_Part_'+str(i+1)].value_counts().index[0]
    cv_name.append(text)
    cv_count.append(value)
cv_new=['General purpose image/video tools','Image segmentation methods','Object detection methods','Image classification and other general purpose networks','Generative Networks','None','Other']
cv_new=numpy.array(cv_new)
cv_count=numpy.array(cv_count)
inds = cv_count.argsort()
sorted_cv_new = cv_new[inds]
sorted_y=cv_count[inds]
    
nlp={}
for i in range(6):
    value=df['Q27_Part_'+str(i+1)].value_counts().values[0]
    text=df['Q27_Part_'+str(i+1)].value_counts().index[0]
    nlp[text]=value
nlp=dict( sorted(nlp.items(), key=operator.itemgetter(1)))
    
trace0=go.Bar(x=sorted_y,
              y=sorted_cv_new,
              name='',
             marker=dict(
             color='#03396c '),showlegend=False,orientation='h')
trace1=go.Bar(x=list(nlp.values()),
              y=list(nlp.keys()),
              name='',
             marker=dict(
             color='#03396c '),showlegend=False,orientation='h')

fig = tools.make_subplots(rows=2, cols=1, specs=[[{}],[{}]],
                          subplot_titles=('Computer Vision Methods(Image data)','NLP Methods(text data)'))
fig.append_trace(trace0, 1, 1)
fig.append_trace(trace1, 2, 1)



fig['layout'].update( height=700,width=900,margin=go.layout.Margin(
        l=100,
        r=50,
        b=50,
        t=50,
        pad=4
    ))
py.offline.iplot(fig)
org_exp=pd.crosstab(df['Q6'],df['Q23'])
org_exp=org_exp[['< 1 years','1-2 years','2-3 years','3-4 years','4-5 years','5-10 years','10-15 years']]

trace1 = go.Bar(
    x=org_exp.columns,
    y=org_exp[org_exp.index=='0-49 employees'].values[0],
    name='0-49 employees',
    marker=dict(
    color="#66545e")
)
trace2 = go.Bar(
    x=org_exp.columns,
    y=org_exp[org_exp.index=='50-249 employees'].values[0],
    name='50-249 employees',
    marker=dict(
    color='#15b2b3')
)
trace3 = go.Bar(
    x=org_exp.columns,
    y=org_exp[org_exp.index=='250-999 employees'].values[0],
    name='250-999 employees',
    marker=dict(
    color='#ff920c')
)
trace4 = go.Bar(
    x=org_exp.columns,
    y=org_exp[org_exp.index=='1000-9,999 employees'].values[0],
    name='1000-9,999 employees',
    marker=dict(
    color='#f05e68')
)
trace5 = go.Bar(
    x=org_exp.columns,
    y=org_exp[org_exp.index=='> 10,000 employees'].values[0],
    name='> 10,000 employees',
    marker=dict(
    color='#00a3e0')
)

data = [trace1, trace2,trace3,trace4,trace5]
layout = go.Layout(
    barmode='group',height=600,width=900,title='Company size according to years of experience',yaxis_title='Number of people',xaxis_title='Years of experience'
)

fig = go.Figure(data=data, layout=layout)
py.offline.iplot(fig, filename='grouped-bar')
comp_dw=pd.crosstab(df['Q6'],df['Q7'])
comp_dw=comp_dw[['0','1-2','3-4','5-9','10-14','15-19','20+']]
trace1 = go.Bar(
    x=comp_dw.columns,
    y=comp_dw[comp_dw.index=='0-49 employees'].values[0],
    name='0-49 employees',
    marker=dict(
    color="#4b3832")
)
trace2 = go.Bar(
    x=comp_dw.columns,
    y=comp_dw[comp_dw.index=='50-249 employees'].values[0],
    name='50-249 employees',
    marker=dict(
    color='#854442')
)
trace3 = go.Bar(
    x=comp_dw.columns,
    y=comp_dw[comp_dw.index=='250-999 employees'].values[0],
    name='250-999 employees',
    marker=dict(
    color='#66545e')
)
trace4 = go.Bar(
    x=comp_dw.columns,
    y=comp_dw[comp_dw.index=='1000-9,999 employees'].values[0],
    name='1000-9,999 employees',
    marker=dict(
    color='#3c2f2f')
)
trace5 = go.Bar(
    x=comp_dw.columns,
    y=comp_dw[comp_dw.index=='> 10,000 employees'].values[0],
    name='> 10,000 employees',
    marker=dict(
    color='#be9b7b')
)

data = [trace1, trace2,trace3,trace4,trace5]
layout = go.Layout(
    barmode='stack',height=600,width=900,title='',plot_bgcolor='#fffff4',xaxis_title='No. of people responsible for DS workloads',yaxis_title='Numbe rof people'
)

fig = go.Figure(data=data, layout=layout)

py.offline.iplot(fig, filename='stacked-bar')
org_exp=pd.concat([pd.crosstab(df['Q32_Part_1'],df['Q6']),pd.crosstab(df['Q32_Part_2'],df['Q6']),pd.crosstab(df['Q32_Part_3'],df['Q6']),pd.crosstab(df['Q32_Part_4'],df['Q6']),pd.crosstab(df['Q32_Part_5'],df['Q6']),pd.crosstab(df['Q32_Part_6'],df['Q6']),pd.crosstab(df['Q32_Part_7'],df['Q6']),pd.crosstab(df['Q32_Part_7'],df['Q6']),pd.crosstab(df['Q32_Part_8'],df['Q6']),pd.crosstab(df['Q32_Part_9'],df['Q6']),pd.crosstab(df['Q32_Part_10'],df['Q6']),pd.crosstab(df['Q32_Part_11'],df['Q6']),pd.crosstab(df['Q32_Part_12'],df['Q6'])])
#org_exp=org_exp[['< 1 years','1-2 years','2-3 years','3-4 years','4-5 years','5-10 years','10-15 years']]

trace1 = go.Bar(
    x=org_exp.index,
    y=org_exp['0-49 employees'].values,
    name='0-49 employees',
    marker=dict(
    color="#66545e")
)

trace2 = go.Bar(
    x=org_exp.index,
    y=org_exp['50-249 employees'].values,
    name='50-249 employees',
    marker=dict(
    color='#15b2b3')
)
trace3 = go.Bar(
    x=org_exp.index,
    y=org_exp['250-999 employees'].values,
    name='250-999 employees',
    marker=dict(
    color='#ff920c')
)
trace4 = go.Bar(
    x=org_exp.index,
    y=org_exp['1000-9,999 employees'].values,
    name='1000-9,999 employees',
    marker=dict(
    color='#f05e68')
)
trace5 = go.Bar(
    x=org_exp.index,
    y=org_exp['> 10,000 employees'].values,
    name='> 10,000 employees',
    marker=dict(
    color='#00a3e0')
)

data = [trace1, trace2,trace3,trace4,trace5]
layout = go.Layout(
    barmode='group',height=600,width=1200,title='ML product used vs Company size',yaxis_title='Number of people',xaxis_title=''
)

fig = go.Figure(data=data, layout=layout)
py.offline.iplot(fig, filename='grouped-bar')
edu_sub=df[(df['Q4']=='Master’s degree') | (df['Q4']=='Bachelor’s degree') ]
edu_comp=pd.crosstab(edu_sub['Q4'],edu_sub['Q10'])
edu_comp=edu_comp[['$0-999','1,000-1,999','2,000-2,999','3,000-3,999','4,000-4,999','5,000-7,499','7,500-9,999','10,000-14,999','15,000-19,999','20,000-24,999','25,000-29,999','30,000-39,999','40,000-49,999','50,000-59,999','60,000-69,999','70,000-79,999','80,000-89,999','90,000-99,999','100,000-124,999','125,000-149,999','150,000-199,999','200,000-249,999','250,000-299,999','250,000-299,999','> $500,000']]

trace1 = go.Bar(
    x=edu_comp.columns,
    y=edu_comp[edu_comp.index=='Bachelor’s degree'].values[0],
    name='Bachelor’s degree',
    marker=dict(
    color="#1a472a")
)
trace2 = go.Bar(
    x=edu_comp.columns,
    y=edu_comp[edu_comp.index=='Master’s degree'].values[0],
    name='Master’s degree',
    marker=dict(
    color='#808080')
)

data = [trace1, trace2]
layout = go.Layout(
    barmode='group',height=600,width=1000,title='Salary according to highest education',xaxis_title='Salary',yaxis_title='Number of people'
)

fig = go.Figure(data=data, layout=layout)
py.offline.iplot(fig, filename='grouped-bar')
comp_exp=pd.crosstab(df['Q23'],df['Q10'])
comp_exp=comp_exp[['$0-999','1,000-1,999','2,000-2,999','3,000-3,999','4,000-4,999','5,000-7,499','7,500-9,999','10,000-14,999','15,000-19,999','20,000-24,999','25,000-29,999','30,000-39,999','40,000-49,999','50,000-59,999','60,000-69,999','70,000-79,999','80,000-89,999','90,000-99,999','100,000-124,999','125,000-149,999','150,000-199,999','200,000-249,999','250,000-299,999','250,000-299,999','> $500,000']]

trace0 = go.Scatter(
    x = comp_exp.columns,
    y = comp_exp[comp_exp.index=='< 1 years'].values[0],
    mode = 'lines+markers',
    name = '< 1 years',
    line = dict(
        color = "#5d5d5d")
)
trace1 = go.Scatter(
    x = comp_exp.columns,
    y = comp_exp[comp_exp.index=='1-2 years'].values[0],
    mode = 'lines+markers',
    name = '1-2 years',
    line= dict( color= "#c9df8a")
)
trace2 = go.Scatter(
    x = comp_exp.columns,
    y = comp_exp[comp_exp.index=='2-3 years'].values[0],
    mode = 'lines+markers',
    name = '2-3 years',
    line= dict( color= "#77ab59")
)
trace3 = go.Scatter(
    x = comp_exp.columns,
    y = comp_exp[comp_exp.index=='3-4 years'].values[0],
    mode = 'lines+markers',
    name = '3-4 years',
    line= dict( color= "#36802d")
)
trace4 = go.Scatter(
    x = comp_exp.columns,
    y = comp_exp[comp_exp.index=='4-5 years'].values[0],
    mode = 'lines+markers',
    name = '4-5 years',
    line= dict( color= "#234d20")
)
trace5 = go.Scatter(
    x = comp_exp.columns,
    y = comp_exp[comp_exp.index=='5-10 years'].values[0],
    mode = 'lines+markers',
    name = '5-10 years',
    line= dict( color= "#a47c48")
)
trace6 = go.Scatter(
    x = comp_exp.columns,
    y = comp_exp[comp_exp.index=='10-15 years'].values[0],
    mode = 'lines+markers',
    name = '10-15 years',
    line= dict( color= "#845422")
)
trace7 = go.Scatter(
    x = comp_exp.columns,
    y = comp_exp[comp_exp.index=='20+ years'].values[0],
    mode = 'lines+markers',
    name = '20+ years',
    line= dict( color= "#000000")
)



data=[trace0,trace1,trace2,trace3,trace4,trace5,trace6,trace7]
layout = dict(title = 'Salary based on years of experience',
              xaxis = dict(title = 'Salary group'),
              
              margin=go.layout.Margin(
        l=50,
        r=50,
        b=200,
        t=100,
        pad=4
    ),height=600,width=600,paper_bgcolor ='#aaaaaa'
              )

fig = dict(data=data, layout=layout)

py.offline.iplot(fig)
new_range=[]
for salary in df['Q10']:
    if salary in ['$0-999','1,000-1,999','2,000-2,999','3,000-3,999','4,000-4,999','5,000-7,499','7,500-9,999','10,000-14,999','15,000-19,999','20,000-24,999']:
        new_range.append('Low Income Range')
    elif salary in ['25,000-29,999','30,000-39,999','40,000-49,999','50,000-59,999','60,000-69,999','70,000-79,999','80,000-89,999','90,000-99,999']:
        new_range.append('Mid Income Range')
    
    else :
        new_range.append('High Income Range')
        
df['salary_range']=new_range
df_sub=pd.crosstab(df['salary_range'],df['Q8'])
df_sub.columns=['do not know','do not use ML','Exploring ML methods','Well established ML methods','Recently statrted using ML methods','use ML methods for getting insights']
trace1 = go.Bar(
    x=df_sub.columns,
    y=df_sub[df_sub.index=='Low Income Range'].values[0],
    name='Low Income Range',
    marker=dict(
    color="#00a0b0")
)
trace2 = go.Bar(
    x=df_sub.columns,
    y=df_sub[df_sub.index=='Mid Income Range'].values[0],
    name='Mid Income Range',
    marker=dict(
    color='#4f372d')
)
trace3 = go.Bar(
    x=df_sub.columns,
    y=df_sub[df_sub.index=='High Income Range'].values[0],
    name='High Income Range',
    marker=dict(
    color='#cc2a36')
)
data = [trace1, trace2,trace3]
layout = go.Layout(
    barmode='group',height=600,width=700,title='Salary Range vs state of ML in the company',xaxis_title='State of DS in the Company',yaxis_title='Number of people'
)

fig = go.Figure(data=data, layout=layout)
py.offline.iplot(fig, filename='grouped-bar')

# df_cv=df[df['DS_field']=='NLP']
# cv_sal_exp=pd.crosstab(df_cv['salary_range'],df['Q23'])
# cv_sal_exp=cv_sal_exp[['< 1 years','1-2 years','2-3 years','3-4 years','4-5 years','5-10 years','10-15 years']]

df_ml=df[df['DS_field']=='ML']
ml_sal_exp=pd.crosstab(df_ml['salary_range'],df['Q23'])
ml_sal_exp=ml_sal_exp[['< 1 years','1-2 years','2-3 years','3-4 years','4-5 years','5-10 years','10-15 years']]
sal_field=pd.crosstab(df['salary_range'],df['DS_field'])
trace1 = go.Bar(
    x=sal_field.columns,
    y=sal_field[sal_field.index=='Low Income Range'].values[0],
    name='Low Income Range',
    marker=dict(
    color="#00a0b0"),showlegend=False
)
trace2 = go.Bar(
    x=sal_field.columns,
    y=sal_field[sal_field.index=='Mid Income Range'].values[0],
    name='Mid Income Range',
    marker=dict(
    color='#4f372d'),showlegend=False
)
trace3 = go.Bar(
    x=sal_field.columns,
    y=sal_field[sal_field.index=='High Income Range'].values[0],
    name='High Income Range',
    marker=dict(
    color='#cc2a36'),showlegend=False
)


trace4 = go.Bar(
    x=ml_sal_exp.columns,
    y=ml_sal_exp[ml_sal_exp.index=='Low Income Range'].values[0],
    name='Low Income Range',
    marker=dict(
    color="#00a0b0")
)
trace5 = go.Bar(
    x=ml_sal_exp.columns,
    y=ml_sal_exp[ml_sal_exp.index=='Mid Income Range'].values[0],
    name='Mid Income Range',
    marker=dict(
    color='#4f372d')
)
trace6 = go.Bar(
    x=ml_sal_exp.columns,
    y=ml_sal_exp[ml_sal_exp.index=='High Income Range'].values[0],
    name='High Income Range',
    marker=dict(
    color='#cc2a36')
)
# trace7 = go.Bar(
#     x=cv_sal_exp.columns,
#     y=cv_sal_exp[cv_sal_exp.index=='Low Income Range'].values[0],
#     name='Low Income Range',
#     marker=dict(
#     color="#00a0b0")
# )
# trace8 = go.Bar(
#     x=cv_sal_exp.columns,
#     y=cv_sal_exp[cv_sal_exp.index=='Mid Income Range'].values[0],
#     name='Mid Income Range',
#     marker=dict(
#     color='#4f372d')
# )
# trace9 = go.Bar(
#     x=cv_sal_exp.columns,
#     y=cv_sal_exp[cv_sal_exp.index=='High Income Range'].values[0],
#     name='High Income Range',
#     marker=dict(
#     color='#cc2a36')
# )


fig = tools.make_subplots(rows=2, cols=1, specs=[[{}],[{}]],
                          subplot_titles=('Salary Range vs DS Field','Expereince vs Salary for people in ML'))
fig.append_trace(trace1, 1, 1)
fig.append_trace(trace2, 1, 1)
fig.append_trace(trace3, 1, 1)
fig.append_trace(trace4, 2, 1)
fig.append_trace(trace5, 2, 1)
fig.append_trace(trace6, 2, 1)
# fig.append_trace(trace7, 3, 1)
# fig.append_trace(trace8, 3, 1)
# fig.append_trace(trace9, 3, 1)


fig['layout'].update( height=600,width=800,margin=go.layout.Margin(
        l=50,
        r=50,
        b=50,
        t=50,
        pad=4
    ))
py.offline.iplot(fig)