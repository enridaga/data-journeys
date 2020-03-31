
# import the necessary libraries
import numpy as np 
import pandas as pd 

# Visualisation libraries
import matplotlib.pyplot as plt
### matplotlib inline
import seaborn as sns
sns.set()
from plotly.offline import init_notebook_mode, iplot 
import plotly.graph_objs as go
import plotly.offline as py
import pycountry
py.init_notebook_mode(connected=True)
import folium 
from folium import plugins

# Graphics in retina format 
### config InlineBackend.figure_format = 'retina' 

# Increase the default plot size and set the color scheme
plt.rcParams['figure.figsize'] = 8, 5
#plt.rcParams['image.cmap'] = 'viridis'


import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# palette of colors to be used for plots
colors = ["steelblue","dodgerblue","lightskyblue","powderblue","cyan","deepskyblue","cyan","darkturquoise","paleturquoise","turquoise"]


# Disable warnings in Anaconda
import warnings
warnings.filterwarnings('ignore')
# Importing the 2017,2018 and 2019 survey dataset

#Importing the 2019 Dataset
df_2019 = pd.read_csv('../input/kaggle-survey-2019/multiple_choice_responses.csv')
df_2019.columns = df_2019.iloc[0]
df_2019=df_2019.drop([0])

#Importing the 2018 Dataset
df_2018 = pd.read_csv('../input/kaggle-survey-2018/multipleChoiceResponses.csv')
df_2018.columns = df_2018.iloc[0]
df_2018=df_2018.drop([0])

#Importing the 2017 Dataset
df_2017=pd.read_csv('../input/kaggle-survey-2017/multipleChoiceResponses.csv',encoding='ISO-8859-1')


# Helper functions

def return_count(data,question_part):
    """Counts occurences of each value in a given column"""
    counts_df = data[question_part].value_counts().to_frame()
    return counts_df

def return_percentage(data,question_part):
    """Calculates percent of each value in a given column"""
    total = data[question_part].count()
    counts_df= data[question_part].value_counts().to_frame()
    percentage_df = (counts_df*100)/total
    return percentage_df


    
def plot_graph(data,question,title,x_axis_title,y_axis_title):
    """ plots a percentage bar graph"""
    df = return_percentage(data,question)
    
    trace1 = go.Bar(
                    x = df.index,
                    y = df[question],
                    #orientation='h',
                    marker = dict(color='dodgerblue',
                                 line=dict(color='black',width=1)),
                    text = df.index)
    data = [trace1]
    layout = go.Layout(barmode = "group",title=title,width=800, height=500,
                       xaxis=dict(type='category',categoryorder='array',categoryarray=salary_order,title=y_axis_title),
                       yaxis= dict(title=x_axis_title))
                       
    fig = go.Figure(data = data, layout = layout)
    iplot(fig)    

# Replacing the ambigious countries name with Standard names
df_2019['In which country do you currently reside?'].replace(
                                                   {'United States of America':'United States',
                                                    'Viet Nam':'Vietnam',
                                                    "People 's Republic of China":'China',
                                                    "United Kingdom of Great Britain and Northern Ireland":'United Kingdom',
                                                    "Hong Kong (S.A.R.)":"Hong Kong"},inplace=True)


df_2018['In which country do you currently reside?'].replace(
                                                   {'United States of America':'United States',
                                                    'Viet Nam':'Vietnam',
                                                    "People 's Republic of China":'China',
                                                    "United Kingdom of Great Britain and Northern Ireland":'United Kingdom',
                                                    "Hong Kong (S.A.R.)":"Hong Kong"},inplace=True)


# Splitting all the datasets genderwise
male_2019 = df_2019[df_2019['What is your gender? - Selected Choice']=='Male']
female_2019 = df_2019[df_2019['What is your gender? - Selected Choice']=='Female']

male_2018 = df_2018[df_2018['What is your gender? - Selected Choice']=='Male']
female_2018 = df_2018[df_2018['What is your gender? - Selected Choice']=='Female']

male_2017 = df_2017[df_2017['GenderSelect']=='Male']
female_2017 = df_2017[df_2017['GenderSelect']=='Female']


# Top-10 Countries with Respondents in 2019
topn = 10
count_male = male_2019['In which country do you currently reside?'].value_counts()[:topn].reset_index()
count_female = female_2019['In which country do you currently reside?'].value_counts()[:topn].reset_index()

pie_men = go.Pie(labels=count_male['index'],values=count_male['In which country do you currently reside?'],name="Men",hole=0.4,domain={'x': [0,0.46]})
pie_women = go.Pie(labels=count_female['index'],values=count_female['In which country do you currently reside?'],name="Women",hole=0.5,domain={'x': [0.52,1]})

layout = dict(title = 'Top-10 Countries with Respondents in 2019', font=dict(size=10), legend=dict(orientation="h"),
              annotations = [dict(x=0.2, y=0.5, text='Men', showarrow=False, font=dict(size=20)),
                             dict(x=0.8, y=0.5, text='Women', showarrow=False, font=dict(size=20)) ])

fig = dict(data=[pie_men, pie_women], layout=layout)
py.iplot(fig)
df_all_surveys = pd.DataFrame(data = [len(df_2017),len(df_2018),len(df_2019)],
                          columns = ['Number of responses'], index = ['2017','2018','2019'])
df_all_surveys.index.names = ['Year of Survey']


x = df_all_surveys['Number of responses'].index
y = df_all_surveys['Number of responses'].values


# Use textposition='auto' for direct text
fig = go.Figure(data=[go.Bar(
            x=['Year 2017','Year 2018','Year 2019'],
            y=y,
            text=y,
            width=0.4,
            textposition='auto',
            marker=dict(color='dodgerblue')
 )])

fig.data[0].marker.line.width = 1
fig.data[0].marker.line.color = "black"
fig.update_layout(yaxis=dict(title='Number of Respondents'),width=700,height=500,
                  title='Total number of respondents over the years',
                  xaxis=dict(title='Years'))
fig.show()

colors1 = ['dodgerblue', 'plum', '#F0A30A','#8c564b'] 
counts = df_2019['What is your gender? - Selected Choice'].value_counts(sort=True)
labels = counts.index
values = counts.values

pie = go.Pie(labels=labels, values=values, marker=dict(colors=colors1,line=dict(color='#000000', width=1)))
layout = go.Layout(title='Gender Distribution in 2019')

fig = go.Figure(data=[pie], layout=layout)
py.iplot(fig)
from plotly.subplots import make_subplots
colors2 = ['dodgerblue', 'plum', '#F0A30A','#8c564b'] 
gender_count_2019 = df_2019['What is your gender? - Selected Choice'].value_counts(sort=True)
gender_count_2018 = df_2018['What is your gender? - Selected Choice'].value_counts(sort=True)
gender_count_2017 = df_2017['GenderSelect'].value_counts(sort=True)


labels = ["Male ", "Female", "Prefer not to say ", "Prefer to self-describe"]
labels1 = ["Male ", "Female","A different identity", "Non-binary","genderqueer, or gender non-conforming"]
# Create subplots: use 'domain' type for Pie subplot
fig = make_subplots(rows=1, cols=3, specs=[[{'type':'domain'}, {'type':'domain'},{'type':'domain'}]])
fig.add_trace(go.Pie(labels=labels1, values=gender_count_2017.values, name="2017",marker=dict(colors=colors2)),
              1, 1)
fig.add_trace(go.Pie(labels=labels, values=gender_count_2018.values, name="2018",marker=dict(colors=colors2)),
              1, 2)
fig.add_trace(go.Pie(labels=labels, values=gender_count_2019.values, name="2019",marker=dict(colors=colors2)),
              1, 3)
# Use `hole` to create a donut-like pie chart
fig.update_traces(hole=.5, hoverinfo="label+percent+name")

fig.data[0].marker.line.width = 1
fig.data[0].marker.line.color = "black"
fig.data[1].marker.line.width = 1
fig.data[1].marker.line.color = "black"
fig.data[2].marker.line.width = 1
fig.data[2].marker.line.color = "black"

fig.update_layout(
    title_text="Gender Distribution over the years",font=dict(size=12), legend=dict(orientation="h"),
    # Add annotations in the center of the donut pies.
    annotations=[dict(text='2017', x=0.11, y=0.5, font_size=20, showarrow=False),
                 dict(text='2018', x=0.5, y=0.5, font_size=20, showarrow=False),
                 dict(text='2019', x=0.88, y=0.5, font_size=20, showarrow=False)])
fig.show()
def get_name(code):
    '''
    Translate code to name of the country
    '''
    try:
        name = pycountry.countries.get(alpha_3=code).name
    except:
        name=code
    return name

country_number = pd.DataFrame(female_2019['In which country do you currently reside?'].value_counts())
country_number['country'] = country_number.index
country_number.columns = ['number', 'country']
country_number.reset_index().drop(columns=['index'], inplace=True)
country_number['country'] = country_number['country'].apply(lambda c: get_name(c))
country_number.head(5)



worldmap = [dict(type = 'choropleth', locations = country_number['country'], locationmode = 'country names',
                 z = country_number['number'], colorscale = "Blues", reversescale = True, 
                 marker = dict(line = dict( width = 0.5)), 
                 colorbar = dict(autotick = False, title = 'Number of respondents'))]

layout = dict(title = 'The Nationality of Female Respondents in 2019', geo = dict(showframe = False, showcoastlines = True, 
                                                                projection = dict(type = 'Mercator')))

fig = dict(data=worldmap, layout=layout)
py.iplot(fig, validate=False)


def plot_graph(data,question,title,x_axis_title):
    df = return_percentage(data,question)
    
    trace1 = go.Bar(
                    y = df.index,
                    x = df[question][0:20],
                    orientation='h',
                    marker = dict(color='dodgerblue',line=dict(color='black',width=1)),
                    text = df.index)
    data = [trace1]
    layout = go.Layout(barmode = "group",title=title,width=800, height=500, 
                       xaxis= dict(title=x_axis_title),
                       yaxis=dict(autorange="reversed"),
                       showlegend=False)
    fig = go.Figure(data = data, layout = layout)
    fig.show()
     
response_count = female_2019['In which country do you currently reside?'].count()
plot_graph(female_2019,'In which country do you currently reside?','Top 20 countries of female respondents','Percentage of Female Respondents')
    

female_country_2019 = female_2019['In which country do you currently reside?']
female_country_2018 = female_2018['In which country do you currently reside?']
female_country_2017 = female_2017['Country']
                                                                  
f_2019 = female_country_2019[(female_country_2019 == 'India') | (female_country_2019 == 'United States')].value_counts()
f_2018 = female_country_2018[(female_country_2018 == 'India') | (female_country_2018 == 'United States')].value_counts()
f_2017 = female_country_2017[(female_country_2017 == 'India') | (female_country_2017 == 'United States')].value_counts()                                                                  
                                         
female_country_count = pd.DataFrame(data = [f_2017,f_2018,f_2019],index = ['2017','2018','2019'])    

female_country_count['total'] = [len(female_2017),len(female_2018),len(female_2019)]
female_country_count['US%'] = female_country_count['United States']/female_country_count['total']*100
female_country_count['India%'] = female_country_count['India']/female_country_count['total']*100

female_country_count[['India%','US%']].plot(kind='bar',color=['dodgerblue','skyblue'],linewidth=1,edgecolor='k')
plt.gcf().set_size_inches(10,8)
plt.title('Pattern of US and Indian Female respondents over the years', fontsize = 15)
plt.xticks(rotation=0,fontsize='10', horizontalalignment='right')
plt.xlabel('Year of Survey',fontsize=15)
plt.ylabel('Percentage of Respondents',fontsize=15)
plt.legend(fontsize=15,bbox_to_anchor=(1.04,0.5), loc="center left",labels=['India','US'])
plt.show()


African_2019 = female_country_2019[(female_country_2019 == 'Algeria') | 
                                   (female_country_2019 == 'Nigeria') |
                                   (female_country_2019 == 'Egypt')   |
                                   (female_country_2019 == 'Kenya')   |
                                   (female_country_2019 == 'South Africa')].value_counts()

African_2018 = female_country_2018[(female_country_2018 == 'Algeria') | 
                                   (female_country_2018 == 'Nigeria') |
                                   (female_country_2018 == 'Egypt')   |
                                   (female_country_2018 == 'Kenya')   |
                                   (female_country_2018 == 'South Africa')].value_counts()




African_2017 = female_country_2017[(female_country_2017 == 'Algeria') | 
                                   (female_country_2017 == 'Nigeria') |
                                   (female_country_2017 == 'Egypt')   |
                                   (female_country_2017 == 'Kenya')   |
                                   (female_country_2017 == 'South Africa')].value_counts()
African_subcontinent_count = pd.DataFrame(data = [African_2017,African_2018,African_2019],index = ['2017','2018','2019']) 


African_subcontinent_count.fillna(0,inplace=True)
African_subcontinent_count.loc[:,'Sum'] = African_subcontinent_count.sum(axis=1)


x = African_subcontinent_count['Sum'].index
y = African_subcontinent_count['Sum'].values

# Use textposition='auto' for direct text
fig1 = go.Figure(data=[go.Bar(
            x=['Year 2017','Year 2018','Year 2019'],
            y=y,
            text=y,
            width=0.4,
            textposition='auto',
            marker=dict(color='dodgerblue'))])

fig1.data[0].marker.line.width = 1
fig1.data[0].marker.line.color = "black"
fig1.update_layout(yaxis=dict(title='Number of Female Respondents'),
                   title='Total African Females respondents over the years',width=800,height=500,
                   xaxis=dict(title='Years'))
fig1.show()
countries = ['South Africa','Egypt','Kenya','Nigeria','Algeria']

African_subcontinent_count[countries].plot(kind='bar',color=colors,linewidth=1,edgecolor='k')
plt.gcf().set_size_inches(10,8)
plt.title('Country wise Female respondents from Africa ', fontsize = 15)
plt.xticks(rotation=0,fontsize='10', horizontalalignment='right')
plt.xlabel('Years',fontsize=15)
plt.ylabel('Number of Female Respondents',fontsize=15)
plt.legend(fontsize=15,bbox_to_anchor=(1.04,0.5), loc="center left")
plt.show()

female_2017['Age in years'] = pd.cut(x=female_2017['Age'], bins=[18,21,25,29,34,39,44,49,54,59,69,79], 
                                                        labels=['18-21',
                                                                '22-24',
                                                                '25-29',
                                                                '30-34',
                                                                '35-39',
                                                                '40-44',
                                                                '45-49',
                                                                '50-54',
                                                                '55-59',
                                                                '60-69',
                                                                '70+'])
                                                                                                  


x = female_2017['Age in years'].value_counts()
y = female_2018['What is your age (# years)?'].value_counts()
z = female_2019['What is your age (# years)?'].value_counts()


w = pd.DataFrame(data = [x,y,z],index = ['2017','2018','2019'])
w.fillna(0,inplace=True)

w.loc['2017'] = w.loc['2017']/len(female_2017)*100
w.loc['2018'] = w.loc['2018']/len(female_2018)*100
w.loc['2019'] = w.loc['2019']/len(female_2019)*100

w.T[['2019']].plot(subplots=True, layout=(1,1),kind='bar',color='dodgerblue',linewidth=1,edgecolor='k',legend=False)
plt.gcf().set_size_inches(10,8)
plt.title('Age wise Distribution of Female Respondents in 2019',fontsize=15)
plt.xticks(rotation=45,fontsize='10', horizontalalignment='right')
plt.yticks( fontsize=10)
plt.xlabel('Age in years',fontsize=15)
plt.ylabel('Percentage of Female Respondents',fontsize=15)
plt.show()
w.T[['2017','2018']].plot(subplots=True,layout=(1,2),color='dodgerblue',kind='bar',linewidth=1,edgecolor='k',legend=False)
plt.gcf().set_size_inches(15,6)
#plt.title('Age wise Distribution of Female Respondents in 2019',fontsize=15)
#plt.xticks(rotation=0,fontsize='10', horizontalalignment='right')
plt.yticks( fontsize=10)
#plt.xlabel('Age in years',fontsize=15)
plt.ylabel('Percentage of Female Respondents',fontsize=15)
plt.show()
## Grouping the Ages


female_2019['Age Group']=np.where(female_2019['What is your age (# years)?'].isin(['18-21']),'18-21','')
female_2019['Age Group']=np.where(female_2019['What is your age (# years)?'].isin(['25-29','22-24']),'22-29',female_2019['Age Group'])
female_2019['Age Group']=np.where(female_2019['What is your age (# years)?'].isin(['30-34','35-39']),'30-39',female_2019['Age Group'])
female_2019['Age Group']=np.where(female_2019['What is your age (# years)?'].isin(['40-44','45-49']),'40-49',female_2019['Age Group'])
female_2019['Age Group']=np.where(female_2019['What is your age (# years)?'].isin(['50-54','55-59']),'50-59',female_2019['Age Group'])
female_2019['Age Group']=np.where(female_2019['What is your age (# years)?'].isin(['60-69']),'60-69',female_2019['Age Group'])
female_2019['Age Group']=np.where(female_2019['What is your age (# years)?'].isin(['70+']),'70s and above',female_2019['Age Group'])


colors = ["steelblue","dodgerblue","lightskyblue","deepskyblue","darkturquoise","paleturquoise","turquoise"]

count_age=female_2019.groupby(['In which country do you currently reside?','Age Group'])['What is your age (# years)?'].count().reset_index()
count_age.columns=['Country','Age Group','Count']
count_age=count_age[count_age['Country'].isin(female_2019['In which country do you currently reside?'].value_counts()[:15].index)]
count_age=count_age[count_age['Country']!='Other']

count_age['Percent'] = (count_age['Count']/len(female_2019))*100
count_age = count_age.sort_values('Percent',ascending=False)

df_age_country = count_age.pivot('Country','Age Group','Percent')
df_age_country.fillna('0',inplace=True)

df_age_country.plot.bar(stacked=True,color=colors,linewidth=1,edgecolor='k')
plt.gcf().set_size_inches(16,8)
plt.xticks( rotation=45,fontsize='15', horizontalalignment='right')
plt.yticks( fontsize=10)
plt.title('Country wise Age Distribution in 2019', fontsize = 15)
#plt.xlabel('Countries',fontsize=15)
plt.ylabel('Percentage of Female Respondents',fontsize=15)
plt.legend(fontsize=15,bbox_to_anchor=(1.04,0.5), loc="centre left")
plt.show()
import textwrap
from  textwrap import fill

x_axis=range(7)
qualification = female_2019['What is the highest level of formal education that you have attained or plan to attain within the next 2 years?'].value_counts().to_frame()
qualification = qualification/len(female_2019)*100
labels = qualification.index

qualification.plot(kind='bar',color='dodgerblue',linewidth=1,edgecolor='k',legend=None)
plt.gcf().set_size_inches(10,8)
plt.title('Educational Qualifications of the Females respondents in 2019', fontsize = 15)
plt.xticks(x_axis, [textwrap.fill(label, 10) for label in labels], 
           rotation = 0, fontsize=12, horizontalalignment="right")
#plt.xlabel('Education Qualification',fontsize=15)
plt.ylabel('Percentage of Female Respondents',fontsize=15)
plt.xlabel('Qualification',fontsize=15)
plt.show()
df_edu_temp = pd.crosstab(female_2019['In which country do you currently reside?'],
              female_2019['What is the highest level of formal education that you have attained or plan to attain within the next 2 years?'])


df_edu = df_edu_temp[(df_edu_temp.index == 'Brazil')| (df_edu_temp.index == 'India') | (df_edu_temp.index == 'Japan') | (df_edu_temp.index == 'Russia') | (df_edu_temp.index == 'United States')
                    |(df_edu_temp.index == 'Canada')| (df_edu_temp.index == 'Germany') | (df_edu_temp.index == "China")
                    |(df_edu_temp.index == 'United Kingdom')].drop('I prefer not to answer',axis=1)

df_edu = df_edu/len(female_2019)*100
df_edu.plot(kind='bar',width=1,cmap='tab10',linewidth=1,edgecolor='k')
plt.gcf().set_size_inches(16,8)
plt.title("Country wise distribution of Females' educational qualification in 2019", fontsize = 20)
plt.xlabel('Countries',fontsize=15)
plt.ylabel('Percentage of Female Respondents',fontsize=15)
plt.xticks( rotation=45,fontsize='15', horizontalalignment='right')
plt.yticks( fontsize=10)
plt.legend(fontsize=15,bbox_to_anchor=(0.5, -0.22), loc="upper center",ncol=4)
plt.show()
from wordcloud import WordCloud
female_title_2017 = female_2017['CurrentJobTitleSelect'].dropna()
female_title_2018 = female_2018['Select the title most similar to your current role (or most recent title if retired): - Selected Choice'].dropna()
female_title_2019 = female_2019['Select the title most similar to your current role (or most recent title if retired): - Selected Choice'].dropna()

fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=[26, 8])
wordcloud1 = WordCloud( background_color='white',
                        width=600,
                        height=400).generate(" ".join(female_title_2017))
ax1.imshow(wordcloud1)
ax1.axis('off')
ax1.set_title('Job Titles in 2017',fontsize=20);

wordcloud2 = WordCloud( background_color='white',
                        width=600,
                        height=400).generate(" ".join(female_title_2018))
ax2.imshow(wordcloud2)
ax2.axis('off')
ax2.set_title('Job Titles in 2018',fontsize=20);

wordcloud3 = WordCloud( background_color='white',
                        width=600,
                        height=400).generate(" ".join(female_title_2019))
ax3.imshow(wordcloud3)
ax3.axis('off')
ax3.set_title('Job Titles in 2019',fontsize=20);
def plot_graph(data,question,title,x_axis_title):
    df = return_percentage(data,question)
    
    trace1 = go.Bar(
                    y = df.index,
                    x = df[question][0:20],
                    orientation='h',
                    marker = dict(color='skyblue',
                                 line=dict(color='black',width=1)),
                    text = df.index)
    data = [trace1]
    layout = go.Layout(barmode = "group",title=title,width=800, height=500, 
                       xaxis= dict(title=x_axis_title),
                       yaxis=dict(autorange="reversed"),
                       showlegend=False)
    fig = go.Figure(data = data, layout = layout)
    iplot(fig)
    
#response_count = len(female_2019['Select the title most similar to your current role (or most recent title if retired): - Selected Choice'])
plot_graph(female_2019,'Select the title most similar to your current role (or most recent title if retired): - Selected Choice','Top 20 roles for female respondents in 2019','Percentage of Female Respondents')
    

df_roles_temp = pd.crosstab(female_2019['In which country do you currently reside?'],
                            female_2019['Select the title most similar to your current role (or most recent title if retired): - Selected Choice'])
df_roles_temp['Data Engineer'] = df_roles_temp['DBA/Database Engineer']+df_roles_temp['Data Engineer']
df_roles_temp['Data/Business Analyst'] = df_roles_temp['Data Analyst']+df_roles_temp['Business Analyst']
df_roles = df_roles_temp[(df_edu_temp.index == 'Brazil')| (df_edu_temp.index == 'India') | (df_edu_temp.index == 'Japan') | (df_edu_temp.index == 'Russia') | (df_edu_temp.index == 'United States')
                    |(df_edu_temp.index == 'Canada')| (df_edu_temp.index == 'Germany') | (df_edu_temp.index == "China")
                    |(df_edu_temp.index == 'United Kingdom')].drop(['Other','DBA/Database Engineer','Data Engineer','Data Analyst','Business Analyst'],axis=1)


df_roles = (df_roles/len(female_2019))*100

ax = df_roles.plot(kind='bar',width=1,cmap='tab20',linewidth=1,edgecolor='k')
plt.gcf().set_size_inches(20,8)
plt.xticks( rotation=45,fontsize='10', horizontalalignment='right')
plt.yticks( fontsize=10)
plt.title('Country wise Job Role Distribution of Females in 2019', fontsize = 15)
#plt.xlabel('Countries',fontsize=15)
plt.ylabel('Percentage of Female Respondents',fontsize=15)
plt.xticks( rotation=45,fontsize='15', horizontalalignment='right')
plt.yticks( fontsize=10)
plt.legend(fontsize=15,bbox_to_anchor=(0.5, -0.22), loc="upper center",ncol=6)
plt.show()

def return_percentage1(data,question_part):
    """Calculates percent of each value in a given column"""
    total = data[question_part].count()
    counts_df= data[question_part].value_counts()
    percentage_df = (counts_df*100)/total
    return percentage_df



female_DS_2017 = return_percentage1(female_2017,'CurrentJobTitleSelect').loc['Data Scientist']
female_DS_2018 = return_percentage1(female_2018,'Select the title most similar to your current role (or most recent title if retired): - Selected Choice').loc['Data Scientist']
female_DS_2019 = return_percentage1(female_2019,'Select the title most similar to your current role (or most recent title if retired): - Selected Choice').loc['Data Scientist']

ds = pd.DataFrame(data = [female_DS_2017,female_DS_2018,female_DS_2019],
                          columns = ['Percentage of total roles'], index = ['2017','2018','2019'])
ds.round(1)
ds.index.names = ['Year of Survey']

x = ds['Percentage of total roles'].index
y = np.round(ds['Percentage of total roles'].values,1)


# Use textposition='auto' for direct text
fig = go.Figure(data=[go.Bar(
            x=['Year 2017','Year 2018','Year 2019'],
            y=y,
            text=y,
            width=0.4,
            textposition='auto',
            marker=dict(color='dodgerblue')
 )])

fig.data[0].marker.line.width = 1
fig.data[0].marker.line.color = "black"
fig.update_layout(yaxis=dict(title='Percentage of Female Respondents'),width=700,height=500,
                  title='Female Data Scientists in the survey',xaxis=dict(title='Years'))
fig.show()
female_2019['What is your current yearly compensation (approximate $USD)?'].fillna('didnot Disclose',inplace=True)

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
                        '> $500,000',
                        'didnot Disclose']

def plot_graph(data,question,title,x_axis_title,y_axis_title):
    df = return_percentage(data,question)
    
    trace1 = go.Bar(
                    x = df.index,
                    y = df[question],
                    #orientation='h',
                    marker = dict(color='dodgerblue',
                                 line=dict(color='black',width=1)),
                    text = df.index)
    data = [trace1]
    layout = go.Layout(barmode = "group",title=title,width=800, height=500,
                       xaxis=dict(type='category',categoryorder='array',categoryarray=salary_order,title=y_axis_title),
                       yaxis= dict(title=x_axis_title))
                       
    fig = go.Figure(data = data, layout = layout)
    iplot(fig)

plot_graph(female_2019,'What is your current yearly compensation (approximate $USD)?','Salary Range of Female respondents in 2019','Percentage of Female Respondents','Annual Salary in USD')
    

# Column of dataset under consideration
q = 'What is your current yearly compensation (approximate $USD)?'

#Preprocessing the 2019 salary data to get standard salary categories
female_2019['Salary Range'] = np.where(female_2019[q].isin(['$0-999','1,000-1,999','2,000-2,999','3,000-3,999',
                     '4,000-4,999','5,000-7,499','7,500-9,999']),'0-10,000',' ')
female_2019['Salary Range'] = np.where(female_2019[q].isin(['10,000-14,999','15,000-19,999',]),'10-20,000',female_2019['Salary Range'])
female_2019['Salary Range'] = np.where(female_2019[q].isin(['20,000-24,999','25,000-29,999',]),'20-30,000',female_2019['Salary Range'])
female_2019['Salary Range'] = np.where(female_2019[q].isin(['30,000-39,999']),'30-40,000',female_2019['Salary Range'])
female_2019['Salary Range'] = np.where(female_2019[q].isin(['40,000-49,999']),'40-50,000',female_2019['Salary Range'])
female_2019['Salary Range'] = np.where(female_2019[q].isin(['50,000-59,999']),'50-60,000',female_2019['Salary Range'])
female_2019['Salary Range'] = np.where(female_2019[q].isin(['60,000-69,999']),'60-70,000',female_2019['Salary Range'])
female_2019['Salary Range'] = np.where(female_2019[q].isin(['70,000-79,999']),'70-80,000',female_2019['Salary Range'])
female_2019['Salary Range'] = np.where(female_2019[q].isin(['80,000-89,999']),'80-90,000',female_2019['Salary Range'])
female_2019['Salary Range'] = np.where(female_2019[q].isin(['90,000-99,999']),'90-100,000',female_2019['Salary Range'])
female_2019['Salary Range'] = np.where(female_2019[q].isin(['100,000-124,999']),'100-125,000',female_2019['Salary Range'])
female_2019['Salary Range'] = np.where(female_2019[q].isin(['125,000-149,999']),'125-150,000',female_2019['Salary Range'])
female_2019['Salary Range'] = np.where(female_2019[q].isin(['150,000-199,999']),'150-200,000',female_2019['Salary Range'])
female_2019['Salary Range'] = np.where(female_2019[q].isin(['200,000-249,999']),'200-250,000',female_2019['Salary Range'])
female_2019['Salary Range'] = np.where(female_2019[q].isin(['250,000-299,999']),'250-300,000',female_2019['Salary Range'])
female_2019['Salary Range'] = np.where(female_2019[q].isin(['300,000-400,000']),'300-500,000',female_2019['Salary Range'])
female_2019['Salary Range'] = np.where(female_2019[q].isin(['400,000-500,000']),'300-500,000',female_2019['Salary Range'])
female_2019['Salary Range'] = np.where(female_2019[q].isin(['> $500,000']),'500,000+',female_2019['Salary Range'])
female_2019['What is your current yearly compensation (approximate $USD)?'].fillna('didnot Disclose',inplace=True)

#Preprocessing the 2018 salary data to get standard salary categories

female_2018[q].replace({'I do not wish to disclose my approximate yearly compensation':'didnot Disclose'},inplace=True)
female_2018[q].fillna('didnot Disclose',inplace=True)

salary_order2 = ['0-10,000',
                '10-20,000',
                '20-30,000',
                '30-40,000',
                '40-50,000',
                '50-60,000',
                '60-70,000',
                '70-80,000',
                '80-90,000',
                '90-100,000',
                '100-125,000',
                '125-150,000',
                '150-200,000',
                '200-250,000',
                '250-300,000',
                '300-400,000',
                '400-500,000',
                '500,000+',
                'didnot Disclose']

   
df1 = return_percentage(female_2019,'Salary Range')
df2 = return_percentage(female_2018,'What is your current yearly compensation (approximate $USD)?')

fig = go.Figure(data=[
    go.Bar(name='Females in 2019', y=df1['Salary Range'], x=df1.index,marker_color='dodgerblue'),
    go.Bar(name='Females in 2018', y=df2[q], x=df2.index,marker_color='skyblue')
])    
fig.update_layout(barmode='group',title='Female respondents salaries in 2018 and 2019',xaxis=dict(title='Annual Salary in USD',categoryarray=salary_order2),yaxis=dict(title='Percentage of Female respondents'))
fig.show()    

male_2019['What is your current yearly compensation (approximate $USD)?'].fillna('didnot Disclose',inplace=True)

male_salary_2019 = return_percentage(male_2019,'What is your current yearly compensation (approximate $USD)?')
female_salary_2019 = return_percentage(female_2019,'What is your current yearly compensation (approximate $USD)?')
df_male_salary = male_salary_2019;df_female_salary = female_salary_2019

fig = go.Figure(data=[
    go.Bar(name='Males in 2019', y=df_male_salary['What is your current yearly compensation (approximate $USD)?'], x=df_male_salary.index,marker_color='dodgerblue'),
    go.Bar(name='Females in 2019',y=df_female_salary['What is your current yearly compensation (approximate $USD)?'], x=df_female_salary.index,marker_color='skyblue')
])    
fig.update_layout(barmode='group',title='Comparison of Male and Female salaries in 2019.',xaxis=dict(title='Annual Salary in USD',categoryarray=salary_order),yaxis=dict(title='Percentage of respondents'))
fig.show()    
def return_percentage_groupbarplot(total_data,data,question_part):
    """Calculates percent of each value in a given column"""
    total = len(total_data)
    counts_df= data[question_part].value_counts()
    percentage_df = (counts_df*100)/total
    return percentage_df

DS_2019 = df_2019[df_2019['Select the title most similar to your current role (or most recent title if retired): - Selected Choice']=='Data Scientist']
female_2019['What is your current yearly compensation (approximate $USD)?'].fillna('didnot Disclose',inplace=True)
male_2019['What is your current yearly compensation (approximate $USD)?'].fillna('didnot Disclose',inplace=True)

female_DS_2019 = female_2019[female_2019['Select the title most similar to your current role (or most recent title if retired): - Selected Choice']=='Data Scientist'] 
male_DS_2019 = male_2019[male_2019['Select the title most similar to your current role (or most recent title if retired): - Selected Choice']=='Data Scientist']  

df_DS_male = return_percentage_groupbarplot(DS_2019,male_DS_2019,'What is your current yearly compensation (approximate $USD)?')
df_DS_female = return_percentage_groupbarplot(DS_2019,female_DS_2019,'What is your current yearly compensation (approximate $USD)?')

fig = go.Figure(data=[
    go.Bar(name='Male', y=df_DS_male.values, x=df_DS_male.index,marker_color='dodgerblue'),
    go.Bar(name='Females', y=df_DS_female.values, x=df_DS_female.index,marker_color='skyblue')
])    
fig.update_layout(barmode='group',title="Salaries of Male & Female Data Scientists'in 2019",xaxis=dict(title='Annual Salary in USD',categoryarray=salary_order),yaxis=dict(title='Percentage of total Data Scientists'))
fig.show() 
df_salary_temp = pd.crosstab(female_DS_2019['In which country do you currently reside?'],female_DS_2019['What is your current yearly compensation (approximate $USD)?'])
df_salary = df_salary_temp[(df_salary_temp.index == 'India')| (df_salary_temp.index == 'United States')
                           |(df_salary_temp.index == 'Canada') | (df_salary_temp.index == 'Germany') |
                           (df_salary_temp.index == 'United Kingdom')].drop('didnot Disclose',axis=1)

df_salary = df_salary/len(female_DS_2019)*100
df_salary.plot(kind='bar',width=1,cmap='tab20',linewidth=1,edgecolor='k')
plt.gcf().set_size_inches(16,8)
plt.title("Country wise distribution of Females' educational qualification in 2019", fontsize = 20)
plt.xlabel('Countries',fontsize=15)
plt.ylabel('Percentage of Female Respondents',fontsize=15)
plt.xticks( rotation=45,fontsize='15', horizontalalignment='right')
plt.yticks( fontsize=10)
plt.legend(fontsize=15,bbox_to_anchor=(0.5, -0.22), loc="upper center",ncol=4)
plt.show()
