
#Render Matplotlib Plots Inline

### matplotlib inline



#Import the standard Python Scientific Libraries

import pandas as pd

import numpy as np

from scipy import stats

import matplotlib.pyplot as plt

import seaborn as sns



#Import Plotly and use it in the Offline Mode

import plotly

import plotly.offline as py

py.init_notebook_mode(connected=True)

import plotly.tools as tls

import plotly.graph_objs as go

import plotly.tools as tls

import plotly.figure_factory as fig_fact

plotly.tools.set_config_file(world_readable=True, sharing='public')



#Suppress Deprecation and Incorrect Usage Warnings 

import warnings

warnings.filterwarnings('ignore')
#Load MCQ Responses into a Pandas DataFrame

mcq = pd.read_csv('../input/multipleChoiceResponses.csv', encoding="ISO-8859-1", low_memory=False)

mcq.shape
#Load Free Form Responses into a Pandas DataFrame

ff = pd.read_csv('../input/freeformResponses.csv', encoding="ISO-8859-1", low_memory=False)

ff.shape
#The Seaborn Countplot function counts the number of instances of each category and renders a barplot.

sns.countplot(y='GenderSelect', data=mcq)
#Create a DataFrame for number of respondents by country

con_df = pd.DataFrame(mcq['Country'].value_counts())

con_df['country'] = con_df.index

con_df.columns = ['num_resp', 'country']

con_df = con_df.reset_index().drop('index', axis=1)

con_df.head(10)
#Create a Choropleth Map of the respondents using Plotly. 

#Find out more at https://plot.ly/python/choropleth-maps/

data = [ dict(

        type = 'choropleth',

        locations = con_df['country'],

        locationmode = 'country names',

        z = con_df['num_resp'],

        text = con_df['country'],

        colorscale = [[0,'rgb(255, 255, 255)'],[1,'rgb(56, 142, 60)']],

        autocolorscale = False,

        reversescale = False,

        marker = dict(

            line = dict (

                color = 'rgb(180,180,180)',

                width = 0.5

            ) ),

        colorbar = dict(

            autotick = False,

            tickprefix = '',

            title = 'Survey Respondents'),

      ) ]



layout = dict(

    title = 'Survey Respondents by Nationality',

    geo = dict(

        showframe = False,

        showcoastlines = False,

        projection = dict(

            type = 'Mercator'

        )

    )

)



fig = dict( data=data, layout=layout )

py.iplot( fig, validate=False, filename='survey-world-map')
#Get Summary Statistics of the Respndents' Ages.

mcq['Age'].describe()
#Plot the Age distribution

fig = fig_fact.create_distplot([mcq[mcq['Age'] > 0]['Age']], ['age'], colors=['#BA68C8'])

py.iplot(fig, filename='Basic Distplot')

#sns.distplot(mcq[mcq['Age'] > 0]['Age'])
sns.countplot(y='FormalEducation', data=mcq)
plt.figure(figsize=(6,8))

sns.countplot(y='MajorSelect', data=mcq)
sns.countplot(y='EmploymentStatus', data=mcq)
sns.countplot(y='Tenure', data=mcq)
sns.countplot(y='LanguageRecommendationSelect', data=mcq)
top_lang = mcq['LanguageRecommendationSelect'].value_counts()

top_lang_dist = []

for lang in top_lang.index:

    top_lang_dist.append(mcq[(mcq['Age'].notnull()) & (mcq['LanguageRecommendationSelect'] == lang)]['Age'])



group_labels = top_lang.index



fig = fig_fact.create_distplot(top_lang_dist, group_labels, show_hist=False)

py.iplot(fig, filename='Language Preferences by Age')

mcq[mcq['CurrentJobTitleSelect'].notnull()]['CurrentJobTitleSelect'].shape
#Plot the number of R and Python users by Occupation

data = mcq[(mcq['CurrentJobTitleSelect'].notnull()) & ((mcq['LanguageRecommendationSelect'] == 'Python') | (mcq['LanguageRecommendationSelect'] == 'R'))]

plt.figure(figsize=(8, 10))

sns.countplot(y="CurrentJobTitleSelect", hue="LanguageRecommendationSelect", data=data)
#Render a bar plot of the 15 most popular ML Tools for next year

data = mcq['MLToolNextYearSelect'].value_counts().head(15)

sns.barplot(y=data.index, x=data)
data = mcq['MLMethodNextYearSelect'].value_counts().head(15)

sns.barplot(y=data.index, x=data)
#Explode the Pandas Dataframe to get the number of times each Learning Platform was mentioned

mcq['LearningPlatformSelect'] = mcq['LearningPlatformSelect'].astype('str').apply(lambda x: x.split(','))

s = mcq.apply(lambda x: pd.Series(x['LearningPlatformSelect']),axis=1).stack().reset_index(level=1, drop=True)

s.name = 'platform'
plt.figure(figsize=(6,8))

data = s[s != 'nan'].value_counts()

sns.barplot(y=data.index, x=data)
use_features = [x for x in mcq.columns if x.find('LearningPlatformUsefulness') != -1]
#Construct a Pandas DataFrame to illustrate the usefulness of various learning platforms.

fdf = {}

for feature in use_features:

    a = mcq[feature].value_counts()

    a = a/a.sum()

    fdf[feature[len('LearningPlatformUsefulness'):]] = a



fdf = pd.DataFrame(fdf).transpose()#.sort_values('Very useful', ascending=False)



#Plot a Heatmap of Learning Platform Usefulness

plt.figure(figsize=(6,12))

sns.heatmap(fdf.sort_values("Very useful", ascending=False), annot=True)



#Plot a grouped barplot of Learning Platform Usefulness

fdf.plot(kind='bar', figsize=(18,8), title="Usefullness of Learning Platforms")

plt.show()
cat_features = [x for x in mcq.columns if x.find('LearningCategory') != -1]
cdf = {}

for feature in cat_features:

    cdf[feature[len('LearningCategory'):]] = mcq[feature].mean()



cdf = pd.Series(cdf)



#Plot a Pie Chart of the contribution of each platform to learning

plt.pie(cdf, labels=cdf.index, 

        autopct='%1.1f%%', shadow=True, startangle=140)

plt.axis('equal')

plt.title("Contribution of each Platform to Learning")

plt.show()
mcq[mcq['HardwarePersonalProjectsSelect'].notnull()]['HardwarePersonalProjectsSelect'].shape
mcq['HardwarePersonalProjectsSelect'] = mcq['HardwarePersonalProjectsSelect'].astype('str').apply(lambda x: x.split(','))

s = mcq.apply(lambda x: pd.Series(x['HardwarePersonalProjectsSelect']),axis=1).stack().reset_index(level=1, drop=True)

s.name = 'hardware'
s[s != 'nan'].value_counts()
plt.figure(figsize=(8,8))

sns.countplot(y='TimeSpentStudying', data=mcq, hue='EmploymentStatus').legend(loc='center left', bbox_to_anchor=(1, 0.5))
mcq['BlogsPodcastsNewslettersSelect'] = mcq['BlogsPodcastsNewslettersSelect'].astype('str').apply(lambda x: x.split(','))
s = mcq.apply(lambda x: pd.Series(x['BlogsPodcastsNewslettersSelect']),axis=1).stack().reset_index(level=1, drop=True)

s.name = 'platforms'
s = s[s != 'nan'].value_counts()
plt.figure(figsize=(6,8))

plt.title("Most Popular Blogs and Podcasts")

sns.barplot(y=s.index, x=s)
mcq['CoursePlatformSelect'] = mcq['CoursePlatformSelect'].astype('str').apply(lambda x: x.split(','))
t = mcq.apply(lambda x: pd.Series(x['CoursePlatformSelect']),axis=1).stack().reset_index(level=1, drop=True)

t.name = 'courses'
t = t[t != 'nan'].value_counts()
plt.title("Most Popular Course Platforms")

sns.barplot(y=t.index, x=t)
job_features = [x for x in mcq.columns if x.find('JobSkillImportance') != -1 and x.find('JobSkillImportanceOther') == -1]
#Get a Pandas DataFrame of Skill Importance of Data Science Jobs

jdf = {}

for feature in job_features:

    a = mcq[feature].value_counts()

    a = a/a.sum()

    jdf[feature[len('JobSkillImportance'):]] = a

#fdf = pd.DataFrame(fdf)

jdf = pd.DataFrame(jdf).transpose()



jdf.plot(kind='bar', figsize=(12,6), title="Skill Importance in Data Science Jobs")
mcq[mcq['CompensationAmount'].notnull()].shape
#Convert all salary values to float. If not possible, convert it to NaN

def clean_salary(x):

    x = x.replace(',', '')

    try:

        return float(x)

    except:

        return np.nan
#Function that outputs salary statistics and plots a salary distribution for that country

def salary_stats(country):

    data = mcq[(mcq['CompensationAmount'].notnull()) & (mcq['Country'] == country) ]

    data['CompensationAmount'] = data['CompensationAmount'].apply(clean_salary)

    print(data[data['CompensationAmount'] < 1e9]['CompensationAmount'].describe())

    sns.distplot(data[data['CompensationAmount'] < 1e9]['CompensationAmount'])
salary_stats('India')
salary_stats('United States')
mcq['PublicDatasetsSelect'] = mcq['PublicDatasetsSelect'].astype('str').apply(lambda x: x.split(','))
q = mcq.apply(lambda x: pd.Series(x['PublicDatasetsSelect']),axis=1).stack().reset_index(level=1, drop=True)

q.name = 'courses'
q = q[q != 'nan'].value_counts()
plt.title("Most Popular Dataset Platforms")

sns.barplot(y=q.index, x=q)
ff['PersonalProjectsChallengeFreeForm'].value_counts().head(15)
time_features = [x for x in mcq.columns if x.find('Time') != -1][4:10]
tdf = {}

for feature in time_features:

    tdf[feature[len('Time'):]] = mcq[feature].mean()



tdf = pd.Series(tdf)

print(tdf)

print()



plt.pie(tdf, labels=tdf.index, 

        autopct='%1.1f%%', shadow=True, startangle=140)

plt.axis('equal')

plt.title("Percentage of Time Spent on Each DS Job")

plt.show()
job_factors = [x for x in mcq.columns if x.find('JobFactor') != -1]
jfdf = {}

for feature in job_factors:

    a = mcq[feature].value_counts()

    a = a/a.sum()

    jfdf[feature[len('JobFactor'):]] = a



jfdf = pd.DataFrame(jfdf).transpose()



plt.figure(figsize=(6,12))

sns.heatmap(jfdf.sort_values('Very Important', ascending=False), annot=True)



jfdf.plot(kind='bar', figsize=(18,6), title="Things to look for while considering Data Science Jobs")

plt.show()
sns.countplot(y='UniversityImportance', data=mcq)
top_uni = mcq['UniversityImportance'].value_counts().head(5)

top_uni_dist = []

for uni in top_uni.index:

    top_uni_dist.append(mcq[(mcq['Age'].notnull()) & (mcq['UniversityImportance'] == uni)]['Age'])



group_labels = top_uni.index



fig = fig_fact.create_distplot(top_uni_dist, group_labels, show_hist=False)

py.iplot(fig, filename='University Importance by Age')
mcq[mcq['FirstTrainingSelect'].notnull()].shape
sns.countplot(y='FirstTrainingSelect', data=mcq)
sns.countplot(y='ProveKnowledgeSelect', data=mcq)
mcq[mcq['AlgorithmUnderstandingLevel'].notnull()].shape
sns.countplot(y='AlgorithmUnderstandingLevel', data=mcq)
plt.title("Best Places to look for a Data Science Job")

sns.countplot(y='JobSearchResource', data=mcq)
plt.title("Top Places to get Data Science Jobs")

sns.countplot(y='EmployerSearchMethod', data=mcq)