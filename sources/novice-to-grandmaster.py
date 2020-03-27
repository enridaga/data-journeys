
import pandas as pd

import seaborn as sns

import matplotlib.pyplot as plt

import squarify

plt.style.use('fivethirtyeight')

import warnings

warnings.filterwarnings('ignore')

import numpy as np

import plotly.offline as py

py.init_notebook_mode(connected=True)

import plotly.graph_objs as go

import plotly.tools as tls

from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))
response=pd.read_csv('../input/multipleChoiceResponses.csv',encoding='ISO-8859-1')
response.head()
print('The total number of respondents:',response.shape[0])

print('Total number of Countries with respondents:',response['Country'].nunique())

print('Country with highest respondents:',response['Country'].value_counts().index[0],'with',response['Country'].value_counts().values[0],'respondents')

print('Youngest respondent:',response['Age'].min(),' and Oldest respondent:',response['Age'].max())
f,ax=plt.subplots(1,2,figsize=(22,8))

response['GenderSelect'].value_counts().plot.pie(ax=ax[0],explode=[0,0.1,0,0],shadow=True,autopct='%1.1f%%')

sns.countplot(y=response['GenderSelect'],ax=ax[1])

plt.subplots_adjust(wspace=0.8)

ax[0].set_ylabel('')

ax[1].set_ylabel('')

plt.show()
resp_coun=response['Country'].value_counts()[:15].to_frame()

sns.barplot(resp_coun['Country'],resp_coun.index,palette='inferno')

plt.title('Top 15 Countries by number of respondents')

plt.xlabel('')

fig=plt.gcf()

fig.set_size_inches(10,10)

plt.show()

tree=response['Country'].value_counts().to_frame()

squarify.plot(sizes=tree['Country'].values,label=tree.index,color=sns.color_palette('RdYlGn_r',52))

plt.rcParams.update({'font.size':20})

fig=plt.gcf()

fig.set_size_inches(40,15)

plt.show()
response['CompensationAmount']=response['CompensationAmount'].str.replace(',','')

response['CompensationAmount']=response['CompensationAmount'].str.replace('-','')

rates=pd.read_csv('../input/conversionRates.csv')

rates.drop('Unnamed: 0',axis=1,inplace=True)

salary=response[['CompensationAmount','CompensationCurrency','GenderSelect','Country','CurrentJobTitleSelect']].dropna()

salary=salary.merge(rates,left_on='CompensationCurrency',right_on='originCountry',how='left')

salary['Salary']=pd.to_numeric(salary['CompensationAmount'])*salary['exchangeRate']

print('Maximum Salary is USD $',salary['Salary'].dropna().astype(int).max())

print('Minimum Salary is USD $',salary['Salary'].dropna().astype(int).min())

print('Median Salary is USD $',salary['Salary'].dropna().astype(int).median())
plt.subplots(figsize=(15,8))

salary=salary[salary['Salary']<1000000]

sns.distplot(salary['Salary'])

plt.title('Salary Distribution',size=15)

plt.show()
f,ax=plt.subplots(1,2,figsize=(25,10))

sal_coun=salary.groupby('Country')['Salary'].median().sort_values(ascending=False)[:15].to_frame()

sns.barplot('Salary',sal_coun.index,data=sal_coun,palette='RdYlGn',ax=ax[0])

ax[0].axvline(salary['Salary'].median(),linestyle='dashed')

ax[0].set_title('Highest Salary Paying Countries')

ax[0].set_xlabel('')

max_coun=salary.groupby('Country')['Salary'].median().to_frame()

max_coun=max_coun[max_coun.index.isin(resp_coun.index)]

sns.barplot('Salary',max_coun.index,data=max_coun,palette='RdYlGn',ax=ax[1])

ax[1].axvline(salary['Salary'].median(),linestyle='dashed')

ax[1].set_title('Compensation of Top 15 Respondent Countries')

ax[1].set_xlabel('')

plt.show()
plt.subplots(figsize=(10,8))

sns.boxplot(y='GenderSelect',x='Salary',data=salary)

plt.ylabel('')

plt.show()
plt.subplots(figsize=(15,8))

response['Age'].hist(bins=50,edgecolor='black')

plt.xticks(list(range(0,80,5)))

plt.title('Age Distribution')

plt.show() 
f,ax=plt.subplots(1,2,figsize=(25,15))

sns.countplot(y=response['MajorSelect'],ax=ax[0])

ax[0].set_title('Major')

ax[0].set_ylabel('')

sns.countplot(y=response['CurrentJobTitleSelect'],ax=ax[1])

ax[1].set_title('Current Job')

ax[1].set_ylabel('')

plt.subplots_adjust(wspace=0.8)

plt.show()
sal_job=salary.groupby('CurrentJobTitleSelect')['Salary'].median().to_frame()

ax=sal_job.plot.barh(width=0.9,color='orange')

plt.title('Compensation By Job Title',size=15)

for i, v in enumerate(sal_job.Salary): 

    ax.text(0.1, i, v,fontsize=10,color='white',weight='bold')

fig=plt.gcf()

fig.set_size_inches(10,10)

plt.show()
f,ax=plt.subplots(1,2,figsize=(25,12))

skills=response['MLSkillsSelect'].str.split(',')

skills_set=[]

for i in skills.dropna():

    skills_set.extend(i)

pd.Series(skills_set).value_counts().sort_values(ascending=True).plot.barh(width=0.9,color=sns.color_palette('inferno',15),ax=ax[0])

ax[0].set_title('ML Skills')

tech=response['MLTechniquesSelect'].str.split(',')

techniques=[]

for i in tech.dropna():

    techniques.extend(i)

pd.Series(techniques).value_counts().sort_values(ascending=True).plot.barh(width=0.9,color=sns.color_palette('inferno',15),ax=ax[1])

ax[1].set_title('ML Techniques used')

plt.subplots_adjust(wspace=0.8)

plt.show()
f,ax=plt.subplots(1,2,figsize=(25,12))

ml_nxt=response['MLMethodNextYearSelect'].str.split(',')

nxt_year=[]

for i in ml_nxt.dropna():

    nxt_year.extend(i)

pd.Series(nxt_year).value_counts()[:15].sort_values(ascending=True).plot.barh(width=0.9,color=sns.color_palette('winter_r',15),ax=ax[0])

tool=response['MLToolNextYearSelect'].str.split(',')

tool_nxt=[]

for i in tool.dropna():

    tool_nxt.extend(i)

pd.Series(tool_nxt).value_counts()[:15].sort_values(ascending=True).plot.barh(width=0.9,color=sns.color_palette('winter_r',15),ax=ax[1])

plt.subplots_adjust(wspace=0.8)

ax[0].set_title('ML Method Next Year')

ax[1].set_title('ML Tool Next Year')

plt.show()
plt.subplots(figsize=(6,8))

learn=response['LearningPlatformSelect'].str.split(',')

platform=[]

for i in learn.dropna():

    platform.extend(i)

pd.Series(platform).value_counts()[:15].sort_values(ascending=True).plot.barh(width=0.9,color=sns.color_palette('winter',15))

plt.title('Best Platforms to Learn',size=15)

plt.show()
plt.subplots(figsize=(10,10))

hard=response['HardwarePersonalProjectsSelect'].str.split(',')

hardware=[]

for i in hard.dropna():

    hardware.extend(i)

pd.Series(hardware).value_counts().sort_values(ascending=True).plot.barh(width=0.9,color=sns.color_palette('inferno',10))

plt.title('Machines Used')

plt.show()
plt.subplots(figsize=(15,15))

data=response['PublicDatasetsSelect'].str.split(',')

dataset=[]

for i in data.dropna():

    dataset.extend(i)

pd.Series(dataset).value_counts().plot.pie(autopct='%1.1f%%',colors=sns.color_palette('Paired',10),startangle=90,wedgeprops = { 'linewidth' : 2, 'edgecolor' : 'white' })

plt.title('Dataset Source')

my_circle=plt.Circle( (0,0), 0.7, color='white')

p=plt.gcf()

p.gca().add_artist(my_circle)

plt.ylabel('')

plt.show()
plt.subplots(figsize=(15,15))

code=response['WorkCodeSharing'].str.split(',')

code_share=[]

for i in code.dropna():

    code_share.extend(i)

pd.Series(code_share).value_counts().plot.pie(autopct='%1.1f%%',shadow=True,colors=sns.color_palette('Set3',10),startangle=90,wedgeprops = { 'linewidth' : 2, 'edgecolor' : 'white' })

plt.title('Code Sharing Medium')

my_circle=plt.Circle( (0,0), 0.65, color='white')

p=plt.gcf()

p.gca().add_artist(my_circle)

plt.ylabel('')

plt.show()
plt.subplots(figsize=(15,15))

challenge=response['WorkChallengesSelect'].str.split(',')

challenges=[]

for i in challenge.dropna():

    challenges.extend(i)

pd.Series(challenges).value_counts().sort_values(ascending=True).plot.barh(width=0.9,color=sns.color_palette('inferno',25))

plt.title('Challenges in Data Science')

plt.show()
satisfy=response.copy()

satisfy['JobSatisfaction'].replace({'10 - Highly Satisfied':'10','1 - Highly Dissatisfied':'1','I prefer not to share':np.NaN},inplace=True)

satisfy.dropna(subset=['JobSatisfaction'],inplace=True)

satisfy['JobSatisfaction']=satisfy['JobSatisfaction'].astype(int)

satisfy_job=satisfy.groupby(['CurrentJobTitleSelect'])['JobSatisfaction'].mean().sort_values(ascending=True).to_frame()

ax=satisfy_job.plot.barh(width=0.9,color='orange')

fig=plt.gcf()

fig.set_size_inches(8,12)

for i, v in enumerate(satisfy_job.JobSatisfaction): 

    ax.text(.1, i, v,fontsize=10,color='white',weight='bold')

plt.title('Job Satisfaction out of 10')

plt.show()
satisfy=response.copy()

satisfy['JobSatisfaction'].replace({'10 - Highly Satisfied':'10','1 - Highly Dissatisfied':'1','I prefer not to share':np.NaN},inplace=True)

satisfy.dropna(subset=['JobSatisfaction'],inplace=True)

satisfy['JobSatisfaction']=satisfy['JobSatisfaction'].astype(int)

satisfy_job=satisfy.groupby(['Country'])['JobSatisfaction'].mean().sort_values(ascending=True).to_frame()

data = [ dict(

        type = 'choropleth',

        autocolorscale = False,

        colorscale = 'Viridis',

        reversescale = True,

        showscale = True,

        locations = satisfy_job.index,

        z = satisfy_job['JobSatisfaction'],

        locationmode = 'country names',

        text = satisfy_job['JobSatisfaction'],

        marker = dict(

            line = dict(color = 'rgb(200,200,200)', width = 0.5)),

            colorbar = dict(autotick = True, tickprefix = '', 

            title = 'Satisfaction')

            )

       ]



layout = dict(

    title = 'Job Satisfaction By Country',

    geo = dict(

        showframe = True,

        showocean = True,

        oceancolor = 'rgb(0,0,255)',

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
resp=response.dropna(subset=['WorkToolsSelect'])

resp=resp.merge(rates,left_on='CompensationCurrency',right_on='originCountry',how='left')

python=resp[(resp['WorkToolsSelect'].str.contains('Python'))&(~resp['WorkToolsSelect'].str.contains('R'))]

R=resp[(~resp['WorkToolsSelect'].str.contains('Python'))&(resp['WorkToolsSelect'].str.contains('R'))]

both=resp[(resp['WorkToolsSelect'].str.contains('Python'))&(resp['WorkToolsSelect'].str.contains('R'))]
response['LanguageRecommendationSelect'].value_counts()[:2].plot.bar()

plt.show()
labels1=python['LanguageRecommendationSelect'].value_counts()[:5].index

sizes1=python['LanguageRecommendationSelect'].value_counts()[:5].values



labels2=R['LanguageRecommendationSelect'].value_counts()[:5].index

sizes2=R['LanguageRecommendationSelect'].value_counts()[:5].values





fig = {

  "data": [

    {

      "values": sizes1,

      "labels": labels1,

      "domain": {"x": [0, .48]},

      "name": "Language",

      "hoverinfo":"label+percent+name",

      "hole": .4,

      "type": "pie"

    },     

    {

      "values": sizes2 ,

      "labels": labels2,

      "text":"CO2",

      "textposition":"inside",

      "domain": {"x": [.54, 1]},

      "name": "Language",

      "hoverinfo":"label+percent+name",

      "hole": .4,

      "type": "pie"

    }],

  "layout": {

        "title":"Language Recommended By Python and R users",

        "annotations": [

            {

                "font": {

                    "size": 30

                },

                "showarrow": False,

                "text": "Python",

                "x": 0.17,

                "y": 0.5

            },

            {

                "font": {

                    "size": 30

                },

                "showarrow": False,

                "text": "R",

                "x": 0.79,

                "y": 0.5}]}}

py.iplot(fig, filename='donut')
f,ax=plt.subplots(1,2,figsize=(18,8))

response['JobSkillImportancePython'].value_counts().plot.pie(ax=ax[0],autopct='%1.1f%%',explode=[0.1,0,0],shadow=True,colors=['g','lightblue','r'])

ax[0].set_title('Python Necessity')

ax[0].set_ylabel('')

response['JobSkillImportanceR'].value_counts().plot.pie(ax=ax[1],autopct='%1.1f%%',explode=[0,0.1,0],shadow=True,colors=['lightblue','g','r'])

ax[1].set_title('R Necessity')

ax[1].set_ylabel('')

plt.show()
plt.subplots(figsize=(10,6))

ax=pd.Series([python.shape[0],R.shape[0],both.shape[0]],index=['Python','R','Both']).plot.bar()

ax.set_title('Number of Users',size=15)

for p in ax.patches:

    ax.annotate(format(p.get_height()), (p.get_x()+0.15, p.get_height()+0.25),size=12)

plt.show()
py_sal=(pd.to_numeric(python['CompensationAmount'].dropna())*python['exchangeRate']).dropna()

py_sal=py_sal[py_sal<1000000]

R_sal=(pd.to_numeric(R['CompensationAmount'].dropna())*R['exchangeRate']).dropna()

R_sal=R_sal[R_sal<1000000]

both_sal=(pd.to_numeric(both['CompensationAmount'].dropna())*both['exchangeRate']).dropna()

both_sal=both_sal[both_sal<1000000]

trying=pd.DataFrame([py_sal,R_sal,both_sal])

trying=trying.transpose()

trying.columns=['Python','R','Both']

print('Median Salary For Individual using Python:',trying['Python'].median())

print('Median Salary For Individual using R:',trying['R'].median())

print('Median Salary For Individual knowing both languages:',trying['Both'].median())

trying.plot.box()

plt.title('Compensation By Language')

fig=plt.gcf()

fig.set_size_inches(10,6)

plt.show()
py1=python.copy()

r=R.copy()

py1['WorkToolsSelect']='Python'

r['WorkToolsSelect']='R'

r_vs_py=pd.concat([py1,r])

r_vs_py=r_vs_py.groupby(['CurrentJobTitleSelect','WorkToolsSelect'])['Age'].count().to_frame().reset_index()

r_vs_py.pivot('CurrentJobTitleSelect','WorkToolsSelect','Age').plot.barh(width=0.8)

fig=plt.gcf()

fig.set_size_inches(10,15)

plt.title('Job Title vs Language Used',size=15)

plt.show()
r_vs_py=pd.concat([py1,r])

r_vs_py=r_vs_py.groupby(['JobFunctionSelect','WorkToolsSelect'])['Age'].count().to_frame().reset_index()

r_vs_py.pivot('JobFunctionSelect','WorkToolsSelect','Age').plot.barh(width=0.8)

fig=plt.gcf()

fig.set_size_inches(10,15)

plt.title('Job Description vs Language Used')

plt.show()
r_vs_py=pd.concat([py1,r])

r_vs_py=r_vs_py.groupby(['Tenure','WorkToolsSelect'])['Age'].count().to_frame().reset_index()

r_vs_py.pivot('Tenure','WorkToolsSelect','Age').plot.barh(width=0.8)

fig=plt.gcf()

fig.set_size_inches(10,10)

plt.title('Job Tenure vs Language Used')

plt.show()
f,ax=plt.subplots(1,2,figsize=(20,15))

py_comp=python['WorkToolsSelect'].str.split(',')

py_comp1=[]

for i in py_comp:

    py_comp1.extend(i)

pd.Series(py_comp1).value_counts()[1:15].sort_values(ascending=True).plot.barh(width=0.9,ax=ax[0],color=sns.color_palette('inferno',15))

R_comp=R['WorkToolsSelect'].str.split(',')

R_comp1=[]

for i in R_comp:

    R_comp1.extend(i)

pd.Series(R_comp1).value_counts()[1:15].sort_values(ascending=True).plot.barh(width=0.9,ax=ax[1],color=sns.color_palette('inferno',15))

ax[0].set_title('Commonly Used Tools with Python')

ax[1].set_title('Commonly Used Tools with R')

plt.subplots_adjust(wspace=0.8)

plt.show()
response['DataScienceIdentitySelect'].value_counts()
plt.subplots(figsize=(10,8))

scientist=response[response['DataScienceIdentitySelect']=='Yes']

scientist['CurrentJobTitleSelect'].value_counts().sort_values(ascending=True).plot.barh(width=0.9,color=sns.color_palette('inferno',15))

plt.title('Job Titles',size=15)

plt.show()
true=response[response['CurrentJobTitleSelect']=='Data Scientist']
scientist=pd.concat([scientist,true])

scientist['CurrentJobTitleSelect'].shape[0]
plt.subplots(figsize=(10,8))

scientist['Country'].value_counts()[:15].sort_values(ascending=True).plot.barh(width=0.9,color=sns.color_palette('inferno',25))

plt.title('Countries By Number Of Data Scientists',size=15)

plt.show()
plt.subplots(figsize=(8,6))

sns.countplot(y=scientist['EmploymentStatus'])

plt.show()
f,ax=plt.subplots(1,2,figsize=(30,15))

past=scientist['PastJobTitlesSelect'].str.split(',')

past_job=[]

for i in past.dropna():

    past_job.extend(i)

pd.Series(past_job).value_counts().sort_values(ascending=True).plot.barh(width=0.9,color=sns.color_palette('summer',25),ax=ax[0])

ax[0].set_title('Previous Job')

sal=scientist['SalaryChange'].str.split(',')

sal_change=[]

for i in sal.dropna():

    sal_change.extend(i)

pd.Series(sal_change).value_counts().sort_values(ascending=True).plot.barh(width=0.9,color=sns.color_palette('summer',10),ax=ax[1])

ax[1].set_title('Salary Change')

plt.subplots_adjust(wspace=0.8)

plt.show()
plt.subplots(figsize=(8,8))

tools=scientist['WorkToolsSelect'].str.split(',')

tools_work=[]

for i in tools.dropna():

    tools_work.extend(i)

pd.Series(tools_work).value_counts()[:15].sort_values(ascending=True).plot.barh(width=0.9,color=sns.color_palette('RdYlGn',15))

plt.show()
course=scientist['CoursePlatformSelect'].str.split(',')

course_plat=[]

for i in course.dropna():

    course_plat.extend(i)

course_plat=pd.Series(course_plat).value_counts()

blogs=scientist['BlogsPodcastsNewslettersSelect'].str.split(',')

blogs_fam=[]

for i in blogs.dropna():

    blogs_fam.extend(i)

blogs_fam=pd.Series(blogs_fam).value_counts()

labels1=course_plat.index

sizes1=course_plat.values



labels2=blogs_fam[:5].index

sizes2=blogs_fam[:5].values





fig = {

  "data": [

    {

      "values": sizes1,

      "labels": labels1,

      "domain": {"x": [0, .48]},

      "name": "MOOC",

      "hoverinfo":"label+percent+name",

      "hole": .4,

      "type": "pie"

    },     

    {

      "values": sizes2 ,

      "labels": labels2,

      "text":"CO2",

      "textposition":"inside",

      "domain": {"x": [.54, 1]},

      "name": "Blog",

      "hoverinfo":"label+percent+name",

      "hole": .4,

      "type": "pie"

    }],

  "layout": {

        "title":"Blogs and Online Platforms",

        "showlegend":False,

        "annotations": [

            {

                "font": {

                    "size": 25

                },

                "showarrow": False,

                "text": "MOOC's",

                "x": 0.17,

                "y": 0.5

            },

            {

                "font": {

                    "size": 25

                },

                "showarrow": False,

                "text": "BLOGS",

                "x": 0.85,

                "y": 0.5}]}}

py.iplot(fig, filename='donut')
import itertools

plt.subplots(figsize=(22,10))

time_spent=['TimeFindingInsights','TimeVisualizing','TimeGatheringData','TimeModelBuilding']

length=len(time_spent)

for i,j in itertools.zip_longest(time_spent,range(length)):

    plt.subplot((length/2),2,j+1)

    plt.subplots_adjust(wspace=0.2,hspace=0.5)

    scientist[i].hist(bins=10,edgecolor='black')

    plt.axvline(scientist[i].mean(),linestyle='dashed',color='r')

    plt.title(i,size=20)

plt.show()
f,ax=plt.subplots(1,2,figsize=(25,12))

sns.countplot(scientist['JobSkillImportanceVisualizations'],ax=ax[0])

ax[0].set_title('Job Importance For Visuals')

ax[0].set_xlabel('')

scientist['WorkDataVisualizations'].value_counts().plot.pie(autopct='%2.0f%%',colors=sns.color_palette('Paired',10),ax=ax[1])

ax[1].set_title('Use Of Visualisations in Projects')

my_circle=plt.Circle( (0,0), 0.7, color='white')

p=plt.gcf()

p.gca().add_artist(my_circle)

plt.ylabel('')

plt.show()
f,ax=plt.subplots(1,2,figsize=(25,12))

sns.countplot(y=scientist['AlgorithmUnderstandingLevel'],ax=ax[0])

sns.countplot(response['JobSkillImportanceStats'],ax=ax[1])

ax[0].set_title('Algorithm Understanding')

ax[0].set_ylabel('')

ax[1].set_title('Knowledge of Stats')

ax[1].set_xlabel('')

plt.show()
plt.subplots(figsize=(25,35))

useful=['LearningPlatformUsefulnessBlogs','LearningPlatformUsefulnessCollege','LearningPlatformUsefulnessCompany','LearningPlatformUsefulnessKaggle','LearningPlatformUsefulnessCourses','LearningPlatformUsefulnessProjects','LearningPlatformUsefulnessTextbook','LearningPlatformUsefulnessYouTube']

length=len(useful)

for i,j in itertools.zip_longest(useful,range(length)):

    plt.subplot((length/2),3,j+1)

    plt.subplots_adjust(wspace=0.2,hspace=0.2)

    scientist[i].value_counts().plot.pie(autopct='%2.0f%%',colors=['g','lightblue','r'],wedgeprops = { 'linewidth' : 2, 'edgecolor' : 'white' })

    plt.title(i,size=25)

    my_circle=plt.Circle( (0,0), 0.7, color='white')

    p=plt.gcf()

    p.gca().add_artist(my_circle)

    plt.xlabel('')

    plt.ylabel('')

plt.show()
f,ax=plt.subplots(1,2,figsize=(22,8))

sns.countplot(y=scientist['ProveKnowledgeSelect'],ax=ax[0])

ax[0].set_title('How to prove my knowledge')

sns.countplot(scientist['JobSkillImportanceKaggleRanking'],ax=ax[1])

ax[1].set_title('Kaggle Rank')

plt.show()