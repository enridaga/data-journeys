
# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(color_codes=True)
import operator
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import plotly.offline as py
py.init_notebook_mode(connected=True)
import plotly.graph_objs as go
import plotly.tools as tls

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.
cvRates = pd.read_csv('../input/conversionRates.csv', encoding="ISO-8859-1")
freeForm = pd.read_csv('../input/freeformResponses.csv', encoding="ISO-8859-1")
data = pd.read_csv('../input/multipleChoiceResponses.csv', encoding="ISO-8859-1")
schema = pd.read_csv('../input/schema.csv', encoding="ISO-8859-1")
plt.figure(figsize=(12,8))
genders = data['GenderSelect'].value_counts()
sns.barplot(y=genders.index, x=genders.values, alpha=0.6)
plt.yticks(range(len(data['GenderSelect'].value_counts().index)), ['Male', 'Female','Different identity','Non-confirming'])
plt.title("Gender Distribution of the suervey participants", fontsize=16)
plt.xlabel("Number of participants", fontsize=16)
plt.ylabel("Gender", fontsize=16)
plt.show();
print('Proportion of women in this survey: {:0.2f}% '.format(100*len(data[data['GenderSelect']=='Female'])/len(data['GenderSelect'].dropna())))
print('Proportion of men in this survey: {:0.2f}% '.format(100*len(data[data['GenderSelect']=='Male'])/len(data['GenderSelect'].dropna())))

print('{} instances seem to be too old (>65 years old)'.format(len(data[data['Age']>65])))
print('{} instances seem to be too young (<15 years old)'.format(len(data[data['Age']<15])))
age=data[(data['Age']>=15) & (data['Age']<=65) ]
plt.figure(figsize=(10,8))
sns.boxplot( y=age['Age'],data=age)
plt.title("Age boxplot", fontsize=16)
plt.ylabel("Age", fontsize=16)
plt.show();

plt.figure(figsize=(12,8))
countries = data['Country'].value_counts().head(30)
sns.barplot(y=countries.index, x=countries.values, alpha=0.6)
plt.title("Country Distribution of the suervey participants", fontsize=16)
plt.xlabel("Number of participants", fontsize=16)
plt.ylabel("Country", fontsize=16)
plt.show();
print('{:0.2f}% of the instances are Americans'.format(100*len(data[data['Country']=='United States'])/len(data)))
print('{:0.2f}% of the instances are Indians'.format(100*len(data[data['Country']=='India'])/len(data)))
edu = data['FormalEducation'].value_counts()
labels = (np.array(edu.index))

values = (np.array((edu / edu.sum())*100))

trace = go.Pie(labels=labels, values=values,
              hoverinfo='label+percent',
               textfont=dict(size=20),
                showlegend=False)

layout = go.Layout(
    title='Formal Education of the survey participants'
)

data_trace = [trace]
fig = go.Figure(data=data_trace, layout=layout)
py.iplot(fig, filename="Formal_Education")

data['MajorSelect']=data['MajorSelect'].replace(to_replace ='Information technology, networking, or system administration',
                                                       value = 'Information tech / System admin', axis=0)
plt.figure(figsize=(12,8))
majors = data['MajorSelect'].value_counts()
sns.barplot(y=majors.index, x=majors.values, alpha=0.6)
plt.title("Majors of the survey participants", fontsize=16)
plt.xlabel("Number of participants", fontsize=16)
plt.ylabel("Major", fontsize=16)
plt.show();
plt.figure(figsize=(10,8))

temp=data['FirstTrainingSelect'].value_counts()
labels = temp.index
sizes = temp.values

plt.pie(sizes, labels=labels, autopct='%1.1f%%')
        
#draw a circle at the center of pie to make it look like a donut
centre_circle = plt.Circle((0,0),0.75,color='black', fc='white',linewidth=1.25)
fig = plt.gcf()
fig.gca().add_artist(centre_circle)


# Set aspect ratio to be equal so that pie is drawn as a circle.
plt.axis('equal')
plt.show();

data['EmploymentStatus']=data['EmploymentStatus'].replace(to_replace ='Independent contractor, freelancer, or self-employed',
                                                       value = 'Independent', axis=0)
plt.figure(figsize=(12,8))
status = data['EmploymentStatus'].value_counts()
sns.barplot(y=status.index, x=status.values, alpha=0.6)
plt.title("Employment status of the survey respondents", fontsize=16)
plt.xlabel("Number of participants", fontsize=16)
plt.ylabel("Employment status", fontsize=16)
plt.show();
print('{:0.2f}% of the instances are employed full-time'.format(100*len(data[data['EmploymentStatus']=='Employed full-time'])/len(data)))
status=['Employed full-time','Independent','Employed part-time']
print('{:0.2f}% of the instances are employed'.format(100*len(data[data.EmploymentStatus.isin(status)])/len(data)))
car = data['CareerSwitcher'].value_counts()
labels = (np.array(car.index))
proportions = (np.array((car / car.sum())*100))
colors = ['#FEBFB3', '#E1396C']

trace = go.Pie(labels=labels, values=proportions,
              hoverinfo='lbal+percent',
              marker = dict(colors=colors, 
                           line=dict(color='#000000', width=2)))
layout = go.Layout(
    title='Working people looking to switch careers to data science'
)

data_trace = [trace]
fig = go.Figure(data=data_trace, layout=layout)
py.iplot(fig, filename="Career_Switcher")
from wordcloud import (WordCloud, get_single_color_func)

#PS : Credits to Andreas Mueller, creator of wordcloud, for the following code of the class 'GroupedColorFunc'.
#He made the code fully public for people who want to use specific color for specific words and made an example.
#Source link : https://amueller.github.io/word_cloud/auto_examples/colored_by_group.html#sphx-glr-auto-examples-colored-by-group-py

class GroupedColorFunc(object):

    def __init__(self, color_to_words, default_color):
        self.color_func_to_words = [
            (get_single_color_func(color), set(words))
            for (color, words) in color_to_words.items()]

        self.default_color_func = get_single_color_func(default_color)

    def get_color_func(self, word):
        """Returns a single_color_func associated with the word"""
        try:
            color_func = next(
                color_func for (color_func, words) in self.color_func_to_words
                if word in words)
        except StopIteration:
            color_func = self.default_color_func

        return color_func

    def __call__(self, word, **kwargs):
        return self.get_color_func(word)(word, **kwargs)


#############################################################
# Get text data from the freeform
text = freeForm[pd.notnull(freeForm["KaggleMotivationFreeForm"])]["KaggleMotivationFreeForm"]

wc = WordCloud(collocations=False,height=800, width=600,  relative_scaling=0.2,random_state=74364).generate(" ".join(text))

color_to_words = {
    # words below will be colored with a green single color function
    '#00ff00': ['data', 'science', 'mining', 'big',
                'bigdata', 'machine', 'learning']
}

# Words that are not in any of the color_to_words values will be colored with grey
default_color = 'grey'

# Create a color function with multiple tones
grouped_color_func = GroupedColorFunc(color_to_words, default_color)

# Apply our color function
wc.recolor(color_func=grouped_color_func)

# Plot
plt.figure(figsize=(12,12))
plt.imshow(wc, interpolation='bilinear')
plt.axis("off");

t2=data[["WorkToolsFrequencyR","WorkToolsFrequencyPython"]].fillna(0)
t2.replace(to_replace=['Rarely','Sometimes','Often','Most of the time'], 
           value=[1,2,3,4], inplace=True)
t2['PythonVsR'] = [ 'R' if (freq1 >2 and freq1 > freq2) else
                    'Python' if (freq1<freq2 and freq2>2) else
                    'Both' if (freq1==freq2 and freq1 >2) else
                    'None' for (freq1,freq2) in zip(t2["WorkToolsFrequencyR"],t2["WorkToolsFrequencyPython"])]
data['PythonVsR']=t2['PythonVsR']

df = data[data['PythonVsR']!='None']
print("Python users: ",len(df[df['PythonVsR']=='Python']))
print("R users: ",len(df[df['PythonVsR']=='R']))
print("Python+R users: ",len(df[df['PythonVsR']=='Both']))

test=df[['Country','PythonVsR']]
test['agg']=np.ones(test.shape[0])
#test.to_excel('countries_excel.xlsx')
%%HTML
<div class='tableauPlaceholder' id='viz1510841993802' style='position: relative'><noscript><a href='#'><img alt='Proportions of Python and R coders by country ' src='https:&#47;&#47;public.tableau.com&#47;static&#47;images&#47;ka&#47;kaggle_kernel_final&#47;Feuille1&#47;1_rss.png' style='border: none' /></a></noscript><object class='tableauViz'  style='display:none;'><param name='host_url' value='https%3A%2F%2Fpublic.tableau.com%2F' /> <param name='embed_code_version' value='3' /> <param name='site_root' value='' /><param name='name' value='kaggle_kernel_final&#47;Feuille1' /><param name='tabs' value='no' /><param name='toolbar' value='yes' /><param name='static_image' value='https:&#47;&#47;public.tableau.com&#47;static&#47;images&#47;ka&#47;kaggle_kernel_final&#47;Feuille1&#47;1.png' /> <param name='animate_transition' value='yes' /><param name='display_static_image' value='yes' /><param name='display_spinner' value='yes' /><param name='display_overlay' value='yes' /><param name='display_count' value='yes' /><param name='filter' value='publish=yes' /></object></div>                <script type='text/javascript'>                    var divElement = document.getElementById('viz1510841993802');                    var vizElement = divElement.getElementsByTagName('object')[0];                    vizElement.style.width='100%';vizElement.style.height=(divElement.offsetWidth*0.75)+'px';                    var scriptElement = document.createElement('script');                    scriptElement.src = 'https://public.tableau.com/javascripts/api/viz_v1.js';                    vizElement.parentNode.insertBefore(scriptElement, vizElement);                </script>
df['Country'].fillna('Missing',inplace=True)
d_country={}
for country in df['Country'].unique(): #modify to unique values
    maskp = (df['Country'] == country )& (df['PythonVsR']=='Python')
    maskr = (df['Country'] == country )& (df['PythonVsR']=='R')
    maskb = (df['Country'] == country )& (df['PythonVsR']=='Both')
    d_country[country]={'Python':100*len(df[maskp])/len(df[df['Country']==country]) , 
                        'R':100*len(df[maskr])/len(df[df['Country']==country]),
                        'Both':100*len(df[maskb])/len(df[df['Country']==country])}
pd.DataFrame(d_country).transpose()

print('Table with percentage of use for each country')
print(pd.DataFrame(d_country).transpose().head(10).round(2))

df['WorkMethodsSelect']=df['WorkMethodsSelect'].fillna('None')
techniques = ['Bayesian Techniques','Data Visualization', 'Logistic Regression','Natural Language Processing',
 'kNN and Other Clustering','Neural Networks','PCA and Dimensionality Reduction',
 'Time Series Analysis', 'Text Analytics','Cross-Validation']

df['WorkAlgorithmsSelect']=df['WorkAlgorithmsSelect'].fillna('None')
algorithms = ['Bayesian Techniques','Decision Trees','Random Forests','Regression/Logistic Regression',
 'CNNs', 'RNNs', 'Gradient Boosted Machines','SVMs','GANs','Ensemble Methods']

d={}
for technique in techniques :
    d[technique]={'Python':0,'R':0,'Both':0}
    for (i,elem) in zip(range(df.shape[0]),df['WorkMethodsSelect']):
        if technique in elem : 
            d[technique][df['PythonVsR'].iloc[i]]+=1
    d[technique]['Python']=100*d[technique]['Python']/len(df[df['PythonVsR']=='Python'])
    d[technique]['R']=100*d[technique]['R']/len(df[df['PythonVsR']=='R'])
    d[technique]['Both']=100*d[technique]['Both']/len(df[df['PythonVsR']=='Both'])
    
d_algo={}
for algo in algorithms :
    d_algo[algo]={'Python':0,'R':0,'Both':0}
    for (i,elem) in zip(range(df.shape[0]),df['WorkAlgorithmsSelect']):
        if algo in elem : 
            d_algo[algo][df['PythonVsR'].iloc[i]]+=1
    d_algo[algo]['Python']=100*d_algo[algo]['Python']/len(df[df['PythonVsR']=='Python'])
    d_algo[algo]['R']=100*d_algo[algo]['R']/len(df[df['PythonVsR']=='R'])
    d_algo[algo]['Both']=100*d_algo[algo]['Both']/len(df[df['PythonVsR']=='Both'])
            
f,ax=plt.subplots(1,2,figsize=(20,10))

(pd.DataFrame(d)).transpose().plot(kind='barh',ax=ax[0])
ax[0].set_title("% of users using each method ")
ax[0].set_xlabel('')
ax[0].set_ylabel("Method")

(pd.DataFrame(d_algo)).transpose().plot(kind='barh',ax=ax[1])
ax[1].set_title("% of users using each algorithm ")
ax[1].set_xlabel('%')
ax[1].set_ylabel("Algorithm")

plt.tight_layout()
plt.show();
df['MLSkillsSelect'].fillna('None',inplace=True)
skills = ['Natural Language Processing', 'Computer Vision', 'Adversarial Learning',
          'Supervised Machine Learning (Tabular Data)', 'Reinforcement learning',
          'Unsupervised Learning', 'Outlier detection (e.g. Fraud detection)',
          'Time Series', 'Recommendation Engines']
d_skills={}
for skill in skills : 
    d_skills[skill]={'Python':0,'R':0,'Both':0}
    for (i,elem) in zip(range(df.shape[0]),df['MLSkillsSelect']):
        if skill in elem : 
            d_skills[skill][df['PythonVsR'].iloc[i]]+=1
    d_skills[skill]['Python']=100*d_skills[skill]['Python']/len(df[df['PythonVsR']=='Python'])
    d_skills[skill]['R']=100*d_skills[skill]['R']/len(df[df['PythonVsR']=='R'])
    d_skills[skill]['Both']=100*d_skills[skill]['Both']/len(df[df['PythonVsR']=='Both'])
    
(pd.DataFrame(d_skills)).transpose().plot(kind='barh',figsize=(12,8))
plt.ylabel("Machine Learning Skill", fontsize=13)
plt.xlabel("% of users", fontsize=13)
plt.title("% of users mastering ML skills by programming language", fontsize=16)
plt.show();

df['WorkDataTypeSelect'].fillna('None',inplace=True)
data_type = ['Relational data', 'Text data', 'Other',
             'Image data', 'Image data', 'Video data']
d_data={}
for dtype in data_type :
    d_data[dtype]={'Python':0,'R':0,'Both':0}
    for (i,elem) in zip(range(df.shape[0]),df['WorkDataTypeSelect']):
        if dtype in elem : 
            d_data[dtype][df['PythonVsR'].iloc[i]]+=1
    d_data[dtype]['Python']=100*d_data[dtype]['Python']/len(df[df['PythonVsR']=='Python'])
    d_data[dtype]['R']=100*d_data[dtype]['R']/len(df[df['PythonVsR']=='R'])
    d_data[dtype]['Both']=100*d_data[dtype]['Both']/len(df[df['PythonVsR']=='Both'])
    
(pd.DataFrame(d_data)).transpose().plot(kind='barh',figsize=(12,8))
plt.ylabel("Data Type", fontsize=13)
plt.xlabel("Percentages", fontsize=13)
plt.title("% of use of types of data by programming language", fontsize=13)
plt.show();
f,ax=plt.subplots(1,2,figsize=(20,10))

sns.countplot(y='EmployerIndustry', hue='PythonVsR',data=df,ax=ax[0])
ax[0].set_title("Number of Python/R coders per industry")
ax[0].set_xlabel('#')
ax[0].set_ylabel("Industry")

d_ind={}
for value in df['PythonVsR'].value_counts().index : 
    temp=df[df['PythonVsR']==value]
    d_ind[value]={}
    for industry in df['EmployerIndustry'].value_counts().index :
        d_ind[value][industry]=100*len(temp[temp['EmployerIndustry']==industry])/len(temp)

(pd.DataFrame(d_ind)).plot(kind='barh',ax=ax[1])
ax[1].set_title('% of Python/R coders per industry')
ax[1].set_xlabel('%')
ax[1].set_ylabel("Industry")

plt.tight_layout()
plt.show();


d_title={}
for value in df['PythonVsR'].value_counts().index : 
    temp=df[df['PythonVsR']==value]
    d_title[value]={}
    for title in df['CurrentJobTitleSelect'].value_counts().index :
        d_title[value][title]=100*len(temp[temp['CurrentJobTitleSelect']==title])/len(temp)

ax = (pd.DataFrame(d_title)).plot(kind='barh',figsize=(12,8))
ax.set_title('% of Python/R coders per role',fontsize=16)
ax.set_xlabel('%')
ax.set_ylabel("Job Titles",fontsize=14)

plt.show();

from scipy.stats import chi2_contingency

conti = pd.crosstab(data['MajorSelect'], data['CurrentJobTitleSelect'])
conti
p_value = chi2_contingency(conti,lambda_='log-likelihood')[1]
p_value
d_title={}
for value in df['PythonVsR'].value_counts().index : 
    temp=df[df['PythonVsR']==value]
    d_title[value]={}
    for title in df['JobFunctionSelect'].value_counts().index :
        d_title[value][title]=100*len(temp[temp['JobFunctionSelect']==title])/len(temp)

ax = (pd.DataFrame(d_title)).plot(kind='barh',figsize=(12,8))
ax.set_title('% of Python/R coders per function',fontsize=16)
ax.set_xlabel('%')
ax.set_ylabel("Main job function",fontsize=14)

plt.show();
d_task={}
tasks=['TimeGatheringData','TimeModelBuilding','TimeProduction','TimeVisualizing','TimeFindingInsights']
for task in tasks : 
    d_task[task]={'Python':df[df['PythonVsR']=='Python'][task].mean(),
                  'R':df[df['PythonVsR']=='R'][task].mean(),
                  'Both':df[df['PythonVsR']=='Both'][task].mean()}
    
(pd.DataFrame(d_task)).transpose().plot(kind='barh',figsize=(12,8))
plt.ylabel("Task", fontsize=13)
plt.xlabel("Percentage of time", fontsize=13)
plt.title("% of time devoted to specific tasks ", fontsize=16)
plt.show();
df['WorkProductionFrequency']=df['WorkProductionFrequency'].fillna("Don't know")
d_prod={}
for value in df['PythonVsR'].value_counts().index : 
    temp=df[df['PythonVsR']==value]
    d_prod[value]={}
    for frequency in df['WorkProductionFrequency'].value_counts().index :
        d_prod[value][frequency]=100*len(temp[temp['WorkProductionFrequency']==frequency])/len(temp)

(pd.DataFrame(d_prod)).plot(kind='barh',figsize=(12,8))
plt.ylabel("Frequency", fontsize=13)
plt.xlabel("Percentages", fontsize=13)
plt.title("Proportion of R/Python coders by frequency of push to production  ", fontsize=16)
plt.show();
d_tenure={}
for value in df['PythonVsR'].value_counts().index : 
    temp=df[df['PythonVsR']==value]
    d_tenure[value]={}
    for dur in df['Tenure'].value_counts().index :
        d_tenure[value][dur]=100*len(temp[temp['Tenure']==dur])/len(temp)

ax = (pd.DataFrame(d_tenure)).plot(kind='barh',figsize=(12,8))
ax.set_title('% of Python/R coders per function',fontsize=16)
ax.set_xlabel('%')
ax.set_ylabel("Main job function",fontsize=14)

plt.show();
df['LanguageRecommendationSelect'].fillna('Other',inplace=True)
plt.figure(figsize=(12,8))
sns.countplot(y='LanguageRecommendationSelect',hue='PythonVsR',data=df)
plt.ylabel("Language", fontsize=13)
plt.xlabel("Number of recommenders", fontsize=13)
plt.title("Recommended language", fontsize=13)
plt.show();

mask1=(df['LanguageRecommendationSelect'] == 'R')& (df['PythonVsR']=='Python')
print('Proportion of Python users who recommend R as the first language to learn: {:0.2f}%'.format(100*len(df[mask1])/len(df[df['PythonVsR']=='Python'])))

mask1=(df['LanguageRecommendationSelect'] == 'Python')& (df['PythonVsR']=='R')
print('Proportion of R users who recommend Python as the first language to learn: {:0.2f}%'.format(100*len(df[mask1])/len(df[df['PythonVsR']=='R'])))
demographic_features = ['GenderSelect','Country','Age',
                        'FormalEducation','MajorSelect','ParentsEducation',
                        'EmploymentStatus', 'CurrentJobTitleSelect',
                        'DataScienceIdentitySelect','CodeWriter',
                        'CurrentEmployerType','JobFunctionSelect',
                        'SalaryChange','RemoteWork','WorkMLTeamSeatSelect',
                        'Tenure','EmployerIndustry','EmployerSize','PythonVsR',
                        'CompensationAmount']
data_dem = data[demographic_features]
data_dem.head(5)
#Convert all salaries to floats
data_dem['CompensationAmount'] = data_dem['CompensationAmount'].fillna(0)
data_dem['CompensationAmount'] = data_dem.CompensationAmount.apply(lambda x: 0 if (pd.isnull(x) or (x=='-') or (x==0))
                                                       else float(x.replace(',','')))
#Remove Outliers
data_dem = data_dem[(data_dem['CompensationAmount']>5000) & (data_dem['CompensationAmount']<1000000)]
data_dem = data_dem[data_dem['Country']=='United States']

plt.subplots(figsize=(15,8))
sns.distplot(data_dem['CompensationAmount'])
plt.title('Income histograms and fitted distribtion',size=15)
plt.show();

print('The median salary for US data scientist: {} USD'.format(data_dem['CompensationAmount'].median()
))
print('The mean salary for US data scientist: {:0.2f} USD'.format(data_dem['CompensationAmount'].mean()
))

plt.figure(figsize=(15,8))
sns.violinplot(x='CompensationAmount', data=data_dem)
plt.title("Salary distribution for US data scientists", fontsize=16)
plt.xlabel("Annual Salary", fontsize=16)
plt.show();
temp=data_dem[data_dem.GenderSelect.isin(['Male','Female'])]
plt.figure(figsize=(10,8))
sns.violinplot( y='CompensationAmount', x='GenderSelect',data=temp)
plt.title("Salary distribution Vs Gender", fontsize=16)
plt.ylabel("Annual Salary", fontsize=16)
plt.xlabel("Gender", fontsize=16)
plt.show();
titles=list(data_dem['FormalEducation'].value_counts().index)
temp=data_dem[data_dem.FormalEducation.isin(titles)]
plt.figure(figsize=(10,8))
sns.boxplot( x='CompensationAmount', y='FormalEducation',data=temp)
plt.title("Salary distribution VS Academic degrees", fontsize=16)
plt.xlabel("Annual Salary", fontsize=16)
plt.ylabel("Academic degree", fontsize=16)
plt.show();
titles=list(data_dem['CurrentJobTitleSelect'].value_counts().index)
temp=data_dem[data_dem.CurrentJobTitleSelect.isin(titles)]
plt.figure(figsize=(10,8))
sns.violinplot( x='CompensationAmount', y='CurrentJobTitleSelect',data=temp)
plt.title("Salary distribution VS Job Titles", fontsize=16)
plt.xlabel("Annual Salary", fontsize=16)
plt.ylabel("Job Titles", fontsize=16)
plt.show();
data_dem['CompensationAmount'] = pd.cut(data_dem['CompensationAmount'],bins=[0,130000,1000000],
                                            include_lowest=True,labels=[1,2])
data_dem['Age'] = pd.cut(data_dem['Age'],bins=[0,18,25,30,35,40,50,60,100],
                           include_lowest=True,labels=[1,2,3,4,5,6,7,8])
data_dem.drop('Country',axis=1,inplace=True)
### NOT WORKING ON KAGGLE SERVERS (no module prince)####
#import prince
#np.random.seed(42)
#mca = prince.MCA(data_viz, n_components=2,use_benzecri_rates=True)
#mca.plot_rows(show_points=True, show_labels=False, color_by='CompensationAmount', ellipse_fill=True)

# I have uploaded an image instead.
"""If you want to execute the following two blocks of code and have the plot above,
install the package 'prince', copy all the code and uncomment it, you'll have the same outputs.
P.S : Don't forger the random seed !"""

#projections=mca.row_principal_coordinates
#projections.columns=['ax1','ax2']
#projections['target']=y.iloc[length]

#msk_p = ((projections['ax1']>-0.70) & (projections['ax1']<-0.45 )) & ((projections['ax2']<0.66) &(projections['ax2']>0.50))
#samples_p=projections[msk_p]
#indexes_p = samples_p.index #[133, 247, 499, 576, 2375, 3578, 3606, 3876, 5758, 6059, 10155, 10514, 11552, 13438, 15631]
#ex_p=data_dem.loc[indexes_p]

ex_p=data_dem.loc[[133, 247, 499, 576, 2375, 3578, 3606, 3876, 5758, 6059, 10155, 10514, 11552, 13438, 15631]]
ex_p.head(10)
#msk_r = ((projections['ax1']>0.2) & (projections['ax1']<0.7 )) & ((projections['ax2']<-0.80) &(projections['ax2']>-1.10))
#samples_r=projections[msk_r]
#indexes_r = samples_r.index  #[445, 3273, 4751, 4803, 4960, 11071, 11528, 13663, 13880]
#ex_r = data_dem.loc[indexes_r]

indexes_r=[445, 3273, 4751, 4803, 4960, 11071, 11528, 13663, 13880]
ex_r = data_dem.loc[indexes_r]

ex_r
temp=data_dem

target = temp['CompensationAmount']
target.replace(to_replace=[1,2], value=[0,1],inplace=True )
temp.drop('CompensationAmount',axis=1,inplace=True)
temp2=pd.get_dummies(data=temp,columns=list(temp))

np.random.seed(42)
perm = np.random.permutation(temp2.shape[0])
X_train , y_train = temp2.iloc[perm[0:round(0.8*temp2.shape[0])]] , target.iloc[perm[0:round(0.8*temp2.shape[0])]]
X_test , y_test = temp2.iloc[perm[round(0.8*temp2.shape[0])::]] , target.iloc[perm[round(0.8*temp2.shape[0])::]]

print('Number of US kagglers with an income lower than 130k$ : {}'.format(len(target)-target.sum()))
print('Number of US kagglers with an income higher than 130k$ : {}'.format(target.sum()))
from sklearn.metrics import f1_score, precision_score, recall_score , accuracy_score, roc_curve, auc
from sklearn.linear_model import LogisticRegression
import operator

#Since we're dealing with a sparse matrix, I think it's better to use l1-penalty that shrinks 
#coefficients to 0 to keep the most important features

clf = LogisticRegression(penalty='l1', C=0.05, tol=0.0001, random_state=42, solver='liblinear')
clf.fit(X_train,y_train)

coefs=np.transpose(clf.coef_)

coef_l1_LR = clf.coef_.ravel()
sparsity_l1_LR = np.mean(coef_l1_LR == 0) * 100
print("Sparsity achieved: %.2f%%" % sparsity_l1_LR)

feature_importance={}
for (feature,k) in zip(list(X_train),range(len(coefs))) : 
        feature_importance[feature]=abs(coefs[k])
sorted_features = sorted(feature_importance.items(), key=operator.itemgetter(1))
top5 = sorted_features[-5::]
top5
clf = LogisticRegression(penalty='l1', C=1, tol=0.0001, random_state=42, solver='liblinear')
clf.fit(X_train,y_train)
y_p = clf.predict(X_test)

accuracy , precision, recall  = accuracy_score(y_test,y_p), precision_score(y_test,y_p) , recall_score(y_test,y_p)
print('Accuracy achieved by the classifier: {:0.2f}%'.format(100*accuracy))
print('Precision achieved by the classifier: {:0.2f}%'.format(100*precision))
print('Recall achieved by the classifier: {:0.2f}%'.format(100*recall))
probs = clf.predict_proba(X_test)
preds = probs[:,1]
fpr, tpr, threshold = roc_curve(y_test, preds)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(15,8))
plt.title('Receiver Operating Characteristic')
plt.plot(fpr, tpr, 'b', label = 'AUC = %0.2f' % roc_auc)
plt.legend(loc = 'lower right')
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()
df_students=data[data['StudentStatus']=='Yes']
df_ds=data[(data['LearningDataScience']=="Yes, but data science is a small part of what I'm focused on learning") |
            (data['LearningDataScience']=="Yes, I'm focused on learning mostly data science skills")]
df_c=data[data['CareerSwitcher']=='Yes']
df_e=data[data['EmploymentStatus']=='Not employed, but looking for work']

learners=pd.concat((df_students,df_ds,df_c,df_e))
learners = learners[~learners.index.duplicated(keep='first')]

print('{} participants on this survey are learners.'.format(len(learners)))
print('In other words, {:0.2f}% of the participants on this survey are learners.'.format(100*len(learners)/len(data)))
sexe = learners['GenderSelect'].value_counts()
labels = (np.array(sexe.index))
proportions = (np.array((sexe / sexe.sum())*100))

trace = go.Pie(labels=labels, values=proportions,
              hoverinfo='lbal+percent')
layout = go.Layout(
    title='Gender distrubiton of learners'
)

data_trace = [trace]
fig = go.Figure(data=data_trace, layout=layout)
py.iplot(fig, filename="Career_Switcher")


print("Learners' median age", learners['Age'].median() )
plt.figure(figsize=(12,8))
countries = learners['Country'].value_counts().head(30)
sns.barplot(y=countries.index, x=countries.values, alpha=0.6)
plt.title("Population of learners in each country", fontsize=16)
plt.xlabel("Number of respondents", fontsize=16)
plt.ylabel("Country", fontsize=16)
plt.show();
d_pcountries = {}
for value in data['Country'].value_counts().index:
    d_pcountries[value]=100*len(learners[learners['Country']==value])/len(data[data['Country']==value])
learners_p=pd.DataFrame.from_dict(d_pcountries, orient='index')
learners_p = learners_p.reset_index(drop=False)
learners_p.rename(columns = {'index':'Country',0:'% of learners'},inplace=True)

LOCDATA="""COUNTRY,GDP (BILLIONS),CODE
Afghanistan,21.71,AFG
Albania,13.40,ALB
Algeria,227.80,DZA
American Samoa,0.75,ASM
Andorra,4.80,AND
Angola,131.40,AGO
Anguilla,0.18,AIA
Antigua and Barbuda,1.24,ATG
Argentina,536.20,ARG
Armenia,10.88,ARM
Aruba,2.52,ABW
Australia,1483.00,AUS
Austria,436.10,AUT
Azerbaijan,77.91,AZE
"Bahamas, The",8.65,BHM
Bahrain,34.05,BHR
Bangladesh,186.60,BGD
Barbados,4.28,BRB
Belarus,75.25,BLR
Belgium,527.80,BEL
Belize,1.67,BLZ
Benin,9.24,BEN
Bermuda,5.20,BMU
Bhutan,2.09,BTN
Bolivia,34.08,BOL
Bosnia and Herzegovina,19.55,BIH
Botswana,16.30,BWA
Brazil,2244.00,BRA
British Virgin Islands,1.10,VGB
Brunei,17.43,BRN
Bulgaria,55.08,BGR
Burkina Faso,13.38,BFA
Burma,65.29,MMR
Burundi,3.04,BDI
Cabo Verde,1.98,CPV
Cambodia,16.90,KHM
Cameroon,32.16,CMR
Canada,1794.00,CAN
Cayman Islands,2.25,CYM
Central African Republic,1.73,CAF
Chad,15.84,TCD
Chile,264.10,CHL
"People 's Republic of China",10360.00,CHN
Colombia,400.10,COL
Comoros,0.72,COM
"Congo, Democratic Republic of the",32.67,COD
"Congo, Republic of the",14.11,COG
Cook Islands,0.18,COK
Costa Rica,50.46,CRI
Cote d'Ivoire,33.96,CIV
Croatia,57.18,HRV
Cuba,77.15,CUB
Curacao,5.60,CUW
Cyprus,21.34,CYP
Czech Republic,205.60,CZE
Denmark,347.20,DNK
Djibouti,1.58,DJI
Dominica,0.51,DMA
Dominican Republic,64.05,DOM
Ecuador,100.50,ECU
Egypt,284.90,EGY
El Salvador,25.14,SLV
Equatorial Guinea,15.40,GNQ
Eritrea,3.87,ERI
Estonia,26.36,EST
Ethiopia,49.86,ETH
Falkland Islands (Islas Malvinas),0.16,FLK
Faroe Islands,2.32,FRO
Fiji,4.17,FJI
Finland,276.30,FIN
France,2902.00,FRA
French Polynesia,7.15,PYF
Gabon,20.68,GAB
"Gambia, The",0.92,GMB
Georgia,16.13,GEO
Germany,3820.00,DEU
Ghana,35.48,GHA
Gibraltar,1.85,GIB
Greece,246.40,GRC
Greenland,2.16,GRL
Grenada,0.84,GRD
Guam,4.60,GUM
Guatemala,58.30,GTM
Guernsey,2.74,GGY
Guinea-Bissau,1.04,GNB
Guinea,6.77,GIN
Guyana,3.14,GUY
Haiti,8.92,HTI
Honduras,19.37,HND
Hong Kong,292.70,HKG
Hungary,129.70,HUN
Iceland,16.20,ISL
India,2048.00,IND
Indonesia,856.10,IDN
Iran,402.70,IRN
Iraq,232.20,IRQ
Ireland,245.80,IRL
Isle of Man,4.08,IMN
Israel,305.00,ISR
Italy,2129.00,ITA
Jamaica,13.92,JAM
Japan,4770.00,JPN
Jersey,5.77,JEY
Jordan,36.55,JOR
Kazakhstan,225.60,KAZ
Kenya,62.72,KEN
Kiribati,0.16,KIR
"Korea, North",28.00,PRK
"Korea, South",1410.00,KOR
Kosovo,5.99,KSV
Kuwait,179.30,KWT
Kyrgyzstan,7.65,KGZ
Laos,11.71,LAO
Latvia,32.82,LVA
Lebanon,47.50,LBN
Lesotho,2.46,LSO
Liberia,2.07,LBR
Libya,49.34,LBY
Liechtenstein,5.11,LIE
Lithuania,48.72,LTU
Luxembourg,63.93,LUX
Macau,51.68,MAC
Macedonia,10.92,MKD
Madagascar,11.19,MDG
Malawi,4.41,MWI
Malaysia,336.90,MYS
Maldives,2.41,MDV
Mali,12.04,MLI
Malta,10.57,MLT
Marshall Islands,0.18,MHL
Mauritania,4.29,MRT
Mauritius,12.72,MUS
Mexico,1296.00,MEX
"Micronesia, Federated States of",0.34,FSM
Moldova,7.74,MDA
Monaco,6.06,MCO
Mongolia,11.73,MNG
Montenegro,4.66,MNE
Morocco,112.60,MAR
Mozambique,16.59,MOZ
Namibia,13.11,NAM
Nepal,19.64,NPL
Netherlands,880.40,NLD
New Caledonia,11.10,NCL
New Zealand,201.00,NZL
Nicaragua,11.85,NIC
Nigeria,594.30,NGA
Niger,8.29,NER
Niue,0.01,NIU
Northern Mariana Islands,1.23,MNP
Norway,511.60,NOR
Oman,80.54,OMN
Pakistan,237.50,PAK
Palau,0.65,PLW
Panama,44.69,PAN
Papua New Guinea,16.10,PNG
Paraguay,31.30,PRY
Peru,208.20,PER
Philippines,284.60,PHL
Poland,552.20,POL
Portugal,228.20,PRT
Puerto Rico,93.52,PRI
Qatar,212.00,QAT
Romania,199.00,ROU
Russia,2057.00,RUS
Rwanda,8.00,RWA
Saint Kitts and Nevis,0.81,KNA
Saint Lucia,1.35,LCA
Saint Martin,0.56,MAF
Saint Pierre and Miquelon,0.22,SPM
Saint Vincent and the Grenadines,0.75,VCT
Samoa,0.83,WSM
San Marino,1.86,SMR
Sao Tome and Principe,0.36,STP
Saudi Arabia,777.90,SAU
Senegal,15.88,SEN
Serbia,42.65,SRB
Seychelles,1.47,SYC
Sierra Leone,5.41,SLE
Singapore,307.90,SGP
Sint Maarten,304.10,SXM
Slovakia,99.75,SVK
Slovenia,49.93,SVN
Solomon Islands,1.16,SLB
Somalia,2.37,SOM
South Africa,341.20,ZAF
South Sudan,11.89,SSD
Spain,1400.00,ESP
Sri Lanka,71.57,LKA
Sudan,70.03,SDN
Suriname,5.27,SUR
Swaziland,3.84,SWZ
Sweden,559.10,SWE
Switzerland,679.00,CHE
Syria,64.70,SYR
Taiwan,529.50,TWN
Tajikistan,9.16,TJK
Tanzania,36.62,TZA
Thailand,373.80,THA
Timor-Leste,4.51,TLS
Togo,4.84,TGO
Tonga,0.49,TON
Trinidad and Tobago,29.63,TTO
Tunisia,49.12,TUN
Turkey,813.30,TUR
Turkmenistan,43.50,TKM
Tuvalu,0.04,TUV
Uganda,26.09,UGA
Ukraine,134.90,UKR
United Arab Emirates,416.40,ARE
United Kingdom,2848.00,GBR
United States,17420.00,USA
Uruguay,55.60,URY
Uzbekistan,63.08,UZB
Vanuatu,0.82,VUT
Venezuela,209.20,VEN
Vietnam,187.80,VNM
Virgin Islands,5.08,VGB
West Bank,6.64,WBG
Yemen,45.45,YEM
Zambia,25.61,ZMB
Zimbabwe,13.74,ZWE
    """

with open("location_map.csv", "w") as ofile:
    ofile.write(LOCDATA)
loc_df = pd.read_csv("./location_map.csv")
new_df = pd.merge(learners_p, loc_df, left_on="Country", right_on="COUNTRY")
new_df = new_df[['Country','CODE','% of learners']]

data_t = [ dict(
        type = 'choropleth',
        locations = new_df['CODE'],
        z = new_df['% of learners'],
        text = new_df['Country'],
        #colorscale = [[0,"rgb(5, 10, 172)"],[10,"rgb(40, 60, 190)"],[20,"rgb(70, 100, 245)"],\
        #    [30,"rgb(90, 120, 245)"],[40,"rgb(200, 200, 200)"],[4500,"rgb(220, 220, 220)"]],
        colorscale = [[0,"rgb(210, 210, 210)"], [4500,"rgb(220, 220, 220)"]],
        autocolorscale = False,
        reversescale = True,
        marker = dict(
            line = dict (
                color = 'rgb(180,180,180)',
                width = 0.5
            ) ),
        colorbar = dict(
            autotick = False,
            title = 'Proportion of learners (in%)'),
      ) ]

layout = dict(
    title = 'Country wise proportion of learners',
    geo = dict(
        showframe = False,
        showcoastlines = True,
        projection = dict(
            type = 'Mercator'
        )
    )
)

fig = dict( data=data_t, layout=layout )
py.iplot( fig, validate=False, filename='d3-world-map' )

edu = learners['FormalEducation'].value_counts()
labels = (np.array(edu.index))
values = (np.array((edu / edu.sum())*100))

trace = go.Pie(labels=labels, values=values,
              hoverinfo='label+percent',
               textfont=dict(size=20),
                showlegend=False)

layout = go.Layout(
    title='Formal Education of learners respondents'
)

data_trace = [trace]
fig = go.Figure(data=data_trace, layout=layout)
py.iplot(fig, filename="Formal_Education2")
d_plat={}
platforms = ['College/University','Kaggle','Online courses','Arxiv','Company internal','Textbook',
             'Personal Projects','Stack Overflow Q&A','Blogs','Tutoring','Conferences','YouTube Videos']
for platform in platforms : 
    d_plat[platform]=0
    for elem in learners['LearningPlatformSelect'].fillna('Other/Missing'):
        if platform in elem : 
            d_plat[platform]+=1
            

s=pd.DataFrame.from_dict(data=d_plat,orient='index')
s.sort_values(by=list(s),axis=0, ascending=True, inplace=True)
ax = s.plot(kind='bar',figsize=(15,8),width=0.8,align='center')
ax.legend_.remove()
ax.set_title("Learners' platforms use",fontsize=16)
ax.set_xlabel("Platforms", fontsize=16)
ax.set_ylabel("Number of users", fontsize=16)
for tick in ax.get_xticklabels():
    tick.set_rotation(0)
    tick.set_fontsize(8)
plt.tight_layout()
data_young = learners[(learners['Age']<=22) ]
d_plat2={}
for platform in platforms : 
    d_plat2[platform]=0
    for elem in data_young['LearningPlatformSelect'].fillna('Other/Missing'):
        if platform in elem : 
            d_plat2[platform]+=1

s=pd.DataFrame.from_dict(data=d_plat2,orient='index')
s.sort_values(by=list(s),axis=0, ascending=True, inplace=True)
ax = s.plot(kind='bar',figsize=(15,8),width=0.8,align='center')
ax.legend_.remove()
ax.set_title("Young Learners' platforms use",fontsize=16)
ax.set_xlabel("Platforms", fontsize=16)
ax.set_ylabel("Number of users", fontsize=16)
for tick in ax.get_xticklabels():
    tick.set_rotation(0)
    tick.set_fontsize(8)
plt.tight_layout()
data_young = learners[(learners['Age']<=22) & (learners['Country']=='United States')]
d_plat2={}
for platform in platforms : 
    d_plat2[platform]=0
    for elem in data_young['LearningPlatformSelect'].fillna('Other/Missing'):
        if platform in elem : 
            d_plat2[platform]+=1


s=pd.DataFrame.from_dict(data=d_plat2,orient='index')
s.sort_values(by=list(s),axis=0, ascending=True, inplace=True)
ax = s.plot(kind='bar',figsize=(15,8),width=0.8,align='center')
ax.legend_.remove()
ax.set_title("USA young learners' platforms use",fontsize=16)
ax.set_xlabel("Platforms", fontsize=16)
ax.set_ylabel("Number of users", fontsize=16)
for tick in ax.get_xticklabels():
    tick.set_rotation(0)
    tick.set_fontsize(8)
plt.tight_layout()
d_useful={}
plat_use=['LearningPlatformUsefulnessArxiv','LearningPlatformUsefulnessBlogs','LearningPlatformUsefulnessCollege',
          'LearningPlatformUsefulnessCompany','LearningPlatformUsefulnessConferences',
          'LearningPlatformUsefulnessKaggle','LearningPlatformUsefulnessCourses','LearningPlatformUsefulnessProjects',
          'LearningPlatformUsefulnessSO','LearningPlatformUsefulnessTextbook','LearningPlatformUsefulnessTutoring',
          'LearningPlatformUsefulnessYouTube']
for plat in plat_use : 
    L=learners[plat].value_counts()
    d_useful[plat]={'Very useful':L.loc['Very useful'],
                    'Somewhat useful':L.loc['Somewhat useful'],
                    'Not Useful':L.loc['Not Useful']}

s=pd.DataFrame.from_dict(data=d_useful,orient='index')
s.sort_values(by=list(s),axis=0, ascending=True, inplace=True)
ax = s.plot(kind='barh',figsize=(15,8),width=0.8,align='center')
ax.set_title("Platforms usefulness according to learners",fontsize=16)
ax.set_ylabel("Platforms", fontsize=16)
ax.set_xlabel("Number of users", fontsize=16)
for tick in ax.get_xticklabels():
    tick.set_rotation(0)
    tick.set_fontsize(13)
plt.tight_layout()
d_online={}
online_plat= ['Coursera','Udacity','edX',
              'DataCamp','Other']
for plat in online_plat : 
    d_online[plat]=0
    for elem in learners['CoursePlatformSelect'].fillna('Missing'):
        if plat in elem :
            d_online[plat]+=1

online = pd.DataFrame.from_dict(d_online,orient='index')

labels = (np.array(online.index))
proportions = np.array((online[0] / online[0].sum())*100)

trace = go.Pie(labels=labels, values=proportions,
              hoverinfo='lbal+percent')

layout = go.Layout(
    title='Online Platforms popularity'
)

data_trace = [trace]
fig = go.Figure(data=data_trace, layout=layout)
py.iplot(fig, filename="Online_plat")

f,ax=plt.subplots(1,2,figsize=(20,10))

tools = learners['MLToolNextYearSelect'].value_counts().head(20)
sns.barplot(y=tools.index, x=tools.values, alpha=0.6,ax=ax[0])
ax[0].set_title('DS tools Kagglers want to learn')
ax[0].set_xlabel('')
ax[0].set_ylabel('Tools')

methods = learners['MLMethodNextYearSelect'].value_counts()
sns.barplot(y=methods.index, x=methods.values, alpha=0.6,ax=ax[1])
ax[1].set_title('ML Methods Kagglers want to learn')
ax[1].set_xlabel('')
ax[1].set_ylabel('ML Methods')

plt.tight_layout()
plt.show()


plt.figure(figsize=(12,8))
time = data['TimeSpentStudying'].value_counts()
sns.barplot(y=time.index, x=time.values, alpha=0.6)
plt.title("Average hours per week spent on DS learning", fontsize=16)
plt.xlabel("Number of learners", fontsize=16)
plt.ylabel("Number of hours per week", fontsize=16)
plt.show();
start = data['LearningDataScienceTime'].value_counts()
labels = (np.array(start.index))
values = (np.array((start / edu.sum())*100))

trace = go.Pie(labels=labels, values=values,
              hoverinfo='label+percent',
               textfont=dict(size=20))

layout = go.Layout(
    title='Years invested in Data Science Learning'
)

data_trace = [trace]
fig = go.Figure(data=data_trace, layout=layout)
py.iplot(fig, filename="nb_yers")
df2=data
d_jobskills={}
job_skills = ['JobSkillImportanceDegree','JobSkillImportancePython','JobSkillImportanceR',
              'JobSkillImportanceKaggleRanking','JobSkillImportanceMOOC']

for skill in job_skills : 
    L=df2[skill].value_counts()
    d_jobskills[skill]={'Necessary':L.loc['Necessary'],
                        'Nice to have':L.loc['Nice to have'],
                        'Unnecessary':L.loc['Unnecessary']}


(pd.DataFrame(d_jobskills)).transpose().plot(kind='barh',figsize=(10,8))
plt.title("Most important skills for a DS Job", fontsize=16)          
plt.xlabel("Number of learners", fontsize=16)
plt.ylabel("Skills", fontsize=16)

plt.show();

plt.figure(figsize=(12,8))
proof = learners['ProveKnowledgeSelect'].value_counts()
sns.barplot(y=proof.index, x=proof.values, alpha=0.6)
plt.title("Most important proofs of DS knowledge", fontsize=16)
plt.xlabel("Number of learners", fontsize=16)
plt.ylabel("Proof of knowledge", fontsize=16)
plt.show();
job_s = learners['JobSearchResource'].value_counts()
labels = (np.array(job_s.index))
values = (np.array((job_s / job_s.sum())*100))

trace = go.Pie(labels=labels, values=values,
              hoverinfo='label+percent',
               textfont=dict(size=20),
                showlegend=False)

layout = go.Layout(
    title='Most used resources for finding a DS job'
)

data_trace = [trace]
fig = go.Figure(data=data_trace, layout=layout)
py.iplot(fig, filename="Job_resource")
job_s = learners['JobHuntTime'].value_counts()
labels = (np.array(job_s.index))
values = (np.array((job_s / job_s.sum())*100))

trace = go.Pie(labels=labels, values=values,
              hoverinfo='label+percent',
               textfont=dict(size=20),
                showlegend=True)

layout = go.Layout(title='Hours per week spent  looking for a data science job?'
)

data_trace = [trace]
fig = go.Figure(data=data_trace, layout=layout)
py.iplot(fig, filename="Job_resource")
d_criterias={}
criterias_job=['JobFactorLearning','JobFactorSalary','JobFactorOffice','JobFactorLanguages',
               'JobFactorCommute','JobFactorManagement','JobFactorExperienceLevel',
               'JobFactorDepartment','JobFactorTitle','JobFactorCompanyFunding','JobFactorImpact',
               'JobFactorRemote','JobFactorIndustry','JobFactorLeaderReputation','JobFactorDiversity',
               'JobFactorPublishingOpportunity']
for criteria in criterias_job : 
    L=df2[criteria].value_counts()
    d_criterias[criteria]={'Very Important':L.loc['Very Important'],
                           'Somewhat important':L.loc['Somewhat important'],
                           'Not important':L.loc['Not important']}
    
s=pd.DataFrame.from_dict(data=d_criterias,orient='index')
s.sort_values(by=list(s),axis=0, ascending=True, inplace=True)
ax = s.plot(kind='barh',figsize=(15,8),width=0.8,align='center')
ax.set_title("Most important factors for learners during job hunting",fontsize=16)
ax.set_ylabel("Factors", fontsize=16)
ax.set_xlabel("Number of learners", fontsize=16)
for tick in ax.get_xticklabels():
    tick.set_rotation(0)
    tick.set_fontsize(13)
plt.tight_layout()