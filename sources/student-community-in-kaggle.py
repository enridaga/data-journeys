
# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import os
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib import rc
from matplotlib import rcParams
from IPython.display import Image


sns.set_style('whitegrid')
### matplotlib inline


# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output
def set_size(w,h, ax=None):
    """ w, h: width, height in inches """
    if not ax: ax=plt.gca()
    l = ax.figure.subplotpars.left
    r = ax.figure.subplotpars.right
    t = ax.figure.subplotpars.top
    b = ax.figure.subplotpars.bottom
    figw = float(w)/(r-l)
    figh = float(h)/(t-b)
    ax.figure.set_size_inches(figw, figh)
### Open datasets
b=pd.read_csv('/kaggle/input/kaggle-survey-2019/multiple_choice_responses.csv')
c=b[1:]
d=pd.read_csv('/kaggle/input/kaggle-survey-2018/multipleChoiceResponses.csv')
da=d[1:]
e=pd.read_csv('/kaggle/input/kaggle-survey-2017/multipleChoiceResponses.csv',encoding='ISO-8859-1')
ea=e[1:]
#Students 
st7=ea[ea['StudentStatus']=='Yes']
st8=da[da['Q6']=='Student']
st9=c[c['Q5']=='Student']

countries=c.Q3.value_counts().reset_index()
countries=pd.DataFrame(countries)
countries=countries.rename({'index':'country', 'Q3':'N° of answers'}, axis='columns')

#India and USA
India=c[c['Q3']=='India']
USA=c[c['Q3']=='United States of America']
both=c[(c['Q3']=='India') | (c['Q3']=='United States of America')]
both17=ea[(ea['Country']=='India') | ((ea['Country']=='United States'))]
both18=da[(da['Q3']=='India') | (da['Q3']=='United States of America')]
India17=ea[ea['Country']=='India']
USA17=ea[ea['Country']=='United States']
India18=c[c['Q3']=='India']
USA18=c[c['Q3']=='United States of America']

group=both.groupby(['Q3']).Q1.value_counts()
group=pd.DataFrame(group)
group=group.rename(columns={'Q1':'counting'})
group=group.reset_index()


#Age

def New_age1(row):
    if row['Age'] == 18.0:
      return '18-21'
    if row['Age'] == 19.0:
      return '18-21'
    if row['Age'] == 20.0:
      return '18-21'
    if row['Age'] == 21.0:
      return '18-21'


    if row['Age'] == 22.0:
      return '22-24'
    if row['Age'] == 23.0:
      return '22-24'
    if row['Age'] == 24.0:
      return '22-24'
  
    if row['Age'] == 25.0:
      return '25-29'
    if row['Age'] == 26.0:
      return '25-29'
    if row['Age'] == 27.0:
      return '25-29'
    if row['Age'] == 28.0:
      return '25-29'
    if row['Age'] == 29.0:
      return '25-29'

    if row['Age'] == 30.0:
      return '30-34'
    if row['Age'] == 31.0:
      return '30-34'
    if row['Age'] == 32.0:
      return '30-34'
    if row['Age'] == 33.0:
      return '30-34'
    if row['Age'] == 34.0:
      return '30-34'

    
    if row['Age'] == 35.0:
      return '35-39'
    if row['Age'] == 36.0:
      return '35-39'
    if row['Age'] == 37.0:
      return '35-39'
    if row['Age'] == 38.0:
      return '35-39'
    if row['Age'] == 39.0:
      return '35-39'
 
    if row['Age'] == 40.0:
      return '40-44'
    if row['Age'] == 41.0:
      return '40-44'
    if row['Age'] == 42.0:
      return '40-44'
    if row['Age'] == 43.0:
      return '40-44'
    if row['Age'] == 44.0:
      return '40-44'

  
    if row['Age'] == 45.0:
      return '45-49'
    if row['Age'] == 46.0:
      return '45-49'
    if row['Age'] == 47.0:
      return '45-49'
    if row['Age'] == 48.0:
      return '45-49'
    if row['Age'] == 49.0:
      return '45-49'



    if row['Age'] == 50.0:
      return '50-54'
    if row['Age'] == 51.0:
      return '50-54'
    if row['Age'] == 52.0:
      return '50-54'
    if row['Age'] == 53.0:
      return '50-54'
    if row['Age'] == 54.0:
      return '50-54'

    if row['Age'] == 55.0:
      return '55-59'
    if row['Age'] == 56.0:
      return '55-59'
    if row['Age'] == 57.0:
      return '55-59'
    if row['Age'] == 58.0:
      return '55-59'
    if row['Age'] == 59.0:
      return '55-59'

    return '+60'

st7['New age'] = st7.apply (lambda row: New_age1(row), axis=1)

### Salarie
group1=c.groupby(['Q5']).Q10.value_counts()
group1=pd.DataFrame(group1)
group1=group1.rename(columns={'Q10':'counting'})
group1=group1.reset_index()

#Just ML companies
just_ML=c[(c['Q8']== 'We have well established ML methods (i.e., models in production for more than 2 years)')
            | (c['Q8']== 'We recently started using ML methods (i.e., models in production for less than 2 years)')
               | (c['Q8']=='We use ML methods for generating insights (but do not put working models into production)')]
import plotly.express as px  # Be sure to import express
fig = px.choropleth(countries,  # Input Pandas DataFrame
                    locations="country",  # DataFrame column with locations
                    color="N° of answers",  # DataFrame column with color values
                    locationmode = 'country names', # Set to plot
                    color_continuous_scale='viridis')
fig.update_layout(
    title_text = '2019 Kaggle survey respondents per country')

fig.show()  # Output the plot to the screen
fig = plt.figure(figsize=(15,5))
ax1 = plt.subplot2grid((1,2),(0,0))
sns.countplot(c.Q1,order=c['Q1'].value_counts().index, palette='viridis')
plt.title('AGE', fontsize=15, weight='bold')

ax1 = plt.subplot2grid((1,2),(0,1))
sns.countplot(c.Q2, palette='viridis')
plt.title('GENDER', fontsize=15, weight='bold' )
plt.show()
plt.figure(figsize=(15,5))
plt.title('AGE OF STUDENT RESPONDENTS', weight='bold', fontsize=18)
st7['New age'].value_counts().plot(color='deeppink',animated=True, linewidth=4,
         marker='h', markerfacecolor='deeppink', markeredgewidth=2,
         markersize=7, markevery=3)
st8['Q2'].value_counts().plot(color='deepskyblue', linewidth=4,
         marker='d', markerfacecolor='lightgreen', markeredgewidth=2,
         markersize=7, markevery=3)
st9['Q1'].value_counts().plot(color='tan', linewidth=4,
         marker='d', markerfacecolor='lightgreen', markeredgewidth=2,
         markersize=7, markevery=3)
legend_properties = {'weight':'bold','size': 20}
plt.legend(['Survey 2017','Survey 2018','Survey 2019'],loc='right', prop=legend_properties)
plt.xlabel('Age intervals', weight='bold', fontsize=15)
plt.xticks(fontsize=15, weight='bold')
plt.yticks(fontsize=15, weight='bold')
plt.ylabel('Respondents', weight='bold', fontsize=15)
plt.show()
df_ide = pd.DataFrame(columns=['IDE','count','percentage'])

for i in range(1,12):
    df_ide = df_ide.append({'IDE':st9['Q13_Part_{}'.format(i)].mode()[0],'count':st9['Q13_Part_{}'.format(i)].count(),'percentage':st9['Q13_Part_{}'.format(i)].count()/len(st9)},ignore_index=True)

df_ide.index = df_ide['IDE']
del(df_ide['IDE'])
df_ide['percentage']=df_ide['percentage']*100
st13=df_ide[['percentage']].sort_values(by='percentage',ascending = False)
# Figure Size
fig, ax = plt.subplots(figsize=(9,6))

# Horizontal Bar Plot
title_cnt=st13.reset_index()
m= ax.barh(title_cnt.iloc[:,0], title_cnt.iloc[:,1], color='silver', height=0.7)
m[1].set_color('crimson')
m[1].set_edgecolor('red')
m[1].set_linewidth(1)

m[7].set_color('gainsboro')



# Remove axes splines
for s in ['top','bottom','left','right']:
    ax.spines[s].set_visible(False)

# Remove x,y Ticks
ax.xaxis.set_ticks_position('none')
ax.yaxis.set_ticks_position('none')

# Add padding between axes and labels
ax.xaxis.set_tick_params(pad=5)
ax.yaxis.set_tick_params(pad=10)

# Add x,y gridlines
ax.grid(b=True, color='grey', linestyle='-.', linewidth=1, alpha=0.3)

# Show top values 
ax.invert_yaxis()

# Add Plot Title
ax.set_title('Students favorite Data science platforms',
             loc='center', pad=10, fontsize=13)
ax.set_xlabel('Percentage')


# Add annotation to bars
for i in ax.patches:
    ax.text(i.get_width()+0.5, i.get_y()+0.5, str(round((i.get_width()), 2)),
            fontsize=10, fontweight='bold', color='grey')
r=[0,1,2,3,4,5,6,7,8,9,10]
plt.yticks(r, ['Coursera', 'University courses (degree)', 'Kaggle Courses',' Udemy', 'DataCamp', 'EdX'
              , 'Udacity', 'None', 'Fast.ai', 'Linkedin Learning', 'DataQuest'],weight='bold')



# Show Plot
plt.show()
#I imported this design from https://bmanohar16.github.io/blog/customizing-bar-plots
# Figure Size
fig, ax = plt.subplots(figsize=(10,6))

# Horizontal Bar Plot
title_cnt=st9.Q4.value_counts().sort_values(ascending=False).reset_index()
n= ax.barh(title_cnt.iloc[:,0], title_cnt.iloc[:,1], color='silver', height=0.7)
n[0].set_color('crimson')

n[5].set_color('gainsboro')



# Remove axes splines
for s in ['top','bottom','left','right']:
    ax.spines[s].set_visible(False)

# Remove x,y Ticks
ax.xaxis.set_ticks_position('none')
ax.yaxis.set_ticks_position('none')

# Add padding between axes and labels
ax.xaxis.set_tick_params(pad=5)
ax.yaxis.set_tick_params(pad=10)

# Add x,y gridlines
ax.grid(b=True, color='grey', linestyle='-.', linewidth=1, alpha=0.3)

# Show top values 
ax.invert_yaxis()

# Add Plot Title
ax.set_title('Students highest education level',
             loc='center', pad=10, fontsize=16)
ax.set_xlabel('# of respondents')


# Add annotation to bars
for i in ax.patches:
    ax.text(i.get_width()+10, i.get_y()+0.5, str(round((i.get_width()), 2)),
            fontsize=10, fontweight='bold', color='grey')
r=[0,1,2,3,4,5,6]
plt.yticks(r,["Bachelor's degree",
           "Master's degree",
           "Doctoral degree",
           "University study without degree",
           "Professional degree",
           "I prefer not to answer",
           "No formal education past high school"],weight='bold')
# Show Plot
plt.show()
# Figure Size
fig, ax = plt.subplots(figsize=(10,5))

# Horizontal Bar Plot
title_cnt=st9.Q15.value_counts().sort_values(ascending=False).reset_index()
n= ax.barh(title_cnt.iloc[:,0], title_cnt.iloc[:,1], color='silver')
n[0].set_color('crimson')

n[3].set_color('gainsboro')



# Remove axes splines
for s in ['top','bottom','left','right']:
    ax.spines[s].set_visible(False)

# Remove x,y Ticks
ax.xaxis.set_ticks_position('none')
ax.yaxis.set_ticks_position('none')

# Add padding between axes and labels
ax.xaxis.set_tick_params(pad=5)
ax.yaxis.set_tick_params(pad=10)

# Add x,y gridlines
ax.grid(b=True, color='grey', linestyle='-.', linewidth=1, alpha=0.3)

# Show top values 
ax.invert_yaxis()

# Add Plot Title
ax.set_title('Students coding experience',
             loc='center', pad=10, fontsize=16)
ax.set_xlabel('# of respondents')


# Add annotation to bars
for i in ax.patches:
    ax.text(i.get_width()+10, i.get_y()+0.5, str(round((i.get_width()), 2)),
            fontsize=10, fontweight='bold', color='grey')

plt.yticks(weight='bold')

# Show Plot
plt.show()
# Figure Size
fig, ax = plt.subplots(figsize=(9,5))

# Horizontal Bar Plot
title_cnt=st9.Q19.value_counts().sort_values(ascending=False).reset_index()
m= ax.barh(title_cnt.iloc[:,0], title_cnt.iloc[:,1], color='silver')
m[0].set_color('crimson')


# Remove axes splines
for s in ['top','bottom','left','right']:
    ax.spines[s].set_visible(False)

# Remove x,y Ticks
ax.xaxis.set_ticks_position('none')
ax.yaxis.set_ticks_position('none')

# Add padding between axes and labels
ax.xaxis.set_tick_params(pad=5)
ax.yaxis.set_tick_params(pad=10)

# Add x,y gridlines
ax.grid(b=True, color='grey', linestyle='-.', linewidth=1, alpha=0.3)

# Show top values 
ax.invert_yaxis()

# Add Plot Title
ax.set_title('Programming language recommendation',
             loc='center', pad=10, fontsize=16)
ax.set_xlabel('# of respondents')


# Add annotation to bars
for i in ax.patches:
    ax.text(i.get_width()+10, i.get_y()+0.5, str(round((i.get_width()), 2)),
            fontsize=10, fontweight='bold', color='grey')
plt.yticks(weight='bold')



# Show Plot
plt.show()
df_ide = pd.DataFrame(columns=['IDE','count','percentage'])

for i in range(1,12):
    df_ide = df_ide.append({'IDE':st9['Q20_Part_{}'.format(i)].mode()[0],'count':st9['Q20_Part_{}'.format(i)].count(),'percentage':st9['Q20_Part_{}'.format(i)].count()/len(st9)},ignore_index=True)

df_ide.index = df_ide['IDE']
del(df_ide['IDE'])
df_ide['percentage']=df_ide['percentage']*100
st20=df_ide[['percentage']].sort_values(by='percentage',ascending = False)

# Figure Size
fig, ax = plt.subplots(figsize=(10,5))

# Horizontal Bar Plot
title_cnt=st20.reset_index()
mn= ax.barh(title_cnt.iloc[:,0], title_cnt.iloc[:,1], color='silver')
mn[0].set_color('crimson')

mn[1].set_color('crimson')

mn[3].set_color('crimson')
mn[4].set_color('gainsboro')
mn[2].set_color('crimson')



# Remove axes splines
for s in ['top','bottom','left','right']:
    ax.spines[s].set_visible(False)

# Remove x,y Ticks
ax.xaxis.set_ticks_position('none')
ax.yaxis.set_ticks_position('none')

# Add padding between axes and labels
ax.xaxis.set_tick_params(pad=5)
ax.yaxis.set_tick_params(pad=10)

# Add x,y gridlines
ax.grid(b=True, color='grey', linestyle='-.', linewidth=1, alpha=0.2)

# Show top values 
ax.invert_yaxis()

# Add Plot Title
ax.set_title('Data visualization libraries used by students',
             loc='center', pad=10, fontsize=16)
ax.set_xlabel('Percentage')


# Add annotation to bars
for i in ax.patches:
    ax.text(i.get_width()+1, i.get_y()+0.5, str(round((i.get_width()), 2)),
            fontsize=10, fontweight='bold', color='grey')
plt.yticks(weight='bold')

fig.text(0.70, 0.655, 'Most used libraries', fontsize=14, color='dimgray',
         ha='center' ,va='top')

plt.axhline(y=3.5, color='black', linestyle='-.')

plt.show()
# Show Plot
plt.show()
df_ide = pd.DataFrame(columns=['IDE','count','percentage'])

for i in range(1,12):
    df_ide = df_ide.append({'IDE':st9['Q24_Part_{}'.format(i)].mode()[0],'count':st9['Q24_Part_{}'.format(i)].count(),'percentage':st9['Q24_Part_{}'.format(i)].count()/len(st9)},ignore_index=True)

df_ide.index = df_ide['IDE']
del(df_ide['IDE'])
df_ide['percentage']=df_ide['percentage']*100
st24=df_ide[['percentage']].sort_values(by='percentage',ascending = False)

# Figure Size
fig, ax = plt.subplots(figsize=(10,6))

# Horizontal Bar Plot
title_cnt=st24.reset_index()
mn= ax.barh(title_cnt.iloc[:,0], title_cnt.iloc[:,1], color='silver')
mn[0].set_color('crimson')

mn[1].set_color('crimson')

mn[3].set_color('crimson')
mn[7].set_color('gainsboro')
mn[2].set_color('crimson')





# Remove axes splines
for s in ['top','bottom','left','right']:
    ax.spines[s].set_visible(False)

# Remove x,y Ticks
ax.xaxis.set_ticks_position('none')
ax.yaxis.set_ticks_position('none')

# Add padding between axes and labels
ax.xaxis.set_tick_params(pad=5)
ax.yaxis.set_tick_params(pad=10)

# Add x,y gridlines
ax.grid(b=True, color='grey', linestyle='-.', linewidth=1, alpha=0.2)

# Show top values 
ax.invert_yaxis()

# Add Plot Title
ax.set_title('ML algorithms used by students',
             loc='center', pad=10, fontsize=16)
ax.set_xlabel('Percentage')


# Add annotation to bars
for i in ax.patches:
    ax.text(i.get_width()+1, i.get_y()+0.5, str(round((i.get_width()), 2)),
            fontsize=10, fontweight='bold', color='grey')
r=[0,1,2,3,4,5,6,7,8,9,10]
plt.yticks(r,["Linear/Logistic Regression",
           "Decision Trees/Random Forests",
           "Convolutional Neural networks",
           "Gradient Boosting Machines",
           "Dense neural networks",
           "Bayesian Approaches",
           "Recurrent Neural networks",
             'None',
             'Generative Adversarial Networks',
             'Transformer Networks (BERT...)',
             'Evolutionary Approaches'],weight='bold', fontsize=11)


fig.text(0.72, 0.655, 'Most popular ML algorithms', fontsize=14, color='dimgray',
         ha='center' ,va='top')

plt.axhline(y=3.5, color='black', linestyle='-.')
plt.show()
# Show Plot
plt.show()

# Figure Size
fig, ax = plt.subplots(figsize=(10,7))

# Horizontal Bar Plot
title_cnt=c.Q5.value_counts().sort_values(ascending=False).reset_index()
mn= ax.barh(title_cnt.iloc[:,0], title_cnt.iloc[:,1], color='crimson')
mn[1].set_color('silver')
mn[3].set_color('gainsboro')
mn[6].set_color('silver')
mn[7].set_color('silver')
mn[8].set_color('silver')
mn[9].set_color('silver')
mn[10].set_color('silver')
mn[11].set_color('silver')



# Remove axes splines
for s in ['top','bottom','left','right']:
    ax.spines[s].set_visible(False)

# Remove x,y Ticks
ax.xaxis.set_ticks_position('none')
ax.yaxis.set_ticks_position('none')

# Add padding between axes and labels
ax.xaxis.set_tick_params(pad=5)
ax.yaxis.set_tick_params(pad=10)

# Add x,y gridlines
ax.grid(b=True, color='grey', linestyle='-.', linewidth=1, alpha=0.2)

# Show top values 
ax.invert_yaxis()

# Add Plot Title
ax.set_title('Job position',
             loc='center', pad=10, fontsize=16)
ax.set_xlabel('# of respondents')


# Add annotation to bars
for i in ax.patches:
    ax.text(i.get_width()+20, i.get_y()+0.5, str(round((i.get_width()), 2)),
            fontsize=10, fontweight='bold', color='grey')
plt.yticks(weight='bold')

# Show Plot
plt.show()


# Figure Size
fig, ax = plt.subplots(figsize=(10,6))

# Horizontal Bar Plot
title_cnt=c.Q8.value_counts().sort_values(ascending=False).reset_index()
mn= ax.barh(title_cnt.iloc[:,0], title_cnt.iloc[:,1], color='silver')
mn[1].set_color('crimson')
mn[2].set_color('crimson')


# Remove axes splines
for s in ['top','bottom','left','right']:
    ax.spines[s].set_visible(False)

# Remove x,y Ticks
ax.xaxis.set_ticks_position('none')
ax.yaxis.set_ticks_position('none')


# Add padding between axes and labels
ax.xaxis.set_tick_params(pad=5)
ax.yaxis.set_tick_params(pad=10)

# Add x,y gridlines
ax.grid(b=True, color='grey', linestyle='-.', linewidth=1, alpha=0.3)

# Show top values 
ax.invert_yaxis()

# Add Plot Title
ax.set_title('Machine learning in companies',
             loc='center', pad=10, fontsize=15)
ax.set_xlabel('# of respondents')
# Add annotation to bars
for i in ax.patches:
    ax.text(i.get_width()+15, i.get_y()+0.5, str(round((i.get_width()), 2)),
            fontsize=12, fontweight='bold', color='grey')
r=[0,1,2,3,4,5]
plt.yticks(r,['We are exploring ML methods','We recently started using ML methods (<2 years)',
            'We have well established ML methods (>2 years)',
            'NO (we do not use ML methods)',
            'We use ML methods for generating insights',
            'I do not know'],
            fontsize=12,weight='bold')
plt.xticks(fontsize=12,weight='bold')


# Show Plot
plt.show()


# Figure Size
fig, ax = plt.subplots(figsize=(10,4))

# Horizontal Bar Plot
title_cnt=just_ML.Q6.value_counts().sort_values(ascending=False).reset_index()
mn= ax.barh(title_cnt.iloc[:,0], title_cnt.iloc[:,1], color='silver')
mn[0].set_color('crimson')
mn[1].set_color('lightskyblue')


# Remove axes splines
for s in ['top','bottom','left','right']:
    ax.spines[s].set_visible(False)

# Remove x,y Ticks
ax.xaxis.set_ticks_position('none')
ax.yaxis.set_ticks_position('none')

# Add padding between axes and labels
ax.xaxis.set_tick_params(pad=5)
ax.yaxis.set_tick_params(pad=10)

# Add x,y gridlines
ax.grid(b=True, color='grey', linestyle='-.', linewidth=0.5, alpha=0.2)

# Show top values 
ax.invert_yaxis()

# Add Plot Title
ax.set_title('Size of the companies with established ML',
             loc='center', pad=10, fontsize=16)
ax.set_xlabel('# of respondents')

# Add annotation to bars
for i in ax.patches:
    ax.text(i.get_width()+10, i.get_y()+0.5, str(round((i.get_width()), 2)),
            fontsize=10, fontweight='bold', color='grey')
plt.yticks(weight='bold')
plt.xticks(weight='bold')





# Show Plot
plt.show()


# Figure Size
fig, ax = plt.subplots(figsize=(10,4))

# Horizontal Bar Plot
title_cnt=just_ML.Q11.value_counts().sort_values(ascending=False).reset_index()
mn= ax.barh(title_cnt.iloc[:,0], title_cnt.iloc[:,1], color='silver')
mn[0].set_color('lightskyblue')
mn[4].set_color('crimson')


# Remove axes splines
for s in ['top','bottom','left','right']:
    ax.spines[s].set_visible(False)

# Remove x,y Ticks
ax.xaxis.set_ticks_position('none')
ax.yaxis.set_ticks_position('none')

# Add padding between axes and labels
ax.xaxis.set_tick_params(pad=5)
ax.yaxis.set_tick_params(pad=10)

# Add x,y gridlines
ax.grid(b=True, color='grey', linestyle='-.', linewidth=1, alpha=0.4)

# Show top values 
ax.invert_yaxis()

# Add Plot Title
ax.set_title('Companies: ML and cloud computing cost',
             loc='center', pad=10, fontsize=16)
plt.yticks(weight='bold')


# Add annotation to bars
for i in ax.patches:
    ax.text(i.get_width()+10, i.get_y()+0.5, str(round((i.get_width()), 2)),
            fontsize=10, fontweight='bold', color='grey')
plt.yticks(weight='bold')
plt.xticks(weight='bold')




# Show Plot
plt.show()

df_ide = pd.DataFrame(columns=['IDE','count','percentage'])

for i in range(1,12):
    df_ide = df_ide.append({'IDE':just_ML['Q29_Part_{}'.format(i)].mode()[0],'count':just_ML['Q29_Part_{}'.format(i)].count(),'percentage':just_ML['Q29_Part_{}'.format(i)].count()/len(just_ML)},ignore_index=True)

df_ide.index = df_ide['IDE']
del(df_ide['IDE'])
df_ide['percentage']=df_ide['percentage']*100
with_ML=df_ide[['percentage']].sort_values(by='percentage',ascending = False)
#I imported this design from https://bmanohar16.github.io/blog/customizing-bar-plots
# Figure Size
fig, ax = plt.subplots(figsize=(10,6))

# Horizontal Bar Plot
title_cnt=with_ML.reset_index()
n= ax.barh(title_cnt.iloc[:,0], title_cnt.iloc[:,1], color='silver')
n[0].set_color('crimson')

n[1].set_color('lightskyblue')




# Remove axes splines
for s in ['top','bottom','left','right']:
    ax.spines[s].set_visible(False)

# Remove x,y Ticks
ax.xaxis.set_ticks_position('none')
ax.yaxis.set_ticks_position('none')

# Add padding between axes and labels
ax.xaxis.set_tick_params(pad=5)
ax.yaxis.set_tick_params(pad=10)

# Add x,y gridlines
ax.grid(b=True, color='grey', linestyle='-.', linewidth=1, alpha=0.3)

# Show top values 
ax.invert_yaxis()

# Add Plot Title
ax.set_title('Cloud services',
             loc='center', pad=10, fontsize=16)
ax.set_xlabel('Percentage')


# Add annotation to bars
for i in ax.patches:
    ax.text(i.get_width()+0.1, i.get_y()+0.5, str(round((i.get_width()), 2)),
            fontsize=10, fontweight='bold', color='grey')
r=[0,1,2,3,4,5,6]
plt.yticks(weight='bold')
# Show Plot
plt.show()