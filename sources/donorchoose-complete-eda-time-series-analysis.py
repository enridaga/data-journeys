
import pandas as pd # package for high-performance, easy-to-use data structures and data analysis
import numpy as np # fundamental package for scientific computing with Python
import matplotlib
import matplotlib.pyplot as plt # for plotting
import seaborn as sns # for making plots with seaborn
color = sns.color_palette()
import plotly.offline as py
py.init_notebook_mode(connected=True)
from plotly.offline import init_notebook_mode, iplot
init_notebook_mode(connected=True)
import plotly.graph_objs as go
import plotly.offline as offline
offline.init_notebook_mode()
from plotly import tools
from mpl_toolkits.basemap import Basemap
from numpy import array
from matplotlib import cm

# import cufflinks and offline mode
import cufflinks as cf
cf.go_offline()

from sklearn import preprocessing
# Supress unnecessary warnings so that presentation looks clean
import warnings
warnings.filterwarnings("ignore")

# Print all rows and columns
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)


donations = pd.read_csv('../input/Donations.csv')
donors = pd.read_csv('../input/Donors.csv', low_memory=False)
schools = pd.read_csv('../input/Schools.csv', error_bad_lines=False)
teachers = pd.read_csv('../input/Teachers.csv', error_bad_lines=False)
projects = pd.read_csv('../input/Projects.csv', error_bad_lines=False, warn_bad_lines=False, parse_dates=["Project Posted Date","Project Fully Funded Date"])
resources = pd.read_csv('../input/Resources.csv', error_bad_lines=False, warn_bad_lines=False)
# Merge donation data with donor data 
donors_donations = donations.merge(donors, on='Donor ID', how='inner')
donations.head()
donors.head()
schools.head()
teachers.head()
projects.head()
resources.head()
donors_donations.head()
projects_schools = projects.merge(schools, on='School ID', how='inner')
projects_schools.head()
donations["Donation Amount"].describe()
schools['School Percentage Free Lunch'].describe()
# checking missing data in donors data 
total = donors.isnull().sum().sort_values(ascending = False)
percent = (donors.isnull().sum()/donors.isnull().count()*100).sort_values(ascending = False)
missing_donors_data  = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
missing_donors_data.head()
# checking missing data in donations data 
total = donations.isnull().sum().sort_values(ascending = False)
percent = (donations.isnull().sum()/donations.isnull().count()*100).sort_values(ascending = False)
missing_donations_data  = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
missing_donations_data.head()
# checking missing data in schools dataset 
total = schools.isnull().sum().sort_values(ascending = False)
percent = (schools.isnull().sum()/schools.isnull().count()*100).sort_values(ascending = False)
missing_schools_data  = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
missing_schools_data.head()
# checking missing data in teachers dataset 
total = teachers.isnull().sum().sort_values(ascending = False)
percent = (teachers.isnull().sum()/teachers.isnull().count()*100).sort_values(ascending = False)
missing_teachers_data  = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
missing_teachers_data.head()
# checking missing data in resources dataset 
total = resources.isnull().sum().sort_values(ascending = False)
percent = (resources.isnull().sum()/resources.isnull().count()*100).sort_values(ascending = False)
missing_resources_data  = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
missing_resources_data.head()
plt.figure(figsize = (12, 8))

sns.distplot(donors_donations['Donation Amount'].dropna())
plt.xlabel('Donation Amount', fontsize=12)
plt.title("Histogram of Donation Amount")
plt.show() 

plt.figure(figsize = (12, 8))
plt.scatter(range(donors_donations.shape[0]), np.sort(donors_donations['Donation Amount'].values))
plt.xlabel('Donation Amount', fontsize=12)
plt.title("Distribution of Donation Amount")
plt.show()
temp = donors_donations["Donor City"].value_counts().head(30)
temp.iplot(kind='bar', xTitle = 'City name', yTitle = "Count", title = 'Top Donor cities')
temp = donors_donations["Donor State"].value_counts().head(30)
temp.iplot(kind='bar', xTitle = 'State name', yTitle = "Count", title = 'Top Donor States')
temp = donors_donations['Donor Cart Sequence'].value_counts().head()
df = pd.DataFrame({'labels': temp.index,
                   'values': temp.values
                  })
df.iplot(kind='pie',labels='labels',values='values', title='Top Donor checked out carts')
temp = projects_schools['Project Subject Category Tree'].value_counts().head(30)
temp.iplot(kind='bar', xTitle = 'Project Subject Category', yTitle = "Count", title = 'Distribution of Project subject categories')
temp = projects_schools['Project Subject Subcategory Tree'].value_counts().head(10)
temp.iplot(kind='bar', xTitle = 'Project Subject Sub-Category', yTitle = "Count", title = 'Distribution of Project subject Sub-categories')
# temp = projects_schools['Project Resource Category'].value_counts().head(30)
# temp.iplot(kind='bar', xTitle = 'Project Resource Category Name', yTitle = "Count", title = 'Distribution of Project Resource categories')
temp = projects_schools['Project Resource Category'].value_counts().head(10)
df = pd.DataFrame({'labels': temp.index,
                   'values': temp.values
                  })
df.iplot(kind='pie',labels='labels',values='values', title='Distribution of Project resource categories')
temp = schools['School Metro Type'].value_counts()
df = pd.DataFrame({'labels': temp.index,
                   'values': temp.values
                  })
df.iplot(kind='pie',labels='labels',values='values', title='Distribution of school Metro Type')
cnt_srs = projects_schools['School City'].value_counts().head(20)
trace = go.Bar(
    y=cnt_srs.index[::-1],
    x=cnt_srs.values[::-1],
    orientation = 'h',
    marker=dict(
        color=cnt_srs.values[::-1],
        colorscale = 'Blues',
        reversescale = True
    ),
)

layout = dict(
    title='Distribution of School cities',
    )
data = [trace]
fig = go.Figure(data=data, layout=layout)
py.iplot(fig, filename="CitySchools")
cnt_srs = projects_schools['School County'].value_counts().head(20)
trace = go.Bar(
    y=cnt_srs.index[::-1],
    x=cnt_srs.values[::-1],
    orientation = 'h',
    marker=dict(
        color=cnt_srs.values[::-1],
        colorscale = 'Blues',
        reversescale = True
    ),
)

layout = dict(
    title='Distribution of School County',
    )
data = [trace]
fig = go.Figure(data=data, layout=layout)
py.iplot(fig, filename="CountySchool")
temp = projects['Project Grade Level Category'].value_counts()
fig = {
  "data": [
    {
      "values": temp.values,
      "labels": temp.index,
      "domain": {"x": [0, .48]},
      "name": "Grade Level Category",
      #"hoverinfo":"label+percent+name",
      "hole": .4,
      "type": "pie"
    },
    
    ],
  "layout": {
        "title":"Distribution of Projects Grade Level Category",
        "annotations": [
            {
                "font": {
                    "size": 20
                },
                "showarrow": False,
                "text": "Grade Level Categories",
                "x": 0.15,
                "y": 0.5
            }
            
        ]
    }
}
iplot(fig, filename='donut')
temp = donors['Donor Is Teacher'].value_counts()
df = pd.DataFrame({'labels': temp.index,
                   'values': temp.values
                  })
df.iplot(kind='pie',labels='labels',values='values', title='Donor is Teacher or not')
temp = donors_donations['Donation Included Optional Donation'].value_counts()
df = pd.DataFrame({'labels': temp.index,
                   'values': temp.values
                  })
df.iplot(kind='pie',labels='labels',values='values', title='Whether or not the donation included an optional donation.')
temp = projects_schools["Project Type"].dropna().value_counts().head(30)
temp.iplot(kind='bar', xTitle = 'Project name', yTitle = "Count", title = 'Types of Projects')
temp = projects_schools['Project Current Status'].value_counts()
df = pd.DataFrame({'labels': temp.index,
                   'values': temp.values
                  })
df.iplot(kind='pie',labels='labels',values='values', title='Projects were fully funded or not.')
from wordcloud import WordCloud, STOPWORDS
names = projects_schools["Project Essay"][~pd.isnull(projects_schools["Project Essay"])].sample(10000)
#print(names)
wordcloud = WordCloud(max_font_size=50, width=600, height=300).generate(' '.join(names))
plt.figure(figsize=(15,8))
plt.imshow(wordcloud)
plt.title("Wordcloud for Top Keywords from project Essay", fontsize=35)
plt.axis("off")
plt.show() 
names = projects_schools["Project Title"][~pd.isnull(projects_schools["Project Title"])].sample(1000)
#print(names)
wordcloud = WordCloud(max_font_size=50, width=600, height=300).generate(' '.join(names))
plt.figure(figsize=(15,8))
plt.imshow(wordcloud)
plt.title("Wordcloud for Top Keywords from project Titles", fontsize=35)
plt.axis("off")
plt.show() 
state_wise = donors_donations.groupby('Donor State', as_index=False).agg({'Donation ID': 'count','Donation Amount':'sum'})   
state_wise.columns = ["State","Donation_num", "Donation_sum"]
state_wise["Donation_avg"]=state_wise["Donation_sum"]/state_wise["Donation_num"]
del state_wise['Donation_num']
for col in state_wise.columns:
    state_wise[col] = state_wise[col].astype(str)
state_wise['text'] = state_wise['State'] + '<br>' +\
    'Average amount per donation: $' + state_wise['Donation_avg']+ '<br>' +\
    'Total donation amount:  $' + state_wise['Donation_sum']
state_codes = {
    'District of Columbia' : 'DC','Mississippi': 'MS', 'Oklahoma': 'OK', 
    'Delaware': 'DE', 'Minnesota': 'MN', 'Illinois': 'IL', 'Arkansas': 'AR', 
    'New Mexico': 'NM', 'Indiana': 'IN', 'Maryland': 'MD', 'Louisiana': 'LA', 
    'Idaho': 'ID', 'Wyoming': 'WY', 'Tennessee': 'TN', 'Arizona': 'AZ', 
    'Iowa': 'IA', 'Michigan': 'MI', 'Kansas': 'KS', 'Utah': 'UT', 
    'Virginia': 'VA', 'Oregon': 'OR', 'Connecticut': 'CT', 'Montana': 'MT', 
    'California': 'CA', 'Massachusetts': 'MA', 'West Virginia': 'WV', 
    'South Carolina': 'SC', 'New Hampshire': 'NH', 'Wisconsin': 'WI',
    'Vermont': 'VT', 'Georgia': 'GA', 'North Dakota': 'ND', 
    'Pennsylvania': 'PA', 'Florida': 'FL', 'Alaska': 'AK', 'Kentucky': 'KY', 
    'Hawaii': 'HI', 'Nebraska': 'NE', 'Missouri': 'MO', 'Ohio': 'OH', 
    'Alabama': 'AL', 'Rhode Island': 'RI', 'South Dakota': 'SD', 
    'Colorado': 'CO', 'New Jersey': 'NJ', 'Washington': 'WA', 
    'North Carolina': 'NC', 'New York': 'NY', 'Texas': 'TX', 
    'Nevada': 'NV', 'Maine': 'ME', 'other': ''}

state_wise['code'] = state_wise['State'].map(state_codes)  
# https://plot.ly/python/choropleth-maps/
data = [ dict(
        type='choropleth',
        autocolorscale = True,
        locations = state_wise['code'], # The variable identifying state
        z = state_wise['Donation_sum'].astype(float), # The variable used to adjust map colors
        locationmode = 'USA-states', 
        text = state_wise['text'], # Text to show when mouse hovers on each state
        colorbar = dict(  
            title = "Donation in USD")  # Colorbar to show beside the map
        ) ]

layout = dict(
        title = 'Donation by State<br>(Hover for breakdown)',
        geo = dict(
            scope='usa',
            projection=dict( type='albers usa' ),
            showlakes = True,
            lakecolor = 'rgb(255, 255, 255)'),
             )
fig = dict(data=data, layout=layout)
iplot(fig)

school_count = schools['School State'].value_counts().reset_index()
school_count.columns = ['state', 'schools']
for col in school_count.columns:
    school_count[col] = school_count[col].astype(str)
school_count['text'] = school_count['state'] + '<br>' + '# of schools: ' + school_count['schools']
# https://plot.ly/python/choropleth-maps/
data = [ dict(
        type='choropleth',
        autocolorscale = True,
        locations = state_wise['code'], # The variable identifying state
        z = school_count['schools'].astype(float), # The variable used to adjust map colors
        locationmode = 'USA-states', 
        text = school_count['text'], # Text to show when mouse hovers on each state
        colorbar = dict(  
            title = "# of Schools")  # Colorbar to show beside the map
        ) ]

layout = dict(
        title = 'Number of schools in different states<br>(Hover for breakdown)',
        geo = dict(
            scope='usa',
            projection=dict( type='albers usa' ),
            showlakes = True,
            lakecolor = 'rgb(255, 255, 255)'),
             )
fig = dict(data=data, layout=layout)
iplot(fig)
schools.groupby('School Metro Type')['School Percentage Free Lunch'].describe()
donations_per_donor = donations.groupby('Donor ID')['Donor Cart Sequence'].max()
donations_per_donor = (donations_per_donor == 1).mean() *100
print("Only one time donation is given by : "+ str(donations_per_donor) +" % donors")
temp = teachers['Teacher Prefix'].value_counts()
fig = {
  "data": [
    {
      "values": temp.values,
      "labels": temp.index,
      "domain": {"x": [0, .48]},
      "name": "Teachers Prefix",
      #"hoverinfo":"label+percent+name",
      "hole": .4,
      "type": "pie"
    },
    
    ],
  "layout": {
        "title":"Distribution of Teachers Prefix",
        "annotations": [
            {
                "font": {
                    "size": 20
                },
                "showarrow": False,
                "text": "Teachers Prefix",
                "x": 0.17,
                "y": 0.5
            }
            
        ]
    }
}
iplot(fig, filename='donut')
# Creating the gender column
gender_mapping = {"Mrs.": "Female", "Ms.":"Female", "Mr.":"Male", "Teacher":"Unknown", "Dr.":"Unknown", np.nan:"Unknown", "Mx.":"Unknown" }
teachers["gender"] = teachers['Teacher Prefix'].map(gender_mapping)
temp = teachers['gender'].value_counts()
fig = {
  "data": [
    {
      "values": temp.values,
      "labels": temp.index,
      "domain": {"x": [0, .48]},
      "name": "Gender",
      #"hoverinfo":"label+percent+name",
      "hole": .4,
      "type": "pie"
    },
    
    ],
  "layout": {
        "title":"Males V.S. Females",
        "annotations": [
            {
                "font": {
                    "size": 20
                },
                "showarrow": False,
                "text": "Gender",
                "x": 0.20,
                "y": 0.5
            }
            
        ]
    }
}
iplot(fig, filename='donut')
temp = resources["Resource Item Name"].dropna().value_counts().head(20)
temp.iplot(kind='bar', xTitle = 'Resource Item Name', yTitle = "Count", title = 'Top requested items')
temp = resources["Resource Vendor Name"].dropna().value_counts().head(20)
temp.iplot(kind='bar', xTitle = 'Resource Vendor Name', yTitle = "Count", title = 'Top Resource Vendor Name')
resources['total_price'] = resources['Resource Quantity'] * resources['Resource Unit Price']
plt.figure(figsize = (12, 8))
plt.scatter(range(resources.shape[0]), np.sort(resources['total_price'].values))
plt.xlabel('Price', fontsize=12)
plt.title("Distribution of resources price")
plt.show()
teachers['Teacher First Project Posted Date'] = pd.to_datetime(teachers['Teacher First Project Posted Date'])
teachers['weekdays'] = teachers['Teacher First Project Posted Date'].dt.dayofweek
teachers['month'] = teachers['Teacher First Project Posted Date'].dt.month 
teachers['year'] = teachers['Teacher First Project Posted Date'].dt.year

dmap = {0:'Monday',1:'Tueday',2:'Wednesday',3:'Thursday',4:'Friday',5:'Saturday',6:'Sunday'}
teachers['weekdays'] = teachers['weekdays'].map(dmap)

month_dict = {1 :"Jan",2 :"Feb",3 :"Mar",4 :"Apr",5 : "May",6 : "Jun",7 : "Jul",8 :"Aug",9 :"Sep",10 :"Oct",11 :"Nov",12 :"Dec"}
teachers['month'] = teachers['month'].map(month_dict)
teachers.head()
temp = teachers['weekdays'].value_counts()
temp = temp.reset_index()
temp.columns = ['weekdays', 'count']
#temp.head()
temp[['weekdays','count']].set_index('weekdays').iplot(kind = 'bar', xTitle = 'Day of Week', yTitle = "# of projects posted", title ="week day wise first project posted by teachers")
temp = teachers['month'].value_counts()
temp = temp.reset_index()
temp.columns = ['month', 'count']
#temp.head()
temp[['month','count']].set_index('month').iplot(kind = 'bar', xTitle = 'Month', yTitle = "# of projects posted", title ="Month wise first project posted by teachers")
temp = teachers.groupby('year').agg({'Teacher ID' : 'count'}).reset_index()
year2002_2017 = temp[~temp.year.isin([2018])] 
year2002_2017 = year2002_2017.sort_values('year').set_index("year")
#temp.head()
# temp = teachers['year'].value_counts()
# temp = temp.reset_index()
# temp.columns = ['year', 'count']
year2002_2017.iplot(kind = 'scatter', xTitle='Year 2002 to Year 2017',  yTitle = "# of teachers first project posted", title ="Trend of teachers who posted their project first time(2002 to 2017)")
#temp = teachers.groupby('year').agg({'Teacher ID' : 'count'}).reset_index()
year2018 = temp[temp.year.isin([2018])] 
year2018 = year2018.sort_values('year').set_index("year")
year2018.iplot(kind = 'scatter', xTitle='Year 2018',  yTitle = "# of teachers first project posted", title ="Trend of teachers who posted their project first time(2002 to 2017)")