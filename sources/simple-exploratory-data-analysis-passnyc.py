

# Import the neccessary modules for data manipulation and visual representation
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as matplot
import seaborn as sns
#%matplotlib inline

from plotly.offline import init_notebook_mode, iplot
init_notebook_mode(connected=True)
import plotly.graph_objs as go
import plotly.figure_factory as ff
from plotly import tools
import plotly.plotly as py
pd.set_option('display.max_columns', None)  
df = pd.read_csv('../input/data-science-for-good/2016 School Explorer.csv')
shsat = pd.read_csv('../input/data-science-for-good/D5 SHSAT Registrations and Testers.csv')
safety = pd.read_csv('../input/ny-2010-2016-school-safety-report/2010-2016-school-safety-report.csv')
df.head(3)
# There are 1272 schools in the data set
df.shape
# Preprocess some data
# Create a function to convert the percentage to a fraction 
def p2f(x):
    return float(x.strip('%'))/100

df['Percent of Students Chronically Absent']=df['Percent of Students Chronically Absent'].astype(str).apply(p2f)
df['Rigorous Instruction %'] = df['Rigorous Instruction %'].astype(str).apply(p2f)
df['Collaborative Teachers %'] = df['Collaborative Teachers %'].astype(str).apply(p2f)
df['Supportive Environment %'] = df['Supportive Environment %'].astype(str).apply(p2f)
df['Effective School Leadership %'] = df['Effective School Leadership %'].astype(str).apply(p2f)
df['Strong Family-Community Ties %'] = df['Strong Family-Community Ties %'].astype(str).apply(p2f)
df['Trust %'] = df['Trust %'].astype(str).apply(p2f)
df['Student Attendance Rate'] = df['Student Attendance Rate'].astype(str).apply(p2f)
df['School Income Estimate'] = df['School Income Estimate'].str.replace(',', '')
df['School Income Estimate'] = df['School Income Estimate'].str.replace('$', '')
df['School Income Estimate'] = df['School Income Estimate'].str.replace(' ', '')
df['School Income Estimate'] = df['School Income Estimate'].astype(float)
df['School Income Estimate'] = df['School Income Estimate'].fillna(0)
df['Economic Need Index'] = df['Economic Need Index'].fillna(0)

# Static Version
df.plot(kind="scatter", x="Longitude", y="Latitude",
    s=df['School Income Estimate']/1210, c="Economic Need Index", cmap=plt.get_cmap("jet"),
        label='Schools', title='New York School Population Map',colorbar=True, alpha=0.4, figsize=(15,7))
plt.legend()
plt.show()
data = [
    {
        'x': df["Longitude"],
        'y': df["Latitude"],
        'text': df["School Name"],
        'mode': 'markers',
        'marker': {
            'color': df["Economic Need Index"],
            'size': df["School Income Estimate"]/4500,
            'showscale': True,
            'colorscale':'Portland'
        }
    }
]

layout= go.Layout(
    title= 'New York School Population (Economic Need Index)',
    xaxis= dict(
        title= 'Longitude'
    ),
    yaxis=dict(
        title='Latitude'
    ))
fig = go.Figure(data=data, layout=layout)
iplot(fig, filename='scatter_hover_labels')
df['Percent Asian'] = df['Percent Asian'].apply(p2f)
df['Percent Black'] = df['Percent Black'].apply(p2f)
df['Percent Hispanic'] = df['Percent Hispanic'].apply(p2f)
df['Percent White'] = df['Percent White'].apply(p2f)
df['Percent Black / Hispanic'] = df['Percent Black / Hispanic'].apply(p2f)
data = [
    {
        'x': df["Longitude"],
        'y': df["Latitude"],
        'text': df["School Name"],
        'mode': 'markers',
        'marker': {
            'color': df["Percent Black"],
            'size': df["School Income Estimate"]/4500,
            'showscale': True,
            'colorscale':'Portland'
        }
    }
]

layout= go.Layout(
    title= 'New York Black Student Ratio Of School',
    xaxis= dict(
        title= 'Longitude'
    ),
    yaxis=dict(
        title='Latitude'
    )
)
fig = go.Figure(data=data, layout=layout)
iplot(fig, filename='scatter_hover_labels')
data = [
    {
        'x': df["Longitude"],
        'y': df["Latitude"],
        'text': df["School Name"],
        'mode': 'markers',
        'marker': {
            'color': df["Percent Hispanic"],
            'size': df["School Income Estimate"]/4500,
            'showscale': True,
            'colorscale':'Portland'
        }
    }
]

layout= go.Layout(
    title= 'New York Hispanic Student Ratio Of School',
    xaxis= dict(
        title= 'Longitude'
    ),
    yaxis=dict(
        title='Latitude'
    )
)
fig = go.Figure(data=data, layout=layout)
iplot(fig, filename='scatter_hover_labels')
# Static Figure
# Set up the matplotlib figure
f, axes = plt.subplots(ncols=2, figsize=(15, 6))

df.plot(kind="scatter", x="Longitude", y="Latitude", ax=axes[0],
    s=df['Economic Need Index']*100, c="Percent Black", cmap=plt.get_cmap("jet"),label='Schools', title='Black Population Percentage of Schools',
    colorbar=True, alpha=0.3, figsize=(15,7))

df.plot(kind="scatter", x="Longitude", y="Latitude", ax=axes[1],
    s=df['Economic Need Index']*100, c="Percent Hispanic", cmap=plt.get_cmap("jet"),label='Schools', title='Hispanic Population Percentage of Schools',
    colorbar=True, alpha=0.3, figsize=(15,7))

plt.legend()
plt.show()
# Set up the matplotlib figure
f, axes = plt.subplots(ncols=2, figsize=(15, 6))

df.plot(kind="scatter", x="Longitude", y="Latitude", ax=axes[0],
    s=df['Economic Need Index']*100, c="Percent Asian", label='Schools', title='Asian Population Percentage of Schools',
    colorbar=True, alpha=0.3, figsize=(15,5))

df.plot(kind="scatter", x="Longitude", y="Latitude", ax=axes[1],
    s=df['Economic Need Index']*130, c="Percent White", label='Schools', title='White Population Percentage of Schools',
    colorbar=True, alpha=1, figsize=(15,5))

plt.legend()
plt.show()

# Set up the matplotlib figure
f, axes = plt.subplots(ncols=2, figsize=(15, 6))

# Graph Employee Satisfaction
sns.distplot(df['Percent Asian'], kde=False, color="g", ax=axes[0], bins=25).set_title('% Asian Distribution')
axes[0].set_ylabel('School Count')

# Graph Employee Evaluation
sns.distplot(df['Percent White'], kde=False, color="r", ax=axes[1], bins=25).set_title('% White Distribution')
axes[1].set_ylabel('Employee Count')
# Set up the matplotlib figure
f, axes = plt.subplots(ncols=2, figsize=(15, 6))

# Graph Employee Average Monthly Hours
sns.distplot(df['Percent Black'], kde=False, color="b", ax=axes[0], bins=35).set_title('% Black Distribution')
axes[0].set_ylabel('Employee Count')

# Graph Employee Average Monthly Hours
sns.distplot(df['Percent Hispanic'], kde=False, color="r", ax=axes[1], bins=35).set_title('% Hispanic Distribution')
axes[1].set_ylabel('Employee Count')

df.head(1)
df2 = df.iloc[:,[16,17,19,20,21,22,23,24,25,26,28,30,32,34,36]]
#Correlation Matrix
corr = df2.corr()
corr = (corr)
sns.heatmap(corr, 
            xticklabels=corr.columns.values,
            yticklabels=corr.columns.values,
            cmap="RdYlGn", center=0)
f, axes = plt.subplots(2, 2, figsize=(19, 9), sharex=True)
sns.despine(left=True)

sns.regplot(x=df["Economic Need Index"], y=df["Percent Asian"], color='purple', ax=axes[0, 0], line_kws={"color": "black"})
sns.regplot(x=df["Economic Need Index"], y=df["Percent White"], color='g', ax=axes[0, 1], line_kws={"color": "black"})
sns.regplot(x=df["Economic Need Index"], y=df["Percent Black"], color='b', ax=axes[1, 0], line_kws={"color": "black"})
sns.regplot(x=df["Economic Need Index"], y=df["Percent Hispanic"], color='r', ax=axes[1, 1], line_kws={"color": "black"})

axes[0,0].set_title('Ecnomic Need Index (Asian)')
axes[0,1].set_title('Ecnomic Need Index (White)')
axes[1,0].set_title('Ecnomic Need Index (Black)')
axes[1,1].set_title('Ecnomic Need Index (Hispanic)')

plt.subplots_adjust(hspace=0.4)

# Create a dataframe of schools with an absent rate of 30% or more.
absent_30 = df[df['Percent of Students Chronically Absent']>=.30]

# Create a dataframe of schools with an absent rate of 11% or less.
absent_11 = df[df['Percent of Students Chronically Absent']<=.11]

# Exploring School Absent Rate 
df['Percent of Students Chronically Absent'].describe()
absent_30.iloc[:,[15,16,17,23,19,20,21,22]].describe()
absent_11.iloc[:,[3,15,16,17,23,19,20,21,22]].describe()
data = [
    {
        'x': absent_30["Longitude"],
        'y': absent_30["Latitude"],
        'text': absent_30["School Name"],
        'mode': 'markers',
        'marker': {
            'color': df["Economic Need Index"],
            'size': df["School Income Estimate"]/4500,
            'showscale': True,
            'colorscale':'Portland'
        }
    }
]

layout= go.Layout(
    title= 'Schools with 30% Absent Rate',
    xaxis= dict(
        title= 'Longitude'
    ),
    yaxis=dict(
        title='Latitude'
    )
)
fig = go.Figure(data=data, layout=layout)
iplot(fig, filename='scatter_hover_labels')
data = [
    {
        'x': absent_11["Longitude"],
        'y': absent_11["Latitude"],
        'text': absent_11["School Name"],
        'mode': 'markers',
        'marker': {
            'color': df["Economic Need Index"],
            'size': df["School Income Estimate"]/4500,
            'showscale': True,
            'colorscale':'Portland'
        }
    }
]

layout= go.Layout(
    title= 'Schools with 10% Absent Rate',
    xaxis= dict(
        title= 'Longitude'
    ),
    yaxis=dict(
        title='Latitude'
    )
)
fig = go.Figure(data=data, layout=layout)
iplot(fig, filename='scatter_hover_labels')
# Set up the matplotlib figure
f, axes = plt.subplots(ncols=2, figsize=(15, 6))

absent_11.plot(kind="scatter", x="Longitude", y="Latitude", ax=axes[0],
    s=absent_11['Economic Need Index']*100, c="School Income Estimate", cmap=plt.get_cmap("jet"),label='Schools', title='Schools with 10% Absent Rate',
    colorbar=True, alpha=0.3, figsize=(15,7))

absent_30.plot(kind="scatter", x="Longitude", y="Latitude", ax=axes[1],
    s=absent_30['Economic Need Index']*100, c="School Income Estimate", cmap=plt.get_cmap("jet"),label='Schools', title='Schools with 30% Absent Rate',
    colorbar=True, alpha=0.4, figsize=(15,7))

plt.show()
# Display the Mean ELA and Math Scores for Black/Hispanic Dominant Schools
df[df['Percent Black / Hispanic'] >= .70][['Average ELA Proficiency','Average Math Proficiency']].mean()
# Display the Mean ELA and Math Scores for White/Asian Dominant Schools
df[df['Percent Black / Hispanic'] <= .30][['Average ELA Proficiency','Average Math Proficiency']].mean()
# Create New Column for Black/Hispanic Dominant Schools
df['Black_Hispanic_Dominant'] = df['Percent Black / Hispanic'] >= .70

#KDEPlot: Kernel Density Estimate Plot
fig = plt.figure(figsize=(15,4))
ax=sns.kdeplot(df.loc[(df['Black_Hispanic_Dominant'] == True),'Average Math Proficiency'] , color='b',shade=True, label='Black/Hispanic Dominant School')
ax=sns.kdeplot(df.loc[(df['Black_Hispanic_Dominant'] == False),'Average Math Proficiency'] , color='r',shade=True, label='Asian/White Dominant School')
plt.title('Average Math Proficiency Distribution by Race')
plt.xlabel('Average Math Proficiency Score')
plt.ylabel('Frequency Count')
#KDEPlot: Kernel Density Estimate Plot
fig = plt.figure(figsize=(15,4))
ax=sns.kdeplot(df.loc[(df['Black_Hispanic_Dominant'] == True),'Average ELA Proficiency'] , color='b',shade=True, label='Black/Hispanic Dominant School')
ax=sns.kdeplot(df.loc[(df['Black_Hispanic_Dominant'] == False),'Average ELA Proficiency'] , color='g',shade=True, label='Asian/White Dominant School')
plt.title('Average ELA Proficiency Distribution by Race')
plt.xlabel('Average ELA Proficiency Score')
plt.ylabel('Frequency Count')
# Create the math scores for each race
asian_math = []
asian_math.append(sum(df['Grade 3 Math 4s - Asian or Pacific Islander']))
asian_math.append(sum(df['Grade 4 Math 4s - Asian or Pacific Islander']))
asian_math.append(sum(df['Grade 5 Math 4s - Asian or Pacific Islander']))
asian_math.append(sum(df['Grade 6 Math 4s - Asian or Pacific Islander']))
asian_math.append(sum(df['Grade 7 Math 4s - Asian or Pacific Islander']))
asian_math.append(sum(df['Grade 8 Math 4s - Asian or Pacific Islander']))

white_math = []
white_math.append(sum(df['Grade 3 Math 4s - White']))
white_math.append(sum(df['Grade 4 Math 4s - White']))
white_math.append(sum(df['Grade 5 Math 4s - White']))
white_math.append(sum(df['Grade 6 Math 4s - White']))
white_math.append(sum(df['Grade 7 Math 4s - White']))
white_math.append(sum(df['Grade 8 Math 4s - White']))

black_math = []
black_math.append(sum(df['Grade 3 Math 4s - Black or African American']))
black_math.append(sum(df['Grade 4 Math 4s - Black or African American']))
black_math.append(sum(df['Grade 5 Math 4s - Black or African American']))
black_math.append(sum(df['Grade 6 Math 4s - Black or African American']))
black_math.append(sum(df['Grade 7 Math 4s - Black or African American']))
black_math.append(sum(df['Grade 8 Math 4s - Black or African American']))

hispanic_math = []
hispanic_math.append(sum(df['Grade 3 Math 4s - Hispanic or Latino']))
hispanic_math.append(sum(df['Grade 4 Math 4s - Hispanic or Latino']))
hispanic_math.append(sum(df['Grade 5 Math 4s - Hispanic or Latino']))
hispanic_math.append(sum(df['Grade 6 Math 4s - Hispanic or Latino']))
hispanic_math.append(sum(df['Grade 7 Math 4s - Hispanic or Latino']))
hispanic_math.append(sum(df['Grade 8 Math 4s - Hispanic or Latino']))
# Create dataframe of math scores
race_mathscores = pd.DataFrame({'Asian Math':asian_math,'Black Math':black_math,'White Math':white_math, 'Hispanic Math':hispanic_math})
race_mathscores['Grade'] = [3,4,5,6,7,8]
race_mathscores
# Create a trace
trace = go.Scatter(
    x = race_mathscores['Grade'],
    y = race_mathscores['Asian Math'],
    name = 'Asian Students'
)

trace2 = go.Scatter(
    x = race_mathscores['Grade'],
    y = race_mathscores['Hispanic Math'],
    name = 'Hispanic Students'
)

trace3 = go.Scatter(
    x = race_mathscores['Grade'],
    y = race_mathscores['Black Math'],
    name = 'Black Students'
)

trace4 = go.Scatter(
    x = race_mathscores['Grade'],
    y = race_mathscores['White Math'],
    name = 'White Students'
)

layout= go.Layout(
    title= 'Student Math 4s Performance By Grade (Race)',
    xaxis= dict(
        title= 'Grade Level'
    ),
    yaxis=dict(
        title='# of Students Receiving Math 4s'
    )
)

data = [trace, trace2, trace3, trace4]

fig = go.Figure(data=data, layout=layout)
iplot(fig, filename='scatter_hover_labels')
# Create the math scores for each race
asian_ELA = []
asian_ELA.append(sum(df['Grade 3 ELA 4s - Asian or Pacific Islander']))
asian_ELA.append(sum(df['Grade 4 ELA 4s - Asian or Pacific Islander']))
asian_ELA.append(sum(df['Grade 5 ELA 4s - Asian or Pacific Islander']))
asian_ELA.append(sum(df['Grade 6 ELA 4s - Asian or Pacific Islander']))
asian_ELA.append(sum(df['Grade 7 ELA 4s - Asian or Pacific Islander']))
asian_ELA.append(sum(df['Grade 8 ELA 4s - Asian or Pacific Islander']))

white_ELA = []
white_ELA.append(sum(df['Grade 3 ELA 4s - White']))
white_ELA.append(sum(df['Grade 4 ELA 4s - White']))
white_ELA.append(sum(df['Grade 5 ELA 4s - White']))
white_ELA.append(sum(df['Grade 6 ELA 4s - White']))
white_ELA.append(sum(df['Grade 7 ELA 4s - White']))
white_ELA.append(sum(df['Grade 8 ELA 4s - White']))

black_ELA = []
black_ELA.append(sum(df['Grade 3 ELA 4s - Black or African American']))
black_ELA.append(sum(df['Grade 4 ELA 4s - Black or African American']))
black_ELA.append(sum(df['Grade 5 ELA 4s - Black or African American']))
black_ELA.append(sum(df['Grade 6 ELA 4s - Black or African American']))
black_ELA.append(sum(df['Grade 7 ELA 4s - Black or African American']))
black_ELA.append(sum(df['Grade 8 ELA 4s - Black or African American']))

hispanic_ELA = []
hispanic_ELA.append(sum(df['Grade 3 ELA 4s - Hispanic or Latino']))
hispanic_ELA.append(sum(df['Grade 4 ELA 4s - Hispanic or Latino']))
hispanic_ELA.append(sum(df['Grade 5 ELA 4s - Hispanic or Latino']))
hispanic_ELA.append(sum(df['Grade 6 ELA 4s - Hispanic or Latino']))
hispanic_ELA.append(sum(df['Grade 7 ELA 4s - Hispanic or Latino']))
hispanic_ELA.append(sum(df['Grade 8 ELA 4s - Hispanic or Latino']))
# Create dataframe of ELA scores
race_ELA = pd.DataFrame({'Asian ELA':asian_ELA,'Black ELA':black_ELA,'White ELA':white_ELA, 'Hispanic ELA':hispanic_ELA})
race_ELA['Grade'] = [3,4,5,6,7,8]
race_ELA
# Create a trace
trace = go.Scatter(
    x = race_ELA['Grade'],
    y = race_ELA['Asian ELA'],
    name = 'Asian Students'
)

trace2 = go.Scatter(
    x = race_ELA['Grade'],
    y = race_ELA['Hispanic ELA'],
    name = 'Hispanic Students'
)

trace3 = go.Scatter(
    x = race_ELA['Grade'],
    y = race_ELA['Black ELA'],
    name = 'Black Students'
)

trace4 = go.Scatter(
    x = race_ELA['Grade'],
    y = race_ELA['White ELA'],
    name = 'White Students'
)

layout= go.Layout(
    title= 'Student ELA 4s Performance By Grade (Race)',
    xaxis= dict(
        title= 'Grade Level'
    ),
    yaxis=dict(
        title='# of Students Receiving ELA 4s'
    )
)

data = [trace, trace2, trace3, trace4]

fig = go.Figure(data=data, layout=layout)
iplot(fig, filename='scatter_hover_labels')
# Overview of summary (Turnover V.S. Non-turnover)
turnover_Summary = df.groupby('Community School?')
turnover_Summary[['Economic Need Index', 'School Income Estimate', 'Percent Asian','Percent Black', 'Percent Hispanic', 
                  'Percent White', 'Average ELA Proficiency', 'Average Math Proficiency']].mean()
# Let's normalize the School Income Estimate to have better visualization
from sklearn import preprocessing

# Impute the missing Income with the median
median = df['School Income Estimate'].median()
df['School Income Estimate'] = df['School Income Estimate'].fillna(median)

# Create x, where x the 'scores' column's values as floats
x = df[['School Income Estimate']].values.astype(float)

# Create a minimum and maximum processor object
min_max_scaler = preprocessing.MinMaxScaler()

# Create an object to transform the data to fit minmax processor
x_scaled = min_max_scaler.fit_transform(x)

# Run the normalizer on the dataframe
school_income_normalized = pd.DataFrame(x_scaled)

df['school_income_normalized'] = school_income_normalized
sns.lmplot(x='school_income_normalized', y='Economic Need Index', data=df,
           fit_reg=False) # No regression line
        
#KDEPlot: Kernel Density Estimate Plot
fig = plt.figure(figsize=(15,4))
ax=sns.kdeplot(df.loc[(df['Community School?'] == 'Yes'),'School Income Estimate'] , color='b',shade=True, label='Community School')
ax=sns.kdeplot(df.loc[(df['Community School?'] == 'No'),'School Income Estimate'] , color='r',shade=True, label='Not Community School')
plt.title('Community School VS Not Community School Income')
#KDEPlot: Kernel Density Estimate Plot
fig = plt.figure(figsize=(15,4))
ax=sns.kdeplot(df.loc[(df['Community School?'] == 'Yes'),'Economic Need Index'] , color='b',shade=True, label='Community School')
ax=sns.kdeplot(df.loc[(df['Community School?'] == 'No'),'Economic Need Index'] , color='r',shade=True, label='Not Community School')
plt.xlabel('Economic Need Index Score')
plt.ylabel('Kernel Density Frequency')
plt.title('Community School VS Not Community School Economic Need Index')
city = df.groupby('City')['Zip'].count().reset_index().sort_values('Zip', ascending=False).reset_index(drop=True)
city_community = df[df['Community School?'] == 'Yes'].groupby('City')['Zip'].count().reset_index().sort_values('Zip', ascending=False).reset_index(drop=True)

city_merge = pd.merge(city, city_community, how='left', on='City')
city_merge.fillna(0, inplace=True)
city_merge['Zip_y'] = city_merge['Zip_y'].astype(int)

top_10_city = city_merge.iloc[:10,]
top_10_city = top_10_city.rename(columns={"Zip_x":'Total Count', "Zip_y":'Community Count'})
top_10_city
sns.set(style="whitegrid")

# Initialize the matplotlib figure
f, ax = plt.subplots(figsize=(13, 7))

# Plot the total schools per city
sns.set_color_codes("pastel")
sns.barplot(x="Total Count", y='City', data=top_10_city,
            label="Total", color="b")

# Plot the total community schools per city
sns.set_color_codes("muted")
sns.barplot(x="Community Count", y="City", data=top_10_city,
            label="Community School", color="b")

# Add a legend and informative axis label
ax.legend(ncol=2, loc="lower right", frameon=True)
ax.set(xlim=(0, 411), ylabel="City Name", title='Number of Schools Per City',
       xlabel="# of Schools")
sns.despine(left=True, bottom=True)
top10_city_list = list(top_10_city.City)

temp=df[df.City.isin(top10_city_list)]
plt.figure(figsize=(10,8))
sns.boxplot( x='School Income Estimate', y='City',data=temp)
plt.title("School Income For The Top 10 Schools", fontsize=16)
plt.xlabel("School Income Estimate", fontsize=16)
plt.ylabel("School Name", fontsize=16)
plt.show();
shsat.head(3)
registration_per_year = pd.DataFrame(shsat.groupby('Year of SHST')['Number of students who registered for the SHSAT'].sum()).reset_index()
registration_per_year
# Testing...
shsat_schools = list(shsat['DBN'].unique())
shsat_schools_list = '|'.join(shsat_schools)

df['Registered SHSAT'] = df['Location Code'].str.contains(shsat_schools_list)

registered_SHSAT_schools = df[df['Registered SHSAT'] == True]
registered_SHSAT_schools.groupby('School Name').mean()

registered_SHSAT_schools.plot(kind="scatter", x="Longitude", y="Latitude",
    s=absent_11['Economic Need Index']*100, c="School Income Estimate", cmap=plt.get_cmap("jet"),label='Schools', title='Schools with 10% Absent Rate',
    colorbar=True, alpha=0.3, figsize=(15,7))
plt.figure(figsize=(10,6))
sns.barplot(y=registration_per_year['Number of students who registered for the SHSAT'], x=registration_per_year['Year of SHST'], alpha=0.6)
plt.title("Number of SHSAT Registrations Per Year", fontsize=16)
plt.xlabel("Year", fontsize=16)
plt.ylabel("# of SHSAT Registration", fontsize=16)
plt.show()
# Create a dataframe of # of students who took the SHSAT
school_take = shsat.groupby('School name')['Number of students who took the SHSAT'].sum().reset_index().sort_values('Number of students who took the SHSAT', ascending=False).reset_index(drop=True)

# Create a dataframe of SHSAT Registration Count
school_registrations = shsat.groupby('School name')['Number of students who registered for the SHSAT'].sum().reset_index().sort_values('Number of students who registered for the SHSAT', ascending=False).reset_index(drop=True)

# Merge the DataFrames
registration_merge = pd.merge(school_registrations, school_take, how='left', on='School name')
registration_merge.head(10)
sns.set(style="whitegrid")

# Initialize the matplotlib figure
f, ax = plt.subplots(figsize=(15, 10))

# Plot the total schools per city
sns.set_color_codes("pastel")
sns.barplot(x="Number of students who registered for the SHSAT", y='School name', data=registration_merge,
            label="# of SHSAT Registrations", color="g")

# Plot the total community schools per city
sns.set_color_codes("muted")
sns.barplot(x="Number of students who took the SHSAT", y="School name", data=registration_merge,
            label="# of Students who Took SHSAT", color="g")

# Add a legend and informative axis label
ax.legend(ncol=2, loc="lower right", frameon=True)
ax.set(xlim=(0, 300), ylabel="School Name", title='SHSAT School Registration Distribution',
       xlabel="# of Registrations")
sns.despine(left=True, bottom=True)
import pandas as pd
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

df_zero = df[df['Economic Need Index'] != 0]
df_zero2 = df_zero[df_zero['School Income Estimate'] != 0]

data = df_zero2[['Economic Need Index', 'School Income Estimate']]

sse = {}
for k in range(1, 10):
    kmeans = KMeans(n_clusters=k, max_iter=1000).fit(data)
    data["clusters"] = kmeans.labels_
    #print(data["clusters"])
    sse[k] = kmeans.inertia_ # Inertia: Sum of distances of samples to their closest cluster center
plt.figure()
plt.plot(list(sse.keys()), list(sse.values()))
plt.xlabel("Number of cluster")
plt.ylabel("SSE")
plt.show()
# Import KMeans Model
from sklearn.cluster import KMeans

# Graph and create 3 clusters of Employee Turnover
kmeans = KMeans(n_clusters=3,random_state=2)
kmeans.fit(df_zero2[["Economic Need Index","School Income Estimate"]])

kmeans_colors = ['green' if c == 0 else 'red' if c == 2 else 'yellow' if c == 3 else 'blue' for c in kmeans.labels_]

fig = plt.figure(figsize=(10, 6))
plt.scatter(x="Economic Need Index",y="School Income Estimate", data=df_zero2,
            alpha=0.25,color = kmeans_colors)
plt.xlabel("Economic Need Index")
plt.ylabel("School Income Estimate")
plt.scatter(x=kmeans.cluster_centers_[:,0],y=kmeans.cluster_centers_[:,1],color="black",marker="X",s=100)
plt.title("Clusters of School Economic Need & Income Estimate")
plt.show()
sat = pd.read_csv('../input/new-york-city-sat-results/2012-sat-results.csv')
sat = sat.rename(columns={"DBN":"Location Code"})

sat_merge = pd.merge(df, sat, on='Location Code', how='left')
sat_merge[sat_merge['SCHOOL NAME'].isnull() == False].shape
sat_merge2 = sat_merge[sat_merge['SCHOOL NAME'].isnull() == False]
sat_merge2['SAT Math Avg. Score'] = sat_merge2['SAT Math Avg. Score'].replace("s","0")
sat_merge2['SAT Critical Reading Avg. Score'] = sat_merge2['SAT Critical Reading Avg. Score'].replace("s","0")
sat_merge2['SAT Writing Avg. Score'] = sat_merge2['SAT Writing Avg. Score'].replace("s","0")
sat_merge2['SAT Critical Reading Avg. Score'] = pd.to_numeric(sat_merge2['SAT Critical Reading Avg. Score'])
sat_merge2['SAT Writing Avg. Score'] = pd.to_numeric(sat_merge2['SAT Writing Avg. Score'])
sat_merge2['SAT Math Avg. Score'] = pd.to_numeric(sat_merge2['SAT Math Avg. Score'])
sat_merge2.head(1)
# Set up the matplotlib figure
f, axes = plt.subplots(ncols=2, figsize=(15, 6))

sat_merge2.plot(kind="scatter", x="Average Math Proficiency", y="SAT Math Avg. Score", ax=axes[0],
    s=sat_merge2['Economic Need Index']*100, c="School Income Estimate", cmap=plt.get_cmap("jet"),label='Schools', title='Math SAT Scores VS Math Proficiency Score',
    colorbar=True, alpha=0.3, figsize=(15,7))

sat_merge2.plot(kind="scatter", x="SAT Critical Reading Avg. Score", y="SAT Writing Avg. Score", ax=axes[1],
    s=sat_merge2['Economic Need Index']*100, c="School Income Estimate", cmap=plt.get_cmap("jet"),label='Schools', title='SAT Writing VS Reading Scores',
    colorbar=True, alpha=0.3, figsize=(15,7))
pop = pd.read_csv('../input/nys-nyserda-low-to-moderate-income-census-populat/nyserda-low-to-moderate-income-new-york-state-census-population-analysis-dataset-average-for-2013-2015.csv')
nyc_pop = pop[pop['County / County Group']=='New York']
nyc_income = pd.DataFrame(nyc_pop['Income Groups'].value_counts()).reset_index()
nyc_income
nyc_pop['Percent of Poverty Level']
plt.figure(figsize=(10,6))
sns.barplot(y=nyc_income['Income Groups'], x=nyc_income['index'], alpha=0.6)
plt.title("Income Groups of NYC", fontsize=16)
plt.xlabel("Income Groups Level", fontsize=16)
plt.ylabel("Count", fontsize=16)
plt.xticks(rotation=90)
plt.show()
race_income = pd.DataFrame(nyc_pop.groupby(['Race / Ethnicity', 'Income Groups'])['Income Groups'].count())
race_income
