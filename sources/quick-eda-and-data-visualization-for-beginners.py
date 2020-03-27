
import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns
import pandas_profiling as pp
import plotly.graph_objs as go
from plotly.offline import iplot
import plotly.express as px
df_train = pd.read_csv('../input/learn-together/train.csv',index_col='Id')
df_test  = pd.read_csv('../input/learn-together/test.csv',index_col='Id')
df_train.head()
print("Train dataset shape: "+ str(df_train.shape))
print("Test dataset shape:  "+ str(df_test.shape))
df_train.info()
df_train.describe().T
print(df_train.iloc[:,10:-1].columns)
pd.unique(df_train.iloc[:,10:-1].values.ravel())
df_train.iloc[:,10:-1] = df_train.iloc[:,10:-1].astype("category")
df_test.iloc[:,10:] = df_test.iloc[:,10:].astype("category")
f,ax = plt.subplots(figsize=(8,6))
sns.heatmap(df_train.corr(),annot=True, 
            linewidths=.5, fmt='.1f', ax=ax) 

plt.show()
df_train.plot(kind='scatter', x='Vertical_Distance_To_Hydrology', 
              y='Horizontal_Distance_To_Hydrology', alpha=0.5, 
              color='darkblue', figsize = (12,9))

plt.title('Vertical And Horizontal Distance To Hydrology')
plt.xlabel("Vertical Distance")
plt.ylabel("Horizontal Distance")

plt.show()
df_train.plot(kind='scatter', x='Aspect', y='Hillshade_3pm', 
              alpha=0.5, color='maroon', figsize = (12,9))

plt.title('Aspect and Hillshade 3pm Relation')
plt.xlabel("Aspect")
plt.ylabel("Hillshade 3pm")

plt.show()
df_train.plot(kind='scatter', x='Hillshade_Noon', y='Hillshade_3pm', 
              alpha=0.5, color='purple', 
              figsize = (12,9))

plt.title('Hillshade Noon and Hillshade 3pm Relation')
plt.xlabel("Hillshade_Noon")
plt.ylabel("Hillshade 3pm")

plt.show()
trace1 = go.Box(
    y=df_train["Vertical_Distance_To_Hydrology"],
    name = 'Vertical Distance',
    marker = dict(color = 'rgb(0,145,119)')
)

trace2 = go.Box(
    y=df_train["Horizontal_Distance_To_Hydrology"],
    name = 'Horizontal Distance',
    marker = dict(color = 'rgb(5, 79, 174)')
)

data = [trace1, trace2]
layout = dict(autosize=False, width=700,height=500, title='Distance To Hydrology', paper_bgcolor='rgb(243, 243, 243)', 
              plot_bgcolor='rgb(243, 243, 243)', margin=dict(l=40,r=30,b=80,t=100,))

fig = dict(data=data, layout=layout)
iplot(fig)
trace1 = go.Box(
    y=df_train["Hillshade_Noon"],
    name = 'Hillshade Noon',
    marker = dict(color = 'rgb(255,111,145)')
)

trace2 = go.Box(
    y=df_train["Hillshade_3pm"],
    name = 'Hillshade 3pm',
    marker = dict(color = 'rgb(132,94,194)')
)

data = [trace1, trace2]
layout = dict(autosize=False, width=700,height=500, title='Hillshade 3pm and Noon', paper_bgcolor='rgb(243, 243, 243)', 
              plot_bgcolor='rgb(243, 243, 243)', margin=dict(l=40,r=30,b=80,t=100,))

fig = dict(data=data, layout=layout)
iplot(fig)
f,ax=plt.subplots(1,2,figsize=(15,7))
df_train.Vertical_Distance_To_Hydrology.plot.hist(ax=ax[0],bins=30,
                                                  edgecolor='black',color='crimson') 
                                       
ax[0].set_title('Vertical Distance To Hydrology')
x1=list(range(-150,350,50))
ax[0].set_xticks(x1)

df_train.Horizontal_Distance_To_Hydrology.plot.hist(ax=ax[1],bins=30,
                                                    edgecolor='black',color='darkmagenta') 
                                                                                                        
ax[1].set_title('Horizontal Distance To Hydrology')
x2=list(range(0,1000,100))
ax[1].set_xticks(x2)

plt.show
soil_types = df_train.iloc[:,14:-1].sum(axis=0)

plt.figure(figsize=(18,9))
sns.barplot(x=soil_types.index, y=soil_types.values, 
            palette="rocket")

plt.xticks(rotation= 75)
plt.ylabel('Total')
plt.title('Count of Soil Types With Value 1',
          color = 'darkred',fontsize=12)

plt.show()
wilderness_areas = df_train.iloc[:,10:14].sum(axis=0)

plt.figure(figsize=(7,5))
sns.barplot(x=wilderness_areas.index, y=wilderness_areas.values, 
            palette="Blues_d")

plt.xticks(rotation=90)
plt.title('Wilderness Areas',color = 'darkred',fontsize=12)
plt.ylabel('Total')

plt.show()
cover_type = df_train["Cover_Type"].value_counts()
df_cover_type = pd.DataFrame({'CoverType': cover_type.index, 'Total':cover_type.values})

fig = px.bar(df_cover_type, x='CoverType', y='Total', 
             height=400, width=650)

fig.show()
f,ax=plt.subplots(1,3,figsize=(21,7))
df_train.plot.scatter(ax=ax[0],x='Cover_Type', y='Horizontal_Distance_To_Fire_Points', 
                      alpha=0.5, color='purple')

ax[0].set_title('Horizontal Distance To Fire Points')
x1=list(range(1,8,1))
ax[0].set_ylabel("")
ax[0].set_xlabel("Cover Type")
df_train.plot.scatter(ax=ax[1],x='Cover_Type', y='Horizontal_Distance_To_Roadways', 
                      alpha=0.5, color='purple')

ax[1].set_title('Horizontal Distance To Roadways')
x2=list(range(1,8,1))
ax[1].set_ylabel("")
ax[1].set_xlabel("Cover Type")
df_train.plot.scatter(ax=ax[2],x='Cover_Type', y='Horizontal_Distance_To_Hydrology', 
                      alpha=0.5, color='purple')

ax[2].set_title('Horizontal Distance To Hydrology')
x2=list(range(1,8,1))
ax[2].set_ylabel("")
ax[2].set_xlabel("Cover Type")

plt.show()
report = pp.ProfileReport(df_train)

report.to_file("report.html")

report