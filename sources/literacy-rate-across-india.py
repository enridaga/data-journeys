
import numpy as np #Linear Algebra
import pandas as pd #To Wrok WIth Data

## I am going to use plotly for visualizations. It creates really cool and interective plots.

import matplotlib.pyplot # Just in case.
import plotly.express as px #Easy way to plot charts
import plotly.graph_objects as go #Does the same thing. Gives more options.
import plotly as ply # The whole package
from plotly.subplots import make_subplots #As the name suggests, for subplots.
import seaborn as sns #Just in case. It can be useful sometimes.
df = pd.read_csv("../input/govt-of-india-literacy-rate/GOI.csv") #Loading the dataset.
df['Total - Per. Change'] = (df.loc[:,'Literacy Rate (Persons) - Total - 2011'] - 
                df.loc[:,'Literacy Rate (Persons) - Total - 2001'])/df.loc[:,'Literacy Rate (Persons) - Total - 2001']
df['Rural - Per. Change'] = (df.loc[:,'Literacy Rate (Persons) - Rural - 2011'] - 
                df.loc[:,'Literacy Rate (Persons) - Rural - 2001'])/df.loc[:,'Literacy Rate (Persons) - Total - 2001']
df['Urban - Per. Change'] = (df.loc[:,'Literacy Rate (Persons) - Urban - 2011'] - 
                df.loc[:,'Literacy Rate (Persons) - Urban - 2001'])/df.loc[:,'Literacy Rate (Persons) - Total - 2001']
## Column names are too long, I don't need that much info in a column name. So, i am altering the names.
new_col=[]
for i in df.columns:
    new_col.append(i.split('(Persons) - ')[-1])
df.columns=new_col
df.head()
India = df[df['Category'] == 'Country'].T
India = India.iloc[2:8,:]
India.reset_index(inplace=True)
India.columns = ['Measure', 'Value']
India.loc[:,'Measure'] = India['Measure'].apply(lambda x : str(x).split(' -')[0])
India_2001 = India.iloc[[0,2,4],:]
India_2011 = India.iloc[[1,3,5],:]
fig = go.Figure(data=[
    go.Bar(name='2001', x=India_2001['Measure'], y=India_2001['Value'], marker_color='rgb(55, 83, 109)'),
    go.Bar(name='2011', x=India_2011['Measure'], y=India_2011['Value'], marker_color='rgb(26, 118, 255)')
])
fig.update_layout(barmode='group', title='Overall Literacy Rate in India :')
fig.show()
df = df.iloc[1:,:] #Removing data for India as a whole country.
df.rename(columns={'Country/ States/ Union Territories Name' :'States/ Union Territories'}, inplace = True) 
df.sort_values(by='Total - 2001', inplace=True)

fig = go.Figure(data = [
    go.Scatter(name='2001', x=df['States/ Union Territories'], y=df['Total - 2001']),
    go.Scatter(name='2011', x=df['States/ Union Territories'], y=df['Total - 2011'])
])

fig.update_layout(barmode='group', title = 'Total Literacy Rate Across Nation :')
fig.show()
lowest_2001 = df.sort_values(by=['Total - 2001']).head()
highest_2001 = df.sort_values(by=['Total - 2001']).tail()

fig = go.Figure(data = [
    go.Line(name = 'Lowest_2001', x=lowest_2001['States/ Union Territories'], y=lowest_2001['Total - 2001']),
    go.Line(name = 'Highest_2001', x=highest_2001['States/ Union Territories'], y=highest_2001['Total - 2001'])
])

fig.update_layout(barmode='group', title = 'Lowest and highest "Total literacy" rate in 2001 :')
fig.show()
lowest_2011 = df.sort_values(by=['Total - 2011']).head()
highest_2011 = df.sort_values(by=['Total - 2011']).tail()

fig = go.Figure(data = [
    go.Line(name = 'Lowest_2011', x=lowest_2011['States/ Union Territories'], y=lowest_2011['Total - 2011']),
    go.Line(name = 'Highest_2011', x=highest_2011['States/ Union Territories'], y=highest_2011['Total - 2011'])
])

fig.update_layout(barmode='group', title = 'Lowest and highest "Total Literacy" literacy rate in 2011 :')
fig.show()
px.bar(df.sort_values(by='Total - Per. Change'),
       x='States/ Union Territories', y='Total - Per. Change',
       color='Total - Per. Change', title='Totel Per. Change')
df.sort_values(by='Rural - 2001', inplace=True)

fig = go.Figure(data = [
    go.Line(name='2001', x=df['States/ Union Territories'], y=df['Rural - 2001']),
    go.Line(name='2011', x=df['States/ Union Territories'], y=df['Rural - 2011'])
])

fig.update_layout(barmode='group', title = 'Literacy rate in rural areas acorss the country :')
fig.show()
lowest_2001 = df.sort_values(by=['Rural - 2001']).head()
highest_2001 = df.sort_values(by=['Rural - 2001']).tail()

fig = go.Figure(data = [
    go.Line(name = 'Lowest_2001', x=lowest_2001['States/ Union Territories'], y=lowest_2001['Rural - 2001']),
    go.Line(name = 'Highest_2001', x=highest_2001['States/ Union Territories'], y=highest_2001['Rural - 2001'])
])

fig.update_layout(barmode='group', title = 'Lowest and highest "Rural literacy" rate in 2001 :')
fig.show()
lowest_2011 = df.sort_values(by=['Rural - 2011']).head()
highest_2011 = df.sort_values(by=['Rural - 2011']).tail()

fig = go.Figure(data = [
    go.Line(name = 'Lowest_2011', x=lowest_2011['States/ Union Territories'], y=lowest_2011['Rural - 2011']),
    go.Line(name = 'Highest_2011', x=highest_2011['States/ Union Territories'], y=highest_2011['Rural - 2011'])
])

fig.update_layout(barmode='group', title = 'Lowest and highest "Rural literacy" rate in 2011 :')
fig.show()
px.bar(df.sort_values(by='Rural - Per. Change'),
       x='States/ Union Territories', y='Rural - Per. Change',
       color='Rural - Per. Change', title='Rural Per. Change')
df.sort_values(by='Urban - 2001', inplace=True)

fig = go.Figure(data = [
    go.Line(name='2001', x=df['States/ Union Territories'], y=df['Urban - 2001']),
    go.Line(name='2011', x=df['States/ Union Territories'], y=df['Urban - 2011'])
])

fig.update_layout(barmode='group')
fig.show()
lowest_2001 = df.sort_values(by=['Urban - 2001']).head()
highest_2001 = df.sort_values(by=['Urban - 2001']).tail()

fig = go.Figure(data = [
    go.Line(name = 'Lowest_2001', x=lowest_2011['States/ Union Territories'], y=lowest_2001['Urban - 2001']),
    go.Line(name = 'Highest_2001', x=highest_2011['States/ Union Territories'], y=highest_2001['Urban - 2001'])
])

fig.update_layout(barmode='group', title = 'Lowest and highest "Urban literacy" rate in 2001 :')
fig.show()
lowest_2011 = df.sort_values(by=['Urban - 2011']).head()
highest_2011 = df.sort_values(by=['Urban - 2011']).tail()

fig = go.Figure(data = [
    go.Line(name = 'Lowest_2011', x=lowest_2011['States/ Union Territories'], y=lowest_2011['Urban - 2011']),
    go.Line(name = 'Highest_2011', x=highest_2011['States/ Union Territories'], y=highest_2011['Urban - 2011'])
])

fig.update_layout(barmode='group', title = 'Lowest and highest "Urban literacy" rate in 2011 :')
fig.show()
px.bar(df.sort_values(by='Urban - Per. Change'),
       x='States/ Union Territories', y='Urban - Per. Change',
       color='Urban - Per. Change', title='Urban Per. Change')
temp_1 = df.groupby(by=['Category'])['Total - 2001'].mean().reset_index().T
temp_2 = df.groupby(by=['Category'])['Total - 2011'].mean().reset_index().T

temp_3 = df.groupby(by=['Category'])['Rural - 2001'].mean().reset_index().T
temp_4 = df.groupby(by=['Category'])['Rural - 2011'].mean().reset_index().T

temp_5 = df.groupby(by=['Category'])['Urban - 2001'].mean().reset_index().T
temp_6 = df.groupby(by=['Category'])['Urban - 2011'].mean().reset_index().T

frames = [temp_1, temp_2, temp_3, temp_4, temp_5, temp_6]
temp = pd.concat(frames)
loc = [0,1,3,5,7,9,11]
temp = temp.iloc[loc,:]
temp = temp.iloc[1:,:]
temp.reset_index(inplace=True)
temp.columns=['Category','State','Union Territory']


fig = go.Figure(data = [
    go.Bar(name='States', y=temp['Category'], x=temp['State'], orientation='h', marker_color='rgb(26, 118, 255)'),
    go.Bar(name='Union Territories', y=temp['Category'], x=temp['Union Territory'], orientation='h', marker_color='rgb(55, 83, 109)')
])
fig.update_layout(barmode='group')
fig.show()
East = ['Arunachal Pradesh','Assam','Jharkhand','West Bengal','Odisha',
        'Mizoram','Meghalaya','Manipur','Sikkim','Tripura','Nagaland']
West = ['Maharashtra','Gujarat','Goa']    
North = ['Uttar Pradesh','Bihar','Jammu & Kashmir','Rajasthan', 'Punjab','Haryana','Madhya Pradesh',
        'Chhattisgarh','Uttarakhand','NCT of Delhi','Tamil Nadu','Chandigarh','Himachal Pradesh',]
South = ['Andhra Pradesh','Karnataka','Kerala']

def zone_applier(x):
    if x in East :
        return 'East'
    elif x in West :
        return 'West'
    elif x in North :
        return 'North'
    else :
        return 'South'

State = df[df['Category']=='State']
State['Zone'] =State['States/ Union Territories'].apply(zone_applier)
State = State.groupby(by='Zone').agg('mean')
State = State.iloc[:,:6]
State.reset_index(inplace=True)

State = State.T.reset_index()
State.columns = State.iloc[0,:]
State = State.iloc[1:,:]

fig = go.Figure(data=[
    go.Bar(name='East', x=State['Zone'], y=State['East']),
    go.Bar(name='West', x=State['Zone'], y=State['West']),
    go.Bar(name='North', x=State['Zone'], y=State['North']),
    go.Bar(name='South', x=State['Zone'], y=State['South'])
])
fig.update_layout(barmode='group', title='Avg. Literacy Rate by Zone:')
fig.show()

fig = make_subplots(rows=2,cols=2)
fig.add_trace(go.Bar(name='East', x=State['Zone'], y=State['East']), row=1,col=1)
fig.add_trace(go.Bar(name='West', x=State['Zone'], y=State['West']), row=1, col=2)
fig.add_trace(go.Bar(name='North', x=State['Zone'], y=State['North']), row=2, col=1)
fig.add_trace(go.Bar(name='South', x=State['Zone'], y=State['South']), row=2, col=2)
fig.show()
