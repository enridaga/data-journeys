

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# charts
import seaborn as sns 
import matplotlib.pyplot as plt
import squarify #TreeMap

# import graph objects as "go"

import plotly.graph_objs as go

#%matplotlib inline

#ignore warning 
import warnings
warnings.filterwarnings("ignore")

# Input data files are available in the "../input/" directory.
import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

data_import = pd.read_csv("/kaggle/input/india-trade-data/2018-2010_import.csv")
data_export = pd.read_csv("/kaggle/input/india-trade-data/2018-2010_export.csv")
data_export.head(5)
data_import.head(5)
data_export.describe()
data_import.describe()
data_export.info()
data_import.info()
data_import.isnull().sum()
data_import[data_import.value==0].head(5)
data_import[data_import.country == "UNSPECIFIED"].head(5)
print("Duplicate imports : "+str(data_import.duplicated().sum()))
print("Duplicate exports : "+str(data_export.duplicated().sum()))

def cleanup(data_df):
    #setting country UNSPECIFIED to nan
    data_df['country']= data_df['country'].apply(lambda x : np.NaN if x == "UNSPECIFIED" else x)
    #ignoring where import value is 0 . 
    data_df = data_df[data_df.value!=0]
    data_df.dropna(inplace=True)
    data_df.year = pd.Categorical(data_df.year)
    data_df.drop_duplicates(keep="first",inplace=True)
    return data_df
data_import = cleanup(data_import)
data_export = cleanup(data_export)
data_import.isnull().sum()
print("Import Commodity Count : "+str(len(data_import['Commodity'].unique())))
print("Export Commodity Count : "+str(len(data_export['Commodity'].unique())))
df = pd.DataFrame(data_import['Commodity'].value_counts())
df.head(20)
print("No of Country were we are importing Comodities are "+str(len(data_import['country'].unique())))
print("No of Country were we are Exporting Comodities are "+str(len(data_export['country'].unique())))
df3 = data_import.groupby('year').agg({'value':'sum'})

df4 = data_export.groupby('year').agg({'value':'sum'})


df3['deficit'] = df4.value - df3.value
df3

# create trace1 
trace1 = go.Bar(
                x = df3.index,
                y = df3.value,
                name = "Import",
                marker = dict(color = 'rgba(0,191,255, 1)',
                             line=dict(color='rgb(0,0,0)',width=1.5)),
                text = df3.value)
# create trace2 
trace2 = go.Bar(
                x = df4.index,
                y = df4.value,
                name = "Export",
                marker = dict(color = 'rgba(1, 255, 130, 1)',
                              line=dict(color='rgb(0,0,0)',width=1.5)),
                text = df4.value)

trace3 = go.Bar(
                x = df3.index,
                y = df3.deficit,
                name = "Trade Deficit",
                marker = dict(color = 'rgba(220, 20, 60, 1)',
                              line=dict(color='rgb(0,0,0)',width=1.5)),
                text = df3.deficit)


data = [trace1, trace2, trace3]
layout = go.Layout(barmode = "group")
fig = go.Figure(data = data, layout = layout)
fig.update_layout(
    title=go.layout.Title(
        text="Yearwise Import/Export/Trade deficit",
        xref="paper",
        x=0
    ),
    xaxis=go.layout.XAxis(
        title=go.layout.xaxis.Title(
            text="Year",
            font=dict(
                family="Courier New, monospace",
                size=18,
                color="#7f7f7f"
            )
        )
    ),
    yaxis=go.layout.YAxis(
        title=go.layout.yaxis.Title(
            text="Value",
            font=dict(
                family="Courier New, monospace",
                size=18,
                color="#7f7f7f"
            )
        )
    )
)




fig.show()
df5 = data_import.groupby('country').agg({'value':'sum'})
df5 = df5.sort_values(by='value', ascending = False)
df5 = df5[:10]

df6 = data_export.groupby('country').agg({'value':'sum'})
df6 = df6.sort_values(by='value', ascending = False)
df6 = df6[:10]
sns.set(rc={'figure.figsize':(15,6)})
ax1 = plt.subplot(121)

sns.barplot(df5.value,df5.index).set_title('Country Wise Import')

ax2 = plt.subplot(122)
sns.barplot(df6.value,df6.index).set_title('Country Wise Export')
plt.tight_layout()
plt.show()
fig = go.Figure()
# Create and style traces
fig.add_trace(go.Scatter(x=df3.index, y=df3.value, name='Import',mode='lines+markers',
                         line=dict(color='firebrick', width=4)))
fig.add_trace(go.Scatter(x=df4.index, y=df4.value, name = 'Export',mode='lines+markers',
                         line=dict(color='royalblue', width=4)))
fig.update_layout(
    title=go.layout.Title(
        text="Yearwise Import/Export",
        xref="paper",
        x=0
    ),
    xaxis=go.layout.XAxis(
        title=go.layout.xaxis.Title(
            text="Year",
            font=dict(
                family="Courier New, monospace",
                size=18,
                color="#7f7f7f"
            )
        )
    ),
    yaxis=go.layout.YAxis(
        title=go.layout.yaxis.Title(
            text="Value",
            font=dict(
                family="Courier New, monospace",
                size=18,
                color="#7f7f7f"
            )
        )
    )
)

fig.show()
df3 = data_import.groupby('Commodity').agg({'value':'sum'})
df3 = df3.sort_values(by='value', ascending = False)
df3 = df3[:10]

df4 = data_export.groupby('Commodity').agg({'value':'sum'})
df4 = df4.sort_values(by='value', ascending = False)
df4 = df4[:10]
sns.set(rc={'figure.figsize':(15,10)})
#ax1 = plt.subplot(121)
sns.barplot(df3.value,df3.index).set_title('Commodity Wise Import')
plt.show()
#ax2 = plt.subplot(122)
sns.barplot(df4.value,df4.index).set_title('Commodity Wise Import')
plt.show()
expensive_import = data_import[data_import.value>1000]
expensive_import.head(10)
#fig, ax = plt.subplots(1,1,figsize=(18,6)) 
plt.figure(figsize=(20,9))
#plt.rcParams['figure.figsize']=(23,10)
# the size of A4 paper
#fig.set_size_inches(11.7, 8.27)
ax = sns.boxplot(x="HSCode", y="value", data=expensive_import).set_title('Expensive Imports HsCode distribution')
plt.show()

df =expensive_import.groupby(['HSCode']).agg({'value': 'sum'})
df = df.sort_values(by='value')
 
value=np.array(df)
commodityCode=df.index
plt.style.use('ggplot')
plt.rcParams['figure.figsize'] = (10.0, 6.0)
squarify.plot(sizes=value, label=commodityCode, alpha=.7 )
plt.axis('off')
plt.title("Expensive Imports HsCode Share")
plt.show()
len(expensive_import['country'].unique())
df1 = expensive_import.groupby(['country']).agg({'value': 'sum'})
df1 = df1.sort_values(by='value')
value=np.array(df1)
country=df1.index
plt.style.use('ggplot')
plt.rcParams['figure.figsize'] = (10.0, 10.0)
squarify.plot(sizes=value, label=country, alpha=.7 )
plt.title("Expensive Imports Countrywise Share")
plt.axis('off')
plt.show()