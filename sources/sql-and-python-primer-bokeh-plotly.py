
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
#%matplotlib inline
import sqlite3
import os
from bokeh.plotting import figure, show
from bokeh.io import output_notebook
output_notebook()
#%matplotlib inline
import plotly.offline as py
py.init_notebook_mode(connected=True)
import plotly.graph_objs as go
import plotly.tools as tls
import warnings
warnings.filterwarnings('ignore')
conn = sqlite3.connect('../input/database.sqlite')
c = conn.cursor()
for row in c.execute(
                    # SQL statement 
                    """
                        SELECT   * 
                        FROM     Country 
                        LIMIT    2
                        
                     """ ):
    print(row)
# Store Country data in a pandas dataframe via a SQL query
Country = pd.read_sql(
                       """
                       
                        SELECT  * 
                        FROM    Country
                        
                       """, con=conn)
# Pandas code
Country.head(5)
# SQL query 
pd.read_sql(
            """
                SELECT   * 
                FROM     Country 
                LIMIT    5 
                
            """, con=conn)
# Pandas Boolean Indexing
Country[Country['CountryCode'] == 'AFG']
# SQL WHERE clause
pd.read_sql(
        """ 
            SELECT   * 
            FROM     Country 
            WHERE    CountryCode = 'AFG'
            
        """, con=conn)
# SQL GROUP BY Clause
pd.read_sql(
        """ 
            SELECT      Region
                        ,COUNT(*) AS [Count]
            FROM        Country 
            GROUP BY    Region
            ORDER BY    2 DESC
            
        """, con=conn)
# Let's do a LEFT JOIN on some subqueries 
pd.read_sql(
        """ 
           
            SELECT      A.CountryCode
                        ,B.LatestPopulationCensus
                        ,B.SourceOfMostRecentIncomeAndExpenditureData
                        ,B.ShortName
            FROM       ( 
                            -- First subquery (i.e the Left table)
                            
                           SELECT      CountryCode
                                        ,LatestPopulationCensus
                                        ,SourceOfMostRecentIncomeAndExpenditureData
                                        ,ShortName
                           FROM        Country
                           WHERE       CountryCode IN ('AFG','ALB', 'ASM', 'BEL')
                        ) AS A
            LEFT JOIN   (
                            -- Second subquery (i.e the right table )
                            
                            SELECT      CountryCode
                                        ,LatestPopulationCensus
                                        ,SourceOfMostRecentIncomeAndExpenditureData
                                        ,ShortName
                            FROM        Country AS A
                            WHERE       CountryCode IN ('AFG','ARM', 'URY', 'BEL')
                            
                          ) AS B
            ON          A.CountryCode = B.CountryCode    
            
        """, con=conn)
# UNION 
pd.read_sql(
        """ 
                           SELECT      CountryCode
                                        ,LatestPopulationCensus
                                        ,SourceOfMostRecentIncomeAndExpenditureData
                                        ,ShortName
                           FROM        Country
                           WHERE       CountryCode IN ('AFG','ALB', 'ASM', 'BEL')
                       
                           UNION
                           
                           SELECT      CountryCode
                                        ,LatestPopulationCensus
                                        ,SourceOfMostRecentIncomeAndExpenditureData
                                        ,ShortName
                           FROM        Country AS A
                           WHERE       CountryCode IN ('AFG','ARM', 'URY', 'BEL')
            
        """, con=conn)
# INTERSECT 
pd.read_sql(
        """ 
                           SELECT      CountryCode
                                        ,LatestPopulationCensus
                                        ,SourceOfMostRecentIncomeAndExpenditureData
                                        ,ShortName
                           FROM        Country
                           WHERE       CountryCode IN ('AFG','ALB', 'ASM', 'BEL')
                       
                           INTERSECT
                           
                           SELECT      CountryCode
                                        ,LatestPopulationCensus
                                        ,SourceOfMostRecentIncomeAndExpenditureData
                                        ,ShortName
                           FROM        Country AS A
                           WHERE       CountryCode IN ('AFG','ARM', 'URY', 'BEL')
            
        """, con=conn)
Indicators = pd.read_sql(""" SELECT   * 
                             FROM     Indicators 
                             WHERE    IndicatorCode IN 
                                      (  'AG.LND.PRCP.MM, AG.LND.FRST.K2'
                                       , 'EG.ELC.ACCS.ZS', 'EG.ELC.FOSL.ZS'
                                       , 'EN.POP.DNST', 'SG.VAW.REAS.ZS'
                                       , 'SM.POP.NETM', 'SP.POP.65UP.TO.ZS'
                                       , 'FI.RES.TOTL.DT.ZS', 'GC.DOD.TOTL.GD.ZS'
                                       , 'MS.MIL.XPND.GD.ZS','SI.POV.GINI'
                                       , 'IP.JRN.ARTC.SC', 'SE.ADT.1524.LT.ZS'
                                      )  
                        """, con=conn)
#Regions = ['ARB', 'EUU', 'LCN' , 'NAC',  'EAS', 'SSF', 'World']
gini = Indicators[Indicators['IndicatorCode']== 'SI.POV.GINI']
gini.CountryCode.unique()
# Plotting a Subplot of the Seaborn regplot
f, ((ax1, ax2, ax3), (ax4,ax5,ax6), (ax7, ax8, ax9)) = plt.subplots(3,3,figsize=(12,10))

# Plot of GINI index of China
points = ax1.scatter(gini[gini['CountryCode'] == 'CHN']["Year"], gini[gini['CountryCode'] == 'CHN']["Value"],
                     c=gini[gini['CountryCode'] == 'CHN']["Value"], s=100, cmap="viridis")
sns.regplot("Year", "Value", data=gini[gini['CountryCode'] == 'CHN'], ax=ax1)
ax1.set_title("GINI Index of China")

# Plot of GINI of Argentina
points = ax2.scatter(gini[gini['CountryCode'] == 'ARG']["Year"], gini[gini['CountryCode'] == 'ARG']["Value"],
                     c=gini[gini['CountryCode'] == 'ARG']["Value"], s=85, cmap="viridis")
sns.regplot("Year", "Value", data=gini[gini['CountryCode'] == 'ARG'], ax=ax2)
ax2.set_title("GINI Index of Argentina")

points = ax3.scatter(gini[gini['CountryCode'] == 'UGA']["Year"], gini[gini['CountryCode'] == 'UGA']["Value"],
                     c=gini[gini['CountryCode'] == 'UGA']["Value"], s=100, cmap="afmhot")
sns.regplot("Year", "Value", data=gini[gini['CountryCode'] == 'UGA'], ax=ax3)
ax3.set_title("GINI Index of Uganda")

points = ax4.scatter(gini[gini['CountryCode'] == 'USA']["Year"], gini[gini['CountryCode'] == 'USA']["Value"],
                     c=gini[gini['CountryCode'] == 'USA']["Value"], s=100, cmap="Purples_r")
sns.regplot("Year", "Value", data=gini[gini['CountryCode'] == 'USA'], ax=ax4)
ax4.set_title("GINI Index of USA")

points = ax5.scatter(gini[gini['CountryCode'] == 'COL']["Year"], gini[gini['CountryCode'] == 'COL']["Value"],
                     c=gini[gini['CountryCode'] == 'COL']["Value"], s=100, cmap="YlOrBr")
sns.regplot("Year", "Value", data=gini[gini['CountryCode'] == 'COL'], ax=ax5)
ax5.set_title("GINI Index of Colombia")

points = ax6.scatter(gini[gini['CountryCode'] == 'RWA']["Year"], gini[gini['CountryCode'] == 'RWA']["Value"],
                     c=gini[gini['CountryCode'] == 'RWA']["Value"], s=100, cmap="Blues")
sns.regplot("Year", "Value", data=gini[gini['CountryCode'] == 'RWA'], ax=ax6)
ax6.set_title("GINI Index of Rwanda")

points = ax7.scatter(gini[gini['CountryCode'] == 'RUS']["Year"], gini[gini['CountryCode'] == 'RUS']["Value"],
                     c=gini[gini['CountryCode'] == 'RUS']["Value"], s=100, cmap="Blues")
sns.regplot("Year", "Value", data=gini[gini['CountryCode'] == 'RUS'], ax=ax7)
ax7.set_title("GINI Index of Russia")

points = ax8.scatter(gini[gini['CountryCode'] == 'ECU']["Year"], gini[gini['CountryCode'] == 'ECU']["Value"],
                     c=gini[gini['CountryCode'] == 'ECU']["Value"], s=100, cmap="winter")
sns.regplot("Year", "Value", data=gini[gini['CountryCode'] == 'ECU'], ax=ax8)
ax8.set_title("GINI Index of Ecuador")

points = ax9.scatter(gini[gini['CountryCode'] == 'CAF']["Year"], gini[gini['CountryCode'] == 'CAF']["Value"],
                     c=gini[gini['CountryCode'] == 'CAF']["Value"], s=100, cmap="magma")
sns.regplot("Year", "Value", data=gini[gini['CountryCode'] == 'CAF'], ax=ax9)
ax9.set_title("GINI Index of Central African Republic")
sns.set_style(style="dark")
plt.tight_layout()
# Barplots of Youth literacy rates in 1990
data = Indicators[Indicators['IndicatorCode'] == 'SE.ADT.1524.LT.ZS'][Indicators['Year'] == 1990]
x, y = (list(x) for x in zip(*sorted(zip(data['Value'].values, data['CountryName'].values), 
                                                            reverse = False)))

# Plotting using Plotly 
trace2 = go.Bar(
    x=x ,
    y=y,
    marker=dict(
        color=x,
        colorscale = 'Portland',
        reversescale = True
    ),
    name='Percentage of Youth Literacy Rate',
    orientation='h',
)

layout = dict(
    title='Barplot of Youth Literacy Rate in 1990',
     width = 680, height = 1500,
    yaxis=dict(
        showgrid=False,
        showline=False,
        showticklabels=True,
#         domain=[0, 0.85],
    ))

fig1 = go.Figure(data=[trace2])
fig1['layout'].update(layout)
py.iplot(fig1, filename='plots')

# Barplot of Youth literacy rates in 2010
data = Indicators[Indicators['IndicatorCode'] == 'SE.ADT.1524.LT.ZS'][Indicators['Year'] == 2010]
x, y = (list(x) for x in zip(*sorted(zip(data['Value'].values, data['CountryName'].values), 
                                                            reverse = False)))

# Plotting using Plotly 
trace2 = go.Bar(
    x=x ,
    y=y,
    marker=dict(
        color=x,
        colorscale = 'Portland',
        reversescale = True
    ),
    name='Percentage of Youth Literacy Rate',
    orientation='h',
)

layout = dict(
    title='Barplot of Youth Literacy Rate in 2010',
     width = 680, height = 1500,
    yaxis=dict(
        showgrid=False,
        showline=False,
        showticklabels=True,
#         domain=[0, 0.85],
    ))

fig1 = go.Figure(data=[trace2])
fig1['layout'].update(layout)
py.iplot(fig1, filename='plots')
# Create some useful helper variables
data = Indicators[Indicators['IndicatorCode'] == 'EG.ELC.ACCS.ZS']
data['Year'] = [str(x) for x in data['Year']]
years = list(data['Year'].unique())
country = [
     u'Jordan', u'Kazakhstan', u'Kenya', u'Kiribati',
       u'Korea, Dem. Rep.', u'Korea, Rep.', u'Kosovo', u'Kuwait',
       u'Kyrgyz Republic', u'Lao PDR', u'Latvia', u'Lebanon', u'Lesotho',
       u'Liberia', u'Libya', u'Liechtenstein', u'Lithuania', u'Luxembourg',
       u'Macao SAR, China', u'Macedonia, FYR', u'Madagascar', u'Malawi',
       u'Malaysia', u'Maldives', u'Mali', u'Malta', u'Marshall Islands',
       u'Mauritania', u'Mauritius', u'Mexico', u'Micronesia, Fed. Sts.',
       u'Moldova', u'Monaco', u'Mongolia', u'Montenegro', u'Morocco',
       u'Mozambique', u'Myanmar', u'Namibia', u'Nepal', u'Netherlands',
       u'New Caledonia', u'New Zealand', u'Nicaragua', u'Niger',
       u'Nigeria', u'Norway', u'Oman', u'Pakistan', u'Palau', u'Panama',
       u'Papua New Guinea', u'Paraguay', u'Peru', u'Philippines',
       u'Poland', u'Portugal', u'Puerto Rico', u'Qatar', u'Romania',
       u'Russian Federation', u'Rwanda', u'Samoa', u'San Marino',
       u'Sao Tome and Principe', u'Saudi Arabia', u'Senegal', u'Serbia',
       u'Seychelles', u'Sierra Leone', u'Singapore', u'Slovak Republic',
       u'Slovenia', u'Solomon Islands', u'Somalia', u'South Africa',
       u'South Sudan', u'Spain', u'Sri Lanka' u'Sudan', u'Suriname',
       u'Swaziland', u'Sweden', u'Switzerland', u'Syrian Arab Republic',
       u'Tajikistan', u'Tanzania', u'Thailand', u'Timor-Leste', u'Togo',
       u'Tonga', u'Trinidad and Tobago', u'Tunisia', u'Turkey',
       u'Turkmenistan', u'Tuvalu', u'Uganda',
       u'Ukraine', u'United Arab Emirates', u'United Kingdom',
       u'United States', u'Uruguay'
]
from math import pi

from bokeh.io import show
from bokeh.models import ColumnDataSource, HoverTool, LinearColorMapper
from bokeh.plotting import figure
from bokeh.models import ColumnDataSource, HoverTool, LinearColorMapper
# data = data.set_index('Year')
# this is the colormap from the original NYTimes plot
colors = ["#75968f", "#a5bab7", "#c9d9d3", "#e2e2e2", "#dfccce", "#ddb7b1", "#cc7878", "#933b41", "#550b1d"]
mapper = LinearColorMapper(palette=colors)
# Set up the data for plotting. We will need to have values for every
# pair of year/month names. Map the rate to a color.
countr = []
year = []
color = []
rate = []
for y in years:
    for m in country:
        countr.append(m)
        year.append(y)
#         d[(d['x']>2) & (d['y']>7)]
        monthly_rate = data[(data['CountryName']==m) & (data['Year']==y)]['Value']
        rate.append(monthly_rate)

source = ColumnDataSource(
    data=dict(country=countr, year=year, rate=rate)
)

TOOLS = "hover,save,pan,box_zoom,wheel_zoom"

p = figure(title="Access to Electricity",
           x_range=years, y_range=list(reversed(country)),
           x_axis_location="above", plot_width=900, plot_height=900,
           tools=TOOLS)

p.grid.grid_line_color = None
p.axis.axis_line_color = None
p.axis.major_tick_line_color = None
p.axis.major_label_text_font_size = "5pt"
p.axis.major_label_standoff = 0
p.xaxis.major_label_orientation = pi / 3

p.rect(x="year", y="country", width=1, height=1,
       source=source,
       fill_color={'field': 'rate', 'transform': mapper},
       line_color=None)

p.select_one(HoverTool).tooltips = [
#     ('date', '@countr @year'),
    ('rate', '@rate'),
]

show(p)      # show the plot
data = Indicators[Indicators['IndicatorCode'] == 'SG.VAW.REAS.ZS']
data = Indicators[Indicators['IndicatorCode'] == 'SE.ADT.1524.LT.ZS']
data['Year'] = [str(x) for x in data['Year']]
years = ['2000',
 '2001',
 '2002',
 '2003',
 '2004',
 '2005',
 '2006',
 '2007',
 '2008',
 '2009',
 '2010',
 '2011',
 '2012',
 '2013',
 '2014']
country = ['Burkina Faso', 'Central African Republic', 'Kuwait', 'Turkey',
       'United Arab Emirates', 'Uruguay', 'Bolivia', 'Cameroon',
       'Egypt, Arab Rep.', 'Iran, Islamic Rep.', 'Mali', 'New Caledonia',
       'Swaziland', 'Tonga', 'Maldives', 'Poland', 'Rwanda', 'Afghanistan',
       'Benin', 'Burundi', 'Guinea-Bissau', 'Jordan', 'Vanuatu', 'Vietnam',
       'American Samoa', 'Argentina', 'Brazil', 'Comoros', 'Guam',
       'Hungary', 'Indonesia', 'Malaysia', 'Mexico', 'Mozambique', 'Palau',
       'Panama', 'Philippines', 'Puerto Rico', 'Singapore', 'South Africa',
       'Thailand', 'Trinidad and Tobago', 'Bahrain', 'Bangladesh',
       'Brunei Darussalam', 'Cuba', 'Dominican Republic', 'Greece',
       'India', 'Italy', 'Macao SAR, China', 'Nepal', 'Pakistan', 'Peru',
       'Portugal', 'Sao Tome and Principe', 'Spain', 'Sri Lanka',
       'Syrian Arab Republic', 'Venezuela, RB', 'Chile', 'China',
       'Ecuador', 'Haiti', 'Morocco', 'Paraguay', 'Zimbabwe', 'Israel',
       'Myanmar', 'Costa Rica', 'Liberia', 'Libya', 'Tunisia', 'Malta',
       'Qatar', 'Algeria', 'Malawi', 'Seychelles', "Cote d'Ivoire",
       'Senegal', 'Tanzania', 'Armenia', 'Belarus', 'Estonia',
       'Kazakhstan', 'Latvia', 'Lithuania', 'Moldova','Lesotho', 'Madagascar', 'Mauritania', 'Mongolia',
       'Papua New Guinea', 'Sudan', 'Togo', 'Uzbekistan', 'Albania',
       'Angola', 'Bulgaria', 'Congo, Dem. Rep.', 'Honduras', 'Nicaragua',
       'Niger', 'Ukraine', 'Eritrea', 'Georgia', 'Oman', 'Sierra Leone',
       'Suriname', 'Bhutan', 'Cayman Islands', 'Lebanon',
       'Korea, Dem. Rep.', 'South Sudan', 'Guyana', 'Timor-Leste',
       'Congo, Rep.', 'Montenegro', 'Serbia', 'Austria']
from math import pi

from bokeh.io import show
from bokeh.models import ColumnDataSource, HoverTool, LinearColorMapper
from bokeh.plotting import figure
from bokeh.models import ColumnDataSource, HoverTool, LinearColorMapper
# data = data.set_index('Year')
# this is the colormap from the original NYTimes plot
colors = ["#75968f", "#a5bab7", "#c9d9d3", "#e2e2e2", "#dfccce", "#ddb7b1", "#cc7878", "#933b41", "#550b1d"]
mapper = LinearColorMapper(palette=colors)
# Set up the data for plotting. We will need to have values for every
# pair of year/month names. Map the rate to a color.
countr = []
year = []
color = []
rate = []
for y in years:
    for m in country:
        countr.append(m)
        year.append(y)
#         d[(d['x']>2) & (d['y']>7)]
        monthly_rate = data[(data['CountryName']==m) & (data['Year']==y)]['Value']
        rate.append(monthly_rate)

source = ColumnDataSource(
    data=dict(country=countr, year=year, rate=rate)
)

TOOLS = "hover,save,pan,box_zoom,wheel_zoom"

p = figure(title="Women who believe Husbands are justified in beating wifes",
           x_range=years, y_range=list(reversed(country)),
           x_axis_location="above", plot_width=900, plot_height=900,
           tools=TOOLS)

p.grid.grid_line_color = None
p.axis.axis_line_color = None
p.axis.major_tick_line_color = None
p.axis.major_label_text_font_size = "5pt"
p.axis.major_label_standoff = 0
p.xaxis.major_label_orientation = pi / 3

p.rect(x="year", y="country", width=1, height=1,
       source=source,
       fill_color={'field': 'rate', 'transform': mapper},
       line_color=None)

p.select_one(HoverTool).tooltips = [
#     ('date', '@countr @year'),
    ('rate', '@rate'),
]

show(p)      # show the plot