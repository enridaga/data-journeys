
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
color = sns.color_palette()
### matplotlib inline

import plotly.offline as py
py.init_notebook_mode(connected=True)
import plotly.graph_objs as go
import plotly.tools as tls

pd.options.mode.chained_assignment = None
pd.options.display.max_columns = 999
from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))
kiva_loans_df = pd.read_csv("../input/kiva_loans.csv")
kiva_loans_df.head()
kiva_mpi_locations_df = pd.read_csv("../input/kiva_mpi_region_locations.csv")
kiva_mpi_locations_df.head()
loan_theme_ids_df = pd.read_csv("../input/loan_theme_ids.csv")
loan_theme_ids_df.head()
loan_themes_by_region_df = pd.read_csv("../input/loan_themes_by_region.csv")
loan_themes_by_region_df.head()
kiva_loans_df.shape
cnt_srs = kiva_loans_df['country'].value_counts().head(50)
trace = go.Bar(
    y=cnt_srs.index[::-1],
    x=cnt_srs.values[::-1],
    orientation = 'h',
    marker=dict(
        color=cnt_srs.values[::-1],
        colorscale = 'Viridis',
        reversescale = True
    ),
)

layout = go.Layout(
    title='Country wise distribution of loans',
    width=700,
    height=1000,
    )
data = [trace]
fig = go.Figure(data=data, layout=layout)
py.iplot(fig, filename="CountryLoan")
con_df = pd.DataFrame(kiva_loans_df['country'].value_counts()).reset_index()
con_df.columns = ['country', 'num_loans']
con_df = con_df.reset_index().drop('index', axis=1)

#Find out more at https://plot.ly/python/choropleth-maps/
data = [ dict(
        type = 'choropleth',
        locations = con_df['country'],
        locationmode = 'country names',
        z = con_df['num_loans'],
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
            title = 'Number of Loans'),
      ) ]

layout = dict(
    title = 'Number of loans by Country',
    geo = dict(
        showframe = False,
        showcoastlines = False,
        projection = dict(
            type = 'Mercator'
        )
    )
)

fig = dict( data=data, layout=layout )
py.iplot( fig, validate=False, filename='loans-world-map')
cnt_srs = kiva_loans_df['sector'].value_counts().head(25)
trace = go.Bar(
    y=cnt_srs.index[::-1],
    x=cnt_srs.values[::-1],
    orientation = 'h',
    marker=dict(
        color=cnt_srs.values[::-1],
        colorscale = 'Rainbow',
        reversescale = True
    ),
)

layout = dict(
    title='Sector wise distribution of loans',
    )
data = [trace]
fig = go.Figure(data=data, layout=layout)
py.iplot(fig, filename="SectorLoan")
cnt_srs = kiva_loans_df['activity'].value_counts().head(25)
trace = go.Bar(
    y=cnt_srs.index[::-1],
    x=cnt_srs.values[::-1],
    orientation = 'h',
    marker=dict(
        color=cnt_srs.values[::-1],
        colorscale = 'Picnic',
        reversescale = True
    ),
)

layout = dict(
    title='Activity wise distribution of loans',
    )
data = [trace]
fig = go.Figure(data=data, layout=layout)
py.iplot(fig, filename="ActivityLoan")
plt.figure(figsize=(8,6))
plt.scatter(range(kiva_loans_df.shape[0]), np.sort(kiva_loans_df.loan_amount.values))
plt.xlabel('index', fontsize=12)
plt.ylabel('loan_amount', fontsize=12)
plt.title("Loan Amount Distribution")
plt.show()
plt.figure(figsize=(8,6))
plt.scatter(range(kiva_loans_df.shape[0]), np.sort(kiva_loans_df.funded_amount.values))
plt.xlabel('index', fontsize=12)
plt.ylabel('loan_amount', fontsize=12)
plt.title("Funded Amount Distribution")
plt.show()
ulimit = np.percentile(kiva_loans_df.loan_amount.values, 99)
llimit = np.percentile(kiva_loans_df.loan_amount.values, 1)
kiva_loans_df['loan_amount_trunc'] = kiva_loans_df['loan_amount'].copy()
kiva_loans_df['loan_amount_trunc'].loc[kiva_loans_df['loan_amount']>ulimit] = ulimit
kiva_loans_df['loan_amount_trunc'].loc[kiva_loans_df['loan_amount']<llimit] = llimit

plt.figure(figsize=(12,8))
sns.distplot(kiva_loans_df.loan_amount_trunc.values, bins=50, kde=False)
plt.xlabel('loan_amount_trunc', fontsize=12)
plt.title("Loan Amount Histogram after outlier truncation")
plt.show()
cnt_srs = kiva_loans_df.term_in_months.value_counts()

trace = go.Bar(
    x=cnt_srs.index,
    y=cnt_srs.values,
    marker=dict(
        color=cnt_srs.values,
        colorscale = 'Picnic',
        reversescale = True
    ),
)

layout = go.Layout(
    title='Repayment Term in Months'
)

data = [trace]
fig = go.Figure(data=data, layout=layout)
py.iplot(fig, filename="RepaymentIntervals")
cnt_srs = kiva_loans_df.repayment_interval.value_counts()

trace = go.Bar(
    x=cnt_srs.index,
    y=cnt_srs.values,
    marker=dict(
        color=cnt_srs.values,
        colorscale = 'Rainbow',
        reversescale = True
    ),
)

layout = go.Layout(
    title='Repayment Interval of loans'
)

data = [trace]
fig = go.Figure(data=data, layout=layout)
py.iplot(fig, filename="RepaymentIntervals")
cnt_srs = kiva_loans_df.lender_count.value_counts()

trace = go.Bar(
    x=cnt_srs.index,
    y=cnt_srs.values,
    marker=dict(
        color=cnt_srs.values,
        colorscale = 'Portland',
        reversescale = True
    ),
)

layout = go.Layout(
    title='Lender Count'
)

data = [trace]
fig = go.Figure(data=data, layout=layout)
py.iplot(fig, filename="LenderCount")
cnt_srs = kiva_loans_df.lender_count.value_counts().head(100)

trace = go.Bar(
    x=cnt_srs.index,
    y=cnt_srs.values,
    marker=dict(
        color=cnt_srs.values,
        colorscale = 'Portland',
        reversescale = True
    ),
)

layout = go.Layout(
    title='Lender Count Top 100'
)

data = [trace]
fig = go.Figure(data=data, layout=layout)
py.iplot(fig, filename="LenderCount")
olist = []
for ll in kiva_loans_df["borrower_genders"].values:
    if str(ll) != "nan":
        olist.extend( [l.strip() for l in ll.split(",")] )
temp_series = pd.Series(olist).value_counts()

labels = (np.array(temp_series.index))
sizes = (np.array((temp_series / temp_series.sum())*100))

trace = go.Pie(labels=labels, values=sizes)
layout = go.Layout(
    title='Borrower Gender'
)
data = [trace]
fig = go.Figure(data=data, layout=layout)
py.iplot(fig, filename="BorrowerGender")