
import numpy as np
import pandas as pd

import plotly.offline as py
py.init_notebook_mode(connected=True)
import plotly.graph_objs as go
import plotly.tools as tls

from kaggle.competitions import twosigmanews

pd.options.mode.chained_assignment = None
pd.options.display.max_columns = 999

# Get 2Sigma environment
env = twosigmanews.make_env()
# Get the data
mt_df, nt_df = env.get_training_data()
mt_df.head()
print("We have {:,} market samples in the training dataset.".format(mt_df.shape[0]))
mt_df.dtypes
mt_df.isna().sum()
mt_df.nunique()
asset1Code = 'AAPL.O'
asset1_df = mt_df[(mt_df['assetCode'] == asset1Code) & (mt_df['time'] > '2015-01-01') & (mt_df['time'] < '2017-01-01')]
# Create a trace
trace1 = go.Scatter(
    x = asset1_df['time'].dt.strftime(date_format='%Y-%m-%d').values,
    y = asset1_df['close'].values
)

layout = dict(title = "Closing prices of {}".format(asset1Code),
              xaxis = dict(title = 'Month'),
              yaxis = dict(title = 'Price (USD)'),
              )
data = [trace1]

py.iplot(dict(data=data, layout=layout), filename='basic-line')
asset1_df['high'] = asset1_df['open']
asset1_df['low'] = asset1_df['close']

for ind, row in asset1_df.iterrows():
    if row['close'] > row['open']:
        asset1_df.loc[ind, 'high'] = row['close']
        asset1_df.loc[ind, 'low'] = row['open']

trace1 = go.Candlestick(
    x = asset1_df['time'].dt.strftime(date_format='%Y-%m-%d').values,
    open = asset1_df['open'].values,
    low = asset1_df['low'].values,
    high = asset1_df['high'].values,
    close = asset1_df['close'].values
)

layout = dict(title = "Candlestick chart for {}".format(asset1Code),
              xaxis = dict(
                  title = 'Month',
                  rangeslider = dict(visible = False)
              ),
              yaxis = dict(title = 'Price (USD)')
             )
data = [trace1]

py.iplot(dict(data=data, layout=layout), filename='basic-line')
mt_df['time'].dt.date.describe()
print("There are {} missing values in the `time` column".format(mt_df['time'].isna().sum()))
mt_df['time'].dt.time.describe()
assetsByTradingDay = mt_df.groupby(mt_df['time'].dt.date)['assetCode'].nunique()
# Create a trace
trace1 = go.Bar(
    x = assetsByTradingDay.index, # asset1_df['time'].dt.strftime(date_format='%Y-%m-%d').values,
    y = assetsByTradingDay.values
)

layout = dict(title = "# of assets by trading days",
              xaxis = dict(title = 'Year'),
              yaxis = dict(title = '# of assets'),
              )
data = [trace1]

py.iplot(dict(data=data, layout=layout), filename='basic-line')
print("There are {:,} unique assets in the training set".format(mt_df['assetCode'].nunique()))
print("There are {} missing values in the `assetCode` column".format(mt_df['time'].isna().sum()))
volumeByAssets = mt_df.groupby(mt_df['assetCode'])['volume'].sum()
highestVolumes = volumeByAssets.sort_values(ascending=False)[0:10]
# Create a trace
trace1 = go.Pie(
    labels = highestVolumes.index,
    values = highestVolumes.values
)

layout = dict(title = "Highest trading volumes")
data = [trace1]

py.iplot(dict(data=data, layout=layout), filename='basic-line')
mt_df['assetName'].describe()
print("There are {:,} records with assetName = `Unknown` in the training set".format(mt_df[mt_df['assetName'] == 'Unknown'].size))
assetNameGB = mt_df[mt_df['assetName'] == 'Unknown'].groupby('assetCode')
unknownAssets = assetNameGB.size().reset_index('assetCode')
print("There are {} unique assets without assetName in the training set".format(unknownAssets.shape[0]))
unknownAssets
mt_df['universe'].nunique()
print("There are {:,} missing values in the `universe` column".format(mt_df['universe'].isna().sum()))
print("There are {:,} missing values in the `volume` column".format(mt_df['volume'].isna().sum()))
mt_df['volume'].describe()
zeroVolume = mt_df[mt_df['volume'] == 0]
print("There are {:,} sample in the training set with zero trading volumes".format(len(zeroVolume)))
print("The scoring function will consider {:,} out of {:,} 'zero trading' training samples".format(len(zeroVolume[zeroVolume['universe'] == 1]), len(zeroVolume)))
volumesByTradingDay = mt_df.groupby(mt_df['time'].dt.date)['volume'].sum()
# Create a trace
trace1 = go.Bar(
    x = volumesByTradingDay.index,
    y = volumesByTradingDay.values
)

layout = dict(title = "Trading volumes by date",
              xaxis = dict(title = 'Year'),
              yaxis = dict(title = 'Volume'),
              )
data = [trace1]

py.iplot(dict(data=data, layout=layout), filename='basic-line')
print("There are {:,} missing values in the `open` column".format(mt_df['open'].isna().sum()))
mt_df['open'].describe()
print("There are {:,} missing values in the `close` column".format(mt_df['close'].isna().sum()))
mt_df['close'].describe()
print("There are {} missing `returnsOpenNextMktres10` values in the training set.".format(mt_df['returnsOpenNextMktres10'].isna().sum()))
# No growth, no decrease
print(len(mt_df[mt_df['returnsOpenNextMktres10'] == 0]))
mt_df['returnsOpenNextMktres10'].describe()
outliers = mt_df[(mt_df['returnsOpenNextMktres10'] > 1) |  (mt_df['returnsOpenNextMktres10'] < -1)]
outliers['returnsOpenNextMktres10'].describe()
# returnsOpenNextMktres10 data without outliers
woOutliers = mt_df[(mt_df['returnsOpenNextMktres10'] < 1) &  (mt_df['returnsOpenNextMktres10'] > -1)]
woOutliers['returnsOpenNextMktres10'].describe()
# Create a trace
trace1 = go.Histogram(
    x = woOutliers.sample(n=10000)['returnsOpenNextMktres10'].values
)

layout = dict(title = "returnsOpenNextMktres10 (random 10.000 sample; without outliers)")
data = [trace1]

py.iplot(dict(data=data, layout=layout), filename='basic-line')