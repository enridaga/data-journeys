
import numpy as np
import pandas as pd
pd.options.mode.chained_assignment = None
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
#%matplotlib inline
import cufflinks as cf
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
import plotly.graph_objs as go
init_notebook_mode(connected=True)
cf.go_offline()
from kaggle.competitions import twosigmanews
env = twosigmanews.make_env()
(market_train_df, news_train_df) = env.get_training_data()
market_train_df['date'] = market_train_df['time'].dt.strftime('%Y-%m-%d')
market_train_df.describe().round(3)
market_train_df[market_train_df['returnsClosePrevRaw1'] < -.7]
someAssetsWithBadData = ['FLEX.O','MAT.O','SHLD.O','ZNGA.O']
someMarketData = market_train_df[(market_train_df['assetCode'].isin(someAssetsWithBadData)) 
                & (market_train_df['time'] >= '2016-07-05')
                & (market_train_df['time'] < '2016-07-08')].sort_values('assetCode')
someMarketData
selectedAssets = market_train_df[(market_train_df['assetCode'].isin(someAssetsWithBadData))
                                 & (market_train_df['time'] >= '2015-01-01')]
selectedAssetsPivot = selectedAssets.pivot(index='date',columns='assetCode',values='close')

flex = go.Scatter(x = selectedAssetsPivot.index,y = selectedAssetsPivot['FLEX.O'],mode = 'lines',name = 'FLEX.O')
mat = go.Scatter(x = selectedAssetsPivot.index,y = selectedAssetsPivot['MAT.O'],mode = 'lines',name = 'MAT.O')
shld = go.Scatter(x = selectedAssetsPivot.index,y = selectedAssetsPivot['SHLD.O'],mode = 'lines',name = 'SHLD.O')
znga = go.Scatter(x = selectedAssetsPivot.index,y = selectedAssetsPivot['ZNGA.O'],mode = 'lines',name = 'ZNGA.O')
data = [flex,mat,shld,znga]
layout = go.Layout(
    title = 'assets with error on 2016-07-06',
    yaxis = dict(title='close price')
)
fig = go.Figure(data=data, layout=layout)
iplot(fig)
market_train_df[(market_train_df['date'] == '2016-07-06')
                & ((market_train_df['close'] == 123.45) 
                   | (market_train_df['close'] == 123.47))]
sampleZynga = market_train_df[(market_train_df['assetCode'] == 'ZNGA.O')
                              & (market_train_df['time'] >= '2016-06-01')
                              & (market_train_df['time'] < '2016-09-01')]

sampleZyngaReturn = go.Scatter(
    x = sampleZynga['date'],
    y = sampleZynga['returnsClosePrevMktres1'],
    mode = 'lines',
    name = 'ZNGA.O'
)
data = [sampleZyngaReturn]
layout = go.Layout(
    title = 'ZNGA.O returnsClosePrevMktres1 metric<br>(errors in return data linger for 20 days after 2016-07-06 price error)',
    yaxis = dict(
        title='return (%)',
        tickformat = ',.0%',
    )
)
fig = go.Figure(data=data, layout=layout)
iplot(fig)
quickZyngaSample = market_train_df[(market_train_df['assetCode'] == 'ZNGA.O')
                                   & ((market_train_df['time'] >= '2016-06-01') & (market_train_df['time'] < '2016-07-06')
                                      | (market_train_df['time'] >= '2016-08-04') & (market_train_df['time'] < '2016-09-01'))]

quickZyngaSample['ones'] = 1
A = quickZyngaSample[['returnsClosePrevRaw1','ones']]
y = quickZyngaSample['returnsClosePrevMktres1']
m, c = np.linalg.lstsq(A,y,rcond=-1)[0]
zyngaFitLine = m*np.array(quickZyngaSample['returnsClosePrevRaw1'])+c

quickZyngaSampleReturnsComparison = go.Scatter(
    x = quickZyngaSample['returnsClosePrevRaw1'],
    y = quickZyngaSample['returnsClosePrevMktres1'],
    mode = 'markers',
    name = 'ZNGA.O raw vs. mktres return'
)
quickZyngaSampleReturnsComparisonBestFitLine = go.Scatter(
    x = quickZyngaSample['returnsClosePrevRaw1'],
    y = zyngaFitLine,
    mode = 'lines',
    name = 'best fit line'
)
data = [quickZyngaSampleReturnsComparison,quickZyngaSampleReturnsComparisonBestFitLine]
layout = go.Layout(
    title = 'ZNGA.O sample of returnsClosePrevRaw1 vs. returnsClosePrevMktres1',
    xaxis = dict(
        title='returnsClosePrevRaw1 (%)',
        tickformat = ',.0%',
    ),
    yaxis = dict(
        title='returnsClosePrevMktres1 (%)',
        tickformat = ',.0%',
    )
)
fig = go.Figure(data=data, layout=layout)
iplot(fig)
def sampleAssetData(assetCode, date, numDays):
    d = datetime.strptime(date,'%Y-%m-%d')
    start = d - timedelta(days=numDays)
    end = d + timedelta(days=numDays)
    return market_train_df[(market_train_df['assetCode'] == assetCode)
                             & (market_train_df['time'] >= start.strftime('%Y-%m-%d'))
                             & (market_train_df['time'] <= end.strftime('%Y-%m-%d'))].copy()
def updateRawReturns(assetData, indices):
    rowsToUpdate1 = assetData[(assetData.index >= indices[0][0]) & (assetData.index <= indices[0][1])]
    for index, row in rowsToUpdate1.iterrows():
        market_train_df.loc[[index],['returnsClosePrevRaw1']] = assetData['close'].pct_change()
        market_train_df.loc[[index],['returnsOpenPrevRaw1']] = assetData['open'].pct_change()
    rowsToUpdate2 = assetData[(assetData.index >= indices[1][0]) & (assetData.index <= indices[1][1])]
    for index, row in rowsToUpdate2.iterrows():
        market_train_df.loc[[index],['returnsClosePrevRaw10']] = assetData['close'].pct_change(periods=10)
        market_train_df.loc[[index],['returnsOpenPrevRaw10']] = assetData['open'].pct_change(periods=10)
def estimateMktresReturn(sampleData, mktresCol, index):
    sampleData['ones'] = 1
    sampleData.dropna(inplace=True)
    rawCol = mktresCol.replace('Mktres','Raw')
    A = sampleData[[rawCol,'ones']]
    y = sampleData[mktresCol]
    m, c = np.linalg.lstsq(A,y,rcond=-1)[0]
    return c + m * market_train_df.loc[index,rawCol]
def updateMktresReturns(assetCode, assetData, indices):
    # update range of values for returnsClosePrevMktres1 & returnsOpenPrevMktres1
    sample1 = assetData[(assetData.index < indices[2][0]) | (assetData.index > indices[2][1])]
    rowsToUpdate1 = assetData[(assetData.index >= indices[2][0]) & (assetData.index <= indices[2][1])]
    for index, row in rowsToUpdate1.iterrows():
        market_train_df.loc[[index],['returnsClosePrevMktres1']] = estimateMktresReturn(sample1,'returnsClosePrevMktres1',index)
        market_train_df.loc[[index],['returnsOpenPrevMktres1']] = estimateMktresReturn(sample1,'returnsOpenPrevMktres1',index)
    # update range of values for returnsClosePrevMktres10 & returnsOpenPrevMktres10
    sample2 = assetData[(assetData.index < indices[3][0]) | (assetData.index > indices[3][1])]
    rowsToUpdate2 = assetData[(assetData.index >= indices[3][0]) & (assetData.index <= indices[3][1])]
    l = []
    for index, row in rowsToUpdate2.iterrows():
        market_train_df.loc[[index],['returnsClosePrevMktres10']] = estimateMktresReturn(sample2,'returnsClosePrevMktres10',index)
        est = estimateMktresReturn(sample2,'returnsOpenPrevMktres10',index)
        l.append(est)
        market_train_df.loc[[index],['returnsOpenPrevMktres10']] = est
    # update range of values for returnsOpenNextMktres10
    rowsToUpdate3 = assetData[(assetData.index >= indices[4][0]) & (assetData.index <= indices[4][1])]
    i = 0
    for index, row in rowsToUpdate3.iterrows():
        market_train_df.loc[[index],['returnsOpenNextMktres10']] = l[i]
        i += 1
def fixBadReturnData(assetCode, badDate, badIndex, badReturnDataRanges, dayWindow):
    # store copy of bad data window
    badDataWindow = sampleAssetData(assetCode,badDate,dayWindow)
    badDataWindow.reset_index(inplace=True)
    # store indices needed to update raw and mktres return data
    newIdx = badDataWindow[badDataWindow['index'] == badIndex].index[0]
    indices = [
        # range of bad data for... returnsClosePrevRaw1 & returnsOpenPrevRaw1
        [badIndex,badDataWindow.loc[newIdx+badReturnDataRanges[0],'index']],
        # returnsClosePrevRaw10 & returnsOpenPrevRaw10
        [badIndex,badDataWindow.loc[newIdx+badReturnDataRanges[1],'index']],
        # returnsClosePrevMktres1 & returnsOpenPrevMktres1
        [badIndex,badDataWindow.loc[newIdx+badReturnDataRanges[2],'index']],
        # returnsClosePrevMktres10 & returnsOpenPrevMktres10
        [badIndex,badDataWindow.loc[newIdx+badReturnDataRanges[3],'index']],
        # returnsOpenNextMktres10
        [badDataWindow.loc[newIdx+badReturnDataRanges[4],'index'],badDataWindow.loc[newIdx+badReturnDataRanges[5],'index']]
    ]
    badDataWindow.set_index('index',inplace=True)
    # correct bad raw return data
    updateRawReturns(badDataWindow,indices)
    # estimate affected mktres return data
    updateMktresReturns(assetCode,badDataWindow,indices)
# bad volume, open, and close for ZNGA.O on 2016-07-06
assetCode = 'ZNGA.O'
badDate = '2016-07-06'
badIndex = market_train_df[(market_train_df['assetCode'] == assetCode) & (market_train_df['date'] == badDate)].index[0]
# correct bad data
market_train_df.loc[[badIndex],['volume']] = 19213100
market_train_df.loc[[badIndex],['open']] = 2.64
market_train_df.loc[[badIndex],['close']] = 2.75
# ranges of affected return data
#   integers specify how many rows of data need to be updated for different return columns, in reference to "badDate" row
badReturnDataRanges = [1,10,20,20,-11,9]
# sample data window (on either side of "badDate")
sampleWindow = 45
# fix bad return data in market_train_df
fixBadReturnData(assetCode,badDate,badIndex,badReturnDataRanges,sampleWindow)
# bad volume, open, and close for FLEX.O on 2016-07-06
assetCode = 'FLEX.O'
badDate = '2016-07-06'
badIndex = market_train_df[(market_train_df['assetCode'] == assetCode) & (market_train_df['date'] == badDate)].index[0]
# correct bad data
market_train_df.loc[[badIndex],['volume']] = 5406600
market_train_df.loc[[badIndex],['open']] = 11.580
market_train_df.loc[[badIndex],['close']] = 11.750
# ranges of affected return data
badReturnDataRanges = [1,10,20,20,-11,9]
# sample data window (on either side of "badDate")
sampleWindow = 45
# fix bad return data in market_train_df
fixBadReturnData(assetCode,badDate,badIndex,badReturnDataRanges,sampleWindow)
# bad volume, open, and close for SHLD.O on 2016-07-06
assetCode = 'SHLD.O'
badDate = '2016-07-06'
badIndex = market_train_df[(market_train_df['assetCode'] == assetCode) & (market_train_df['date'] == badDate)].index[0]
# correct bad data
market_train_df.loc[[badIndex],['volume']] = 279300
market_train_df.loc[[badIndex],['open']] = 12.8900
market_train_df.loc[[badIndex],['close']] = 13.1400
# ranges of affected return data
badReturnDataRanges = [1,10,20,20,-11,9]
# sample data window (on either side of "badDate")
sampleWindow = 45
# fix bad return data in market_train_df
fixBadReturnData(assetCode,badDate,badIndex,badReturnDataRanges,sampleWindow)
# bad volume, open, and close for MAT.O on 2016-07-06
assetCode = 'MAT.O'
badDate = '2016-07-06'
badIndex = market_train_df[(market_train_df['assetCode'] == assetCode) & (market_train_df['date'] == badDate)].index[0]
# correct bad data
market_train_df.loc[[badIndex],['volume']] = 3242100
market_train_df.loc[[badIndex],['open']] = 32.13
market_train_df.loc[[badIndex],['close']] = 31.52
# ranges of affected return data
badReturnDataRanges = [1,10,20,20,-11,9]
# sample data window (on either side of "badDate")
sampleWindow = 45
# fix bad return data in market_train_df
fixBadReturnData(assetCode,badDate,badIndex,badReturnDataRanges,sampleWindow)
# bad volume, open, and close for BBBY.O on 2016-07-06
assetCode = 'BBBY.O'
badDate = '2016-07-06'
badIndex = market_train_df[(market_train_df['assetCode'] == assetCode) & (market_train_df['date'] == badDate)].index[0]
# correct bad data
market_train_df.loc[[badIndex],['volume']] = 4205500
market_train_df.loc[[badIndex],['open']] = 42.23
market_train_df.loc[[badIndex],['close']] = 43.55
# ranges of affected return data
badReturnDataRanges = [1,10,20,20,-11,9]
# sample data window (on either side of "badDate")
sampleWindow = 45
# fix bad return data in market_train_df
fixBadReturnData(assetCode,badDate,badIndex,badReturnDataRanges,sampleWindow)
# bad volume, open, and close for DISH.O on 2016-07-06
assetCode = 'DISH.O'
badDate = '2016-07-06'
badIndex = market_train_df[(market_train_df['assetCode'] == assetCode) & (market_train_df['date'] == badDate)].index[0]
# correct bad data
market_train_df.loc[[badIndex],['volume']] = 2303300
market_train_df.loc[[badIndex],['open']] = 50.06
market_train_df.loc[[badIndex],['close']] = 51.33
# ranges of affected return data
badReturnDataRanges = [1,10,20,20,-11,9]
# sample data window (on either side of "badDate")
sampleWindow = 45
# fix bad return data in market_train_df
fixBadReturnData(assetCode,badDate,badIndex,badReturnDataRanges,sampleWindow)
# bad volume, open, and close for NDAQ.O on 2016-07-06
assetCode = 'NDAQ.O'
badDate = '2016-07-06'
badIndex = market_train_df[(market_train_df['assetCode'] == assetCode) & (market_train_df['date'] == badDate)].index[0]
# correct bad data
market_train_df.loc[[badIndex],['volume']] = 733400
market_train_df.loc[[badIndex],['open']] = 64.64
market_train_df.loc[[badIndex],['close']] = 64.74
# ranges of affected return data
badReturnDataRanges = [1,10,20,20,-11,9]
# sample data window (on either side of "badDate")
sampleWindow = 45
# fix bad return data in market_train_df
fixBadReturnData(assetCode,badDate,badIndex,badReturnDataRanges,sampleWindow)
# bad volume, open, and close for PCAR.O on 2016-07-06
assetCode = 'PCAR.O'
badDate = '2016-07-06'
badIndex = market_train_df[(market_train_df['assetCode'] == assetCode) & (market_train_df['date'] == badDate)].index[0]
# correct bad data
market_train_df.loc[[badIndex],['volume']] = 2394300
market_train_df.loc[[badIndex],['open']] = 50.16
market_train_df.loc[[badIndex],['close']] = 50.79
# ranges of affected return data
badReturnDataRanges = [1,10,20,20,-11,9]
# sample data window (on either side of "badDate")
sampleWindow = 45
# fix bad return data in market_train_df
fixBadReturnData(assetCode,badDate,badIndex,badReturnDataRanges,sampleWindow)
# bad volume, open, and close for PZZA.O on 2016-07-06
assetCode = 'PZZA.O'
badDate = '2016-07-06'
badIndex = market_train_df[(market_train_df['assetCode'] == assetCode) & (market_train_df['date'] == badDate)].index[0]
# correct bad data
market_train_df.loc[[badIndex],['volume']] = 185100
market_train_df.loc[[badIndex],['open']] = 67.86
market_train_df.loc[[badIndex],['close']] = 67.91
# ranges of affected return data
badReturnDataRanges = [1,10,20,20,-11,9]
# sample data window (on either side of "badDate")
sampleWindow = 45
# fix bad return data in market_train_df
fixBadReturnData(assetCode,badDate,badIndex,badReturnDataRanges,sampleWindow)
market_train_df.describe().round(3)
market_train_df[market_train_df['open'] > 2000]
market_train_df.loc[market_train_df['assetCode'] == 'WW.N','assetCode'] = 'TW.N'
twSample = market_train_df[(market_train_df['assetCode'] == 'TW.N') & (market_train_df['time'] < '2012-01-01')]
twSample['month'] = twSample['time'].dt.strftime('%Y') + '-' + twSample['time'].dt.strftime('%m')
twSample['month'].iplot(kind='hist',bins=48,layout=dict(title='monthly TW.N (and WW.N) records in market training dataset'))
# dropping Towers Watson data affected by 2010-01-04 open price error
market_train_df = market_train_df[~((market_train_df['assetCode'] == 'TW.N')
                                  & (market_train_df['time'] >= '2009-12-16')
                                  & (market_train_df['time'] < '2010-01-08'))]
market_train_df.describe().round(3)
market_train_df[market_train_df['returnsOpenPrevRaw1'] > 10]
market_train_df[((market_train_df['assetCode'] == 'PBRa.N') & (market_train_df['time'] >= '2007-05-02') & (market_train_df['time'] < '2007-05-05'))
                | ((market_train_df['assetCode'] == 'EXH.N') & (market_train_df['time'] >= '2007-08-22') & (market_train_df['time'] < '2007-08-25'))
                | ((market_train_df['assetCode'] == 'ATPG.O') & (market_train_df['time'] >= '2007-10-29') & (market_train_df['time'] < '2007-11-01'))
                | ((market_train_df['assetCode'] == 'TEO.N') & (market_train_df['time'] >= '2007-02-26') & (market_train_df['time'] < '2007-03-01'))
               ].sort_values('assetCode')
market_train_df = market_train_df[market_train_df['time'].dt.year >= 2009]
market_train_df.describe().round(3)
market_train_df[(market_train_df['returnsOpenPrevRaw1'] > 3)]
paragonSample = market_train_df[(market_train_df['assetCode'] == 'PGN.N')
                                & (market_train_df['time'] >= '2011-08-01')
                                & (market_train_df['time'] < '2012-06-01')]

paragonOpen = go.Scatter(x = paragonSample['date'],y = paragonSample['open'],mode = 'lines',name = 'open price')
paragonClose = go.Scatter(x = paragonSample['date'],y = paragonSample['close'],mode = 'lines',name = 'close price')
paragonOpenPrevRaw1 = go.Scatter(x = paragonSample['date'],y = paragonSample['returnsOpenPrevRaw1'],mode = 'lines',name = 'returnsOpenPrevRaw1',yaxis='y2')
data = [paragonOpen,paragonClose,paragonOpenPrevRaw1]
layout = go.Layout(
    title = 'PGN.N stock price vs. OpenPrevRaw1 return',
    yaxis = dict(
        title='Stock Price',
        showgrid=False,
        zeroline=False
    ),
    yaxis2 = dict(
        title = 'return (%)',
        overlaying = 'y',
        showgrid=False,
        zeroline=False,
        side = 'right',
        tickformat = ',.0%'
    ),
    legend=dict(orientation="h",x=.25)
)
fig = go.Figure(data=data, layout=layout)
iplot(fig)
# bad return data for PGN.N from 2011-10-19 to 2012-03-28
assetCode = 'PGN.N'
# fix first section of incorrect raw returns
badDate1 = '2011-12-23'
badIndex1 = market_train_df[(market_train_df['assetCode'] == assetCode) & (market_train_df['date'] == badDate1)].index[0]
dayWindow1 = 20
badReturnDataRanges1 = [0,9]
badDataWindow1 = sampleAssetData(assetCode,badDate1,dayWindow1)
badDataWindow1.reset_index(inplace=True)
newIdx1 = badDataWindow1[badDataWindow1['index'] == badIndex1].index[0]
rawIndices1 = [
    [badIndex1,badDataWindow1.loc[newIdx1+badReturnDataRanges1[0],'index']],
    [badIndex1,badDataWindow1.loc[newIdx1+badReturnDataRanges1[1],'index']]
]
badDataWindow1.set_index('index',inplace=True)
updateRawReturns(badDataWindow1,rawIndices1)
# fix second section of incorrect raw returns
badDate2 = '2012-02-15'
badIndex2 = market_train_df[(market_train_df['assetCode'] == assetCode) & (market_train_df['date'] == badDate2)].index[0]
dayWindow2 = 20
badReturnDataRanges2 = [0,9]
badDataWindow2 = sampleAssetData(assetCode,badDate2,dayWindow2)
badDataWindow2.reset_index(inplace=True)
newIdx2 = badDataWindow2[badDataWindow2['index'] == badIndex2].index[0]
rawIndices2 = [
    [badIndex2,badDataWindow2.loc[newIdx2+badReturnDataRanges2[0],'index']],
    [badIndex2,badDataWindow2.loc[newIdx2+badReturnDataRanges2[1],'index']]
]
badDataWindow2.set_index('index',inplace=True)
updateRawReturns(badDataWindow2,rawIndices2)
# fix mktres returns
badDate3 = '2012-01-09'
badIndex3 = market_train_df[(market_train_df['assetCode'] == assetCode) & (market_train_df['date'] == badDate3)].index[0]
dayWindow3 = 120
badReturnDataRanges3 = [-44,46,55,-55,44]
badDataWindow3 = sampleAssetData(assetCode,badDate3,dayWindow3)
badDataWindow3.reset_index(inplace=True)
newIdx3 = badDataWindow3[badDataWindow3['index'] == badIndex3].index[0]
indices3 = [[],[],
    [badDataWindow3.loc[newIdx3+badReturnDataRanges3[0],'index'],badDataWindow3.loc[newIdx3+badReturnDataRanges3[1],'index']],
    [badDataWindow3.loc[newIdx3+badReturnDataRanges3[0],'index'],badDataWindow3.loc[newIdx3+badReturnDataRanges3[2],'index']],
    [badDataWindow3.loc[newIdx3+badReturnDataRanges3[3],'index'],badDataWindow3.loc[newIdx3+badReturnDataRanges3[4],'index']]
]
badDataWindow3.set_index('index',inplace=True)
updateMktresReturns(assetCode,badDataWindow3,indices3)
paragonSample = market_train_df[(market_train_df['assetCode'] == 'PGN.N')
                                & (market_train_df['time'] >= '2011-08-01')
                                & (market_train_df['time'] < '2012-06-01')]

paragonOpen = go.Scatter(x = paragonSample['date'],y = paragonSample['open'],mode = 'lines',name = 'open price')
paragonClose = go.Scatter(x = paragonSample['date'],y = paragonSample['close'],mode = 'lines',name = 'close price')
paragonOpenPrevRaw1 = go.Scatter(x = paragonSample['date'],y = paragonSample['returnsOpenPrevRaw1'],mode = 'lines',name = 'returnsOpenPrevRaw1',yaxis='y2')
paragonClosePrevMktres10 = go.Scatter(x = paragonSample['date'],y = paragonSample['returnsClosePrevMktres10'],mode = 'lines',name = 'returnsClosePrevMktres10',yaxis='y2')
paragonOpenNextMktres10 = go.Scatter(x = paragonSample['date'],y = paragonSample['returnsOpenNextMktres10'],mode = 'lines',name = 'returnsOpenNextMktres10',yaxis='y2')
data = [paragonOpen,paragonClose,paragonOpenPrevRaw1,paragonClosePrevMktres10,paragonOpenNextMktres10]
layout = go.Layout(
    title = 'PGN.N stock price vs. selected return metrics',
    yaxis = dict(
        title='Stock Price',
        showgrid=False
    ),
    yaxis2 = dict(
        title = 'return (%)',
        overlaying = 'y',
        showgrid=False,
        side = 'right',
        tickformat = ',.0%'
    ),
    legend=dict(orientation="h",x=.15)
)
fig = go.Figure(data=data, layout=layout)
iplot(fig)
qorvo = market_train_df[(market_train_df['assetCode'] == 'QRVO.O')]

qorvoOpen = go.Scatter(x = qorvo['date'],y = qorvo['open'],mode = 'lines',name = 'open price',yaxis= 'y2')
qorvoClose = go.Scatter(x = qorvo['date'],y = qorvo['close'],mode = 'lines',name = 'close price',yaxis= 'y2')
qorvoOpenPrevRaw10 = go.Scatter(x = qorvo['date'],y = qorvo['returnsOpenPrevRaw10'],mode = 'lines',name = 'returnsOpenPrevRaw10')
qorvoOpenPrevMktres10 = go.Scatter(x = qorvo['date'],y = qorvo['returnsOpenPrevMktres10'],mode = 'lines',name = 'returnsOpenPrevMktres10')
qorvoOpenNextMktres10 = go.Scatter(x = qorvo['date'],y = qorvo['returnsOpenNextMktres10'],mode = 'lines',name = 'returnsOpenNextMktres10')
data = [qorvoOpen,qorvoClose,qorvoOpenPrevRaw10,qorvoOpenPrevMktres10,qorvoOpenNextMktres10]
layout = go.Layout(
    title = 'QRVO.O stock price vs. returns<br>(notice the bad returns data through 2015-02-13)',
    yaxis = dict(
        title='return metrics (%)',
        tickformat = ',.0%',
        showgrid=False,
        zeroline=False
    ),
    yaxis2 = dict(
        title = 'Stock Price',
        overlaying = 'y',
        showgrid=False,
        zeroline=False,
        side = 'right'
    )
)
fig = go.Figure(data=data, layout=layout)
iplot(fig)
# dropping Qorvo data through 2015-02-13
market_train_df = market_train_df[~((market_train_df['assetCode'] == 'QRVO.O')
                                  & (market_train_df['time'] < '2015-02-14'))]
qorvo = market_train_df[(market_train_df['assetCode'] == 'QRVO.O')]

qorvoOpenPrevRaw10 = go.Scatter(x = qorvo['date'],y = qorvo['returnsOpenPrevRaw10'],mode = 'lines',name = 'returnsOpenPrevRaw10')
qorvoOpenPrevMktres10 = go.Scatter(x = qorvo['date'],y = qorvo['returnsOpenPrevMktres10'],mode = 'lines',name = 'returnsOpenPrevMktres10')
qorvoOpenNextMktres10 = go.Scatter(x = qorvo['date'],y = qorvo['returnsOpenNextMktres10'],mode = 'lines',name = 'returnsOpenNextMktres10')
data = [qorvoOpenPrevRaw10,qorvoOpenPrevMktres10,qorvoOpenNextMktres10]
layout = go.Layout(
    title = 'QRVO.O returns',
    yaxis = dict(
        title='return (%)',
        tickformat = ',.0%',
        showgrid=False,
        zeroline=False
    )
)
fig = go.Figure(data=data, layout=layout)
iplot(fig)
# bad volume, open, and close for TECD.O on 2015-03-16
assetCode = 'TECD.O'
badDate = '2015-03-16'
badIndex = market_train_df[(market_train_df['assetCode'] == assetCode) & (market_train_df['date'] == badDate)].index[0]
# correct bad data
market_train_df.loc[[badIndex],['open']] = 56.18
# ranges of affected return data
badReturnDataRanges = [1,10,20,20,-11,9]
# sample data window (on either side of "badDate")
sampleWindow = 45
# fix bad return data in market_train_df
fixBadReturnData(assetCode,badDate,badIndex,badReturnDataRanges, sampleWindow)
ebrnSample = market_train_df[(market_train_df['assetCode'] == 'EBR.N')
                             & (market_train_df['time'] >= '2014-01-01')]

ebrnClose = go.Scatter(x = ebrnSample['date'],y = ebrnSample['close'],mode = 'lines',name = 'close price',yaxis= 'y2')
ebrnClosePrevRaw1 = go.Scatter(x = ebrnSample['date'],y = ebrnSample['returnsClosePrevRaw1'],mode = 'lines',name = 'returnsClosePrevRaw1')
ebrnOpenNextMktres10 = go.Scatter(x = ebrnSample['date'],y = ebrnSample['returnsOpenNextMktres10'],mode = 'lines',name = 'returnsOpenNextMktres10')
data = [ebrnClose,ebrnClosePrevRaw1,ebrnOpenNextMktres10]
layout = go.Layout(
    title = 'EBR.N close price vs. returns<br>(notice gap in data from May to Oct 2016 and bad returns data in Oct 2016)',
    yaxis = dict(
        title='return metrics (%)',
        tickformat = ',.0%',
        showgrid=False,
        zeroline=False
    ),
    yaxis2 = dict(
        title = 'Close Price',
        overlaying = 'y',
        showgrid=False,
        zeroline=False,
        side = 'right'
    )
)
fig = go.Figure(data=data, layout=layout)
iplot(fig)
# dropping EBR.N data in Oct 2016
market_train_df = market_train_df[~((market_train_df['assetCode'] == 'EBR.N')
                                  & (market_train_df['time'] >= '2016-10-01'))]
ebrn = market_train_df[(market_train_df['assetCode'] == 'EBR.N')]
ebrn.iplot(kind='line',x='date',y=['returnsClosePrevRaw1','returnsOpenNextMktres10'],layout=dict(title='EBR.N selected return metrics',yaxis=dict(tickformat = ',.0%')))
market_train_df.describe().round(3)
market_train_df[market_train_df['returnsClosePrevMktres10'] < -1]
hgsi = market_train_df[(market_train_df['assetCode'] == 'HGSI.O')
                       & (market_train_df['time'] < '2010-07-01')]

hgsiClose = go.Scatter(x = hgsi['date'],y = hgsi['close'],mode = 'lines',name = 'close price',yaxis= 'y2')
hgsiOpenPrevMktres10 = go.Scatter(x = hgsi['date'],y = hgsi['returnsOpenPrevMktres10'],mode = 'lines',name = 'returnsOpenPrevMktres10')
hgsiClosePrevMktres10 = go.Scatter(x = hgsi['date'],y = hgsi['returnsClosePrevMktres10'],mode = 'lines',name = 'returnsClosePrevMktres10')
data = [hgsiClose,hgsiOpenPrevMktres10,hgsiClosePrevMktres10]
layout = go.Layout(
    title = 'HGSI.O stock price vs. 10-day Prev returns<br>(notice the large swings and incorrect return values in Aug 2009)',
    yaxis = dict(
        title='return metrics (%)',
        tickformat = ',.0%',
        showgrid=False,
        zeroline=False
    ),
    yaxis2 = dict(
        title = 'Stock Price',
        overlaying = 'y',
        showgrid=False,
        zeroline=False,
        side = 'right'
    )
)
fig = go.Figure(data=data, layout=layout)
iplot(fig)
# bad return data for HGSI.O around June to August 2009
assetCode = 'HGSI.O'
badDate1 = '2009-08-03'
badIndex1 = market_train_df[(market_train_df['assetCode'] == assetCode) & (market_train_df['date'] == badDate1)].index[0]
dayWindow1 = 120
badReturnDataRanges1 = [-7,11,-14,19,-25,8]
badDataWindow1 = sampleAssetData(assetCode,badDate1,dayWindow1)
badDataWindow1.reset_index(inplace=True)
newIdx1 = badDataWindow1[badDataWindow1['index'] == badIndex1].index[0]
indices1 = [[],[],
    [badDataWindow1.loc[newIdx1+badReturnDataRanges1[0],'index'],badDataWindow1.loc[newIdx1+badReturnDataRanges1[1],'index']],
    [badDataWindow1.loc[newIdx1+badReturnDataRanges1[2],'index'],badDataWindow1.loc[newIdx1+badReturnDataRanges1[3],'index']],
    [badDataWindow1.loc[newIdx1+badReturnDataRanges1[4],'index'],badDataWindow1.loc[newIdx1+badReturnDataRanges1[5],'index']]
]
badDataWindow1.set_index('index',inplace=True)
updateMktresReturns(assetCode,badDataWindow1,indices1)
# dropping HGSI.O data in Feb and Mar 2016
market_train_df = market_train_df[~((market_train_df['assetCode'] == 'HGSI.O')
                                  & (market_train_df['time'] < '2009-04-01'))]
hgsi = market_train_df[(market_train_df['assetCode'] == 'HGSI.O')
                       & (market_train_df['time'] < '2010-07-01')]

hgsiClose = go.Scatter(x = hgsi['date'],y = hgsi['close'],mode = 'lines',name = 'close price',yaxis= 'y2')
hgsiOpenPrevMktres10 = go.Scatter(x = hgsi['date'],y = hgsi['returnsOpenPrevMktres10'],mode = 'lines',name = 'returnsOpenPrevMktres10')
hgsiClosePrevMktres10 = go.Scatter(x = hgsi['date'],y = hgsi['returnsClosePrevMktres10'],mode = 'lines',name = 'returnsClosePrevMktres10')
data = [hgsiClose,hgsiOpenPrevMktres10,hgsiClosePrevMktres10]
layout = go.Layout(
    title = 'HGSI.O stock price vs. 10-day Prev returns',
    yaxis = dict(
        title='return metrics (%)',
        tickformat = ',.0%',
        showgrid=False,
        zeroline=False
    ),
    yaxis2 = dict(
        title = 'Stock Price',
        overlaying = 'y',
        showgrid=False,
        zeroline=False,
        side = 'right'
    )
)
fig = go.Figure(data=data, layout=layout)
iplot(fig)
apple = market_train_df[market_train_df['assetCode'] == 'AAPL.O']
apple.iplot(kind='line',x='date',y='open')
appleNews = news_train_df[news_train_df['assetName'] == 'Apple Inc']
list(appleNews[(appleNews['headline'].str.contains('stock split')) & (appleNews['relevance'] >= 0.6)].head()['headline'])
apple[(apple['time'] > '2014-06-01') & (apple['time'] < '2014-06-16')]
apple['adjOpen'] = np.where(apple['time'] < '2014-06-09',apple['open']/7.0,apple['open'])
apple['MA10'] = apple['adjOpen'].rolling(window=10).mean()
apple['MA50'] = apple['adjOpen'].rolling(window=50).mean()
apple['MA200'] = apple['adjOpen'].rolling(window=200).mean()
apple.iplot(kind='line',x='date',y=['adjOpen','MA10','MA50','MA200'])