
# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.
#Importing Librabries 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns 

# TIME SERIES
from statsmodels.tsa.arima_model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from pandas.plotting import autocorrelation_plot
from statsmodels.tsa.stattools import adfuller, acf, pacf,arma_order_select_ic
import statsmodels.formula.api as smf
import statsmodels.tsa.api as smt
import statsmodels.api as sm
import scipy.stats as scs

# warnings 
import warnings
warnings.filterwarnings("ignore")
Path = '../input/competitive-data-science-predict-future-sales/'
shops = pd.read_csv(Path +'shops.csv')
sales_train = pd.read_csv(Path+'sales_train.csv',parse_dates=True)
item_categories = pd.read_csv(Path+'item_categories.csv')
test = pd.read_csv(Path+'test.csv')
item = pd.read_csv(Path+'items.csv')
print(shops.shape,sales_train.shape,item_categories.shape , test.shape,item.shape)
by_cat = item.groupby(['item_category_id']).size().sort_values(ascending=False).head(20).plot.bar()
plt.xlabel('Items per category')
plt.ylabel('No. of Times')
total_sale = sales_train.groupby(['date_block_num'])['item_cnt_day'].sum()
#total_sale.head()
plt.figure(figsize=(8,8))
plt.title('Total sales of company')
plt.xlabel('Time (January 2013 is 0, February 2013 is 1,..., October 2015 is 33)')
plt.ylabel('sales')
plt.plot(total_sale)
#checking timeseries stationarity 
#1. plotting rolling statistics curve
#2. dickey fuller test 
plt.figure(figsize=(16,6))
plt.plot(total_sale, color='blue',label='Original')
plt.plot(total_sale.rolling(window= 12,center= False).mean(),label='Rolling Mean')
plt.plot(total_sale.rolling(window=12,center= False).std(),label='Rolling std')
plt.legend()
sales_train.head()
#formatting the date column correctly
#sales_train.date=sales_train.date.apply(lambda x:datetime.datetime.strptime(x, '%d.%m.%Y'))

sales_train['date'] = pd.to_datetime(sales_train['date'], format="%d.%m.%Y")
sales_train['DayOfWeekNum'] = sales_train['date'].dt.dayofweek
sales_train['DayOfWeek'] = sales_train['date'].dt.weekday_name
sales_train['MonthDayNum'] = sales_train['date'].dt.day
import statsmodels.api as sm
tse = sm.tsa.seasonal_decompose(total_sale.values,freq=12,model='multiplicative')
tse.plot()
tse = sm.tsa.seasonal_decompose(total_sale.values,freq=12,model='additive')
tse.plot()
# Stationarity tests
def test_stationarity(timeseries):
    
    #Perform Dickey-Fuller test:
    print('Results of Dickey-Fuller Test:')
    dftest = adfuller(timeseries, autolag='AIC')
    dfoutput = pd.Series(dftest[0:4], index=['Test Statistic','p-value','#Lags Used','Number of Observations Used'])
    for key,value in dftest[4].items():
        dfoutput['Critical Value (%s)'%key] = value
    print (dfoutput)

test_stationarity(total_sale)
# to remove trend
from pandas import Series as Series
# create a differenced series
def difference(dataset, interval=1):
    diff = list()
    for i in range(interval, len(dataset)):
        value = dataset[i] - dataset[i - interval]
        diff.append(value)
    return Series(diff)

# invert differenced forecast
def inverse_difference(last_ob, value):
    return value + last_ob
new_ts1=difference(total_sale,12) 
new_ts=difference(total_sale)

fig, axs = plt.subplots(3,figsize=(16,16))
axs[0].plot(total_sale)
axs[0].set_title('original')

axs[1].plot(new_ts)
axs[1].set_title('After De-trend')

axs[2].plot(new_ts1)# assuming the seasonality is 12 months long
axs[2].set_title('After De-serialization')

# now testing the stationarity again after de-seasonality
test_stationarity(new_ts1)
lag_acf = acf(total_sale, nlags=20)
lag_pacf = pacf(total_sale, nlags=20, method='ols')
#Plot ACF: 
plt.subplot(121) 
plt.plot(lag_acf)
plt.axhline(y=0,linestyle='--',color='gray')
plt.axhline(y=-1.96/np.sqrt(len(total_sale)),linestyle='--',color='gray')
plt.axhline(y=1.96/np.sqrt(len(total_sale)),linestyle='--',color='gray')
plt.title('Autocorrelation Function')
#Plot PACF:
plt.subplot(122)
plt.plot(lag_pacf)
plt.axhline(y=0,linestyle='--',color='gray')
plt.axhline(y=-1.96/np.sqrt(len(total_sale)),linestyle='--',color='gray')
plt.axhline(y=1.96/np.sqrt(len(total_sale)),linestyle='--',color='gray')
plt.title('Partial Autocorrelation Function')
plt.tight_layout()
from statsmodels.tsa.arima_model import ARIMA
model = ARIMA(total_sale, order=(2, 1, 0))  
results_AR = model.fit(disp=-1)  
plt.plot(total_sale)
plt.plot(results_AR.fittedvalues, color='red')
plt.title('RSS: %.4f'% sum((results_AR.fittedvalues-total_sale)**2))