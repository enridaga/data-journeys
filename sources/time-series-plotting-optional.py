
import pandas as pd
pd.set_option('max_columns', None)
import numpy as np
stocks = pd.read_csv("../input/nyse/prices.csv", parse_dates=['date'])
stocks = stocks[stocks['symbol'] == "GOOG"].set_index('date')
stocks.head()
shelter_outcomes = pd.read_csv(
    "../input/austin-animal-center-shelter-outcomes-and/aac_shelter_outcomes.csv", 
    parse_dates=['date_of_birth', 'datetime']
)
shelter_outcomes = shelter_outcomes[
    ['outcome_type', 'age_upon_outcome', 'datetime', 'animal_type', 'breed', 
     'color', 'sex_upon_outcome', 'date_of_birth']
]
shelter_outcomes.head()
shelter_outcomes['date_of_birth'].value_counts().sort_values().plot.line()
shelter_outcomes['date_of_birth'].value_counts().resample('Y').sum().plot.line()
stocks['volume'].resample('Y').mean().plot.bar()
from pandas.plotting import lag_plot

lag_plot(stocks['volume'].tail(250))
from pandas.plotting import autocorrelation_plot

autocorrelation_plot(stocks['volume'])
import pandas as pd

crypto = pd.read_csv("../input/all-crypto-currencies/crypto-markets.csv")
crypto = crypto[crypto['name'] == 'Bitcoin']
crypto['date'] = pd.to_datetime(crypto['date'])
crypto.head()
from IPython.display import HTML

HTML("""
<ol>
<li>Time-series data is really a special case of interval data.</li>
<br/>
<li>Resampling is often useful in data visualization because it can help clean up and denoise our plots by aggregating on a different level.</li>
<br/>
<li>Lag is the time-difference for each observation in the dataset. Autocorrelation is correlation applied to lag.</li>
</ol>
""")
shelter_outcomes['datetime'].value_counts().resample('Y').count().plot.line()
lag_plot(crypto['volume'].tail(250))
autocorrelation_plot(crypto['volume'])