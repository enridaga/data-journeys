


import pandas as pd

reviews = pd.read_csv("../input/wine-reviews/winemag-data-130k-v2.csv", index_col=0)

pd.set_option('max_rows', 5)
reviews
reviews.country
reviews['country']
reviews['country'][0]
reviews.iloc[0]
reviews.iloc[:, 0]
reviews.iloc[:3, 0]
reviews.iloc[1:3, 0]
reviews.iloc[[0, 1, 2], 0]
reviews.iloc[-5:]
reviews.loc[0, 'country']
reviews.loc[:, ['taster_name', 'taster_twitter_handle', 'points']]
reviews.set_index("title")
reviews.country == 'Italy'
reviews.loc[reviews.country == 'Italy']
reviews.loc[(reviews.country == 'Italy') & (reviews.points >= 90)]
reviews.loc[(reviews.country == 'Italy') | (reviews.points >= 90)]
reviews.loc[reviews.country.isin(['Italy', 'France'])]
reviews.loc[reviews.price.notnull()]
reviews['critic'] = 'everyone'

reviews['critic']
reviews['index_backwards'] = range(len(reviews), 0, -1)

reviews['index_backwards']