


import pandas as pd

pd.set_option('max_rows', 5)

import numpy as np

reviews = pd.read_csv("../input/wine-reviews/winemag-data-130k-v2.csv", index_col=0)
reviews
reviews.points.describe()
reviews.taster_name.describe()
reviews.points.mean()
reviews.taster_name.unique()
reviews.taster_name.value_counts()
review_points_mean = reviews.points.mean()

reviews.points.map(lambda p: p - review_points_mean)
def remean_points(row):

    row.points = row.points - review_points_mean

    return row



reviews.apply(remean_points, axis='columns')
reviews.head(1)
review_points_mean = reviews.points.mean()

reviews.points - review_points_mean
reviews.country + " - " + reviews.region_1