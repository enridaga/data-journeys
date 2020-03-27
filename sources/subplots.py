
import pandas as pd
reviews = pd.read_csv("../input/wine-reviews/winemag-data_first150k.csv", index_col=0)
reviews.head(3)
import matplotlib.pyplot as plt
fig, axarr = plt.subplots(2, 1, figsize=(12, 8))
axarr
fig, axarr = plt.subplots(2, 1, figsize=(12, 8))

reviews['points'].value_counts().sort_index().plot.bar(
    ax=axarr[0]
)

reviews['province'].value_counts().head(20).plot.bar(
    ax=axarr[1]
)
fig, axarr = plt.subplots(2, 2, figsize=(12, 8))
axarr
fig, axarr = plt.subplots(2, 2, figsize=(12, 8))

reviews['points'].value_counts().sort_index().plot.bar(
    ax=axarr[0][0]
)

reviews['province'].value_counts().head(20).plot.bar(
    ax=axarr[1][1]
)
fig, axarr = plt.subplots(2, 2, figsize=(12, 8))

reviews['points'].value_counts().sort_index().plot.bar(
    ax=axarr[0][0], fontsize=12, color='mediumvioletred'
)
axarr[0][0].set_title("Wine Scores", fontsize=18)

reviews['variety'].value_counts().head(20).plot.bar(
    ax=axarr[1][0], fontsize=12, color='mediumvioletred'
)
axarr[1][0].set_title("Wine Varieties", fontsize=18)

reviews['province'].value_counts().head(20).plot.bar(
    ax=axarr[1][1], fontsize=12, color='mediumvioletred'
)
axarr[1][1].set_title("Wine Origins", fontsize=18)

reviews['price'].value_counts().plot.hist(
    ax=axarr[0][1], fontsize=12, color='mediumvioletred'
)
axarr[0][1].set_title("Wine Prices", fontsize=18)

plt.subplots_adjust(hspace=.3)

import seaborn as sns
sns.despine()
from IPython.display import HTML
HTML("""
<ol>
<li>The plot consists of one overall figure composed of one or more axes.</li>
<li>The subplots function takes the number of rows as the first parameter, and the number of columns as the second.</li>
<li>The subplots function returns a figure and an array of axes.</li>
</ol>
""")
import pandas as pd
import matplotlib.pyplot as plt
pokemon = pd.read_csv("../input/pokemon/Pokemon.csv")
pokemon.head(3)
plt.subplots(2, 1, figsize=(8, 8))
fig, axarr = plt.subplots(2, 1, figsize=(8, 8))
pokemon['Attack'].plot.hist(ax=axarr[0], title='Pokemon Attack Ratings')
pokemon['Defense'].plot.hist(ax=axarr[1], title='Pokemon Defense Ratings')