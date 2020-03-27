
import pandas as pd
reviews = pd.read_csv("../input/wine-reviews/winemag-data_first150k.csv", index_col=0)
import seaborn as sns
sns.countplot(reviews['points'])
sns.kdeplot(reviews.query('price < 200').price)
reviews[reviews['price'] < 200]['price'].value_counts().sort_index().plot.line()
sns.kdeplot(reviews[reviews['price'] < 200].loc[:, ['price', 'points']].dropna().sample(5000))
sns.distplot(reviews['points'], bins=10, kde=False)
sns.jointplot(x='price', y='points', data=reviews[reviews['price'] < 100])
sns.jointplot(x='price', y='points', data=reviews[reviews['price'] < 100], kind='hex', 
              gridsize=20)
df = reviews[reviews.variety.isin(reviews.variety.value_counts().head(5).index)]

sns.boxplot(
    x='variety',
    y='points',
    data=df
)
sns.violinplot(
    x='variety',
    y='points',
    data=reviews[reviews.variety.isin(reviews.variety.value_counts()[:5].index)]
)
reviews.head()
from IPython.display import HTML
HTML("""
<ol>
<li>A seaborn countplot is like a pandas bar plot.</li>
<li>A seaborn jointplot is like a pandas hex plot.</li>
<li>KDEPlots work by aggregating data into a smooth curve. This is great for interval data but doesn't always work quite as well for ordinal categorical data.</li>
<li>The top of the box is the 75th percentile. The bottom of the box is the 25th percentile. The median, the 50th percentile, is the line in the center of the box. So 50% of the data in the distribution is located within the box!</li>
</ol>
""")
pokemon = pd.read_csv("../input/pokemon/Pokemon.csv", index_col=0)
pokemon.head()
sns.countplot(pokemon['Generation'])
sns.distplot(pokemon['HP'])
sns.jointplot(x='Attack', y='Defense', data=pokemon)
sns.jointplot(x='Attack', y='Defense', data=pokemon, kind='hex')
sns.kdeplot(pokemon['HP'], pokemon['Attack'])
sns.boxplot(x='Legendary', y='Attack', data=pokemon)
sns.violinplot(x='Legendary', y='Attack', data=pokemon)