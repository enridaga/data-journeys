
import pandas as pd
reviews = pd.read_csv("../input/wine-reviews/winemag-data_first150k.csv", index_col=0)
reviews.head(3)
reviews['province'].value_counts().head(10).plot.bar()
(reviews['province'].value_counts().head(10) / len(reviews)).plot.bar()
reviews['points'].value_counts().sort_index().plot.bar()
reviews['points'].value_counts().sort_index().plot.line()
raw = """
<ol>
<li>This is a simple nominal categorical variable. Five bars will fit easily into a display, so a bar chart will do!</li>
<br/>
<li>This example is similar: nominal categorical variables. There are probably more than five American car manufacturers, so the chart will be a little more crowded, but a bar chart will still do it.</li>
<br/>
<li>This is an ordinal categorical variable. We have a lot of potential values between 0 and 100, so a bar chart won't have enough room. A line chart is better.</li>
<br/>
<li>
<p>Number 4 is a lot harder. City streets are obviously ordinary categorical variables, so we *ought* to use a bar chart; but there are a lot of streets out there! We couldn't possibly fit all of them into a display.</p>
<p>Sometimes, your data will have too many points to do something "neatly", and that's OK. If you organize the data by value count and plot a line chart over that, you'll learn valuable information about *percentiles*: that a street in the 90th percentile has 20 restaurants, for example, or one in the 50th just 6. This is basically a form of aggregation: we've turned streets into percentiles!</p> 
<p>The lesson: your *interpretation* of the data is more important than the tool that you use.</p></li>
</ol>
"""

from IPython.display import HTML
HTML(raw)
reviews['points'].value_counts().sort_index().plot.area()
reviews[reviews['price'] < 200]['price'].plot.hist()
reviews['price'].plot.hist()
reviews[reviews['price'] > 1500]
reviews['points'].plot.hist()
raw = """
<ol>
<li>Example number 1 is a nominal categorical example, and hence, a pretty straightfoward bar graph target.</li>
<br/>
<li>Example 2 is a large nominal categorical variable. A basketball game team can score between 50 and 150 points, too much for a bar chart; a line chart is a good way to go. A histogram could also work.</li>
<br/>
<li>Example 3 is an interval variable: a single building can have anywhere between 1 and 1000 or more apartment units. A line chart could work, but a histogram would probably work better! Note that this distribution is going to have a lot of skew (there is only a handful of very, very large apartment buildings).</li>
</ol>
"""

from IPython.display import HTML
HTML(raw)
pd.set_option('max_columns', None)
pokemon = pd.read_csv("../input/pokemon/pokemon.csv")
pokemon.head(3)
pokemon['type1'].value_counts().plot.bar()
pokemon['hp'].value_counts().sort_index().plot.line()
pokemon['weight_kg'].plot.hist()