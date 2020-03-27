
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
plt.style.use('ggplot')
df = pd.read_csv('../input/goodreadsbooks/books.csv', error_bad_lines = False)
print("Number of total books: {}".format(df['title'].count()))
print("Number of unique books: {}".format(df['title'].value_counts().count()))
most_rated = df.groupby('title')['text_reviews_count'].sum().sort_values(ascending=False).head(5)
print(most_rated)
plt.figure(figsize=(15,10))
sns.barplot(most_rated, most_rated.index, palette='rocket')