
import pandas as pd
pd.set_option('max_columns', None)
df = pd.read_csv("../input/fifa-18-demo-player-dataset/CompleteDataset.csv", index_col=0)

import re
import numpy as np

footballers = df.copy()
footballers['Unit'] = df['Value'].str[-1]
footballers['Value (M)'] = np.where(footballers['Unit'] == '0', 0, 
                                    footballers['Value'].str[1:-1].replace(r'[a-zA-Z]',''))
footballers['Value (M)'] = footballers['Value (M)'].astype(float)
footballers['Value (M)'] = np.where(footballers['Unit'] == 'M', 
                                    footballers['Value (M)'], 
                                    footballers['Value (M)']/1000)
footballers = footballers.assign(Value=footballers['Value (M)'],
                                 Position=footballers['Preferred Positions'].str.split().str[0])
footballers.head()
import seaborn as sns
df = footballers[footballers['Position'].isin(['ST', 'GK'])]
g = sns.FacetGrid(df, col="Position")
df = footballers[footballers['Position'].isin(['ST', 'GK'])]
g = sns.FacetGrid(df, col="Position")
g.map(sns.kdeplot, "Overall")
df = footballers

g = sns.FacetGrid(df, col="Position", col_wrap=6)
g.map(sns.kdeplot, "Overall")
df = footballers[footballers['Position'].isin(['ST', 'GK'])]
df = df[df['Club'].isin(['Real Madrid CF', 'FC Barcelona', 'Atlético Madrid'])]

g = sns.FacetGrid(df, row="Position", col="Club")
g.map(sns.violinplot, "Overall")
df = footballers[footballers['Position'].isin(['ST', 'GK'])]
df = df[df['Club'].isin(['Real Madrid CF', 'FC Barcelona', 'Atlético Madrid'])]

g = sns.FacetGrid(df, row="Position", col="Club", 
                  row_order=['GK', 'ST'],
                  col_order=['Atlético Madrid', 'FC Barcelona', 'Real Madrid CF'])
g.map(sns.violinplot, "Overall")
sns.pairplot(footballers[['Overall', 'Potential', 'Value']])
from IPython.display import HTML
HTML("""
<ol>
<li>You should try to keep your grid variables down to five or so. Otherwise the plots get too small.</li>
<li>It's (1) a multivariate technique which (2) is very easy to use.</li>
<li>Pair plots are most useful when just starting out with a dataset, because they help contextualize relationships within it.</li>
</ol>
""")
import pandas as pd
import seaborn as sns

pokemon = pd.read_csv("../input/pokemon/Pokemon.csv", index_col=0)
pokemon.head(3)
g = sns.FacetGrid(pokemon, row="Legendary")
g.map(sns.kdeplot, "Attack")
g = sns.FacetGrid(pokemon, col="Legendary", row="Generation")
g.map(sns.kdeplot, "Attack")
sns.pairplot(pokemon[['HP', 'Attack', 'Defense']])