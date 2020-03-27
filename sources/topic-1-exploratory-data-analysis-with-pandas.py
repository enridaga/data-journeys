
import numpy as np
import pandas as pd
pd.set_option("display.precision", 2)
df = pd.read_csv('../input/telecom_churn.csv')
df.head()
print(df.shape)
print(df.columns)
print(df.info())
df['Churn'] = df['Churn'].astype('int64')
df.describe()
df.describe(include=['object', 'bool'])
df['Churn'].value_counts()
df['Churn'].value_counts(normalize=True)
df.sort_values(by='Total day charge', ascending=False).head()
df.sort_values(by=['Churn', 'Total day charge'],
        ascending=[True, False]).head()
df['Churn'].mean()
df[df['Churn'] == 1].mean()
df[df['Churn'] == 1]['Total day minutes'].mean()
df[(df['Churn'] == 0) & (df['International plan'] == 'No')]['Total intl minutes'].max()
df.loc[0:5, 'State':'Area code']
df.iloc[0:5, 0:3]
df[-1:]
df.apply(np.max) 
df[df['State'].apply(lambda state: state[0] == 'W')].head()
d = {'No' : False, 'Yes' : True}
df['International plan'] = df['International plan'].map(d)
df.head()
df = df.replace({'Voice mail plan': d})
df.head()
columns_to_show = ['Total day minutes', 'Total eve minutes', 
                   'Total night minutes']

df.groupby(['Churn'])[columns_to_show].describe(percentiles=[])
columns_to_show = ['Total day minutes', 'Total eve minutes', 
                   'Total night minutes']

df.groupby(['Churn'])[columns_to_show].agg([np.mean, np.std, np.min, 
                                            np.max])
pd.crosstab(df['Churn'], df['International plan'])
pd.crosstab(df['Churn'], df['Voice mail plan'], normalize=True)
df.pivot_table(['Total day calls', 'Total eve calls', 'Total night calls'],
               ['Area code'], aggfunc='mean')
total_calls = df['Total day calls'] + df['Total eve calls'] + \
              df['Total night calls'] + df['Total intl calls']
df.insert(loc=len(df.columns), column='Total calls', value=total_calls) 
# loc parameter is the number of columns after which to insert the Series object
# we set it to len(df.columns) to paste it at the very end of the dataframe
df.head()
df['Total charge'] = df['Total day charge'] + df['Total eve charge'] + \
                     df['Total night charge'] + df['Total intl charge']
df.head()
# get rid of just created columns
df.drop(['Total charge', 'Total calls'], axis=1, inplace=True) 
# and here’s how you can delete rows
df.drop([1, 2]).head() 
pd.crosstab(df['Churn'], df['International plan'], margins=True)
# some imports to set up plotting 
import matplotlib.pyplot as plt
# pip install seaborn 
import seaborn as sns
# Graphics in retina format are more sharp and legible
%config InlineBackend.figure_format = 'retina'
sns.countplot(x='International plan', hue='Churn', data=df);
pd.crosstab(df['Churn'], df['Customer service calls'], margins=True)
sns.countplot(x='Customer service calls', hue='Churn', data=df);
df['Many_service_calls'] = (df['Customer service calls'] > 3).astype('int')

pd.crosstab(df['Many_service_calls'], df['Churn'], margins=True)
sns.countplot(x='Many_service_calls', hue='Churn', data=df);
pd.crosstab(df['Many_service_calls'] & df['International plan'] , df['Churn'])