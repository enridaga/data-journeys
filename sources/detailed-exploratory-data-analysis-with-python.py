
import pandas as pd

import seaborn as sns

import matplotlib.pyplot as plt



# Comment this if the data visualisations doesn't work on your side

### matplotlib inline



plt.style.use('bmh')
df = pd.read_csv('../input/train.csv')

df.head()
df.info()
# df.count() does not include NaN values

df2 = df[[column for column in df if df[column].count() / len(df) >= 0.3]]

del df2['Id']

print("List of dropped columns:", end=" ")

for c in df.columns:

    if c not in df2.columns:

        print(c, end=", ")

print('\n')

df = df2
print(df['SalePrice'].describe())

plt.figure(figsize=(9, 8))

sns.distplot(df['SalePrice'], color='g', bins=100, hist_kws={'alpha': 0.4});
list(set(df.dtypes.tolist()))
df_num = df.select_dtypes(include = ['float64', 'int64'])

df_num.head()
df_num.hist(figsize=(16, 20), bins=50, xlabelsize=8, ylabelsize=8); # ; avoid having the matplotlib verbose informations
df_num_corr = df_num.corr()['SalePrice'][:-1] # -1 because the latest row is SalePrice

golden_features_list = df_num_corr[abs(df_num_corr) > 0.5].sort_values(ascending=False)

print("There is {} strongly correlated values with SalePrice:\n{}".format(len(golden_features_list), golden_features_list))
for i in range(0, len(df_num.columns), 5):

    sns.pairplot(data=df_num,

                x_vars=df_num.columns[i:i+5],

                y_vars=['SalePrice'])
import operator



individual_features_df = []

for i in range(0, len(df_num.columns) - 1): # -1 because the last column is SalePrice

    tmpDf = df_num[[df_num.columns[i], 'SalePrice']]

    tmpDf = tmpDf[tmpDf[df_num.columns[i]] != 0]

    individual_features_df.append(tmpDf)



all_correlations = {feature.columns[0]: feature.corr()['SalePrice'][0] for feature in individual_features_df}

all_correlations = sorted(all_correlations.items(), key=operator.itemgetter(1))

for (key, value) in all_correlations:

    print("{:>15}: {:>15}".format(key, value))
golden_features_list = [key for key, value in all_correlations if abs(value) >= 0.5]

print("There is {} strongly correlated values with SalePrice:\n{}".format(len(golden_features_list), golden_features_list))
corr = df_num.drop('SalePrice', axis=1).corr() # We already examined SalePrice correlations

plt.figure(figsize=(12, 10))



sns.heatmap(corr[(corr >= 0.5) | (corr <= -0.4)], 

            cmap='viridis', vmax=1.0, vmin=-1.0, linewidths=0.1,

            annot=True, annot_kws={"size": 8}, square=True);
quantitative_features_list = ['LotFrontage', 'LotArea', 'MasVnrArea', 'BsmtFinSF1', 'BsmtFinSF2', 'TotalBsmtSF', '1stFlrSF',

    '2ndFlrSF', 'LowQualFinSF', 'GrLivArea', 'BsmtFullBath', 'BsmtHalfBath', 'FullBath', 'HalfBath',

    'BedroomAbvGr', 'KitchenAbvGr', 'TotRmsAbvGrd', 'Fireplaces', 'GarageCars', 'GarageArea', 'WoodDeckSF', 'OpenPorchSF', 

    'EnclosedPorch', '3SsnPorch', 'ScreenPorch', 'PoolArea', 'MiscVal', 'SalePrice']

df_quantitative_values = df[quantitative_features_list]

df_quantitative_values.head()
features_to_analyse = [x for x in quantitative_features_list if x in golden_features_list]

features_to_analyse.append('SalePrice')

features_to_analyse
fig, ax = plt.subplots(round(len(features_to_analyse) / 3), 3, figsize = (18, 12))



for i, ax in enumerate(fig.axes):

    if i < len(features_to_analyse) - 1:

        sns.regplot(x=features_to_analyse[i],y='SalePrice', data=df[features_to_analyse], ax=ax)
# quantitative_features_list[:-1] as the last column is SalePrice and we want to keep it

categorical_features = [a for a in quantitative_features_list[:-1] + df.columns.tolist() if (a not in quantitative_features_list[:-1]) or (a not in df.columns.tolist())]

df_categ = df[categorical_features]

df_categ.head()
df_not_num = df_categ.select_dtypes(include = ['O'])

print('There is {} non numerical features including:\n{}'.format(len(df_not_num.columns), df_not_num.columns.tolist()))
plt.figure(figsize = (10, 6))

ax = sns.boxplot(x='BsmtExposure', y='SalePrice', data=df_categ)

plt.setp(ax.artists, alpha=.5, linewidth=2, edgecolor="k")

plt.xticks(rotation=45)
plt.figure(figsize = (12, 6))

ax = sns.boxplot(x='SaleCondition', y='SalePrice', data=df_categ)

plt.setp(ax.artists, alpha=.5, linewidth=2, edgecolor="k")

plt.xticks(rotation=45)
fig, axes = plt.subplots(round(len(df_not_num.columns) / 3), 3, figsize=(12, 30))



for i, ax in enumerate(fig.axes):

    if i < len(df_not_num.columns):

        ax.set_xticklabels(ax.xaxis.get_majorticklabels(), rotation=45)

        sns.countplot(x=df_not_num.columns[i], alpha=0.7, data=df_not_num, ax=ax)



fig.tight_layout()