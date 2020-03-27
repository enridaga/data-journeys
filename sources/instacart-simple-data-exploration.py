
#import the needed librairies



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

#%matplotlib inline

import matplotlib.pyplot as plt  # Matlab-style plotting

import seaborn as sns

color = sns.color_palette()

import warnings

warnings.filterwarnings('ignore') #Supress unnecessary warnings for readability and cleaner presentation





pd.set_option('display.float_format', lambda x: '%.3f' % x) #Limiting floats output to 3 decimal points





from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8")) #check the files available in the directory
#Now let's get and put the data in  pandas dataframe



order_products_train = pd.read_csv('../input/order_products__train.csv')

order_products_prior = pd.read_csv('../input/order_products__prior.csv')

orders = pd.read_csv('../input/orders.csv')

products = pd.read_csv('../input/products.csv')

aisles = pd.read_csv('../input/aisles.csv')

departments = pd.read_csv('../input/departments.csv')
print("The order_products_train size is : ", order_products_train.shape)

print("The order_products_prior size is : ", order_products_prior.shape)
#display first five rows.



order_products_train.head(5)
#display first five rows. 



order_products_prior.head(5)
order_products_all = pd.concat([order_products_train, order_products_prior], axis=0)



print("The order_products_all size is : ", order_products_all.shape)
#display first five rows.



order_products_all.head(5)
total = order_products_all.isnull().sum().sort_values(ascending=False)

percent = (order_products_all.isnull().sum()/order_products_all.isnull().count()).sort_values(ascending=False)

missing_data = pd.concat([total, percent], axis=1, keys=['Total Missing', 'Percent'])

missing_data

# Check the number of unique orders and unique products

orders_Unique = len(set(order_products_all.order_id))

products_Unique = len(set(order_products_all.product_id))

print("There are %s orders for %s products" %(orders_Unique, products_Unique))
grouped = order_products_all.groupby("order_id")["add_to_cart_order"].aggregate("max").reset_index()

grouped = grouped.add_to_cart_order.value_counts()



sns.set_style('whitegrid')

f, ax = plt.subplots(figsize=(15, 12))

plt.xticks(rotation='vertical')

sns.barplot(grouped.index, grouped.values)



plt.ylabel('Number of Orders', fontsize=13)

plt.xlabel('Number of products added in order', fontsize=13)

plt.show()
grouped = order_products_all.groupby("product_id")["reordered"].aggregate({'Total_reorders': 'count'}).reset_index()

grouped = pd.merge(grouped, products[['product_id', 'product_name']], how='left', on=['product_id'])

grouped = grouped.sort_values(by='Total_reorders', ascending=False)[:10]

grouped
grouped  = grouped.groupby(['product_name']).sum()['Total_reorders'].sort_values(ascending=False)



sns.set_style('darkgrid')

f, ax = plt.subplots(figsize=(12, 10))

plt.xticks(rotation='vertical')

sns.barplot(grouped.index, grouped.values)

plt.ylabel('Number of Reorders', fontsize=13)

plt.xlabel('Most ordered Products', fontsize=13)

plt.show()
grouped = order_products_all.groupby("reordered")["product_id"].aggregate({'Total_products': 'count'}).reset_index()

grouped['Ratios'] = grouped["Total_products"].apply(lambda x: x /grouped['Total_products'].sum())

grouped
grouped  = grouped.groupby(['reordered']).sum()['Total_products'].sort_values(ascending=False)



sns.set_style('whitegrid')

f, ax = plt.subplots(figsize=(5, 8))

sns.barplot(grouped.index, grouped.values, palette='RdBu_r')

plt.ylabel('Number of Products', fontsize=13)

plt.xlabel('Reordered or Not Reordered', fontsize=13)

plt.ticklabel_format(style='plain', axis='y')

plt.show()
grouped = order_products_all.groupby("product_id")["reordered"].aggregate({'reorder_sum': sum,'reorder_total': 'count'}).reset_index()

grouped['reorder_probability'] = grouped['reorder_sum'] / grouped['reorder_total']

grouped = pd.merge(grouped, products[['product_id', 'product_name']], how='left', on=['product_id'])

grouped = grouped[grouped.reorder_total > 75].sort_values(['reorder_probability'], ascending=False)[:10]

grouped
grouped  = grouped.groupby(['product_name']).sum()['reorder_probability'].sort_values(ascending=False)



sns.set_style('darkgrid')

f, ax = plt.subplots(figsize=(12, 10))

plt.xticks(rotation='vertical')

sns.barplot(grouped.index, grouped.values)

plt.ylim([0.85,0.95])

plt.ylabel('Reorder probability', fontsize=13)

plt.xlabel('Most reordered products', fontsize=12)

plt.show()

print("The orders data size is : ", orders.shape)

print("Columns in orders data  are : ",  orders.columns.values)
#display first five rows of our dataset.



orders.head(5)
orders_na = (orders.isnull().sum() / len(orders)) * 100

orders_na = orders_na.drop(orders_na[orders_na == 0].index).sort_values(ascending=False)

orders_na
grouped = orders.groupby("order_id")["order_hour_of_day"].aggregate("sum").reset_index()

grouped = grouped.order_hour_of_day.value_counts()



sns.set_style('darkgrid')

f, ax = plt.subplots(figsize=(15, 10))

sns.barplot(grouped.index, grouped.values)

plt.ylabel('Number of orders', fontsize=13)

plt.xlabel('Hours of order in a day', fontsize=13)

plt.show()
grouped = orders.groupby("order_id")["order_dow"].aggregate("sum").reset_index()

grouped = grouped.order_dow.value_counts()



f, ax = plt.subplots(figsize=(10, 10))

sns.barplot(grouped.index, grouped.values)

plt.ylabel('Number of orders', fontsize=13)

plt.xlabel('Days of order in a week', fontsize=13)

plt.show()
grouped = orders.groupby("order_id")["days_since_prior_order"].aggregate("sum").reset_index()

grouped = grouped.days_since_prior_order.value_counts()



from matplotlib.ticker import FormatStrFormatter

f, ax = plt.subplots(figsize=(15, 10))

sns.barplot(grouped.index, grouped.values)

ax.xaxis.set_major_formatter(FormatStrFormatter('%.0f'))

plt.ylabel('Number of orders', fontsize=13)

plt.xlabel('Period of reorder', fontsize=13)

plt.show()
grouped = orders.groupby("eval_set")["order_id"].aggregate({'Total_orders': 'count'}).reset_index()

grouped['Ratio'] = grouped["Total_orders"].apply(lambda x: x /grouped['Total_orders'].sum())

grouped
grouped  = grouped.groupby(['eval_set']).sum()['Total_orders'].sort_values(ascending=False)



sns.set_style('whitegrid')

f, ax = plt.subplots(figsize=(8, 8))

sns.barplot(grouped.index, grouped.values, palette='coolwarm')

plt.ylabel('Number of Orders', fontsize=13)

plt.xlabel('datasets', fontsize=13)

plt.show()
print("Number of unique customers in the whole dataset : ",len(set(orders.user_id)))
grouped = orders.groupby("eval_set")["user_id"].apply(lambda x: len(x.unique()))



plt.figure(figsize=(7,8))

sns.barplot(grouped.index, grouped.values, palette='coolwarm')

plt.ylabel('Number of users', fontsize=13)

plt.xlabel('Eval set', fontsize=13)

plt.title("Number of unique customers in each dataset")

plt.show()
grouped = orders.groupby('user_id')['order_id'].apply(lambda x: len(x.unique())).reset_index()

grouped = grouped.groupby('order_id').aggregate("count")



sns.set_style("whitegrid")

f, ax = plt.subplots(figsize=(15, 12))

sns.barplot(grouped.index, grouped.user_id)

plt.ylabel('Numbers of Customers')

plt.xlabel('Number of Orders per customer')

plt.xticks(rotation='vertical')

plt.show()
#display first five rows of our dataset.

products.head(5)
#display first five rows of our dataset.

departments.head(5)
#display first five rows of our dataset.

aisles.head(5)
items  = pd.merge(left =pd.merge(left=products, right=departments, how='left'), right=aisles, how='left')

items.head()
grouped = items.groupby("department")["product_id"].aggregate({'Total_products': 'count'}).reset_index()

grouped['Ratio'] = grouped["Total_products"].apply(lambda x: x /grouped['Total_products'].sum())

grouped.sort_values(by='Total_products', ascending=False, inplace=True)

grouped
grouped  = grouped.groupby(['department']).sum()['Total_products'].sort_values(ascending=False)



sns.set_style("darkgrid")

f, ax = plt.subplots(figsize=(12, 15))

plt.xticks(rotation='vertical')

sns.barplot(grouped.index, grouped.values)

plt.ylabel('Number of products', fontsize=13)

plt.xlabel('Departments', fontsize=13)

plt.show()
grouped = items.groupby(["department", "aisle"])["product_id"].aggregate({'Total_products': 'count'}).reset_index()

grouped.sort_values(by='Total_products', ascending=False, inplace=True)

fig, axes = plt.subplots(7,3, figsize=(20,45), gridspec_kw =  dict(hspace=1.4))

for (aisle, group), ax in zip(grouped.groupby(["department"]), axes.flatten()):

    g = sns.barplot(group.aisle, group.Total_products , ax=ax)

    ax.set(xlabel = "Aisles", ylabel=" Number of products")

    g.set_xticklabels(labels = group.aisle,rotation=90, fontsize=12)

    ax.set_title(aisle, fontsize=15)
grouped = items.groupby("aisle")["product_id"].aggregate({'Total_products': 'count'}).reset_index()

grouped['Ratio'] = grouped["Total_products"].apply(lambda x: x /grouped['Total_products'].sum())

grouped = grouped.sort_values(by='Total_products', ascending=False)[:20]

grouped
grouped  = grouped.groupby(['aisle']).sum()['Total_products'].sort_values(ascending=False)



f, ax = plt.subplots(figsize=(12, 15))

plt.xticks(rotation='vertical')

sns.barplot(grouped.index, grouped.values)

plt.ylabel('Number of products', fontsize=13)

plt.xlabel('Aisles', fontsize=13)

plt.show()


users_flow = orders[['user_id', 'order_id']].merge(order_products_train[['order_id', 'product_id']],

                                          how='inner', left_on='order_id', right_on='order_id')



users_flow = users_flow.merge(items, how='inner', left_on='product_id',

                                         right_on='product_id')
grouped = users_flow.groupby("department")["order_id"].aggregate({'Total_orders': 'count'}).reset_index()

grouped['Ratio'] = grouped["Total_orders"].apply(lambda x: x /grouped['Total_orders'].sum())

grouped.sort_values(by='Total_orders', ascending=False, inplace=True)

grouped
grouped  = grouped.groupby(['department']).sum()['Total_orders'].sort_values(ascending=False)



f, ax = plt.subplots(figsize=(12, 15))

plt.xticks(rotation='vertical')

sns.barplot(grouped.index, grouped.values)

plt.ylabel('Number of Orders', fontsize=13)

plt.xlabel('Departments', fontsize=13)

plt.show()
grouped = users_flow.groupby(["department", "aisle"])["order_id"].aggregate({'Total_orders': 'count'}).reset_index()

grouped.sort_values(by='Total_orders', ascending=False, inplace=True)

fig, axes = plt.subplots(7,3, figsize=(20,45), gridspec_kw =  dict(hspace=1.4))

for (aisle, group), ax in zip(grouped.groupby(["department"]), axes.flatten()):

    g = sns.barplot(group.aisle, group.Total_orders , ax=ax)

    ax.set(xlabel = "Aisles", ylabel=" Number of Orders")

    g.set_xticklabels(labels = group.aisle,rotation=90, fontsize=12)

    ax.set_title(aisle, fontsize=15)
grouped = users_flow.groupby("aisle")["order_id"].aggregate({'Total_orders': 'count'}).reset_index()

grouped['Ratio'] = grouped["Total_orders"].apply(lambda x: x /grouped['Total_orders'].sum())

grouped.sort_values(by='Total_orders', ascending=False, inplace=True )

grouped.head(10)
grouped  = grouped.groupby(['aisle']).sum()['Total_orders'].sort_values(ascending=False)[:15]



f, ax = plt.subplots(figsize=(12, 15))

plt.xticks(rotation='vertical')

sns.barplot(grouped.index, grouped.values)

plt.ylabel('Number of Orders', fontsize=13)

plt.xlabel('Aisles', fontsize=13)

plt.show()
