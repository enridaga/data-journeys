
import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns

color = sns.color_palette()



### matplotlib inline



pd.options.mode.chained_assignment = None  # default='warn'
from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))
order_products_train_df = pd.read_csv("../input/order_products__train.csv")

order_products_prior_df = pd.read_csv("../input/order_products__prior.csv")

orders_df = pd.read_csv("../input/orders.csv")

products_df = pd.read_csv("../input/products.csv")

aisles_df = pd.read_csv("../input/aisles.csv")

departments_df = pd.read_csv("../input/departments.csv")
orders_df.head()
order_products_prior_df.head()
order_products_train_df.head()
cnt_srs = orders_df.eval_set.value_counts()



plt.figure(figsize=(12,8))

sns.barplot(cnt_srs.index, cnt_srs.values, alpha=0.8, color=color[1])

plt.ylabel('Number of Occurrences', fontsize=12)

plt.xlabel('Eval set type', fontsize=12)

plt.title('Count of rows in each dataset', fontsize=15)

plt.xticks(rotation='vertical')

plt.show()
def get_unique_count(x):

    return len(np.unique(x))



cnt_srs = orders_df.groupby("eval_set")["user_id"].aggregate(get_unique_count)

cnt_srs
cnt_srs = orders_df.groupby("user_id")["order_number"].aggregate(np.max).reset_index()

cnt_srs = cnt_srs.order_number.value_counts()



plt.figure(figsize=(12,8))

sns.barplot(cnt_srs.index, cnt_srs.values, alpha=0.8, color=color[2])

plt.ylabel('Number of Occurrences', fontsize=12)

plt.xlabel('Maximum order number', fontsize=12)

plt.xticks(rotation='vertical')

plt.show()
plt.figure(figsize=(12,8))

sns.countplot(x="order_dow", data=orders_df, color=color[0])

plt.ylabel('Count', fontsize=12)

plt.xlabel('Day of week', fontsize=12)

plt.xticks(rotation='vertical')

plt.title("Frequency of order by week day", fontsize=15)

plt.show()
plt.figure(figsize=(12,8))

sns.countplot(x="order_hour_of_day", data=orders_df, color=color[1])

plt.ylabel('Count', fontsize=12)

plt.xlabel('Hour of day', fontsize=12)

plt.xticks(rotation='vertical')

plt.title("Frequency of order by hour of day", fontsize=15)

plt.show()
grouped_df = orders_df.groupby(["order_dow", "order_hour_of_day"])["order_number"].aggregate("count").reset_index()

grouped_df = grouped_df.pivot('order_dow', 'order_hour_of_day', 'order_number')



plt.figure(figsize=(12,6))

sns.heatmap(grouped_df)

plt.title("Frequency of Day of week Vs Hour of day")

plt.show()
plt.figure(figsize=(12,8))

sns.countplot(x="days_since_prior_order", data=orders_df, color=color[3])

plt.ylabel('Count', fontsize=12)

plt.xlabel('Days since prior order', fontsize=12)

plt.xticks(rotation='vertical')

plt.title("Frequency distribution by days since prior order", fontsize=15)

plt.show()
# percentage of re-orders in prior set #

order_products_prior_df.reordered.sum() / order_products_prior_df.shape[0]
# percentage of re-orders in train set #

order_products_train_df.reordered.sum() / order_products_train_df.shape[0]
grouped_df = order_products_prior_df.groupby("order_id")["reordered"].aggregate("sum").reset_index()

grouped_df["reordered"].ix[grouped_df["reordered"]>1] = 1

grouped_df.reordered.value_counts() / grouped_df.shape[0]
grouped_df = order_products_train_df.groupby("order_id")["reordered"].aggregate("sum").reset_index()

grouped_df["reordered"].ix[grouped_df["reordered"]>1] = 1

grouped_df.reordered.value_counts() / grouped_df.shape[0]
grouped_df = order_products_train_df.groupby("order_id")["add_to_cart_order"].aggregate("max").reset_index()

cnt_srs = grouped_df.add_to_cart_order.value_counts()



plt.figure(figsize=(12,8))

sns.barplot(cnt_srs.index, cnt_srs.values, alpha=0.8)

plt.ylabel('Number of Occurrences', fontsize=12)

plt.xlabel('Number of products in the given order', fontsize=12)

plt.xticks(rotation='vertical')

plt.show()
products_df.head()
aisles_df.head()
departments_df.head()
order_products_prior_df = pd.merge(order_products_prior_df, products_df, on='product_id', how='left')

order_products_prior_df = pd.merge(order_products_prior_df, aisles_df, on='aisle_id', how='left')

order_products_prior_df = pd.merge(order_products_prior_df, departments_df, on='department_id', how='left')

order_products_prior_df.head()
cnt_srs = order_products_prior_df['product_name'].value_counts().reset_index().head(20)

cnt_srs.columns = ['product_name', 'frequency_count']

cnt_srs
cnt_srs = order_products_prior_df['aisle'].value_counts().head(20)

plt.figure(figsize=(12,8))

sns.barplot(cnt_srs.index, cnt_srs.values, alpha=0.8, color=color[5])

plt.ylabel('Number of Occurrences', fontsize=12)

plt.xlabel('Aisle', fontsize=12)

plt.xticks(rotation='vertical')

plt.show()
plt.figure(figsize=(10,10))

temp_series = order_products_prior_df['department'].value_counts()

labels = (np.array(temp_series.index))

sizes = (np.array((temp_series / temp_series.sum())*100))

plt.pie(sizes, labels=labels, 

        autopct='%1.1f%%', startangle=200)

plt.title("Departments distribution", fontsize=15)

plt.show()
grouped_df = order_products_prior_df.groupby(["department"])["reordered"].aggregate("mean").reset_index()



plt.figure(figsize=(12,8))

sns.pointplot(grouped_df['department'].values, grouped_df['reordered'].values, alpha=0.8, color=color[2])

plt.ylabel('Reorder ratio', fontsize=12)

plt.xlabel('Department', fontsize=12)

plt.title("Department wise reorder ratio", fontsize=15)

plt.xticks(rotation='vertical')

plt.show()
grouped_df = order_products_prior_df.groupby(["department_id", "aisle"])["reordered"].aggregate("mean").reset_index()



fig, ax = plt.subplots(figsize=(12,20))

ax.scatter(grouped_df.reordered.values, grouped_df.department_id.values)

for i, txt in enumerate(grouped_df.aisle.values):

    ax.annotate(txt, (grouped_df.reordered.values[i], grouped_df.department_id.values[i]), rotation=45, ha='center', va='center', color='green')

plt.xlabel('Reorder Ratio')

plt.ylabel('department_id')

plt.title("Reorder ratio of different aisles", fontsize=15)

plt.show()
order_products_prior_df["add_to_cart_order_mod"] = order_products_prior_df["add_to_cart_order"].copy()

order_products_prior_df["add_to_cart_order_mod"].ix[order_products_prior_df["add_to_cart_order_mod"]>70] = 70

grouped_df = order_products_prior_df.groupby(["add_to_cart_order_mod"])["reordered"].aggregate("mean").reset_index()



plt.figure(figsize=(12,8))

sns.pointplot(grouped_df['add_to_cart_order_mod'].values, grouped_df['reordered'].values, alpha=0.8, color=color[2])

plt.ylabel('Reorder ratio', fontsize=12)

plt.xlabel('Add to cart order', fontsize=12)

plt.title("Add to cart order - Reorder ratio", fontsize=15)

plt.xticks(rotation='vertical')

plt.show()
order_products_train_df = pd.merge(order_products_train_df, orders_df, on='order_id', how='left')

grouped_df = order_products_train_df.groupby(["order_dow"])["reordered"].aggregate("mean").reset_index()



plt.figure(figsize=(12,8))

sns.barplot(grouped_df['order_dow'].values, grouped_df['reordered'].values, alpha=0.8, color=color[3])

plt.ylabel('Reorder ratio', fontsize=12)

plt.xlabel('Day of week', fontsize=12)

plt.title("Reorder ratio across day of week", fontsize=15)

plt.xticks(rotation='vertical')

plt.ylim(0.5, 0.7)

plt.show()
grouped_df = order_products_train_df.groupby(["order_hour_of_day"])["reordered"].aggregate("mean").reset_index()



plt.figure(figsize=(12,8))

sns.barplot(grouped_df['order_hour_of_day'].values, grouped_df['reordered'].values, alpha=0.8, color=color[4])

plt.ylabel('Reorder ratio', fontsize=12)

plt.xlabel('Hour of day', fontsize=12)

plt.title("Reorder ratio across hour of day", fontsize=15)

plt.xticks(rotation='vertical')

plt.ylim(0.5, 0.7)

plt.show()


grouped_df = order_products_train_df.groupby(["order_dow", "order_hour_of_day"])["reordered"].aggregate("mean").reset_index()

grouped_df = grouped_df.pivot('order_dow', 'order_hour_of_day', 'reordered')



plt.figure(figsize=(12,6))

sns.heatmap(grouped_df)

plt.title("Reorder ratio of Day of week Vs Hour of day")

plt.show()