
import numpy as np

from itertools import combinations, groupby

from collections import Counter



# Sample data

orders = np.array([[1,'apple'], [1,'egg'], [1,'milk'], [2,'egg'], [2,'milk']], dtype=object)



# Generator that yields item pairs, one at a time

def get_item_pairs(order_item):

    

    # For each order, generate a list of items in that order

    for order_id, order_object in groupby(orders, lambda x: x[0]):

        item_list = [item[1] for item in order_object]      

    

        # For each item list, generate item pairs, one at a time

        for item_pair in combinations(item_list, 2):

            yield item_pair                                      





# Counter iterates through the item pairs returned by our generator and keeps a tally of their occurrence

Counter(get_item_pairs(orders))

import pandas as pd

import numpy as np

import sys

from itertools import combinations, groupby

from collections import Counter

from IPython.display import display
# Function that returns the size of an object in MB

def size(obj):

    return "{0:.2f} MB".format(sys.getsizeof(obj) / (1000 * 1000))
orders = pd.read_csv('../input/order_products__prior.csv')

print('orders -- dimensions: {0};   size: {1}'.format(orders.shape, size(orders)))

display(orders.head())
# Convert from DataFrame to a Series, with order_id as index and item_id as value

orders = orders.set_index('order_id')['product_id'].rename('item_id')

display(orders.head(10))

type(orders)
print('dimensions: {0};   size: {1};   unique_orders: {2};   unique_items: {3}'

      .format(orders.shape, size(orders), len(orders.index.unique()), len(orders.value_counts())))
# Returns frequency counts for items and item pairs

def freq(iterable):

    if type(iterable) == pd.core.series.Series:

        return iterable.value_counts().rename("freq")

    else: 

        return pd.Series(Counter(iterable)).rename("freq")



    

# Returns number of unique orders

def order_count(order_item):

    return len(set(order_item.index))





# Returns generator that yields item pairs, one at a time

def get_item_pairs(order_item):

    order_item = order_item.reset_index().as_matrix()

    for order_id, order_object in groupby(order_item, lambda x: x[0]):

        item_list = [item[1] for item in order_object]

              

        for item_pair in combinations(item_list, 2):

            yield item_pair

            



# Returns frequency and support associated with item

def merge_item_stats(item_pairs, item_stats):

    return (item_pairs

                .merge(item_stats.rename(columns={'freq': 'freqA', 'support': 'supportA'}), left_on='item_A', right_index=True)

                .merge(item_stats.rename(columns={'freq': 'freqB', 'support': 'supportB'}), left_on='item_B', right_index=True))





# Returns name associated with item

def merge_item_name(rules, item_name):

    columns = ['itemA','itemB','freqAB','supportAB','freqA','supportA','freqB','supportB', 

               'confidenceAtoB','confidenceBtoA','lift']

    rules = (rules

                .merge(item_name.rename(columns={'item_name': 'itemA'}), left_on='item_A', right_on='item_id')

                .merge(item_name.rename(columns={'item_name': 'itemB'}), left_on='item_B', right_on='item_id'))

    return rules[columns]               
def association_rules(order_item, min_support):



    print("Starting order_item: {:22d}".format(len(order_item)))





    # Calculate item frequency and support

    item_stats             = freq(order_item).to_frame("freq")

    item_stats['support']  = item_stats['freq'] / order_count(order_item) * 100





    # Filter from order_item items below min support 

    qualifying_items       = item_stats[item_stats['support'] >= min_support].index

    order_item             = order_item[order_item.isin(qualifying_items)]



    print("Items with support >= {}: {:15d}".format(min_support, len(qualifying_items)))

    print("Remaining order_item: {:21d}".format(len(order_item)))





    # Filter from order_item orders with less than 2 items

    order_size             = freq(order_item.index)

    qualifying_orders      = order_size[order_size >= 2].index

    order_item             = order_item[order_item.index.isin(qualifying_orders)]



    print("Remaining orders with 2+ items: {:11d}".format(len(qualifying_orders)))

    print("Remaining order_item: {:21d}".format(len(order_item)))





    # Recalculate item frequency and support

    item_stats             = freq(order_item).to_frame("freq")

    item_stats['support']  = item_stats['freq'] / order_count(order_item) * 100





    # Get item pairs generator

    item_pair_gen          = get_item_pairs(order_item)





    # Calculate item pair frequency and support

    item_pairs              = freq(item_pair_gen).to_frame("freqAB")

    item_pairs['supportAB'] = item_pairs['freqAB'] / len(qualifying_orders) * 100



    print("Item pairs: {:31d}".format(len(item_pairs)))





    # Filter from item_pairs those below min support

    item_pairs              = item_pairs[item_pairs['supportAB'] >= min_support]



    print("Item pairs with support >= {}: {:10d}\n".format(min_support, len(item_pairs)))





    # Create table of association rules and compute relevant metrics

    item_pairs = item_pairs.reset_index().rename(columns={'level_0': 'item_A', 'level_1': 'item_B'})

    item_pairs = merge_item_stats(item_pairs, item_stats)

    

    item_pairs['confidenceAtoB'] = item_pairs['supportAB'] / item_pairs['supportA']

    item_pairs['confidenceBtoA'] = item_pairs['supportAB'] / item_pairs['supportB']

    item_pairs['lift']           = item_pairs['supportAB'] / (item_pairs['supportA'] * item_pairs['supportB'])

    

    

    # Return association rules sorted by lift in descending order

    return item_pairs.sort_values('lift', ascending=False)
%%time

rules = association_rules(orders, 0.01)  
# Replace item ID with item name and display association rules

item_name   = pd.read_csv('../input/products.csv')

item_name   = item_name.rename(columns={'product_id':'item_id', 'product_name':'item_name'})

rules_final = merge_item_name(rules, item_name).sort_values('lift', ascending=False)

display(rules_final)