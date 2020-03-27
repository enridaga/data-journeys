
import pandas as pd

pd.set_option('max_rows', 5)

from learntools.core import binder; binder.bind(globals())

from learntools.pandas.creating_reading_and_writing import *

print("Setup complete.")
# Your code goes here. Create a dataframe matching the above diagram and assign it to the variable fruits.

fruits = ____



q1.check()

fruits
#q1.hint()

#q1.solution()
# Your code goes here. Create a dataframe matching the above diagram and assign it to the variable fruit_sales.

fruit_sales = ____



q2.check()

fruit_sales
#q2.hint()

#q2.solution()
ingredients = ____



q3.check()

ingredients
#q3.hint()

#q3.solution()
reviews = ____



q4.check()

reviews
#q4.hint()

#q4.solution()
animals = pd.DataFrame({'Cows': [12, 20], 'Goats': [22, 19]}, index=['Year 1', 'Year 2'])

animals
# Your code goes here



q5.check()
#q5.hint()

#q5.solution()