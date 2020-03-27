
# Code you have previously used to load data

import pandas as pd



# Path of the file to read

iowa_file_path = '../input/home-data-for-ml-course/train.csv'



home_data = pd.read_csv(iowa_file_path)



# Set up code checking

from learntools.core import binder

binder.bind(globals())

from learntools.machine_learning.ex3 import *



print("Setup Complete")
# print the list of columns in the dataset to find the name of the prediction target

#y = _



step_1.check()
# The lines below will show you a hint or the solution.

# step_1.hint() 

# step_1.solution()

# Create the list of features below

# feature_names = ___



# select data corresponding to features in feature_names

#X = _



step_2.check()
# step_2.hint()

# step_2.solution()
# Review data

# print description or statistics from X

#print(_)



# print the top few lines

#print(_)

# from _ import _

#specify the model. 

#For model reproducibility, set a numeric value for random_state when specifying the model

iowa_model = _



# Fit the model

_



step_3.check()
# step_3.hint()

# step_3.solution()
predictions = _

print(predictions)

step_4.check()
# step_4.hint()

# step_4.solution()