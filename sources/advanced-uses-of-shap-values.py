
import numpy as np

import pandas as pd

from sklearn.model_selection import train_test_split

from sklearn.ensemble import RandomForestClassifier



data = pd.read_csv('../input/fifa-2018-match-statistics/FIFA 2018 Statistics.csv')

y = (data['Man of the Match'] == "Yes")  # Convert from string "Yes"/"No" to binary

feature_names = [i for i in data.columns if data[i].dtype in [np.int64, np.int64]]

X = data[feature_names]

train_X, val_X, train_y, val_y = train_test_split(X, y, random_state=1)

my_model = RandomForestClassifier(random_state=0).fit(train_X, train_y)
import shap  # package used to calculate Shap values



# Create object that can calculate shap values

explainer = shap.TreeExplainer(my_model)



# calculate shap values. This is what we will plot.

# Calculate shap_values for all of val_X rather than a single row, to have more data for plot.

shap_values = explainer.shap_values(val_X)



# Make plot. Index of [1] is explained in text below.

shap.summary_plot(shap_values[1], val_X)
import shap  # package used to calculate Shap values



# Create object that can calculate shap values

explainer = shap.TreeExplainer(my_model)



# calculate shap values. This is what we will plot.

shap_values = explainer.shap_values(X)



# make plot.

shap.dependence_plot('Ball Possession %', shap_values[1], X, interaction_index="Goal Scored")