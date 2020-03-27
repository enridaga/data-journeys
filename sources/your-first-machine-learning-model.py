
import pandas as pd



melbourne_file_path = '../input/melbourne-housing-snapshot/melb_data.csv'

melbourne_data = pd.read_csv(melbourne_file_path) 

melbourne_data.columns
# The Melbourne data has some missing values (some houses for which some variables weren't recorded.)

# We'll learn to handle missing values in a later tutorial.  

# Your Iowa data doesn't have missing values in the columns you use. 

# So we will take the simplest option for now, and drop houses from our data. 

# Don't worry about this much for now, though the code is:



# dropna drops missing values (think of na as "not available")

melbourne_data = melbourne_data.dropna(axis=0)
y = melbourne_data.Price
melbourne_features = ['Rooms', 'Bathroom', 'Landsize', 'Lattitude', 'Longtitude']
X = melbourne_data[melbourne_features]
X.describe()
X.head()
from sklearn.tree import DecisionTreeRegressor



# Define model. Specify a number for random_state to ensure same results each run

melbourne_model = DecisionTreeRegressor(random_state=1)



# Fit model

melbourne_model.fit(X, y)
print("Making predictions for the following 5 houses:")

print(X.head())

print("The predictions are")

print(melbourne_model.predict(X.head()))