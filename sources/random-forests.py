
import pandas as pd

    

# Load data

melbourne_file_path = '../input/melbourne-housing-snapshot/melb_data.csv'

melbourne_data = pd.read_csv(melbourne_file_path) 

# Filter rows with missing values

melbourne_data = melbourne_data.dropna(axis=0)

# Choose target and features

y = melbourne_data.Price

melbourne_features = ['Rooms', 'Bathroom', 'Landsize', 'BuildingArea', 

                        'YearBuilt', 'Lattitude', 'Longtitude']

X = melbourne_data[melbourne_features]



from sklearn.model_selection import train_test_split



# split data into training and validation data, for both features and target

# The split is based on a random number generator. Supplying a numeric value to

# the random_state argument guarantees we get the same split every time we

# run this script.

train_X, val_X, train_y, val_y = train_test_split(X, y,random_state = 0)
from sklearn.ensemble import RandomForestRegressor

from sklearn.metrics import mean_absolute_error



forest_model = RandomForestRegressor(random_state=1)

forest_model.fit(train_X, train_y)

melb_preds = forest_model.predict(val_X)

print(mean_absolute_error(val_y, melb_preds))
