
import pandas as pd

melbourne_file_path = '../input/melbourne-housing-snapshot/melb_data.csv'
melbourne_data = pd.read_csv(melbourne_file_path) 
print(melbourne_data.columns)
# store the series of prices separately as melbourne_price_data.
melbourne_price_data = melbourne_data.Price
# the head command returns the top few lines of data.
print(melbourne_price_data.head())
columns_of_interest = ['Landsize', 'BuildingArea']
two_columns_of_data = melbourne_data[columns_of_interest]
two_columns_of_data.describe()