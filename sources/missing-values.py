


import pandas as pd

from sklearn.model_selection import train_test_split



# Load the data

data = pd.read_csv('../input/melbourne-housing-snapshot/melb_data.csv')



# Select target

y = data.Price



# To keep things simple, we'll use only numerical predictors

melb_predictors = data.drop(['Price'], axis=1)

X = melb_predictors.select_dtypes(exclude=['object'])



# Divide data into training and validation subsets

X_train, X_valid, y_train, y_valid = train_test_split(X, y, train_size=0.8, test_size=0.2,

                                                      random_state=0)


from sklearn.ensemble import RandomForestRegressor

from sklearn.metrics import mean_absolute_error



# Function for comparing different approaches

def score_dataset(X_train, X_valid, y_train, y_valid):

    model = RandomForestRegressor(n_estimators=10, random_state=0)

    model.fit(X_train, y_train)

    preds = model.predict(X_valid)

    return mean_absolute_error(y_valid, preds)
# Get names of columns with missing values

cols_with_missing = [col for col in X_train.columns

                     if X_train[col].isnull().any()]



# Drop columns in training and validation data

reduced_X_train = X_train.drop(cols_with_missing, axis=1)

reduced_X_valid = X_valid.drop(cols_with_missing, axis=1)



print("MAE from Approach 1 (Drop columns with missing values):")

print(score_dataset(reduced_X_train, reduced_X_valid, y_train, y_valid))
from sklearn.impute import SimpleImputer



# Imputation

my_imputer = SimpleImputer()

imputed_X_train = pd.DataFrame(my_imputer.fit_transform(X_train))

imputed_X_valid = pd.DataFrame(my_imputer.transform(X_valid))



# Imputation removed column names; put them back

imputed_X_train.columns = X_train.columns

imputed_X_valid.columns = X_valid.columns



print("MAE from Approach 2 (Imputation):")

print(score_dataset(imputed_X_train, imputed_X_valid, y_train, y_valid))
# Make copy to avoid changing original data (when imputing)

X_train_plus = X_train.copy()

X_valid_plus = X_valid.copy()



# Make new columns indicating what will be imputed

for col in cols_with_missing:

    X_train_plus[col + '_was_missing'] = X_train_plus[col].isnull()

    X_valid_plus[col + '_was_missing'] = X_valid_plus[col].isnull()



# Imputation

my_imputer = SimpleImputer()

imputed_X_train_plus = pd.DataFrame(my_imputer.fit_transform(X_train_plus))

imputed_X_valid_plus = pd.DataFrame(my_imputer.transform(X_valid_plus))



# Imputation removed column names; put them back

imputed_X_train_plus.columns = X_train_plus.columns

imputed_X_valid_plus.columns = X_valid_plus.columns



print("MAE from Approach 3 (An Extension to Imputation):")

print(score_dataset(imputed_X_train_plus, imputed_X_valid_plus, y_train, y_valid))
# Shape of training data (num_rows, num_columns)

print(X_train.shape)



# Number of missing values in each column of training data

missing_val_count_by_column = (X_train.isnull().sum())

print(missing_val_count_by_column[missing_val_count_by_column > 0])