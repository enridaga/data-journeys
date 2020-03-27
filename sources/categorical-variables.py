


import pandas as pd

from sklearn.model_selection import train_test_split



# Read the data

data = pd.read_csv('../input/melbourne-housing-snapshot/melb_data.csv')



# Separate target from predictors

y = data.Price

X = data.drop(['Price'], axis=1)



# Divide data into training and validation subsets

X_train_full, X_valid_full, y_train, y_valid = train_test_split(X, y, train_size=0.8, test_size=0.2,

                                                                random_state=0)



# Drop columns with missing values (simplest approach)

cols_with_missing = [col for col in X_train_full.columns if X_train_full[col].isnull().any()] 

X_train_full.drop(cols_with_missing, axis=1, inplace=True)

X_valid_full.drop(cols_with_missing, axis=1, inplace=True)



# "Cardinality" means the number of unique values in a column

# Select categorical columns with relatively low cardinality (convenient but arbitrary)

low_cardinality_cols = [cname for cname in X_train_full.columns if X_train_full[cname].nunique() < 10 and 

                        X_train_full[cname].dtype == "object"]



# Select numerical columns

numerical_cols = [cname for cname in X_train_full.columns if X_train_full[cname].dtype in ['int64', 'float64']]



# Keep selected columns only

my_cols = low_cardinality_cols + numerical_cols

X_train = X_train_full[my_cols].copy()

X_valid = X_valid_full[my_cols].copy()
X_train.head()
# Get list of categorical variables

s = (X_train.dtypes == 'object')

object_cols = list(s[s].index)



print("Categorical variables:")

print(object_cols)


from sklearn.ensemble import RandomForestRegressor

from sklearn.metrics import mean_absolute_error



# Function for comparing different approaches

def score_dataset(X_train, X_valid, y_train, y_valid):

    model = RandomForestRegressor(n_estimators=100, random_state=0)

    model.fit(X_train, y_train)

    preds = model.predict(X_valid)

    return mean_absolute_error(y_valid, preds)
drop_X_train = X_train.select_dtypes(exclude=['object'])

drop_X_valid = X_valid.select_dtypes(exclude=['object'])



print("MAE from Approach 1 (Drop categorical variables):")

print(score_dataset(drop_X_train, drop_X_valid, y_train, y_valid))
from sklearn.preprocessing import LabelEncoder



# Make copy to avoid changing original data 

label_X_train = X_train.copy()

label_X_valid = X_valid.copy()



# Apply label encoder to each column with categorical data

label_encoder = LabelEncoder()

for col in object_cols:

    label_X_train[col] = label_encoder.fit_transform(X_train[col])

    label_X_valid[col] = label_encoder.transform(X_valid[col])



print("MAE from Approach 2 (Label Encoding):") 

print(score_dataset(label_X_train, label_X_valid, y_train, y_valid))
from sklearn.preprocessing import OneHotEncoder



# Apply one-hot encoder to each column with categorical data

OH_encoder = OneHotEncoder(handle_unknown='ignore', sparse=False)

OH_cols_train = pd.DataFrame(OH_encoder.fit_transform(X_train[object_cols]))

OH_cols_valid = pd.DataFrame(OH_encoder.transform(X_valid[object_cols]))



# One-hot encoding removed index; put it back

OH_cols_train.index = X_train.index

OH_cols_valid.index = X_valid.index



# Remove categorical columns (will replace with one-hot encoding)

num_X_train = X_train.drop(object_cols, axis=1)

num_X_valid = X_valid.drop(object_cols, axis=1)



# Add one-hot encoded columns to numerical features

OH_X_train = pd.concat([num_X_train, OH_cols_train], axis=1)

OH_X_valid = pd.concat([num_X_valid, OH_cols_valid], axis=1)



print("MAE from Approach 3 (One-Hot Encoding):") 

print(score_dataset(OH_X_train, OH_X_valid, y_train, y_valid))