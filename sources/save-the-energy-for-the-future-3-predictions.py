
# for data manipulation
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# pandas options
pd.set_option('display.max_columns', 50)
pd.set_option('display.float_format', lambda x: '%.3f' % x)
pd.set_option('mode.use_inf_as_na', True)
pd.options.mode.chained_assignment = None

# for date manipulation
from datetime import datetime

# for visualization: matplotlib
from matplotlib import pyplot as plt
from IPython.core.pylabtools import figsize
### matplotlib inline
# to display visuals in the notebook

# for visualization: seaborn
import seaborn as sns
sns.set_context(font_scale=2)

# for data preprocessing
from sklearn.preprocessing import LabelEncoder
from itertools import combinations
from sklearn.model_selection import KFold

# for building the model and calculate RMSE
import lightgbm as lgb
from sklearn.metrics import mean_squared_error
from math import sqrt

# to cleanup memory usage
import gc

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.
## Function to reduce the DF size and reduce test dataframe size
def reduce_memory_usage(df, verbose=True):
    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
    start_mem = df.memory_usage().sum() / 1024**2    
    for col in df.columns:
        col_type = df[col].dtypes
        if col_type in numerics:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)  
            else:
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)    
    end_mem = df.memory_usage().sum() / 1024**2
    if verbose: print('Mem. usage decreased to {:5.2f} Mb ({:.1f}% reduction)'.format(end_mem, 100 * (start_mem - end_mem) / start_mem))
# load training data created in the second notebook into dataframes
X = pd.read_csv("/kaggle/input/save-the-energy-for-the-future-2-fe-lightgbm/X.csv")
y = pd.read_csv("/kaggle/input/save-the-energy-for-the-future-2-fe-lightgbm/y.csv", header=None)

# rename target as log_meter_reading
y.rename(columns = {0: "log_meter_reading"}, 
         inplace=True)

reduce_memory_usage(X)
reduce_memory_usage(y)
# create categorical features
categorical_features = ['building_id', 'site_id', 'meter',
                        'primary_use', 'cloud_coverage', 'wind_compass_direction',
                        'day_of_week', 'hour','is_weekend', 'season']

# initial parameters of light gbm algorithm
initial_params = {"objective": "regression",
                  "boosting": "gbdt",
                  "num_leaves": 60,
                  "learning_rate": 0.05,
                  "feature_fraction": 0.85,
                  "reg_lambda": 2,
                  "metric": {'rmse'}
}
# cretae kfold object and empty model and evaluation lists
kf = KFold(n_splits=4, shuffle=False, random_state=42)

# save 4 model as a list
models = []

# dynamically split X and y with the k-fold split indexes
for train_index,valid_index in kf.split(X):
    X_train_kf = X.loc[train_index]
    y_train_kf = y.loc[train_index]
    
    X_valid_kf = X.loc[valid_index]
    y_valid_kf = y.loc[valid_index]
    
    d_train = lgb.Dataset(X_train_kf, 
                          label=y_train_kf,
                          categorical_feature=categorical_features, 
                          free_raw_data=False)
    
    d_valid = lgb.Dataset(X_valid_kf, 
                          label=y_valid_kf,
                          categorical_feature=categorical_features, 
                          free_raw_data=False)
    
    model = lgb.train(initial_params, 
                      train_set=d_train, 
                      num_boost_round=1000, 
                      valid_sets=[d_train, d_valid],
                      verbose_eval=100, 
                      early_stopping_rounds=500)
    
    models.append(model)
    
    del X_train_kf, y_train_kf, X_valid_kf, y_valid_kf, d_train, d_valid
    gc.collect()
del X
del y
gc.collect()
# load test  data
building = pd.read_csv("/kaggle/input/ashrae-energy-prediction/building_metadata.csv")
weather_test = pd.read_csv("/kaggle/input/ashrae-energy-prediction/weather_test.csv")
test = pd.read_csv("/kaggle/input/ashrae-energy-prediction/test.csv")
# merge dataframes on test dataframe
test = test.merge(building, on = "building_id", how = "left")
test = test.merge(weather_test, on = ["site_id", "timestamp"], how="left")

# convert timestamp column to date time data type column
test["timestamp"] = pd.to_datetime(test["timestamp"],
                                   format='%Y-%m-%d %H:%M:%S')

# delete the other ones to save space from the memory
del weather_test
del building
gc.collect()
reduce_memory_usage(test)
# create functions
def convert_direction(series):
    if series <= 90:
        return 0
    # as norteast direction
    elif series <= 180:
        return 1
    # as southeast direction
    elif series <= 270:
        return 2
    # as southwest direction
    elif series <= 360:
        return 3
    # as northwest direction

def convert_season(month):
    if (month <= 2) | (month == 12):
        return 0
    # as winter
    elif month <= 5:
        return 1
    # as spring
    elif month <= 8:
        return 2
    # as summer
    elif month <= 11:
        return 3
    # as fall
# add wind_direction
test['wind_compass_direction'] = test.wind_direction.apply(convert_direction)
test.drop(columns=['wind_direction'], inplace=True)

# transform primary_use
le = LabelEncoder()
le_primary_use = le.fit_transform(test.primary_use)
test['primary_use'] = le_primary_use
del le, le_primary_use
gc.collect()

# add building age column
current_year = datetime.now().year
test['building_age'] = current_year - test['year_built']
test.drop(columns=['year_built'], inplace=True)

# add month, day of week, day of month, hour, season and year
test['month'] = test['timestamp'].dt.month.astype(np.int8)
test['day_of_week'] = test['timestamp'].dt.dayofweek.astype(np.int8)
test['day_of_month']= test['timestamp'].dt.day.astype(np.int8)
test['hour'] = test['timestamp'].dt.hour
test['is_weekend'] = test.day_of_week.apply(lambda x: 1 if x>=5 else 0)
test['season'] = test.month.apply(convert_season)
test["year"] = test["timestamp"].dt.year

# create list of weather variables
# create weather variables combinations
weather_variables = ["air_temperature", "cloud_coverage", "dew_temperature",
                     "precip_depth_1_hr", "sea_level_pressure", "wind_speed"]

for i, j in combinations(weather_variables, 2):
    test["mean" + i + "_" + j] = (test[i] + test[j]) / 2
#feature_set = ['building_age', 'le_primary_use', 'cloud_coverage',
#               'is_weekend','wind_speed', 'day_of_week',
#               'wind_compass_direction', 'sea_level_pressure', 'air_temperature',
#               'day_of_month', 'dew_temperature', 'hour', 
#               'month', 'meter', 'building_id', 
#               'site_id', 'floor_count', 'square_feet', 'year']
test.columns
test.drop(columns = ["row_id", 
                     "timestamp", 
                     "floor_count", 
                     "meanprecip_depth_1_hr_wind_speed", 
                     "meanair_temperature_cloud_coverage", 
                     "meandew_temperature_wind_speed", 
                     "meanair_temperature_dew_temperature"], 
          inplace=True)
print("Number of unique columns in the test dataset:", test.shape[1])
# split test set based on year and 75 and 25 percent sequentially for imputing reasons
test_2017 = test[test['year'] == 2017]
test_2017_1 = test_2017[:int(3 * test_2017.shape[0] / 4)]
test_2017_2 = test_2017[int(3 * test_2017.shape[0] / 4):]

test_2018 = test[test['year'] == 2018]
test_2018_1 = test_2018[:int(3 * test_2018.shape[0] / 4)]
test_2018_2 = test_2018[int(3 * test_2018.shape[0] / 4):]

del test
del test_2017
del test_2018
gc.collect()
# function to impute missing values with median values of the training set
def my_median_imputer(df_train, df_valid):
    for col in df_train.columns:
        col_median = df_train[col].median()
        df_train.fillna(col_median, inplace=True)
        df_valid.fillna(col_median, inplace=True)
my_median_imputer(test_2017_1, test_2017_2)
my_median_imputer(test_2018_1, test_2018_2)
# add dataframes back
X_test_2017 = pd.concat([test_2017_1,
                         test_2017_2])
X_test_2018 = pd.concat([test_2018_1,
                         test_2018_2])
reduce_memory_usage(X_test_2017)
reduce_memory_usage(X_test_2018)

del test_2017_1
del test_2017_2
del test_2018_1
del test_2018_2
gc.collect()
# drop year column since it is not a feature
X_test_2017.drop(columns=["year"], 
                 inplace=True)

X_test_2018.drop(columns=["year"], 
                 inplace=True)
print('2017 Test Data Shape:', X_test_2017.shape)
print('2018 Test Data Shape:', X_test_2018.shape)
X_test_2017.isna().sum()
X_test_2018.isna().sum()
X_test_2017.memory_usage()
X_test_2018.memory_usage()
X_test_2017.dtypes
# features that datatypes to be converted
int_features = ['building_age', 'primary_use', 'cloud_coverage', 
                'is_weekend',  'wind_compass_direction']

for feature in int_features:
    X_test_2017[feature] = X_test_2017[feature].astype('int8')
    X_test_2018[feature] = X_test_2018[feature].astype('int8')
X_test_2017.memory_usage()
gc.collect()
predictions_2017 = []

for model in models:
    if  predictions_2017 == []:
        predictions_2017 = (np
                            .expm1(model
                                   .predict(X_test_2017, 
                                            num_iteration=model.best_iteration)) / len(models))
    else:
        predictions_2017 += (np
                             .expm1(model
                                    .predict(X_test_2017,
                                             num_iteration=model.best_iteration)) / len(models))
del X_test_2017
gc.collect()
predictions_2018 = []

for model in models:
    if  predictions_2018 == []:
        predictions_2018 = (np
                            .expm1(model
                                   .predict(X_test_2018, 
                                            num_iteration=model.best_iteration)) / len(models))
    else:
        predictions_2018 += (np
                             .expm1(model
                                    .predict(X_test_2018, 
                                             num_iteration=model.best_iteration)) / len(models))
for model in models:
    lgb.plot_importance(model)
    plt.show()
# to fetch row_ids
sample_submission = pd.read_csv("/kaggle/input/ashrae-energy-prediction/sample_submission.csv")
row_ids = sample_submission.row_id

del sample_submission
gc.collect()
# make sure of the shape of predictions
predictions_2017.shape
predictions_2018.shape
# split row_id's with the indexes of 2017 and 2018 predictions
row_ids_2017 = row_ids[:predictions_2017.shape[0]]
row_ids_2018 = row_ids[predictions_2018.shape[0]:]
submission_2017 = pd.DataFrame({"row_id": row_ids_2017, 
                                "meter_reading": np.clip(predictions_2017, 0, a_max=None)})

submission_2018 = pd.DataFrame({"row_id": row_ids_2018, 
                                "meter_reading": np.clip(predictions_2018, 0, a_max=None)})
submission = pd.concat([submission_2017,
                        submission_2018])

del submission_2017, submission_2018
gc.collect()
submission
submission.to_csv("submission.csv", index=False)
del models
gc.collect()