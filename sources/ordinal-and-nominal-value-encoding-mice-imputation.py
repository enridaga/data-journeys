
# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.
from sklearn import preprocessing
from sklearn.preprocessing import OneHotEncoder, StandardScaler, LabelEncoder
from sklearn.compose import make_column_transformer
import matplotlib.pyplot as plt
import seaborn as sns
from fancyimpute import IterativeImputer
#Lets start-off by loading data
train = pd.read_csv("/kaggle/input/house-prices-advanced-regression-techniques/train.csv")
test = pd.read_csv("/kaggle/input/house-prices-advanced-regression-techniques/test.csv")

n_train = train.shape[0]
n_test = test.shape[0]

test_id = test["Id"]

train.drop("Id", axis = 1, inplace = True)
test.drop("Id", axis = 1, inplace = True)

all_data = pd.concat((train, test)).reset_index(drop=True)
all_data.drop(['SalePrice'], axis=1, inplace=True)

print("AllData")
(all_data_rows, all_data_columns) = all_data.shape
print(" Number of rows: {} \n Number of columns: {}".format(all_data_rows, all_data_columns))
print(train.sample(3))

def display_missing(df):
    for col in df.columns.tolist():
        print("{} column missing values: {} / {}".format(col, df[col].isnull().sum(),df.shape[0]))

display_missing(all_data)
def filterProblematicColumns(df,threshold):
    listOfColumnNames = []
    for col in df.columns.tolist():
        if df[col].isnull().sum()> threshold:
            listOfColumnNames.append(col)
            print(col)
    
    return listOfColumnNames

portion = 0.2
threshold = all_data.shape[0] * portion


columnsToDrop = filterProblematicColumns(all_data, threshold)

all_data = all_data.drop(columns=columnsToDrop)
columns_with_missing_values = all_data.loc[:, all_data.isnull().any()]
missing_columns = columns_with_missing_values.columns.tolist()

print("Columns with Missing Values: ","\n", "\n", missing_columns, "\n")
print(columns_with_missing_values.describe())
print(all_data.shape)
print("\n", "--------------", "\n")

numcols = all_data.select_dtypes(include = np.number).columns

#Lets start by plotting a heatmap to determine if any variables are correlated
plt.figure(figsize = (12,8))
sns.heatmap(data= all_data[numcols].corr())
plt.show()
plt.gcf().clear()

def corr_missing_values(df, columns): 
    for column in columns:
        df_corr = df.corr().abs().unstack().sort_values(kind="quicksort", ascending=False).reset_index()
        df_corr.rename(columns={"level_0": "Feature 1", "level_1": "Feature 2", 0: 'Correlation Coefficient'}, inplace=True)
        print(df_corr[df_corr['Feature 1'] == column])
        print("")

corr_missing_values(all_data, [x for x in missing_columns if x in numcols])

from sklearn.preprocessing import OrdinalEncoder
from sklearn.preprocessing import LabelEncoder

numeric_columns = all_data.select_dtypes(include = np.number).columns.tolist()
nominal_columns = ["MSZoning","Street","LandContour","LotConfig","Neighborhood","Condition1","Condition2","BldgType","HouseStyle","RoofStyle","RoofMatl","Exterior1st","Exterior2nd","MasVnrType","Foundation","Heating","CentralAir","Electrical","GarageType","SaleCondition"]
ordinal_columns = ["LotShape","Utilities","LandSlope","ExterQual","ExterCond","BsmtQual","BsmtCond","BsmtExposure","BsmtFinType1","BsmtFinType2","HeatingQC","KitchenQual","Functional","GarageFinish","GarageQual","GarageCond","PavedDrive","SaleType"]

#Check if numbers match, to make sure no columns are left out
print(all_data.shape[1])
print(len(numeric_columns), len(nominal_columns), len(ordinal_columns))

## Ordinal Encoding (by skipping null values)

ordinal_enc_dict = {}
for col_name in ordinal_columns:
    ordinal_enc_dict[col_name] = OrdinalEncoder()

    col = all_data[col_name]
    col_not_null = col[col.notnull()]
    reshaped_vals = col_not_null.values.reshape(-1,1)

    encoded_vals = ordinal_enc_dict[col_name].fit_transform(reshaped_vals)
    all_data.loc[col.notnull(), col_name] = np.squeeze(encoded_vals)

#Check if the values are encoded and no column has been skipped.   
print(all_data[ordinal_columns].head())
print(display_missing(all_data[nominal_columns]))
## Imputation with mode
nom_cols_withnull = all_data[nominal_columns].columns[all_data[nominal_columns].isnull().any()].tolist()

most_common_imputed = all_data[nom_cols_withnull].apply(lambda x: x.fillna(x.value_counts().index[0]))

for col_name in most_common_imputed.columns:
    all_data[col_name] = most_common_imputed[col_name]
nom_df = pd.get_dummies(all_data[nominal_columns], prefix=nominal_columns)

for col_name in nom_df.columns:
    all_data[col_name] = nom_df[col_name]

all_data = all_data.drop(columns= nominal_columns)

print(all_data)
from fancyimpute import IterativeImputer


MICE_imputer = IterativeImputer()
ordinal_mice = all_data.copy(deep = True)

ordinal_mice.iloc[:,:] = np.round(MICE_imputer.fit_transform(ordinal_mice))

for col_name in ordinal_columns:
    all_data[col_name] = ordinal_mice[col_name]

for col_name in numeric_columns:
    all_data[col_name] = ordinal_mice[col_name]


if all_data.isnull().values.any():
    print("Yuh artık!")
    print("GOSHHH!!!!!")
    print("Breakdown loading...")
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()


y = train["SalePrice"]
X = all_data.loc[:n_train-1,:]
test = all_data.loc[n_train:,:]


X.iloc[:,:] = scaler.fit_transform(X.loc[:,:])
test.iloc[:,:] = scaler.fit_transform(test.loc[:,:])

X_train, X_test, y_train, y_test = train_test_split(X, y.values, test_size=0.20, random_state=1)

from sklearn.linear_model import Ridge, Lasso, LinearRegression
from sklearn.kernel_ridge import KernelRidge
from sklearn.model_selection import GridSearchCV
import numpy as np
from sklearn import metrics
from sklearn.metrics import mean_squared_error



# create and fit a ridge regression model, testing each alpha
ridge = Ridge()
lasso = Lasso()
lr = LinearRegression()

kernel_ridge = KernelRidge()

param_grid_kr = {'alpha': [0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1],
              'kernel':['polynomial'], 
              'degree':[2,3,4,5,6,7,8],
              'coef0':[0,1,1.5,2,2.5,3,3.5,10]}
### 
#kernel_ridge = GridSearchCV(kernel_ridge, 
 #                param_grid = param_grid_kr, 
  #               scoring = "neg_mean_squared_error", 
   #             cv = 5,
    #             n_jobs = -1,
     #            verbose = 1)

#kernel_ridge.fit(X,y)
#print(pd.DataFrame(kernel_ridge.cv_results_))
#k_best = kernel_ridge.best_estimator_
#kernel_ridge.best_score_

param_grid = {"alpha": [0.001,0.003,0.01,0.3,0.1,0.3,1,3,10,30,100,300,1000,3000,100000]}

print("-----------Stats for Ridge-----------------", "\n")
grid_ridge = GridSearchCV(ridge, param_grid = param_grid, cv = 10, scoring = "neg_mean_squared_error", n_jobs=-1) 
grid_ridge.fit(X, y)
print(pd.DataFrame(grid_ridge.cv_results_))

print("-----------Stats for Lasso-----------------", "\n")
grid_lasso = GridSearchCV(lasso, param_grid = param_grid, cv = 10, scoring = "neg_mean_squared_error", n_jobs=-1) 
grid_lasso.fit(X_train, y_train)
print(pd.DataFrame(grid_lasso.cv_results_))

#print(kernel_ridge.best_score_)
#print(kernel_ridge.best_estimator_.alpha)

print("-----------Scoreboard for Ridge-----------------", "\n")
print(grid_ridge.best_score_)
print(grid_ridge.best_estimator_.alpha)

print("-----------Scoreboard for Lasso-----------------", "\n")

print(grid_lasso.best_score_)
print(grid_lasso.best_estimator_.alpha)
test_ridge = Ridge(alpha = 300)
test_ridge.fit(X,y)
predictions = test_ridge.predict(test)


#output = pd.DataFrame({'Id': test_id, 'SalePrice': predictions})
#output.to_csv('first_draft.csv', index=False)


