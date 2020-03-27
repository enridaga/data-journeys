
import pandas as pd
import numpy as np

"../input/"
X_full = pd.read_csv("../input/learn-together/train.csv", 
                index_col=0)
test = pd.read_csv("../input/learn-together/test.csv", 
                   index_col=0)

TARGET = 'Cover_Type'
#X_full[TARGET] = X_full[TARGET].transform(str)
X = X_full.copy()
y = X_full[TARGET]
print(X.shape)
X.head()
y.value_counts()
X.dtypes
X.describe()
X.isna().sum().sum()
import seaborn as sns
import matplotlib.pyplot as plt

X['Euclidean_distance_to_hydro'] = (X.Vertical_Distance_To_Hydrology**2 + X.Horizontal_Distance_To_Hydrology**2)**.5

f, axes = plt.subplots(1, 3, figsize=(15, 5), sharex=True, sharey=True)

sns.distplot(X.Horizontal_Distance_To_Hydrology, color="b", ax=axes[0])
sns.distplot(X.Vertical_Distance_To_Hydrology, color="b", ax=axes[1])
sns.distplot(X['Euclidean_distance_to_hydro'], color="g", ax=axes[2])

def euclidean(df):
    df['Euclidean_distance_to_hydro'] = (df.Vertical_Distance_To_Hydrology**2 
                                         + df.Horizontal_Distance_To_Hydrology**2)**.5

    return df

X = euclidean(X)
test = euclidean(test)

from itertools import combinations

def distances(df):
    cols = [
        'Horizontal_Distance_To_Roadways',
        'Horizontal_Distance_To_Fire_Points',
        'Horizontal_Distance_To_Hydrology',
    ]
    
    df['distance_mean'] = df[cols].mean(axis=1)
    df['distance_sum'] = df[cols].sum(axis=1)
    df['distance_road_fire'] = df[cols[:2]].mean(axis=1)
    df['distance_hydro_fire'] = df[cols[1:]].mean(axis=1)
    df['distance_road_hydro'] = df[[cols[0], cols[2]]].mean(axis=1)
    
    df['distance_sum_road_fire'] = df[cols[:2]].sum(axis=1)
    df['distance_sum_hydro_fire'] = df[cols[1:]].sum(axis=1)
    df['distance_sum_road_hydro'] = df[[cols[0], cols[2]]].sum(axis=1)
    
    df['distance_dif_road_fire'] = df[cols[0]] - df[cols[1]]
    df['distance_dif_hydro_road'] = df[cols[2]] - df[cols[0]]
    df['distance_dif_hydro_fire'] = df[cols[2]] - df[cols[1]]
    
    # Vertical distances measures
    colv = ['Elevation', 'Vertical_Distance_To_Hydrology']
    
    df['Vertical_dif'] = df[colv[0]] - df[colv[1]]
    df['Vertical_sum'] = df[colv].sum(axis=1)
    
    return df

X = distances(X)
test = distances(test)
f, axes = plt.subplots(1, 3, figsize=(15, 5), sharex=True, sharey=True)

sns.distplot(X['Hillshade_9am'], color="y", ax=axes[0])
sns.distplot(X['Hillshade_Noon'], color="b", ax=axes[1])
sns.distplot(X['Hillshade_3pm'], color="g", ax=axes[2])

X[['Hillshade_9am', 'Hillshade_Noon', 'Hillshade_3pm']].kurt()
def shade(df):
    SHADES = ['Hillshade_9am', 'Hillshade_Noon', 'Hillshade_3pm']
    
    df['shade_noon_diff'] = df['Hillshade_9am'] - df['Hillshade_Noon']
    df['shade_3pm_diff'] = df['Hillshade_Noon'] - df['Hillshade_3pm']
    df['shade_all_diff'] = df['Hillshade_9am'] - df['Hillshade_3pm']
    df['shade_sum'] = df[SHADES].sum(axis=1)
    df['shade_mean'] = df[SHADES].mean(axis=1)
    return df

X = shade(X)
test = shade(test)
# create a dict that map soil type with rockness
# 0=unknow 1=complex 2=rubbly, 3=stony, 
# 4=very stony, 5=extremely stony 6=extremely bouldery
soils = [
    [7, 15, 8, 14, 16, 17,
     19, 20, 21, 23], #unknow and complex 
    [3, 4, 5, 10, 11, 13],   # rubbly
    [6, 12],    # stony
    [2, 9, 18, 26],      # very stony
    [1, 24, 25, 27, 28, 29, 30,
     31, 32, 33, 34, 36, 37, 38, 
     39, 40, 22, 35], # extremely stony and bouldery
]

soil_dict = dict()
for index, values in enumerate(soils):
    for v in values:
        soil_dict[v] = index
        
        
def soil(df, soil_dict=soil_dict):
    df['Rocky'] =  sum(i * df['Soil_Type'+ str(i)] for i in range(1, 41))
    df['Rocky'] = df['Rocky'].map(soil_dict) 

    return df

X = soil(X)
test = soil(test)
sns.violinplot(x=TARGET, y='Rocky', data=X)

sns.violinplot(x=TARGET, y='Elevation', data=X)
f, axes = plt.subplots(1, 1, figsize=(15, 15), sharex=True, sharey=True)
sns.scatterplot(x='Hillshade_9am', y='Elevation', 
                hue=TARGET, data=X, y_jitter=True)
f, axes = plt.subplots(1, 1, figsize=(15, 15), sharex=True, sharey=True)
sns.scatterplot(x='Hillshade_Noon', y='Elevation', 
                hue=TARGET, data=X, y_jitter=True)
f, axes = plt.subplots(1, 1, figsize=(15, 15), sharex=True, sharey=True)
sns.scatterplot(x='Hillshade_3pm', y='Elevation', 
                hue=TARGET, data=X, y_jitter=True)

f, axes = plt.subplots(1, 1, figsize=(15, 15), sharex=True, sharey=True)
sns.scatterplot(x='Euclidean_distance_to_hydro', y='Elevation', 
                hue=TARGET, data=X)
def elevation(df):
    df['ElevationHydro'] = df['Elevation'] - 0.25 * df['Euclidean_distance_to_hydro']
    return df

X = elevation(X)
test = elevation(test)
f, axes = plt.subplots(1, 1, figsize=(15, 15), sharex=True, sharey=True)
sns.scatterplot(x='Euclidean_distance_to_hydro', y='ElevationHydro', 
                hue=TARGET, data=X)
f, axes = plt.subplots(1, 1, figsize=(15, 15), sharex=True, sharey=True)
sns.scatterplot(x='Vertical_Distance_To_Hydrology', y='Elevation', 
                hue=TARGET, data=X)
def elevationV(df):
    df['ElevationV'] = df['Elevation'] - df['Vertical_Distance_To_Hydrology']
    return df

X = elevationV(X)
test = elevationV(test)
f, axes = plt.subplots(1, 1, figsize=(15, 15), sharex=True, sharey=True)
sns.scatterplot(x='Vertical_Distance_To_Hydrology', y='ElevationV', 
                hue=TARGET, data=X)
f, axes = plt.subplots(1, 1, figsize=(15, 15), sharex=True, sharey=True)
sns.scatterplot(x='Horizontal_Distance_To_Hydrology', y='Elevation', 
                hue=TARGET, data=X)
def elevationH(df):
    df['ElevationH'] = df['Elevation'] - 0.19 * df['Horizontal_Distance_To_Hydrology']
    return df

X = elevationH(X)
test = elevationH(test)
f, axes = plt.subplots(1, 1, figsize=(15, 15), sharex=True, sharey=True)
sns.scatterplot(x='Horizontal_Distance_To_Hydrology', y='ElevationH', 
                hue=TARGET, data=X)
f, axes = plt.subplots(1, 1, figsize=(15, 15), sharex=True, sharey=True)
sns.scatterplot(x='Horizontal_Distance_To_Roadways', y='Elevation', 
                hue=TARGET, data=X)
f, axes = plt.subplots(1, 1, figsize=(15, 15), sharex=True, sharey=True)
sns.scatterplot(x='Horizontal_Distance_To_Fire_Points', y='Elevation', 
                hue=TARGET, data=X)
f, axes = plt.subplots(1, 1, figsize=(15, 15), sharex=True, sharey=True)
sns.scatterplot(x='distance_road_fire', y='Elevation', 
                hue=TARGET, data=X)
def kernel_features(df):
    df['Elevation2'] = df['Elevation']**2
    df['ElevationLog'] = np.log1p(df['Elevation'])
    return df

X = kernel_features(X)
test = kernel_features(test)
f, axes = plt.subplots(1, 1, figsize=(15, 15), sharex=True, sharey=True)
sns.scatterplot(x='Aspect', y='Elevation', 
                hue=TARGET, data=X)
f, axes = plt.subplots(1, 1, figsize=(15, 15), sharex=True, sharey=True)
sns.scatterplot(x='Slope', y='Elevation', 
                hue=TARGET, data=X)
X[['Slope', 'Aspect']].describe()

f, axes = plt.subplots(3, 1, figsize=(15, 15), sharex=True, sharey=True)
sns.scatterplot(y='Slope', x='Hillshade_9am', 
                hue='Aspect', data=X, ax=axes[0])
sns.scatterplot(y='Slope', x='Hillshade_Noon', 
                hue='Aspect', data=X, ax=axes[1])
sns.scatterplot(y='Slope', x='Hillshade_3pm', 
                hue='Aspect', data=X, ax=axes[2])

def degree(df):
    df['Aspect_cos'] = np.cos(np.radians(df.Aspect))
    df['Aspect_sin'] = np.sin(np.radians(df.Aspect))
    #df['Slope_sin'] = np.sin(np.radians(df.Slope))
    df['Aspectcos_Slope'] = df.Slope * df.Aspect_cos
    #df['Aspectsin_Slope'] = df.Slope * df.Aspect_sin
    
    return df

X = degree(X)
test = degree(test)



f, axes = plt.subplots(3, 1, figsize=(15, 15), sharex=True, sharey=True)
sns.scatterplot(y='Slope', x='Hillshade_9am', 
                hue='Aspect_sin', data=X, ax=axes[0])
sns.scatterplot(y='Slope', x='Hillshade_Noon', 
                hue='Aspect_sin', data=X, ax=axes[1])
sns.scatterplot(y='Slope', x='Hillshade_3pm', 
                hue='Aspect_sin', data=X, ax=axes[2])

from bisect import bisect

cardinals = [i for i in range(45, 361, 90)]

points = ['N', 'E', 'S', 'W']

def cardinal(df):
    df['Cardinal'] = df.Aspect.apply(lambda x: points[bisect(cardinals, x) % 4])
    return df

X = cardinal(X)
test = cardinal(test)


f, axes = plt.subplots(5, 1, figsize=(15, 25), sharex=True, sharey=True)
sns.scatterplot(x='Slope', y='Hillshade_3pm', 
                hue=TARGET, data=X[X.Cardinal=='E'], ax=axes[0])
sns.scatterplot(x='Slope', y='Hillshade_3pm', 
                hue=TARGET, data=X[X.Cardinal=='W'], ax=axes[1])
sns.scatterplot(x='Slope', y='Hillshade_3pm', 
                hue=TARGET, data=X[X.Cardinal=='N'], ax=axes[2])
sns.scatterplot(x='Slope', y='Hillshade_3pm', 
                hue=TARGET, data=X[X.Cardinal=='S'], ax=axes[3])
sns.scatterplot(x='Slope', y='Hillshade_3pm', 
                hue=TARGET, data=X, ax=axes[4])
def cardinal_num(df):
    d = {'N': 0, 'E': 1, 'S': 0, 'W':-1}
    df['Cardinal'] = df.Cardinal.apply(lambda x: d[x])
    return df

X = cardinal_num(X)
test = cardinal_num(test)
X['w'] = sum(i * X['Wilderness_Area'+ str(i)] for i in range(1, 5))
test['w'] = sum(i * test['Wilderness_Area'+ str(i)] for i in range(1, 5))

cols = [
    'ElevationH', 'Vertical_dif', 'Euclidean_distance_to_hydro', 
    'Aspectcos_Slope', 'distance_dif_hydro_road', 'Hillshade_9am'
]
n = ['Elev', 'Vert_d', 'Eucli', 'AspSlo', 'dist_hr', 'hillshade']

stats = X.groupby('w')[cols].describe()

for i, col in enumerate(cols):
    name = n[i] + '_mean'
    d = {r: stats[col]['50%'][r] for r in range(1, 5)}
    X[name] = X.w.apply(lambda r: d[r])
    test[name] = test.w.apply(lambda r: d[r])

    
X.drop('w', inplace=True, axis=1)
test.drop('w', inplace=True, axis=1)

# drop label 
if TARGET in X.columns:
    X.drop(TARGET, axis=1, inplace=True)

from sklearn.ensemble import (
    RandomForestClassifier,
    ExtraTreesClassifier,
    AdaBoostClassifier,
)
from lightgbm import LGBMClassifier
from mlxtend.classifier import StackingCVClassifier
import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)

SEED = 2007

models = {
    'LGBM': LGBMClassifier(n_estimators=370,
                           metric='multi_logloss',
                           num_leaves=100,
                           verbosity=0,
                           random_state=SEED,
                           n_jobs=-1), 
    'Random Forest': RandomForestClassifier(n_estimators=500,
                                            n_jobs=-1,
                                            random_state=SEED),
    'Extra Tree': ExtraTreesClassifier(
           max_depth=400, 
           n_estimators=450, n_jobs=-1,
           oob_score=False, random_state=SEED, 
           warm_start=True)

}

clf = models['Random Forest']

def feature_importances(clf, X, y, figsize=(18, 6)):
    clf = clf.fit(X, y)
    
    importances = pd.DataFrame({'Features': X.columns, 
                                'Importances': clf.feature_importances_})
    
    importances.sort_values(by=['Importances'], axis='index', ascending=False, inplace=True)

    fig = plt.figure(figsize=figsize)
    sns.barplot(x='Features', y='Importances', data=importances)
    plt.xticks(rotation='vertical')
    plt.show()
    return importances
    
importances = feature_importances(clf, X, y)    
def select(importances, edge):
    c = importances.Importances >= edge
    cols = importances[c].Features.values
    return cols

col = select(importances, 0.0003)
X = X[col]
test = test[col]    
# cross validation
from sklearn.model_selection import KFold, cross_val_score

# model selection functions

cv = KFold(n_splits=5, shuffle=True, random_state=SEED)

def cross_val(models, X=X, y=y):
    r = dict()
    for name, model in models.items():
        cv_results = cross_val_score(model, X, y,
                             cv=cv, 
                             scoring='accuracy')
        r[name] = cv_results
        print(name, 'Accuracy Mean {0:.4f}, Std {1:.4f}'.format(
              cv_results.mean(), cv_results.std()))
    return r
    
def choose_best(results):
    errors = dict()

    for name, arr in results.items():
        errors[name] = arr.mean()

    best_model =  [m for m, e in errors.items() 
                   if e == max(errors.values())][0]
    return best_model
results = cross_val(models)

best_model_name = choose_best(results)


model = models[best_model_name]
def predict(model, filename, X=X, y=y, test=test):
    model.fit(X, y)
    predicts = model.predict(test)

    output = pd.DataFrame({'ID': test.index,
                       TARGET: predicts})
    output.to_csv(filename+'.csv', index=False)
    return predicts

estimators = [m for m in models.values()]

stack = StackingCVClassifier(classifiers=estimators,
                             meta_classifier=model,
                             cv=cv,
                             use_probas=True,
                             use_features_in_secondary=True,
                             verbose=1,
                             random_state=SEED,
                             n_jobs=-1)

predict_stack = predict(stack, 'stacked')
print('Ready!')