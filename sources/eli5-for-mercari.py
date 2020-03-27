
import eli5

import numpy as np

import pandas as pd

from sklearn.model_selection import KFold

from sklearn.linear_model import Ridge

from sklearn.pipeline import FeatureUnion

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

from sklearn.metrics import mean_squared_log_error
train = pd.read_table('../input/train.tsv')
y_train = np.log1p(train['price'])

train['category_name'] = train['category_name'].fillna('Other').astype(str)

train['brand_name'] = train['brand_name'].fillna('missing').astype(str)

train['shipping'] = train['shipping'].astype(str)

train['item_condition_id'] = train['item_condition_id'].astype(str)

train['item_description'] = train['item_description'].fillna('None')
train.head()
%%time



# we need a custom pre-processor to extract correct field,

# but want to also use default scikit-learn preprocessing (e.g. lowercasing)

default_preprocessor = CountVectorizer().build_preprocessor()

def build_preprocessor(field):

    field_idx = list(train.columns).index(field)

    return lambda x: default_preprocessor(x[field_idx])

    

vectorizer = FeatureUnion([

    ('name', CountVectorizer(

        ngram_range=(1, 2),

        max_features=50000,

        preprocessor=build_preprocessor('name'))),

    ('category_name', CountVectorizer(

        token_pattern='.+',

        preprocessor=build_preprocessor('category_name'))),

    ('brand_name', CountVectorizer(

        token_pattern='.+',

        preprocessor=build_preprocessor('brand_name'))),

    ('shipping', CountVectorizer(

        token_pattern='\d+',

        preprocessor=build_preprocessor('shipping'))),

    ('item_condition_id', CountVectorizer(

        token_pattern='\d+',

        preprocessor=build_preprocessor('item_condition_id'))),

    ('item_description', TfidfVectorizer(

        ngram_range=(1, 3),

        max_features=100000,

        preprocessor=build_preprocessor('item_description'))),

])

X_train = vectorizer.fit_transform(train.values)

X_train
%%time



def get_rmsle(y_true, y_pred):

    return np.sqrt(mean_squared_log_error(np.expm1(y_true), np.expm1(y_pred)))



cv = KFold(n_splits=10, shuffle=True, random_state=42)

for train_ids, valid_ids in cv.split(X_train):

    model = Ridge(

        solver='auto',

        fit_intercept=True,

        alpha=0.5,

        max_iter=100,

        normalize=False,

        tol=0.05)

    model.fit(X_train[train_ids], y_train[train_ids])

    y_pred_valid = model.predict(X_train[valid_ids])

    rmsle = get_rmsle(y_pred_valid, y_train[valid_ids])

    print(f'valid rmsle: {rmsle:.5f}')

    break
eli5.show_weights(model, vec=vectorizer)
eli5.show_weights(model, vec=vectorizer, top=100, feature_filter=lambda x: x != '<BIAS>')
eli5.show_prediction(model, doc=train.values[100], vec=vectorizer)
eli5.show_prediction(model, doc=train.values[1], vec=vectorizer)
eli5.show_prediction(model, doc=train.values[2], vec=vectorizer)