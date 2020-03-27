
# https://www.kaggle.com/arunkumarramanan/market-data-nn-baseline
import plotly.offline as py
py.init_notebook_mode(connected=True)
import plotly.graph_objs as go
import plotly.tools as tls
import warnings
# from plotly.tools import FigureFactory as FF 
warnings.filterwarnings('ignore')
import pandas as pd
import numpy as np

#=================================================
#***********************************import keras
from keras import initializers
from keras import regularizers
from keras import constraints
from keras.utils import conv_utils
from keras.utils.data_utils import get_file
from keras.engine.topology import get_source_inputs
from keras.engine import InputSpec
from keras import backend as K
from keras.layers import LeakyReLU
from keras.layers import ZeroPadding2D
from keras.losses import binary_crossentropy
import keras.callbacks as callbacks
from keras.callbacks import Callback
from keras.applications.xception import Xception
from keras.layers import multiply

import keras
from keras import optimizers
from keras.legacy import interfaces
from keras.utils.generic_utils import get_custom_objects

from keras.engine.topology import Input
from keras.engine.training import Model
from keras.layers.convolutional import Conv2D, UpSampling2D, Conv2DTranspose
from keras.layers.core import Activation, SpatialDropout2D
from keras.layers.merge import concatenate
from keras.layers.normalization import BatchNormalization
from keras.layers.pooling import MaxPooling2D
from keras.layers import Input,Dropout,BatchNormalization,Activation,Add
from keras.regularizers import l2
from keras.layers.core import Dense, Lambda
from keras.layers.merge import concatenate, add
from keras.layers import GlobalAveragePooling2D, Reshape, Dense, multiply, Permute
from keras.optimizers import SGD
from kaggle.competitions import twosigmanews
# You can only call make_env() once, so don't lose it!
env = twosigmanews.make_env()
print('Done!')
(market_train_df, news_train_df) = env.get_training_data()
market_train, news_train = market_train_df.copy(), news_train_df.copy()
market_train_df1, news_train_df1 = market_train_df.copy(), news_train_df.copy()
import pandas as pd
import matplotlib.pyplot as plt
plt.style.use("fivethirtyeight")
#%matplotlib inline
import seaborn as sns
import numpy as np
import plotly.figure_factory as ff


######### Function
def mis_value_graph(data):
#     data.isnull().sum().plot(kind="bar", figsize = (20,10), fontsize = 20)
#     plt.xlabel("Columns", fontsize = 20)
#     plt.ylabel("Value Count", fontsize = 20)
#     plt.title("Total Missing Value By Column", fontsize = 20)
#     for i in range(len(data)):
#          colors.append(generate_color())
            
    data = [
    go.Bar(
        x = data.columns,
        y = data.isnull().sum(),
        name = 'Unknown Assets',
        textfont=dict(size=20),
        marker=dict(
#         color= colors,
        line=dict(
            color=generate_color(),
            width=2,
        ), opacity = 0.45
    )
    ),
    ]
    layout= go.Layout(
        title= '"Total Missing Value By Column"',
        xaxis= dict(title='Columns', ticklen=5, zeroline=False, gridwidth=2),
        yaxis=dict(title='Value Count', ticklen=5, gridwidth=2),
        showlegend=True
    )
    fig= go.Figure(data=data, layout=layout)
    py.iplot(fig, filename='skin')
    

def mis_impute(data):
    for i in data.columns:
        if data[i].dtype == "object":
            data[i] = data[i].fillna("other")
        elif (data[i].dtype == "int64" or data[i].dtype == "float64"):
            data[i] = data[i].fillna(data[i].mean())
        else:
            pass
    return data


import random

def generate_color():
    color = '#{:02x}{:02x}{:02x}'.format(*map(lambda x: random.randint(0, 255), range(3)))
    return color
mis_value_graph(market_train_df)
market_train_df = mis_impute(market_train_df)
market_train_df.isna().sum().to_frame()
# https://www.kaggle.com/pestipeti/simple-eda-two-sigma
best_asset_volume = market_train_df.groupby("assetCode")["close"].count().to_frame().sort_values(by=['close'],ascending= False)
best_asset_volume = best_asset_volume.sort_values(by=['close'])
largest_by_volume = list(best_asset_volume.nlargest(10, ['close']).index)
# largest_by_volume
for i in largest_by_volume:
    asset1_df = market_train_df[(market_train_df['assetCode'] == i) & (market_train_df['time'] > '2015-01-01') & (market_train_df['time'] < '2017-01-01')]
    # Create a trace
    trace1 = go.Scatter(
        x = asset1_df['time'].dt.strftime(date_format='%Y-%m-%d').values,
        y = asset1_df['close'].values,
        line = dict(color = generate_color()),opacity = 0.8
    )

    layout = dict(title = "Closing prices of {}".format(i),
                  xaxis = dict(title = 'Month'),
                  yaxis = dict(title = 'Price (USD)'),
                  )

    data = [trace1]
    py.iplot(dict(data=data, layout=layout), filename='basic-line')
for i in largest_by_volume:

    asset1_df['high'] = asset1_df['open']
    asset1_df['low'] = asset1_df['close']

    for ind, row in asset1_df.iterrows():
        if row['close'] > row['open']:
            
            asset1_df.loc[ind, 'high'] = row['close']
            asset1_df.loc[ind, 'low'] = row['open']

    trace1 = go.Candlestick(
        x = asset1_df['time'].dt.strftime(date_format='%Y-%m-%d').values,
        open = asset1_df['open'].values,
        low = asset1_df['low'].values,
        high = asset1_df['high'].values,
        close = asset1_df['close'].values,
        increasing=dict(line=dict(color= generate_color())),
        decreasing=dict(line=dict(color= generate_color())))

    layout = dict(title = "Candlestick chart for {}".format(i),
                  xaxis = dict(
                      title = 'Month',
                      rangeslider = dict(visible = False)
                  ),
                  yaxis = dict(title = 'Price (USD)')
                 )
    data = [trace1]

    py.iplot(dict(data=data, layout=layout), filename='basic-line')
assetsByTradingDay = market_train_df.groupby(market_train_df['time'].dt.date)['assetCode'].nunique()
# Create a trace
trace1 = go.Bar(
    x = assetsByTradingDay.index, # asset1_df['time'].dt.strftime(date_format='%Y-%m-%d').values,
    y = assetsByTradingDay.values, 
    marker=dict(
        color= generate_color(),
        line=dict(
            color=generate_color(),
            width=1.5,
        ), opacity = 0.8
    )
)

layout = dict(title = "Assets by trading days",
              xaxis = dict(title = 'Year'),
              yaxis = dict(title = 'Assets'))
data = [trace1]

py.iplot(dict(data=data, layout=layout), filename='basic-line')
for i in range(1,100,10):
    volumeByAssets = market_train_df.groupby(market_train_df['assetCode'])['volume'].sum()
    highestVolumes = volumeByAssets.sort_values(ascending=False)[i:i+9]
    # Create a trace
    colors = ['#FEBFB3', '#E1396C', '#96D38C', '#D0F9B1']
    trace1 = go.Pie(
        labels = highestVolumes.index,
        values = highestVolumes.values,
        textfont=dict(size=20),
        marker=dict(colors=colors,line=dict(color='#000000', width=2)), hole = 0.45)
    layout = dict(title = "Highest trading volumes for range of {} to {}".format(i, i+9))
    data = [trace1]
    py.iplot(dict(data=data, layout=layout), filename='basic-line')

assetNameGB = market_train_df[market_train_df['assetName'] == 'Unknown'].groupby('assetCode')
unknownAssets = assetNameGB.size().reset_index('assetCode')
unknownAssets.columns = ['assetCode',"value"]
unknownAssets = unknownAssets.sort_values("value", ascending= False)
unknownAssets.head(5)

colors = []
for i in range(len(unknownAssets)):
     colors.append(generate_color())

        
data = [
    go.Bar(
        x = unknownAssets.assetCode.head(25),
        y = unknownAssets.value.head(25),
        name = 'Unknown Assets',
        textfont=dict(size=20),
        marker=dict(
        color= colors,
        line=dict(
            color='#000000',
            width=2,
        ), opacity = 0.45
    )
    ),
    ]
layout= go.Layout(
    title= 'Unknown Assets by Asset code',
    xaxis= dict(title='Columns', ticklen=5, zeroline=False, gridwidth=2),
    yaxis=dict(title='Value Count', ticklen=5, gridwidth=2),
    showlegend=True
)
fig= go.Figure(data=data, layout=layout)
py.iplot(fig, filename='skin')
mis_value_graph(news_train_df)
news_train_df = mis_impute(news_train_df)
news_train_df.isna().sum().to_frame()
print("News data shape",news_train_df.shape)
news_train_df.head()
# news_train_df['urgency'].value_counts()
news_sentiment_count = news_train_df.groupby(["urgency","assetName"])[["sentimentNegative","sentimentNeutral","sentimentPositive"]].count()
news_sentiment_count = news_sentiment_count.reset_index()
trace = go.Table(
    header=dict(values=list(news_sentiment_count.columns),
                fill = dict(color='rgba(55, 128, 191, 0.7)'),
                align = ['left'] * 5),
    cells=dict(values=[news_sentiment_count.urgency,news_sentiment_count.assetName,news_sentiment_count["sentimentNegative"], news_sentiment_count["sentimentPositive"], news_sentiment_count["sentimentNeutral"]],
               fill = dict(color='rgba(245, 246, 249, 1)'),
               align = ['left'] * 5))

data = [trace] 
py.iplot(data, filename = 'pandas_table')
trace0 = go.Bar(
    x= news_sentiment_count.assetName.head(30),
    y=news_sentiment_count.sentimentNegative.values,
    name='sentimentNegative',
    textfont=dict(size=20),
        marker=dict(
        color= generate_color(),
        opacity = 0.87
    )
)
trace1 = go.Bar(
    x= news_sentiment_count.assetName.head(30),
    y=news_sentiment_count.sentimentNeutral.values,
    name='sentimentNeutral',
    textfont=dict(size=20),
        marker=dict(
        color= generate_color(),
        opacity = 0.87
    )
)
trace2 = go.Bar(
    x= news_sentiment_count.assetName.head(30),
    y=news_sentiment_count.sentimentPositive.values,
    name='sentimentPositive',
    textfont=dict(size=20),
    marker=dict(
        color= generate_color(),
        opacity = 0.87
    )
)

data = [trace0, trace1, trace2]
layout = go.Layout(
    xaxis=dict(tickangle=-45),
    barmode='group',
)

fig = go.Figure(data=data, layout=layout)
py.iplot(fig, filename='angled-text-bar')
news_sentiment_urgency = news_train_df.groupby(["urgency"])[["sentimentNegative","sentimentNeutral","sentimentPositive"]].count()
news_sentiment_urgency = news_sentiment_urgency.reset_index()
trace = go.Table(
    header=dict(values=list(news_sentiment_urgency.columns),
                fill = dict(color='rgba(55, 128, 191, 0.7)'),
                align = ['left'] * 5),
    cells=dict(values=[news_sentiment_urgency.urgency,news_sentiment_urgency["sentimentNegative"], news_sentiment_urgency["sentimentPositive"], news_sentiment_urgency["sentimentNeutral"]],
               fill = dict(color='rgba(245, 246, 249, 1)'),
               align = ['left'] * 5))

data = [trace] 
py.iplot(data, filename = 'pandas_table')
trace0 = go.Bar(
    x= news_sentiment_urgency.urgency.values,
    y=news_sentiment_urgency.sentimentNegative.values,
    name='sentimentNegative',
    textfont=dict(size=20),
        marker=dict(
        color= generate_color(),
            line=dict(
            color='#000000',
            width=2,
        ),
        opacity = 0.87
    )
)
trace1 = go.Bar(
    x= news_sentiment_urgency.urgency.values,
    y=news_sentiment_urgency.sentimentNegative.values,
    name='sentimentNeutral',
    textfont=dict(size=20),
        marker=dict(
        color= generate_color(),
        line=dict(
            color='#000000',
            width=2,
        ),
        opacity = 0.87
    )
)
trace2 = go.Bar(
    x= news_sentiment_urgency.urgency.values,
    y=news_sentiment_urgency.sentimentNegative.values,
    name='sentimentPositive',
    textfont=dict(size=20),
    marker=dict(
        line=dict(
            color='#000000',
            width=2,
        ),
        color= generate_color(),
        opacity = 0.87
    )
)
data = [trace0, trace1, trace2]
layout = go.Layout(
    xaxis=dict(tickangle=-45),
    barmode='group',
)

fig = go.Figure(data=data, layout=layout)
py.iplot(fig, filename='angled-text-bar')
cat_cols = ['assetCode']
num_cols = ['volume', 'close', 'open', 'returnsClosePrevRaw1', 'returnsOpenPrevRaw1', 'returnsClosePrevMktres1',
                    'returnsOpenPrevMktres1', 'returnsClosePrevRaw10', 'returnsOpenPrevRaw10', 'returnsClosePrevMktres10',
                    'returnsOpenPrevMktres10']
from sklearn.model_selection import train_test_split
train_indices, val_indices = train_test_split(market_train_df1.index.values,test_size=0.25, random_state=23)
def encode(encoder, x):
    len_encoder = len(encoder)
    try:
        id = encoder[x]
    except KeyError:
        id = len_encoder
    return id

encoders = [{} for cat in cat_cols]


for i, cat in enumerate(cat_cols):
    print('encoding %s ...' % cat, end=' ')
    encoders[i] = {l: id for id, l in enumerate(market_train_df1.loc[train_indices, cat].astype(str).unique())}
    market_train_df1[cat] = market_train_df1[cat].astype(str).apply(lambda x: encode(encoders[i], x))
    print('Done')

embed_sizes = [len(encoder) + 1 for encoder in encoders] #+1 for possible unknown assets
from sklearn.preprocessing import StandardScaler
 
market_train_df1[num_cols] = market_train_df1[num_cols].fillna(0)
print('scaling numerical columns')

scaler = StandardScaler()

#col_mean = market_train[col].mean()
#market_train[col].fillna(col_mean, inplace=True)
scaler = StandardScaler()
market_train_df1[num_cols] = scaler.fit_transform(market_train_df1[num_cols])
# %%time
# def data_prep(market_train,news_train):
#     market_train.time = market_train.time.dt.date
#     news_train.time = news_train.time.dt.hour
#     news_train.sourceTimestamp= news_train.sourceTimestamp.dt.hour
#     news_train.firstCreated = news_train.firstCreated.dt.date
#     news_train['assetCodesLen'] = news_train['assetCodes'].map(lambda x: len(eval(x)))
#     news_train['assetCodes'] = news_train['assetCodes'].map(lambda x: list(eval(x))[0])
#     kcol = ['firstCreated', 'assetCodes']
#     news_train = news_train.groupby(kcol, as_index=False).mean()
#     market_train = pd.merge(market_train, news_train, how='left', left_on=['time', 'assetCode'], 
#                             right_on=['firstCreated', 'assetCodes'])
#     lbl = {k: v for v, k in enumerate(market_train['assetCode'].unique())}
#     market_train['assetCodeT'] = market_train['assetCode'].map(lbl)
    
    
#     market_train = market_train.dropna(axis=0)
    
#     return market_train

# market_train = data_prep(market_train_df, news_train_df)
# market_train.shape
# %%time
# from datetime import datetime, date
# # The target is binary
# market_train = market_train.loc[market_train['time_x']>=date(2009, 1, 1)]
# up = market_train.returnsOpenNextMktres10 >= 0
# fcol = [c for c in market_train if c not in ['assetCode', 'assetCodes', 'assetCodesLen', 'assetName', 'audiences', 
#                                              'firstCreated', 'headline', 'headlineTag', 'marketCommentary', 'provider', 
#                                              'returnsOpenNextMktres10', 'sourceId', 'subjects', 'time', 'time_x', 'universe','sourceTimestamp']]
# %%time
# # We still need the returns for model tuning
# X = market_train[fcol].values
# up = up.values
# r = market_train.returnsOpenNextMktres10.values

# # Scaling of X values
# # It is good to keep these scaling values for later
# mins = np.min(X, axis=0)
# maxs = np.max(X, axis=0)
# rng = maxs - mins
# X = 1 - ((maxs - X) / rng)

# # Sanity check
# assert X.shape[0] == up.shape[0] == r.shape[0]
# %%time
# # from xgboost import XGBClassifier
# from sklearn import model_selection
# from sklearn.metrics import accuracy_score
# import time

# X_train, X_test, up_train, up_test, r_train, r_test = model_selection.train_test_split(X, up, r, test_size=0.25, random_state=99)
# xgb_up = XGBClassifier(n_jobs=4,n_estimators=250,max_depth=8,eta=0.1)
# t = time.time()
# print('Fitting Up')
# xgb_up.fit(X_train,up_train)
# print(f'Done, time = {time.time() - t}')
# from sklearn.metrics import accuracy_score
# accuracy_score(xgb_up.predict(X_test),up_test)
from keras.models import Model
from keras.layers import Input, Dense, Embedding, Concatenate, Flatten, BatchNormalization
from keras.losses import binary_crossentropy, mse

categorical_inputs = []
for cat in cat_cols:
    categorical_inputs.append(Input(shape=[1], name=cat))

categorical_embeddings = []
for i, cat in enumerate(cat_cols):
    categorical_embeddings.append(Embedding(embed_sizes[i], 10)(categorical_inputs[i]))

#categorical_logits = Concatenate()([Flatten()(cat_emb) for cat_emb in categorical_embeddings])
categorical_logits = Flatten()(categorical_embeddings[0])
categorical_logits = Dense(32,activation='relu')(categorical_logits)
categorical_logits =Dropout(0.5)(categorical_logits)
categorical_logits =BatchNormalization()(categorical_logits)
categorical_logits = Dense(32,activation='relu')(categorical_logits)

numerical_inputs = Input(shape=(11,), name='num')
numerical_logits = numerical_inputs
numerical_logits = BatchNormalization()(numerical_logits)

numerical_logits = Dense(128,activation='relu')(numerical_logits)
numerical_logits=Dropout(0.5)(numerical_logits)
numerical_logits = BatchNormalization()(numerical_logits)
numerical_logits = Dense(128,activation='relu')(numerical_logits)
numerical_logits = Dense(64,activation='relu')(numerical_logits)

logits = Concatenate()([numerical_logits,categorical_logits])
logits = Dense(64,activation='relu')(logits)
out = Dense(1, activation='sigmoid')(logits)

model = Model(inputs = categorical_inputs + [numerical_inputs], outputs=out)
model.compile(optimizer='adam',loss=binary_crossentropy)
def get_input(market_train, indices):
    X_num = market_train.loc[indices, num_cols].values
    X = {'num':X_num}
    for cat in cat_cols:
        X[cat] = market_train.loc[indices, cat_cols].values
    y = (market_train.loc[indices,'returnsOpenNextMktres10'] >= 0).values
    r = market_train.loc[indices,'returnsOpenNextMktres10'].values
    u = market_train.loc[indices, 'universe']
    d = market_train.loc[indices, 'time'].dt.date
    return X,y,r,u,d

# r, u and d are used to calculate the scoring metric
X_train,y_train,r_train,u_train,d_train = get_input(market_train_df1, train_indices)
X_valid,y_valid,r_valid,u_valid,d_valid = get_input(market_train_df1, val_indices)
# https://www.kaggle.com/guowenrui/market-nn-if-you-like-you-can-use-it-and-upvote
class SWA(keras.callbacks.Callback):
    
    def __init__(self, filepath, swa_epoch):
        super(SWA, self).__init__()
        self.filepath = filepath
        self.swa_epoch = swa_epoch 
    
    def on_train_begin(self, logs=None):
        self.nb_epoch = self.params['epochs']
        print('Stochastic weight averaging selected for last {} epochs.'
              .format(self.nb_epoch - self.swa_epoch))
        
    def on_epoch_end(self, epoch, logs=None):
        
        if epoch == self.swa_epoch:
            self.swa_weights = self.model.get_weights()
            
        elif epoch > self.swa_epoch:    
            for i in range(len(self.swa_weights)):
                self.swa_weights[i] = (self.swa_weights[i] * 
                    (epoch - self.swa_epoch) + self.model.get_weights()[i])/((epoch - self.swa_epoch)  + 1)  

        else:
            pass
        
    def on_train_end(self, logs=None):
        self.model.set_weights(self.swa_weights)
        print('Final model parameters set to stochastic weight average.')
        self.model.save_weights(self.filepath)
        print('Final stochastic averaged weights saved to file.')
        
class SnapshotCallbackBuilder:
    def __init__(self, nb_epochs, nb_snapshots, init_lr=0.1):
        self.T = nb_epochs
        self.M = nb_snapshots
        self.alpha_zero = init_lr

    def get_callbacks(self, model_prefix='Model'):

        callback_list = [
            callbacks.ModelCheckpoint("model.hdf5",monitor='val_my_iou_metric', 
                                   mode = 'max', save_best_only=True, verbose=1),
            swa,
            callbacks.LearningRateScheduler(schedule=self._cosine_anneal_schedule)
        ]

        return callback_list

    def _cosine_anneal_schedule(self, t):
        cos_inner = np.pi * (t % (self.T // self.M))  # t - 1 is used when t has 1-based indexing.
        cos_inner /= self.T // self.M
        cos_out = np.cos(cos_inner) + 1
        return float(self.alpha_zero / 2 * cos_out)
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

check_point = ModelCheckpoint('model.hdf5',verbose=True, save_best_only=True)
early_stop = EarlyStopping(patience=5,verbose=True)
model.fit(X_train,y_train.astype(int),
          validation_data=(X_valid,y_valid.astype(int)),
          epochs=5,
          verbose=True,
          callbacks=[early_stop,check_point]) 
from sklearn.metrics import accuracy_score
# distribution of confidence that will be used as submission
model.load_weights('model.hdf5')
confidence_valid = model.predict(X_valid)[:,0]*2 -1
print(accuracy_score(confidence_valid>0,y_valid))
plt.hist(confidence_valid, bins='auto')
plt.title("predicted confidence")
plt.show()
# calculation of actual metric that is used to calculate final score
r_valid = r_valid.clip(-1,1) # get rid of outliers. Where do they come from??
x_t_i = confidence_valid * r_valid * u_valid
data = {'day' : d_valid, 'x_t_i' : x_t_i}
df = pd.DataFrame(data)
x_t = df.groupby('day').sum().values.flatten()
mean = np.mean(x_t)
std = np.std(x_t)
score_valid = mean / std
print(score_valid)
# import matplotlib.pyplot as plt
# import seaborn as sns

# df = pd.DataFrame({'imp': xgb_up.feature_importances_, 'col':fcol})
# df = df.sort_values(['imp','col'], ascending=[True, False])
# # _ = df.plot(kind='barh', x='col', y='imp', figsize=(7,12))


# #plt.savefig('lgb_gain.png')
# trace = go.Table(
#     header=dict(values=list(df.columns),
#                 fill = dict(color='rgba(55, 128, 191, 0.7)'),
#                 align = ['left'] * 5),
#     cells=dict(values=[df.imp,df.col],
#                fill = dict(color='rgba(245, 246, 249, 1)'),
#                align = ['left'] * 5))

# data = [trace] 
# py.iplot(data, filename = 'pandas_table')
# data = [df]
# for dd in data:  
#     colors = []
#     for i in range(len(dd)):
#          colors.append(generate_color())

#     data = [
#         go.Bar(
#         orientation = 'h',
#         x=dd.imp,
#         y=dd.col,
#         name='Features',
#         textfont=dict(size=20),
#             marker=dict(
#             color= colors,
#             line=dict(
#                 color='#000000',
#                 width=0.5
#             ),
#             opacity = 0.87
#         )
#     )
#     ]
#     layout= go.Layout(
#         title= 'Feature Importance of XGBOOST',
#         xaxis= dict(title='Columns', ticklen=5, zeroline=True, gridwidth=2),
#         yaxis=dict(title='Value Count', ticklen=5, gridwidth=2),
#         showlegend=True
#     )

#     py.iplot(dict(data=data,layout=layout), filename='horizontal-bar')
days = env.get_prediction_days()
import time
n_days = 0
prep_time = 0
prediction_time = 0
packaging_time = 0
predicted_confidences = np.array([])
for (market_obs_df, news_obs_df, predictions_template_df) in days:
    n_days +=1
    print(n_days,end=' ')
    
    t = time.time()

    market_obs_df['assetCode_encoded'] = market_obs_df[cat].astype(str).apply(lambda x: encode(encoders[i], x))

    market_obs_df[num_cols] = market_obs_df[num_cols].fillna(0)
    market_obs_df[num_cols] = scaler.transform(market_obs_df[num_cols])
    X_num_test = market_obs_df[num_cols].values
    X_test = {'num':X_num_test}
    X_test['assetCode'] = market_obs_df['assetCode_encoded'].values
    
    prep_time += time.time() - t
    
    t = time.time()
    market_prediction = model.predict(X_test)[:,0]*2 -1
    predicted_confidences = np.concatenate((predicted_confidences, market_prediction))
    prediction_time += time.time() -t
    
    t = time.time()
    preds = pd.DataFrame({'assetCode':market_obs_df['assetCode'],'confidence':market_prediction})
    # insert predictions to template
    predictions_template_df = predictions_template_df.merge(preds,how='left').drop('confidenceValue',axis=1).fillna(0).rename(columns={'confidence':'confidenceValue'})
    env.predict(predictions_template_df)
    packaging_time += time.time() - t

env.write_submission_file()
total = prep_time + prediction_time + packaging_time
print(f'Preparing Data: {prep_time:.2f}s')
print(f'Making Predictions: {prediction_time:.2f}s')
print(f'Packing: {packaging_time:.2f}s')
print(f'Total: {total:.2f}s')
# sub  = pd.read_csv("submission.csv")
# sub.head()
# import matplotlib.pyplot as plt
# #%matplotlib inline
# from xgboost import plot_importance
# plt.figure(num=None, figsize=(15, 10), dpi=80, facecolor='w', edgecolor='k')
# plt.bar(range(len(xgb_up.feature_importances_)), xgb_up.feature_importances_)
# plt.xticks(range(len(xgb_up.feature_importances_)), fcol, rotation='vertical');
# distribution of confidence as a sanity check: they should be distributed as above
plt.hist(predicted_confidences, bins='auto')
plt.title("predicted confidence")
plt.show()