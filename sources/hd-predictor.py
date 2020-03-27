
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

seed = 51
data = pd.read_csv('/kaggle/input/heart-disease-uci/heart.csv')
data.sample(5)
data.info()
data['chol_age'] = data['chol']/data['age']
data.sample(5)
from sklearn.preprocessing import RobustScaler

data['age'] = RobustScaler().fit_transform(data['age'].values.reshape(-1, 1))
data['chol_age'] = RobustScaler().fit_transform(data['chol_age'].values.reshape(-1, 1))
data['trestbps'] = RobustScaler().fit_transform(data['trestbps'].values.reshape(-1, 1))
data['chol'] = RobustScaler().fit_transform(data['chol'].values.reshape(-1, 1))
data['thalach'] = RobustScaler().fit_transform(data['thalach'].values.reshape(-1, 1))
data['oldpeak'] = RobustScaler().fit_transform(data['oldpeak'].values.reshape(-1, 1))

data.sample(10)
data['cp'][data['cp'] == 0] = 'asymptomatic'
data['cp'][data['cp'] == 1] = 'atypical angina'
data['cp'][data['cp'] == 2] = 'non-anginal pain'
data['cp'][data['cp'] == 3] = 'typical angina'

data['restecg'][data['restecg'] == 0] = 'left ventricular hypertrophy'
data['restecg'][data['restecg'] == 1] = 'normal'
data['restecg'][data['restecg'] == 2] = 'ST-T wave abnormality '

data['slope'][data['slope'] == 0] = 'down'
data['slope'][data['slope'] == 1] = 'flat'
data['slope'][data['slope'] == 2] = 'up'
corr = data.corr()
corr.sort_values(["target"], ascending = False, inplace = True)
corr.target
from sklearn.preprocessing import OneHotEncoder

OH_cols = ['cp', 'slope', 'restecg','thal']

# Apply one-hot encoder to each column with categorical data
OH_encoder = OneHotEncoder(handle_unknown='ignore', sparse=False)
OH_cols_data = pd.DataFrame(OH_encoder.fit_transform(data[OH_cols]))

# One-hot encoding put in generic column names, use feature names instead
OH_cols_data.columns = OH_encoder.get_feature_names(OH_cols)

# # remove the original columns
# for c in OH_cols:
#     cols_to_use.remove(c)
    
# # Add one-hot columns to cols_to_use
# for c in OH_cols_data.columns:
#     cols_to_use.append(c)

# # print(cols_to_use)

# One-hot encoding removed index; put it back
OH_cols_data.index = data.index

# Remove categorical columns (will replace with one-hot encoding)
num_data = data.drop(OH_cols, axis=1)

# Add one-hot encoded columns to numerical features
OH_data = pd.concat([num_data, OH_cols_data], axis=1)

data = OH_data
corr = data.corr()
corr.sort_values(["target"], ascending = False, inplace = True)
corr.target
from sklearn.model_selection import train_test_split

X = data.drop(['target'], axis=1)
y = data['target']

def setup_data(X_in, y_in):
    return train_test_split(X_in, y_in, test_size=0.2, random_state=seed)
import tensorflow
tensorflow.random.set_seed(seed) 
from tensorflow.keras.layers import Input, Dense, ELU, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.regularizers import l2

input = Input(shape=X.shape[1])

m = Dense(1024)(input)
m = ELU()(m)
m = Dropout(0.33)(m)

m = Dense(1024)(m)
m = ELU()(m)
m = Dropout(0.33)(m)

m = Dense(1024)(m)
m = ELU()(m)
m = Dropout(0.33)(m)

m = Dense(1024)(m)
m = ELU()(m)
m = Dropout(0.33)(m)

m = Dense(1024)(m)
m = ELU()(m)
m = Dropout(0.33)(m)

m = Dense(1024)(m)
m = ELU()(m)
m = Dropout(0.33)(m)

m = Dense(1024)(m)
m = ELU()(m)
m = Dropout(0.33)(m)

m = Dense(1024)(m)
m = ELU()(m)
m = Dropout(0.33)(m)

m = Dense(1024)(m)
m = ELU()(m)
m = Dropout(0.33)(m)

m = Dense(1024)(m)
m = ELU()(m)
m = Dropout(0.33)(m)

m = Dense(1024)(m)
m = ELU()(m)
m = Dropout(0.33)(m)

m = Dense(1024)(m)
m = ELU()(m)
m = Dropout(0.33)(m)

#####

m = Dense(1024)(m)
m = ELU()(m)
m = Dropout(0.33)(m)

m = Dense(1024)(m)
m = ELU()(m)
m = Dropout(0.33)(m)

m = Dense(1024)(m)
m = ELU()(m)
m = Dropout(0.33)(m)

m = Dense(1024)(m)
m = ELU()(m)
m = Dropout(0.33)(m)

m = Dense(1024)(m)
m = ELU()(m)
m = Dropout(0.33)(m)

m = Dense(1024)(m)
m = ELU()(m)
m = Dropout(0.33)(m)

m = Dense(1024)(m)
m = ELU()(m)
m = Dropout(0.33)(m)

m = Dense(1024)(m)
m = ELU()(m)
m = Dropout(0.33)(m)

m = Dense(1024)(m)
m = ELU()(m)
m = Dropout(0.33)(m)

m = Dense(1024)(m)
m = ELU()(m)
m = Dropout(0.33)(m)

m = Dense(1024)(m)
m = ELU()(m)
m = Dropout(0.33)(m)

# m = Dense(16, kernel_regularizer=l2(0.01))(m)
# m = ELU()(m)

output = Dense(1, activation='sigmoid')(m)

model = Model(inputs=[input], outputs=[output])

model.summary()
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

model.compile(optimizer=Adam(lr=0.001), loss='binary_crossentropy', metrics=['acc'])

es = EarlyStopping(monitor='val_loss', patience=200, verbose=1, restore_best_weights=True)

rlp = ReduceLROnPlateau(monitor='val_loss', patience=9, verbose=1, factor=0.5, cooldown=5, min_lr=1e-10)
X_remainder, X_test, y_remainder, y_test = setup_data(X,y)
X_train, X_validation, y_train, y_validation = setup_data(X_remainder, y_remainder)

history = model.fit(X_train,
    y_train,
    batch_size=64,
    epochs=200,
    verbose=2,
    callbacks=[es, rlp],
    validation_data=(X_validation, y_validation),
    shuffle=True
         ).history
import matplotlib.pyplot as plt
import seaborn as sns

fig, (ax1, ax2) = plt.subplots(2, 1, sharex='col', figsize=(20, 14))

ax1.plot(history['loss'], label='Train loss')
ax1.plot(history['val_loss'], label='Validation loss')
ax1.legend(loc='best')
ax1.set_title('Loss')

ax2.plot(history['acc'], label='Train accuracy')
ax2.plot(history['val_acc'], label='Validation accuracy')
ax2.legend(loc='best')
ax2.set_title('Accuracy')

plt.xlabel('Epochs')
sns.despine()
plt.show()
rlp = ReduceLROnPlateau(monitor='val_loss', patience=9, verbose=0, factor=0.5, cooldown=5, min_lr=1e-10)

for z in range(5):

    X_train, X_validation, y_train, y_validation = setup_data(X_remainder, y_remainder)

    history = model.fit(X_train,
        y_train,
        batch_size=64,
        epochs=200,
        verbose=0,
        callbacks=[es, rlp],
        validation_data=(X_validation, y_validation),
        shuffle=True
             ).history
model.evaluate(X, y, batch_size=64, verbose=1)
history = model.fit(X_remainder,
    y_remainder,
    batch_size=64,
    epochs=200,
    verbose=2,
    callbacks=[es, rlp],
    shuffle=True
         ).history
model.evaluate(X_test, y_test, verbose=0)
from sklearn.metrics import confusion_matrix

y_prob = model.predict(X_test)
y_pred = np.around(y_prob)
confusion_matrix = confusion_matrix(y_test, y_pred)
confusion_matrix
total=sum(sum(confusion_matrix))

sensitivity = confusion_matrix[0,0]/(confusion_matrix[0,0]+confusion_matrix[1,0])
print('Sensitivity : ', sensitivity )

specificity = confusion_matrix[1,1]/(confusion_matrix[1,1]+confusion_matrix[0,1])
print('Specificity : ', specificity)