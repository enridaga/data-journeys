
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
#%matplotlib notebook
!ls ../input/allgrades
data = pd.read_excel('../input/allgrades/AllGrades.xlsx')
data.head()
set_class = list(set(data['Class']))
set_class.sort()
set_class
data['Class'] = data['Class'].apply(set_class.index)
data.head()
y = data['Final Exam']
x_1 = data['Test1']
x_2 = data['Test2']
fig = plt.figure()

ax = fig.add_subplot(111, projection='3d')
ax.scatter(x_1, y, x_2, c='red')
plt.show()
def loss_function(y_true,y_predict):
    loss = 1/2 * (y_predict - y_true)**2
    total_loss = sum(loss)
    return sum(loss)

X = data[['Test3']]
one = np.ones((X.shape[0], 1))
Xbar = np.concatenate((one, X), axis = 1)
A = np.dot(Xbar.T, Xbar)
b = np.dot(Xbar.T, y)
w = np.dot(np.linalg.pinv(A), b)
print('w = ', w)

# Plot
x0 = np.linspace(0, 10, 2)
y0 = w[0] + w[1]*x0
plt.plot(X, y.T, 'ro')   
plt.plot(x0, y0)          
plt.xlabel('Test3')
plt.ylabel('Final Exam')
plt.show()
predict = w[0] + w[1]*data['Test3']
predict.head()
total_loss = loss_function(data['Final Exam'],predict)
print('Total loss : {}'.format(total_loss))
X = data[['Test3','Test2']]
one = np.ones((X.shape[0], 1))
Xbar = np.concatenate((one, X), axis = 1)
A = np.dot(Xbar.T, Xbar)
b = np.dot(Xbar.T, data['Final Exam'])
w = np.dot(np.linalg.pinv(A), b) #pseudo inverse
print('w = ', w)
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(X['Test2'], y, X['Test3'], c='red')
x_0 = np.linspace(0, 10, 2)
y_0 = np.linspace(0, 10, 2)
# z_0 = w_0 + w_1 *x_0 + w[2] * z_0
Z = w[0] * data['Test3'] + w[1] * data['Test2'] + w[2]
ax.plot_trisurf(X['Test2'], X['Test3'], Z, alpha=0.2)

plt.show()
predict = w[0] + w[1]*data['Test3'] + w[2] * data['Test2']
predict.head()
total_loss = loss_function(data['Final Exam'],predict)
print('Total loss : {}'.format(total_loss))
X = data[['Class','Assignment1','Assignment2','Assignment3','Test1','Test2','Test3','Final Exam']]
one = np.ones((X.shape[0], 1))
Xbar = np.concatenate((one, X), axis = 1)
A = np.dot(Xbar.T, Xbar)
b = np.dot(Xbar.T, y)
w = np.dot(np.linalg.pinv(A), b)
print('w = ', w)
predict = w[0] + (w[1:] * X).sum(axis=1)
total_loss = loss_function(data['Final Exam'],predict)
print('Total loss : {}'.format(total_loss))
import keras
import tensorflow as tf
model = keras.Sequential([
    keras.layers.Dense(64, activation='linear', input_shape=[7]),
    keras.layers.Dense(1)
  ])
 
def K_loss_function(y_true,y_predict):
    return keras.backend.abs(y_true-y_predict)

model.compile(loss=K_loss_function,
                optimizer='sgd',
                metrics=['mean_absolute_error'])
X = data[['Class','Assignment1','Assignment2','Assignment3','Test1','Test2','Test3']]
Y = data['Final Exam']
model.fit(X,Y,epochs=15)
predict = model.predict(X)
total_loss = loss_function(data['Final Exam'],predict[:,0])
print('Total loss : {}'.format(total_loss))

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
allgrades = pd.read_excel('../input/allgrades/AllGrades.xlsx')
allgrades.head()
allgrades.info()
allgrades.loc[allgrades['Class'] == 'A'].mean()
allgrades.loc[allgrades['Class'] == 'A'].max()
allgrades.loc[(allgrades['Class'] == 'A') & (allgrades['Final Exam'] == 2.2)]
sns.barplot(data=allgrades, x='Class', y='Final Exam')
# map class to int
# mapping_dict = {
#     'A':0,
#     'B':1,
#     'C':2,
#     'D':3,
#     'E':4,
#     'G':5
# }

mapping_dict = {
    'B':0,
    'C':1,
    'G':2,
    'E':3,
    'A':4,
    'D':5
}
int_class = allgrades['Class'].map(mapping_dict)
allgrades['Class'] = int_class
allgrades.head()
# build model
my_model = LinearRegression()
# train test split
X = allgrades.drop('Final Exam', axis=1)
y = allgrades['Final Exam']
X_train, X_test, y_train, y_test = train_test_split(X, y, 
                                                    test_size=0.3,
                                                    random_state=1234)
# training
my_model.fit(X_train, y_train)
# evaluation
# on train set:
y_train_pred = my_model.predict(X_train)
print('Training MSE: ', mean_squared_error(y_train, y_train_pred))
# on test set
y_test_pred = my_model.predict(X_test)
print('Testing MSE: ', mean_squared_error(y_test, y_test_pred))
1.75**0.5
print('Training Loss BinhNA: ', 2*loss_function(y_train, y_train_pred)/len(allgrades))