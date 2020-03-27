
import numpy as np
import pandas as pd
from pandas import DataFrame
import scipy.stats as stats
import matplotlib.pyplot as plt
import sklearn
import seaborn as sea
import matplotlib.pyplot as plt
data = pd.read_csv("../input/housesalesprediction/kc_house_data.csv")
data.head()
data.shape
type(data)
data.columns[:]
def correlation_heatmap(data):
    _, ax = plt.subplots(figsize = (15, 10))
    colormap= sea.diverging_palette(220, 10, as_cmap = True)
    sea.heatmap(data.corr(), annot=True, cmap = colormap)

correlation_heatmap(data)
data.corr()
grafico = plt.figure(figsize=(9,6))
plt.scatter(data['sqft_living'], data['price'])
plt.xlabel('Área do Terreno')
plt.ylabel('Valor da casa')
plt.title('Valor da Casa x Área do Terreno')
sea.jointplot('sqft_living','price', data, kind = 'reg')
X = data.iloc[:,5:6].values
y = data.iloc[:,2].values
from sklearn.model_selection import train_test_split
X_treino, X_teste, y_treino, y_teste = train_test_split(X, y, 
                                                        test_size = 0.30, 
                                                        random_state = 0)
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()

regressor.fit(X_treino,y_treino)
score = regressor.score(X_treino,y_treino)
previssoes = regressor.predict(X_teste)

df1 = {'Valor da Casa': y_teste,
      'Previssões': previssoes}
frame = DataFrame(df1)
frame.head()
frame['Resultado'] = abs(y_teste - previssoes)
frame.head()
from sklearn.metrics import mean_absolute_error
mae = mean_absolute_error(y_teste, previssoes)
print('Coeficiente: \n', regressor.coef_)

# MSE (mean square error)
print("MSE: %.2f" % mae)

# Score de variação: 1 representa predição perfeita
print('Score de variação: %.6f' % regressor.score(X, y))
from sklearn.linear_model import LinearRegression
X1 = data.iloc[:, 3:19].values
y1 = data.iloc[:, 2].values
data.head()
regressor = LinearRegression()
X1_treino, X1_teste, y1_treino, y1_teste = train_test_split(X1, y1, 
                                                        test_size = 0.30, 
                                                        random_state = 0)
regressor.fit(X1_treino,y1_treino)
#Score do modelo treino
score1 = regressor.score(X1_treino, y1_treino)
score1
regressor.fit(X1_teste,y1_teste)
#Score do modelo treste
score2 = regressor.score(X1_teste, y1_teste)
score2
mae1 = mean_absolute_error(y1_teste, regressor.predict(X1_teste))
print('Coeficiente: \n', regressor.coef_)

# MSE (mean square error)
print("MSE: %.2f" % mae1)

# Score de variação: 1 representa predição perfeita
print('Score de variação: %.6f' % regressor.score(X1, y1))
