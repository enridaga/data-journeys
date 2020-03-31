
#based on http://bokeh.pydata.org/en/latest/docs/gallery/les_mis.html

from bokeh.plotting import figure, show, output_notebook
from bokeh.models import HoverTool, ColumnDataSource
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def get_product_agg(cols):
    df_train = pd.read_csv('../input/train.csv', usecols = ['Semana', 'Producto_ID'] + cols,
                           dtype  = {'Semana': 'int32',
                                     'Producto_ID':'int32',
                                     'Venta_hoy':'float32',
                                     'Venta_uni_hoy': 'int32',
                                     'Dev_uni_proxima':'int32',
                                     'Dev_proxima':'float32',
                                     'Demanda_uni_equil':'int32'})
    agg  = df_train.groupby(['Semana', 'Producto_ID'], as_index=False).agg(['count','sum', 'min', 'max','median','mean'])
    agg.columns  =  ['_'.join(col).strip() for col in agg.columns.values]
    del(df_train)
    return agg
agg1 = get_product_agg(['Demanda_uni_equil','Dev_uni_proxima'])
agg1.shape
agg1.head()
products  =  pd.read_csv("../input/producto_tabla.csv")
products.head()
### matplotlib inline
def get_top(agg, cols, sort_by, n = 20):
    df = (agg.loc[:, cols]
                         .groupby(level=1)
                         .sum()
                         .sort_values(by = sort_by, ascending = False)
                         .head(20))

    df = pd.merge(df.reset_index(), products, on='Producto_ID', how='left')
    
    fig = plt.figure(figsize = (16,20))
    #fig.set_title('Top 20 products')
    df.plot(y = cols, x = 'NombreProducto', kind = 'barh')
    plt.show()
    return df
top20_prods_by_demand = get_top(agg1,  ['Demanda_uni_equil_sum', 'Dev_uni_proxima' ], 'Demanda_uni_equil_sum')
sns.regplot(x = 'Demanda_uni_equil_count', y='Demanda_uni_equil_sum', data  = agg1)
agg2 = get_product_agg(['Venta_uni_hoy','Dev_uni_proxima'])
agg2.head()
agg2['pct_returns'] = \
        agg2['Dev_uni_proxima_sum'] / agg2['Venta_uni_hoy_sum'].map(lambda x: 1 if x==0 else x)

top20_prods_by_pct_returns = get_top(agg2,['pct_returns'], 'pct_returns')

products[products.NombreProducto == 'Paletina para Cafe NES 3509']
agg2.loc[(slice(None), 3509),:]
agg1.drop((9, 3509), inplace=True)
agg2.drop((9, 3509), inplace=True)
top20_prods_by_pct_returns = get_top(agg2,['pct_returns'], 'pct_returns')