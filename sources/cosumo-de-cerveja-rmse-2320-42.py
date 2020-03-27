
import warnings
import datetime
import itertools
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn import metrics
from sklearn import linear_model
from sklearn.preprocessing import PolynomialFeatures

from bokeh.layouts import row
from bokeh.transform import transform
from bokeh.palettes import Viridis3, Viridis256
from bokeh.plotting import figure, output_notebook, show
from bokeh.models import BasicTicker, ColorBar, LinearColorMapper, ColumnDataSource, PrintfTickFormatter
output_notebook()
warnings.filterwarnings("ignore")
x_prediction = 'consumption_beer'
x_column = ['consumption_beer']
isweekend = { 0:"n", 1:"y" }
dayofweek = {0:'mon', 1:'tues', 2:'weds', 3:'thurs', 4:'fri', 5:'sat', 6:'sun'}
holidaylist = {
    datetime.date(2015, 1, 1).ctime() : "y",
    datetime.date(2015,2, 16).ctime() : "y",
    datetime.date(2015,2, 17).ctime() : "y",
    datetime.date(2015,4, 3).ctime() : "y",
    datetime.date(2015,4, 21).ctime() : "y",
    datetime.date(2015,5, 1).ctime() : "y",
    datetime.date(2015,6, 4).ctime() : "y",
    datetime.date(2015,9, 7).ctime() : "y",
    datetime.date(2015,10, 12).ctime() : "y",
    datetime.date(2015,11, 2).ctime() : "y",
    datetime.date(2015,11, 15).ctime() : "y",
    datetime.date(2015,12, 25).ctime() : "y"
}
##Load data from CSV and prepare all instances
def load_data():
    path="../input/beer-consumption-sao-paulo/Consumo_cerveja.csv"
    df_HE = pd.read_csv(path, parse_dates=['Data'], decimal=",", thousands=".")
    df_HE.dropna(how='all', inplace=True)
    df_HE.rename(columns={
        'Data': 'date', 
        'Temperatura Media (C)':'avg_temperature',
        'Temperatura Minima (C)': 'min_temperature', 
        'Temperatura Maxima (C)':'max_temperature',
        'Precipitacao (mm)':'precipitation', 
        'Final de Semana':'weekend', 
        'Consumo de cerveja (litros)': 'consumption_beer'
        }, inplace=True)

    df_not_HE = df_HE.copy()

    df_HE = hot_enconding(df_HE, 'date', get_weekday, None, dayofweek)
    df_HE = hot_enconding(df_HE, 'date', get_holiday, "hld", None, True)
    df_HE = hot_enconding(df_HE, 'weekend', get_weekend, "wkd", isweekend, True)
    
    df_not_HE['weekday'] = pd.Categorical(df_not_HE['date'].apply(get_weekday)).codes
    df_not_HE['holiday'] = pd.Categorical(df_not_HE['date'].apply(get_holiday)).codes
    df_not_HE['weekend'] = pd.Categorical(df_not_HE['weekend'].apply(get_weekend)).codes

    df_not_HE.drop('date', axis=1, inplace=True)

    return df_HE, df_not_HE

##Check if date is a holiday and return 'Y' if true
def get_holiday(date):
  return holidaylist.get(date.ctime(), "n")

##Convert a day of week in label of week
def get_weekday(date):
    return dayofweek.get(date.dayofweek, "not")

##Check if date is a weekend and return Y if true
def get_weekend(end):
    return isweekend.get(end, "n")

##Hot Encoding to transforme data in columns
def hot_enconding(df, columnname, functoapply, prefix = None, orderref=None, drop=False):
    columnref = columnname
    if not drop:
        columnname = columnname + '_'

    df[columnname] = df[columnref].apply(functoapply)
    hot_encoded = pd.get_dummies(df[columnname], prefix=prefix)

    if prefix == None:
        prefix = ""
    else:
        prefix = prefix + "_"

    if orderref != None:
        hot_encoded = hot_encoded[[prefix + x for x in orderref.values()]]

    df = df.drop(columnname, axis=1)
    df = df.join(hot_encoded)

    return df

##Split csv data in two dataframes, train and test 
def train_test_split(df, test_size=0.2):
    split_row = len(df) - int(test_size * len(df))
    train_data = df.iloc[:split_row]
    test_data = df.iloc[split_row:]

    return train_data, test_data

##Build a regressin polynomial model
def build_regression(df, yc, xc):
    y = np.array(df[yc])
    x = np.array(df[xc])

    poly = PolynomialFeatures(degree=2)
    y_ = poly.fit_transform(y)
    clf = linear_model.LinearRegression()
    clf.fit(y_, x)

    return clf, poly

##Test a model
def test_regression(df_test, df_train, yc, xc):
    nmcolumnpred = 'consumption_pred_beer'
    nmcolumnreal = 'consumption_beer'
    reg, poly = build_regression(df_train, yc, xc)
    df_test[nmcolumnpred] = -1.0

    for i in range(len(df_test)):
        arr = [np.array(df_test[yc].iloc[i])]
        predict = poly.fit_transform(arr)
        res_predict = reg.predict(predict)
        index = test.index.values[i]
        df_test.at[index, nmcolumnpred] = res_predict

    real = df_test[nmcolumnreal]
    predict = df_test[nmcolumnpred]
    df_test.drop(nmcolumnpred, axis=1, inplace=True)

    return np.sqrt(metrics.mean_squared_error(real.values, predict.values)), real, predict        
    
##Test all features combination until find the best rmse
def test_all_combination_rmse(x_column, test_comb, df_train, df_test):
    print(datetime.datetime.now().ctime())
    minor = 100000
    best_comp = []
    count_test = 1

    for L in range(0, len(test_comb)+1):
        for subset in itertools.combinations(test_comb, L):
            count_test += 1
            if count_test % 1000 == 0:
                print("Quantidade de teste:" + str(count_test) + " Menor valor do RMSE: " + str(minor))

            y_columns = np.array(subset)
            if len(y_columns) > 0:
                mrse, r, p = test_regression(df_test, df_train, y_columns, x_column)
                if  mrse < minor:
                    minor = mrse
                    best_comp = y_columns

    print(datetime.datetime.now().ctime())
    return best_comp
def line_plot(line1, line2=None, label1=None, label2=None, color1="blue", color2="red", title="", p = None):
    x1 = line1.index.values
    y1 = line1.values

    if p == None:
        p = figure(plot_width=1200, plot_height=400)
        p.title.text = title

    p.legend.click_policy="hide"
    p.legend.location = "top_left"

    p.line(x1, y1, line_width=1.5, line_color=color1, legend=label1)

    if line2 is not None:
        y2 = line2.values
        x2 = line2.index.values
        p.line(x2, y2, line_width=1.5, line_color=color2, legend=label2)

    return p

def plot_correlation(data, title="Correlation plot"):
    df = data.corr()
    df.index.name = 'AllColumns1'
    df.columns.name = 'AllColumns2'

    df = df.stack().rename("value").reset_index()

    mapper = LinearColorMapper(palette=Viridis256, low=df.value.min(), high=df.value.max())

    TOOLS = ""
    p = figure(
        tools=TOOLS,
        plot_width=700,
        plot_height=550,
        # plot_width=500,
        # plot_height=390,
        title=title,
        x_range=list(df.AllColumns1.drop_duplicates()),
        y_range=list(df.AllColumns2.drop_duplicates()),
        toolbar_location="right",
        x_axis_location="below")

    p.xaxis.major_label_orientation = np.pi/8
    p.yaxis.major_label_orientation = np.pi/8

    p.rect(
        x="AllColumns1",
        y="AllColumns2",
        width=1,
        height=1,
        source=ColumnDataSource(df),
        line_color=None,
        fill_color=transform('value', mapper))

    color_bar = ColorBar(color_mapper=mapper, location=(0, 0), ticker=BasicTicker(desired_num_ticks=10))

    p.add_layout(color_bar, 'right')

    return p
##Two dataframes, first with hot encoding and second without hot encoding
df_consumption_HE, df_consumption_NotHE = load_data()
#Split on dataframe with hot encoding
train, test = train_test_split(df_consumption_HE, 0.2)

#Split the dataframe without hot enconding
train2, test2 = train_test_split(df_consumption_NotHE, 0.2)
#Testing for get best features combinations. This execution takes 1 hour
# test_combination = ['avg_temperature', 'min_temperature', 'max_temperature', 'precipitation', 'mon', 'tues', 'weds', 'thurs', 'fri', 'sat', 'sun', 'hld_n', 'hld_y', 'wkd_n', 'wkd_y']
# print(test_all_combination_rmse(x_column, test_combination, train, test))
##Heatmap for feature with best RMSE
columns = ['avg_temperature', 'max_temperature', 'precipitation', 'mon', 'tues', 'weds', 'sun', 'hld_n', 'hld_y', 'wkd_y', 'consumption_beer']
p1 = plot_correlation(df_consumption_HE[columns], title="RMSE better return feature correlation");

##Heatmap for all features
columns = ['avg_temperature', 'min_temperature', 'max_temperature', 'precipitation', 'mon', 'tues', 'weds', 'thurs', 'fri', 'sat', 'sun', 'hld_y', 'hld_n', 'wkd_y', 'wkd_n', 'consumption_beer']
p2 = plot_correlation(df_consumption_HE[columns], title="Correlation of all features");

show(row(p1, p2))
##Heatmap for all features of the data without hot enconding
columns = ['avg_temperature', 'min_temperature', 'max_temperature', 'precipitation', 'weekend', 'consumption_beer', 'weekday', 'holiday']
p1 = plot_correlation(df_consumption_NotHE[columns], title="Correlation of all using data without hot enconding features");

show(p1)
##Show chart of the all data, splited and for training
line_plotted = line_plot(line1=train[x_prediction], line2=test[x_prediction], label1="Alvos", label2="Teste", color1=Viridis256[128], color2=Viridis256[255], title="Consumption Beer [Training and Test]")
show(line_plotted)
##Test all columns 
y_columns = ['avg_temperature', 'min_temperature', 'max_temperature', 'precipitation', 'mon', 'tues', 'weds', 'thurs', 'fri', 'sat', 'sun', 'hld_y', 'hld_n', 'wkd_y', 'wkd_n']
rmse_all, sr, sp = test_regression(test, train, y_columns, x_column)
line_plotted = line_plot(line1=sr, line2=sp, label1="Targets", label2="Test [Full]", color1=Viridis256[0], color2=Viridis256[255], title="Consumption Beer [Polynomial Regression]")

##Test best features
y_columns = ['avg_temperature', 'max_temperature', 'precipitation', 'mon', 'tues', 'weds', 'sun', 'hld_n', 'hld_y', 'wkd_y']
rmse_best_feature, sr, sp = test_regression(test, train, y_columns, x_column)
line_plotted = line_plot(line1=sp, label1="Test [Best Features]", color1=Viridis256[128], p=line_plotted)
y_columns = ['avg_temperature', 'min_temperature', 'max_temperature', 'precipitation', 'weekend', 'weekday', 'holiday']
rmse_not_he, sr, sp = test_regression(test2, train2, y_columns, x_column)
line_plotted = line_plot(line1=sp, label1="Test [Without Hot Enconded Features]", color1="red", p=line_plotted)

#y_columns = ['avg_temperature', 'max_temperature', 'precipitation', 'weekend']
y_columns = ['avg_temperature', 'min_temperature', 'max_temperature', 'weekend']
rmse_best_not_he, sr, sp = test_regression(test2, train2, y_columns, x_column)
line_plotted = line_plot(line1=sp, label1="Test [Without Hot Enconded Features With Best Correlations]", color1="blue", p=line_plotted)
##RMSE Bar chart
labels = ['All Feature', 'Best Feature', 'All Feature (NOT HE)', 'Best Feature (NOT HE)']

rmse_bar = figure(x_range=labels, plot_width=1200, plot_height=400, title="RMSE")
rmse_bar.vbar(x=labels, top=[rmse_all, rmse_best_feature, rmse_not_he, rmse_best_not_he], fill_color=[Viridis256[255], Viridis256[128], 'red', 'blue'], line_color=None, width=0.9)

rmse_bar.xgrid.grid_line_color = None
rmse_bar.y_range.start = 0
show(line_plotted)
show(rmse_bar)
print(rmse_best_feature)