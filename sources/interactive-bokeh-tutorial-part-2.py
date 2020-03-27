
# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))
# bokeh packages
from bokeh.io import output_file,show,output_notebook,push_notebook
from bokeh.plotting import *
from bokeh.models import ColumnDataSource,HoverTool,CategoricalColorMapper
from bokeh.layouts import row,column,gridplot,widgetbox
from bokeh.models.widgets import Tabs,Panel
output_notebook()
# Any results you write to the current directory are saved as output.
# As you can see from info method. There are 16598.
# However, Year has 16327 entries. That means Year has NAN value.
# Also Year should be integer but it is given as float. Therefore we will convert it.
# In addition, publisher has NAN values.
data = pd.read_csv("../input/vgsales.csv")
data.info()
# Lets start with dropping nan values
data.dropna(how="any",inplace = True)
data.info()
# Then convert data from float to int
data.Year = data.Year.astype(int)
data.head()     # head method always gives you overview of data.
# bokeh packages
# I make year as index
# Video game sales start from 1980 to 2020.
from IPython.html.widgets import interact
data = data.set_index("Year")
x = data.loc[1980].EU_Sales
y = data.loc[1980].Global_Sales
output_notebook()
# initial source
source = ColumnDataSource(data={
    "x": data.loc[1980].EU_Sales,
    "y": data.loc[1980].Global_Sales,
    "Genre" : data.loc[1980].Genre,
    "Publisher" : data.loc[1980].Publisher,
        "Platform" : data.loc[1980].Platform
})
# color map
factors = data.Genre.unique().tolist()
colors = ["red","green","blue","black","orange","brown","grey","purple","yellow","cyan","pink","peru"]
color_mapper = CategoricalColorMapper(factors=factors,palette=colors)

# hover tool
hover = HoverTool(tooltips = [("Genre of game","@Genre"),("Publisher of game","@Publisher"),("Platform of game","@Platform")])

# plotting
plot=figure(title ="Video Game Sales",tools=[hover,"crosshair","pan","box_zoom"])
plot.circle("x","y",source=source,color=dict(field="Genre", transform=color_mapper), legend='Genre',hover_color ="red")

# this is different from what we learn up to now.
# update method: When slider is changed or when different value from drop down tool is chosen this method is called.
# In this method x and y axis are updated from drop dawn value and year is updated from slider value.
def update(x_axis ,y_axis,year=1980):
    c1 = x_axis
    c2 = y_axis
    new_data = {
        'x'       : data.loc[year,c1],
        'y'       : data.loc[year,c2],
        "Genre" : data.loc[year].Genre,
        "Publisher" : data.loc[year].Publisher,
        "Platform" : data.loc[year].Platform
    }
    source.data = new_data
    plot.xaxis.axis_label = c1
    plot.yaxis.axis_label = c2
    push_notebook()  # this push method is vital for this update method
    
show(plot,notebook_handle=True) 

# interact with update method
interact(update,x_axis=["EU_Sales", "JP_Sales", "Global_Sales","Other_Sales"], y_axis=["EU_Sales", "JP_Sales", "Global_Sales","Other_Sales"],year = (1980,2020))