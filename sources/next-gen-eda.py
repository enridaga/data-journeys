
### %HTML
<style type="text/css">
     

    
div.h2 {
    background-color: #159957;
    background-image: linear-gradient(120deg, #155799, #159957);
    text-align: left;
    color: white;              
    padding:9px;
    padding-right: 100px; 
    font-size: 20px; 
    max-width: 1500px; 
    margin: auto; 
    margin-top: 40px; 
}
                                     
                                      
body {
  font-size: 12px;
}    
     
                                    
                                      
div.h3 {
    color: #159957; 
    font-size: 18px; 
    margin-top: 20px; 
    margin-bottom:4px;
}
   
                                      
div.h4 {
    color: #159957;
    font-size: 15px; 
    margin-top: 20px; 
    margin-bottom: 8px;
}
   
                                      
span.note {
    font-size: 5; 
    color: gray; 
    font-style: italic;
}
  
                                      
hr {
    display: block; 
    color: gray
    height: 1px; 
    border: 0; 
    border-top: 1px solid;
}
  
                                      
hr.light {
    display: block; 
    color: lightgray
    height: 1px; 
    border: 0; 
    border-top: 1px solid;
}   
    
                                      
table.dataframe th 
{
    border: 1px darkgray solid;
    color: black;
      <table align="left">
    ...
  </table>
    background-color: white;
}
    
                                      
table.dataframe td 
{
    border: 1px darkgray solid;
    color: black;
    background-color: white;
    font-size: 11px;
    text-align: center;
} 
   
            
                                      
table.rules th 
{
    border: 1px darkgray solid;
    color: black;
    background-color: white;
    font-size: 11px;
    align: left;
}
       
                                      
table.rules td 
{
    border: 1px darkgray solid;
    color: black;
    background-color: white;
    font-size: 13px;
    text-align: center;
} 
   
                                      
                                      
table.rules tr.best
{
    color: green;
}    
    
                                      
.output { 
    align-items: left; 
}
        
                                      
.output_png {
    display: table-cell;
    text-align: left;
    margin:auto;
}                                          
                                                                    
                                      
                                      
</style>  
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#  Reference: 
#      - I really liked the way JohnM's punt kaggle submission had the headers, extremely aesthetically pleasing
#        and aids viewing - borrowing his div.h header concept (so much nicer looking than using conventional
#        ## headers etc), and adding a 'cayman' color theme to it, as a nod to R ...  
#        Isn't it nice looking ?  ->  https://jasonlong.github.io/cayman-theme/
#      - I would strongly suggest we follow JohnM's push into professoinal looking css-based headers, we can't 
#        keep using old-fashioned markdown for headers, its so limited... just my personal opinion
#
# -%%HTML
# <style type="text/css">
#
# div.h2 {
#     background-color: steelblue; 
#     color: white; 
#     padding: 8px; 
#     padding-right: 300px; 
#     font-size: 20px; 
#     max-width: 1500px; 
#     margin: auto; 
#     margin-top: 50px;
# }
# etc
# etc
# --- end reference ---


#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# UNCOMMENT THIS OUT WHEN YOU ARE READY TO OFFICIALLY SUBMIT ! 
# from kaggle.competitions import nflrush
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#T2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
### matplotlib inline
import matplotlib.patches as patches
import seaborn as sns  #I will mainly be using seaborn and bokeh
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# UNCOMMENT THIS OUT WHEN YOU ARE READY TO OFFICIALLY SUBMIT ! 
# from kaggle.competitions import nflrush
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#I wont be using plotly, and being honest it has its strong points, but
#I actually prefer bokeh now 
# import plotly as py
# import plotly.express as px
# import plotly.graph_objs as go
# from plotly.subplots import make_subplots
# from plotly.offline import download_plotlyjs
# from plotly.offline import init_notebook_mode
# from plotly.offline import plot,iplot
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#init_notebook_mode(connected=True)  # remove  
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
import warnings
warnings.filterwarnings('ignore')
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#import sparklines
import colorcet as cc
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
from IPython.display import HTML
from IPython.display import Image
from IPython.display import display
from IPython.core.display import display
from IPython.core.display import HTML
from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity = "all"
from PIL import Image
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
import scipy 
from scipy import constants
import math
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ styles ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
import colorcet as cc
plt.style.use('seaborn') 
color_pal = [x['color'] for x in plt.rcParams['axes.prop_cycle']]
##%config InlineBackend.figure_format = 'retina'   < - keep in case 
### config InlineBackend.figure_format = 'svg' 
#USE THIS in some form:
# th_props = [('font-size', '13px'), ('background-color', 'white'), ('color', '#666666')]
# td_props = [('font-size', '15px'), ('background-color', 'white')]
#styles = [dict(selector="td", props=td_props), dict(selector="th", props=th_props)]
# pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', 500)
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
###?sns.set_context('paper')  #Everything is smaller, use ? 
SMALL_SIZE = 8
MEDIUM_SIZE = 10
BIGGER_SIZE = 12
plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title
##This helps set size of all fontssns.set(font_scale=1.5)
#~~~~~~~~~~~~~~~~~~~~~~~~~ B O K E H ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
from bokeh.io import show
from bokeh.io import push_notebook
from bokeh.io import output_notebook
from bokeh.io import output_file
from bokeh.io import curdoc
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
from bokeh.plotting import show                  
from bokeh.plotting import figure                  
from bokeh.plotting import output_notebook 
from bokeh.plotting import output_file
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
from bokeh.models import ColumnDataSource
from bokeh.models import Circle
from bokeh.models import Grid 
from bokeh.models import LinearAxis
from bokeh.models import Plot
from bokeh.models import Slider
from bokeh.models import CategoricalColorMapper
from bokeh.models import FactorRange
from bokeh.models.tools import HoverTool
from bokeh.models import FixedTicker
from bokeh.models import PrintfTickFormatter
from bokeh.models.glyphs import HBar
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
from bokeh.core.properties import value
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
from bokeh.palettes import Blues4
from bokeh.palettes import Spectral5
from bokeh.palettes import Blues8
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
from bokeh.layouts import row
from bokeh.layouts import column
from bokeh.layouts import gridplot
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
from bokeh.sampledata.perceptions import probly
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
from bokeh.transform import factor_cmap
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ M L  ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
from sklearn import preprocessing
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import KFold
from sklearn.feature_selection import SelectFromModel
import gc, pickle, tqdm, os, datetime
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ 1. kaggle import raw data ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
df = pd.read_csv('../input/nfl-big-data-bowl-2020/train.csv', low_memory=False)
gold = pd.read_csv('../input/nfl-big-data-bowl-2020/train.csv', low_memory=False)
dontbreak = pd.read_csv('../input/nfl-big-data-bowl-2020/train.csv', low_memory=False)
from kaggle.competitions import nflrush
# #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ 2. laptop import raw data ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# df = pd.read_csv('input/nfl-big-data-bowl-2020/train.csv', low_memory=False)
# gold = pd.read_csv('input/nfl-big-data-bowl-2020/train.csv', low_memory=False)
# dontbreak = pd.read_csv('input/nfl-big-data-bowl-2020/train.csv', low_memory=False)
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
killed_columns=['xyz','etc']
def drop_these_columns(your_df,your_list):
    #KILL KOLUMNS
    your_df.drop(your_list,axis=1,inplace=True)
    return(your_df)
YRS = dontbreak[dontbreak.NflId==dontbreak.NflIdRusher].copy()
YR1 = YRS[YRS.Season==2017]
YR2 = YRS[YRS.Season==2018]
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# df_play.drop('Yards', axis=1)
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
##base = df[df["NflIdRusher == NflId"]]
##killed_kolumns = ["GameId","PlayId","Team","Yards","TimeHandoff","TimeSnap"]
#ingplus.com/plots/top-50-matplotlib-visualizations-the-master-plots-python#
#NOTES:
#          sns.despine(bottom=True, left=True)
#  ax.set_title("Rankings Given by Wine Magazine", fontsize=20)

# df04 = tf.groupby('PossessionTeam')['Yards'].agg(sum).sort_values(ascending=False)
# df04 = pd.DataFrame(df04)
# df04['Team'] = df04.index
# df04
#
#
#Some Links:
# Source:  http://www.ncaa.org/about/resources/research/estimated-probability-competing-professional-athletics
#
#
# >>> df = pd.DataFrame(np.random.randn(10, 4))
# >>> df.style.set_table_styles(
# ...     [{'selector': 'tr:hover',
# ...       'props': [('background-color', 'yellow')]}]
# ... )
# sns.despine(left=True)
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#   KEEP: 
#
#      FULL WIDTH SCREEN
#          display(HTML("<style>.container { width:99% !important; }</style>"))   
#
## Set CSS properties for th elements in dataframe
# th_props = [
#   ('font-size', '11px'),
#   ('text-align', 'center'),
#   ('font-weight', 'bold'),
#   ('color', '#6d6d6d'),
#   ('background-color', '#f7f7f9')
#   ]
# # Set CSS properties for td elements in dataframe
# td_props = [
#   ('font-size', '11px')
#   ]
# # Set table styles
# styles = [
#   dict(selector="th", props=th_props),
#   dict(selector="td", props=td_props)
#   ]
# (df.style
#     .applymap(color_negative_red, subset=['total_amt_usd_diff','total_amt_usd_pct_diff'])
#     .format({'total_amt_usd_pct_diff': "{:.2%}"})
#     .set_table_styles(styles))
#
#   df.style.set_properties(**{'text-align': 'right'})
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# value_counts=> dfvc1
# ref = pd.DataFrame({'AlphaCol':dfvc1.index, 'Count':dfvc1.values}).sort_values("AlphaCol")
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# GOLD:
# cm = sns.light_palette("green", as_cmap=True)
# s = df.style.background_gradient(cmap=cm)
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# (monthly_sales
#  .style
#  .format(format_dict)
#  .hide_index()
#  .highlight_max(color='lightgreen')
#  .highlight_min(color='#cd4f39'))
# # USE: 
# dfStyler = df.style.set_properties(**{'text-align': 'left'})
# dfStyler.set_table_styles([dict(selector='th', props=[('text-align', 'left')])])

# filter = df['COUNTRY_FLYING_MISSION'].isin(('USA','GREAT BRITAIN'))
# df = df[filter]
from bokeh.transform import factor_cmap
from bokeh.palettes import Blues8
from bokeh.palettes import Blues, Spectral6, Viridis, Viridis256, GnBu, Viridis256
from bokeh.palettes import Category20b,Category20c,Plasma,Inferno,Category20
from bokeh.palettes import cividis, inferno, grey
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# SUPERHBAR:  i started learning bokeh two days ago, so this quality sucks 
# To draw horizontal bars by specifying a (center) y-coordinate, height, and left and right
# endpoints, use the hbar() glyph function:
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
my_data = df[['PlayerCollegeName','NflId', 'DisplayName']].drop_duplicates().copy()

college_attended = my_data["PlayerCollegeName"].value_counts()

df_cc = pd.DataFrame({'CollegeName':college_attended.index, 'Count':college_attended.values}).sort_values("Count", ascending = False)

#df_cc.Count.astype('int', inplace=True)

df_cc = df_cc[df_cc.CollegeName != 'Louisiana State']

df_cc.at[42,'Count']=51

#df_cc[df_cc.CollegeName=='LSU']['Count']

#THIS IS UNBELIEVABLE.  SOMEONE COUNTED LSU AND LOUSISIANA STATE AS DIFF COLLEGES ! ! ! ! THATS A BIG 
#MISTAKE.  LSU HAS A MASSIVE NUMBER OF PLAYERS CURRENTLY IN THE NFL, and so consolidating the values...

df_cc.sort_values('Count',ascending=False, inplace=True)

#pd.set_option('display.max_rows', 500)
df_cc.index = df_cc.index + 1

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
mysource = ColumnDataSource(df_cc)
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
p = figure(
  y_range=df_cc.CollegeName[:50],    # I need to enter the SAME thing here as y points, i find that odd 
  # wait:  can i set this as the range, but not below ? ? ? 
  # i think caegorical just list in a list the categories here 
  title = '\nNFL Player Count by College Attended\n',
  x_axis_label ='# of NFL players that attended the college prior\n',
  plot_width=600,
  plot_height=700,
  tools="hover",       # or tools="" 
  toolbar_location=None,   
  #background_fill_color="#efe8e2")
  #min_border=0))
)
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~TEMP KILL~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~    
# p.hbar(
#     y='CollegeName',  # center of your y coordinate launcher, 40 points as def above ... 
#     left=0, # or left=20, etc
#     right='Count',    # right is 40 points... 
#     height=0.8,
#     alpha=.6,
#     #color='orange',    #color=Spectral3  #color=Blues8,   
#     #background_fill_color="#efe8e2", 
#     #     fill_color=Blues8,
#     #     fill_alpha=0.4, 
#     source = mysource,
#     fill_alpha=0.9,
#     line_color='blue'   # line_coolor='red'
# ) 
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
p.hbar(
    y='CollegeName',  # center of your y coordinate launcher, 40 points as def above ... 
    left=0, # or left=20, etc
    right='Count',    # right is 40 points... 
    height=0.8,
    alpha=.6,
    #color='orange',    #color=Spectral3  #color=Blues8,   
    #background_fill_color="#efe8e2", 
    #     fill_color=Blues8,
    #     fill_alpha=0.4, 
    
    fill_color=factor_cmap(
        'CollegeName',
        palette=grey(50), #inferno(50),  #cividis(50),  #d3['Category20b'][4],  #Category20b(2),  #[2],   #Category20b,   #Viridis256,    #GnBu[8], #,#Spectral6,             #viridis(50),  #[3], #Spectral6,  #|Blues[2],
        factors=df_cc.CollegeName[:50].tolist()     #'CollegeName'  #but i think i need this: car_list
    ),

    source = mysource,
    fill_alpha=0.9,
    #line_color='blue'  
) 










#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# TITLE: 
# p.title.text = 'Current frame:'
# p.title.text_color = TEXT_COLOR
# p.title.text_font = TEXT_FONT
p.title.text_font_size = '11pt'
# p.title.text_font_style = 'normal'
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# AXES: 
# p.xaxis.major_tick_line_color = None  # turn off x-axis major ticks
# p.xaxis.minor_tick_line_color = None  # turn off x-axis minor ticks
# p.yaxis.major_tick_line_color = None  # turn off y-axis major ticks
# p.yaxis.minor_tick_line_color = None  # turn off y-axis minor ticks
# p.xaxis.axis_line_color = None    # or 'red'
# p.yaxis.axis_line_color = GRID_COLOR 
#
# X-TICKS:
# p.xaxis[0].ticker = FixedTicker(ticks=[0, 1])
# p.xaxis.major_tick_line_color = GRID_COLOR
# p.xaxis.major_label_text_font_size = '7pt'
# p.xaxis.major_label_text_font = TEXT_FONT
# p.xaxis.major_label_text_color = None   #TEXT_COLOR
#
# Y-TICKS:
# p.yaxis[0].ticker = FixedTicker(ticks=np.arange(1, len(labels) + 1, 1).tolist())
# p.yaxis.major_label_text_font_size = '0pt'
p.yaxis.major_tick_line_color = None
p.axis.minor_tick_line_color = None  # turn off y-axis minor ticks

# p.xaxis.major_label_text_font_size = '0pt'  # preferred method for removing tick labels
# p.yaxis.major_label_text_font_size = '0pt'  # preferred method for removing tick labels
p.yaxis.major_tick_line_color = None  # turn off y-axis major ticks
p.yaxis.minor_tick_line_color = None  # turn off y-axis minor ticks
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# GRID:
# p.xgrid.grid_line_color = None
p.ygrid.grid_line_color = None   
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# # LEGENDgend.location = 'top_left'
# p.legend.orientation='vertical'
# p.legend.location='top_right'
# p.legend.label_text_font_size='10px'
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
### NOTES here> 
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# HOVER:
#     hover.names = ['bars']
#     hover.tooltips = [
#         ('Event', '@label'),
#         ('Probability', '@pretty_value')]
#
hover = HoverTool()
#p.select(HoverTool).tooltips = [("x1","@x1"), ("x2","@x2")]
#
# hover.tooltips = [
#         ('Event', '@label')
#         #('Probability', '@pretty_value'),
#     ]
# hover.tooltips = [
#     ("Total:", "@Count")
#     #("x1", "@x1"),
#     #("Totals", "@TONS_HE High Explosive / @TONS_IC Incendiary / @TONS_FRAG Fragmentation")
#     ]
###########################hover.mode = 'vline'
#????curdoc().add_root(p)
# hover.tooltips = """
#     <div>
#         <br>
#         <h4>@CollegeName:</h4>
#         <div><strong>Count: &ensp; </strong>@Count</div>
#     </div>
# """
hover.tooltips = [
    ("College Name:", "@CollegeName"),
    ("Ranking by Count", "$index"),
    ("Number of gradutes that entered the NFL:", "@Count"),
]
#<div><strong>HP: </strong>@Horsepower</div>       
p.add_tools(hover)
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
output_notebook(hide_banner=True)
show(p); 
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# hover.tooltips = [
#     ("index", "$index"),
#     ("(x,y)", "($x, $y)"),
#     ("radius", "@radius"),
#     ("fill color", "$color[hex, swatch]:fill_color"),
#     ("foo", "@foo"),
#     ("bar", "@bar"),
# ]
#
#

# IF YOU WANT ALL BLUES, KEEP THIS: 
# #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# # SUPERHBAR:  i started learning bokeh two days ago, so this quality sucks 
# # To draw horizontal bars by specifying a (center) y-coordinate, height, and left and right
# # endpoints, use the hbar() glyph function:
# #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# my_data = df[['PlayerCollegeName','NflId', 'DisplayName']].drop_duplicates().copy()

# college_attended = my_data["PlayerCollegeName"].value_counts()

# df_cc = pd.DataFrame({'CollegeName':college_attended.index, 'Count':college_attended.values}).sort_values("Count", ascending = False)

# #df_cc.Count.astype('int', inplace=True)

# df_cc = df_cc[df_cc.CollegeName != 'Louisiana State']

# df_cc.at[42,'Count']=51

# #df_cc[df_cc.CollegeName=='LSU']['Count']

# #THIS IS UNBELIEVABLE.  SOMEONE COUNTED LSU AND LOUSISIANA STATE AS DIFF COLLEGES ! ! ! ! THATS A BIG 
# #MISTAKE.  LSU HAS A MASSIVE NUMBER OF PLAYERS CURRENTLY IN THE NFL, and so consolidating the values...

# df_cc.sort_values('Count',ascending=False, inplace=True)

# #pd.set_option('display.max_rows', 500)
# df_cc.index = df_cc.index + 1

# #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# mysource = ColumnDataSource(df_cc)
# #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# p = figure(
#   y_range=df_cc.CollegeName[:50],    # I need to enter the SAME thing here as y points, i find that odd 
#   # wait:  can i set this as the range, but not below ? ? ? 
#   # i think caegorical just list in a list the categories here 
#   title = '\nNFL Player Count by College Attended\n',
#   x_axis_label ='# of NFL players that attended the college prior\n',
#   plot_width=600,
#   plot_height=700,
#   tools="hover",       # or tools="" 
#   toolbar_location=None,   
#   #background_fill_color="#efe8e2")
#   #min_border=0))
# )
# #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~    
# p.hbar(
#     y='CollegeName',  # center of your y coordinate launcher, 40 points as def above ... 
#     left=0, # or left=20, etc
#     right='Count',    # right is 40 points... 
#     height=0.8,
#     alpha=.6,
#     #color='orange',    #color=Spectral3  #color=Blues8,   
#     #background_fill_color="#efe8e2", 
#     #     fill_color=Blues8,
#     #     fill_alpha=0.4, 
#     source = mysource, 
#     line_color='blue'   # line_coolor='red'
# ) 
# #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# # TITLE: 
# # p.title.text = 'Current frame:'
# # p.title.text_color = TEXT_COLOR
# # p.title.text_font = TEXT_FONT
# p.title.text_font_size = '11pt'
# # p.title.text_font_style = 'normal'
# #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# # AXES: 
# # p.xaxis.major_tick_line_color = None  # turn off x-axis major ticks
# # p.xaxis.minor_tick_line_color = None  # turn off x-axis minor ticks
# # p.yaxis.major_tick_line_color = None  # turn off y-axis major ticks
# # p.yaxis.minor_tick_line_color = None  # turn off y-axis minor ticks
# # p.xaxis.axis_line_color = None    # or 'red'
# # p.yaxis.axis_line_color = GRID_COLOR 
# #
# # X-TICKS:
# # p.xaxis[0].ticker = FixedTicker(ticks=[0, 1])
# # p.xaxis.major_tick_line_color = GRID_COLOR
# # p.xaxis.major_label_text_font_size = '7pt'
# # p.xaxis.major_label_text_font = TEXT_FONT
# # p.xaxis.major_label_text_color = None   #TEXT_COLOR
# #
# # Y-TICKS:
# # p.yaxis[0].ticker = FixedTicker(ticks=np.arange(1, len(labels) + 1, 1).tolist())
# # p.yaxis.major_label_text_font_size = '0pt'
# p.yaxis.major_tick_line_color = None
# p.axis.minor_tick_line_color = None  # turn off y-axis minor ticks

# # p.xaxis.major_label_text_font_size = '0pt'  # preferred method for removing tick labels
# # p.yaxis.major_label_text_font_size = '0pt'  # preferred method for removing tick labels
# p.yaxis.major_tick_line_color = None  # turn off y-axis major ticks
# p.yaxis.minor_tick_line_color = None  # turn off y-axis minor ticks
# #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# # GRID:
# # p.xgrid.grid_line_color = None
# p.ygrid.grid_line_color = None   
# #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# # # LEGENDgend.location = 'top_left'
# # p.legend.orientation='vertical'
# # p.legend.location='top_right'
# # p.legend.label_text_font_size='10px'
# #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# ### NOTES here> 
# #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# # HOVER:
# #     hover.names = ['bars']
# #     hover.tooltips = [
# #         ('Event', '@label'),
# #         ('Probability', '@pretty_value')]
# #
# hover = HoverTool()
# #p.select(HoverTool).tooltips = [("x1","@x1"), ("x2","@x2")]
# #
# # hover.tooltips = [
# #         ('Event', '@label')
# #         #('Probability', '@pretty_value'),
# #     ]
# # hover.tooltips = [
# #     ("Total:", "@Count")
# #     #("x1", "@x1"),
# #     #("Totals", "@TONS_HE High Explosive / @TONS_IC Incendiary / @TONS_FRAG Fragmentation")
# #     ]
# ###########################hover.mode = 'vline'
# #????curdoc().add_root(p)
# # hover.tooltips = """
# #     <div>
# #         <br>
# #         <h4>@CollegeName:</h4>
# #         <div><strong>Count: &ensp; </strong>@Count</div>
# #     </div>
# # """
# hover.tooltips = [
#     ("College Name:", "@CollegeName"),
#     ("Ranking by Count", "$index"),
#     ("Number of gradutes that entered the NFL:", "@Count"),
# ]
# #<div><strong>HP: </strong>@Horsepower</div>       
# p.add_tools(hover)
# #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# output_notebook(hide_banner=True)
# show(p); 
# #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# # hover.tooltips = [
# #     ("index", "$index"),
# #     ("(x,y)", "($x, $y)"),
# #     ("radius", "@radius"),
# #     ("fill color", "$color[hex, swatch]:fill_color"),
# #     ("foo", "@foo"),
# #     ("bar", "@bar"),
# # ]
refer = pd.DataFrame(df.columns)
refer.columns=['Mapper']
refer.index.name='Ref:'
refer.style.set_properties(**{'text-align': 'left'}).set_table_styles([dict(selector='th', props=[('text-align', 'center')])])
df.head(1).T. \
style.set_properties(**{'text-align': 'left'}).set_table_styles([dict(selector='th', props=[('text-align', 'center')])])
#--- raw counts for above ---
# len(df_train.Season.unique())
# len(df_train.NflId.unique())
# len(df_train.PlayId.unique())
# len(df_train[df_train.Season==2017].PlayId.unique())
# len(df_train[df_train.Season==2018].PlayId.unique())

def create_field(linenumbers=True,figsize=(10,5)):

    field = patches.Rectangle((0, 0), 100, 53.3, linewidth=5.5,
                             edgecolor='black', facecolor='grey', zorder=0)
    
    fig, ax = plt.subplots(1, figsize=figsize)
    ax.add_patch(field)

    plt.plot([10, 10, 10, 20, 20, 30, 30, 40, 40, 50, 50, 60, 60, 70, 70, 80,
              80, 90, 90, 100, 100, 110, 110, 120, 0, 0, 120, 120],
             [0, 0, 53.3, 53.3, 0, 0, 53.3, 53.3, 0, 0, 53.3, 53.3, 0, 0, 53.3,
              53.3, 0, 0, 53.3, 53.3, 0, 0, 53.3, 53.3, 53.3, 0, 0, 53.3],
             color='white')
    plt.plot([50, 50], [0, 53.3], color='darkgrey')
    
    plt.xlim(-10, 120); plt.ylim(0, 58.3)
    plt.axis('off')
    
        
    return fig, ax
create_field(); 


######################################################################### 
#                                                                       #
#   Creating an example visualization to illustrate the core problem    #
#                                                                       #
#########################################################################

# #Styling
# sns.set_context('paper')
# sns.set(font_scale=1)
# sns.set_style("white", {'grid.linestyle': '--'})
# plt.style.use('seaborn-white')

sns.set_style("white", {'grid.linestyle': '--'})


#Creating a synthetic dataset
synthetic_data   = [12,15,19,21,25,29,35,45,65,90,105,190,305,405,420,430,1700,2300,2450,2855,3105]
synthetic_points = ['U','T','S','R','Q','P','O','N','M','L','K','J','I','H','G','F','E','D','C','B','A']
     
#Creating core dataframe
mich24 = pd.DataFrame(synthetic_data,index=synthetic_points)
mich24.columns =['Count']
mich24 = mich24.sort_values(['Count'], ascending=False)
plt.figure(figsize=(12,7))

ax = sns.barplot(mich24.index, 
                 mich24.Count, 
                 color='gray', 
                 alpha=.6, 
                 linewidth=.1, 
                 edgecolor="red",
                 saturation=80)

ax.get_yaxis().set_major_formatter(plt.FuncFormatter(lambda x, loc: "{:,}".format(int(x))))
ax.set(xlabel="\n\n\n", ylabel='Count\n')
ax.set_xticklabels(mich24.index, color = 'black', alpha=.8)

for item in ax.get_xticklabels(): 
    item.set_rotation(0)
    
for i, v in enumerate(mich24["Count"].iteritems()):        
    ax.text(i ,v[1], "{:,}".format(v[1]), color='gray', va ='bottom', rotation=0, ha='center')
    

ax.tick_params(axis='x', which='major', pad=9)    
plt.tick_params(axis='x',which='both',bottom=False,top=False,labelbottom=True)  
#################################################plt.tight_layout()

plt.axvline(4.5, 0,0.95, linewidth=1.4, color="#00274C", label="= Proposed 'Charbonnet Cut'", linestyle="--")

plt.legend(loc='center', fontsize=13)

#  plt.text(3+0.2, 4.5, "An annotation", horizontalalignment='left', size='medium', color='black', weight='semibold')

plt.text(0, -425, "\nThis is a synthetic dataset I created to illustrate a core problem seen when plotting histograms/boxplots with highly variable data", fontsize=11)

#Remove unnecessary chart junk   
ax.spines["top"].set_visible(False)    
ax.spines["bottom"].set_visible(False)    
ax.spines["right"].set_visible(False)    
ax.spines["left"].set_visible(False) 
# #sns.despine()

plt.title('\n\n\n\nCreating a splitting point can lead to better visualization, if we also plot the second tier/level data...''\n\n',fontsize=12, loc="left")    

plt.text(6.2,700,"|--- This region contains a lack of **visual** insight, we should split data based on Charbonnet Cut ---|", fontsize=10)

plt.show();



sns.set_style("white", {'grid.linestyle': '--'})

#Creating a synthetic dataset
synthetic_data   = [12,15,19,21,25,29,35,45,65,90,105,190,305,405,420,430]
synthetic_points = ['U','T','S','R','Q','P','O','N','M','L','K','J','I','H','G','F']
     
#Creating core dataframe
mich24 = pd.DataFrame(synthetic_data,index=synthetic_points)
mich24.columns =['Count']
mich24 = mich24.sort_values(['Count'], ascending=False)
plt.figure(figsize=(12,7))

ax = sns.barplot(mich24.index, 
                 mich24.Count, 
                 color='gray', 
                 alpha=.6, 
                 linewidth=.1, 
                 edgecolor="red",
                 saturation=80)

ax.get_yaxis().set_major_formatter(plt.FuncFormatter(lambda x, loc: "{:,}".format(int(x))))
ax.set(xlabel="\n\n\n", ylabel='Count\n')
ax.set_xticklabels(mich24.index, color = 'black', alpha=.8)

for item in ax.get_xticklabels(): 
    item.set_rotation(0)
    
for i, v in enumerate(mich24["Count"].iteritems()):        
    ax.text(i ,v[1], "{:,}".format(v[1]), color='gray', va ='bottom', rotation=0, ha='center')
    

ax.tick_params(axis='x', which='major', pad=9)    
plt.tick_params(axis='x',which='both',bottom=False,top=False,labelbottom=True)  
plt.tight_layout()

plt.legend(loc='center', fontsize=13)

#  plt.text(3+0.2, 4.5, "An annotation", horizontalalignment='left', size='medium', color='black', weight='semibold')

plt.text(0, -65, "\nNow we can see the relationship in the heights of the 'second tier' (east of the Charbonnet Cut) data...", fontsize=12, color="#00274C")

#Remove unnecessary chart junk   
ax.spines["top"].set_visible(False)    
ax.spines["bottom"].set_visible(False)    
ax.spines["right"].set_visible(False)    
ax.spines["left"].set_visible(False) 
# #sns.despine()

plt.title('2nd Tier data has been plotted, and now we can see the relationships without data being drowned out...\n\n',fontsize=12, loc="left")    
plt.show();


##sns.palplot(sns.color_palette("RdBu_r", 7))

tf = df.query("NflIdRusher == NflId")

sns.set_style("white", {'grid.linestyle': '--'})

fig, ax = plt.subplots(figsize=(11,8))
ax.set_xlim(-10,26)

###ccc['colors'] = ['red' if x <= 0 else 'green' for x in ccc['Yards']]
###specialcolors = ['red' if x <= 0 else 'green' for x in tf.Yards]
c = [ 'r' if i < 0 else 'b' for i in tf.Yards]


sns.distplot(tf.Yards, kde=False, color='b', bins=100, 
            hist_kws={"linewidth": .9, 'edgecolor':'black'})

#########################ax.set_xlim(80,100)

## Remove the x-tick labels:  plt.xticks([])
plt.yticks([])
## This method also hides the tick marks

plt.title('\nCombined Data: Overall distribution of yards gained during an individual running play\n',fontsize=12)
plt.xlabel('\nYards (yd) Gained $\Rightarrow$\n', fontsize=9)
sns.despine(top=True, right=True, left=True, bottom=True)
ax.xaxis.set_major_locator(plt.MultipleLocator(1))
plt.axvline(x=0, color='blue', linestyle="--", linewidth=.5)
plt.text(-4.55, 2914, r'Line of Scrimmage >', {'color': 'blue', 'fontsize': 9})
plt.tight_layout()
plt.show();

plt.style.use('dark_background')

#aaa is our temp df
aaa = gold
aaa['IsRunner'] = aaa.NflId == aaa.NflIdRusher
#bbb is now the unique run play runners in the year 2018 only
bbb = aaa[aaa.IsRunner & (aaa.Season == 2018)]
#ccc is now a specific actual game
ccc=bbb[bbb.GameId==2018121601] # we grab random game #1 
ccc = ccc[['Yards']][:]
ccc['colors'] = ['red' if x <= 0 else 'green' for x in ccc['Yards']]
##ccc.sort_values('Yards', inplace=True)
ccc.reset_index(inplace=True)
plt.figure(figsize=(9,14))
plt.hlines(y=ccc.index, xmin=0, xmax=ccc.Yards, color=ccc.colors, alpha=0.8, linewidth=9)
plt.gca().set(ylabel='$Play$\n', xlabel='\n$Yards$')
plt.yticks(fontsize=6)
plt.title('\nPositive and Negative Yards for random NFL game #1 - (2018 Season)\n', fontdict={'size':10})
plt.grid(linestyle='--', alpha=0.5)
sns.despine(top=True, right=True, left=True, bottom=True)
plt.gca().invert_yaxis()
plt.tight_layout()
plt.show();

plt.style.use('dark_background')

ccc=bbb[bbb.GameId==2018121500]
ccc = ccc[['Yards']][:]
ccc['colors'] = ['red' if x <= 0 else 'green' for x in ccc['Yards']]
##ccc.sort_values('Yards', inplace=True)
ccc.reset_index(inplace=True)
plt.figure(figsize=(9,12), dpi= 300)
plt.hlines(y=ccc.index, xmin=0, xmax=ccc.Yards, color=ccc.colors, alpha=0.8, linewidth=9)
plt.gca().set(ylabel='$Play$\n', xlabel='\n$Yards$')
plt.yticks(fontsize=6)
plt.title('\nPositive and Negative Yards for random NFL game #2 - (2018 Season)\n', fontdict={'size':10})
plt.grid(linestyle='--', alpha=0.5)
sns.despine(top=True, right=True, left=True, bottom=True)
plt.gca().invert_yaxis()
plt.tight_layout()
plt.show();
plt.style.use('dark_background')

ccc=bbb[bbb.GameId==2018121501]
ccc = ccc[['Yards']][:]
ccc['colors'] = ['red' if x <= 0 else 'green' for x in ccc['Yards']]
##ccc.sort_values('Yards', inplace=True)
ccc.reset_index(inplace=True)
plt.figure(figsize=(9,12), dpi= 300)
plt.hlines(y=ccc.index, xmin=0, xmax=ccc.Yards, color=ccc.colors, alpha=0.8, linewidth=9)
plt.gca().set(ylabel='$Play$\n', xlabel='\n$Yards$')
plt.yticks(fontsize=6)
plt.title('\nPositive and Negative Yards for random NFL game #3 - (2018 Season)\n', fontdict={'size':10})
plt.grid(linestyle='--', alpha=0.5)
sns.despine(top=True, right=True, left=True, bottom=True)
plt.gca().invert_yaxis()
plt.tight_layout()
plt.show();
cm = sns.light_palette("green", as_cmap=True)
tom = df.query("NflIdRusher == NflId")
tom = tom[tom.Season==2018] 
# tom.columns
# Index(['GameId', 'PlayId', 'Team', 'X', 'Y', 'S', 'A', 'Dis', 'Orientation',
#        'Dir', 'NflId', 'DisplayName', 'JerseyNumber', 'Season', 'YardLine',
#        'Quarter', 'GameClock', 'PossessionTeam', 'Down', 'Distance',
#        'FieldPosition', 'HomeScoreBeforePlay', 'VisitorScoreBeforePlay',
#        'NflIdRusher', 'OffenseFormation', 'OffensePersonnel',
#        'DefendersInTheBox', 'DefensePersonnel', 'PlayDirection', 'TimeHandoff',
#        'TimeSnap', 'Yards', 'PlayerHeight', 'PlayerWeight', 'PlayerBirthDate',
#        'PlayerCollegeName', 'Position', 'HomeTeamAbbr', 'VisitorTeamAbbr',
#        'Week', 'Stadium', 'Location', 'StadiumType', 'Turf', 'GameWeather',
#        'Temperature', 'Humidity', 'WindSpeed', 'WindDirection'],
#       dtype='object')
# define this dict that will help normalize the data
fixup = {"ARZ":"ARI","BLT":"BAL","CLV":"CLE","HST":"HOU"}
tom.PossessionTeam.replace(fixup, inplace=True)

#tom.groupby('PossessionTeam')['Yards'].agg(max).sort_values(ascending=False)[:10].values.tolist()
# [99, 97, 92, 90, 78, 77, 75, 71, 70, 67]
#tom.groupby('PossessionTeam')['Yards'].agg(max).sort_values(ascending=False)[:10].values.tolist()
#tom.groupby('PossessionTeam')['Yards'].agg(max).sort_values()
#.agg(max).sort_values(ascending=False)[:10].values.tolist()
tom.groupby(['PossessionTeam'], as_index=False)['Yards'].agg(max).set_index('PossessionTeam').sort_values('Yards', ascending=False)[:10].style.set_caption('TOP TEN LONGEST RUNS:').background_gradient(cmap=cm)
# tf = df.query("NflIdRusher == NflId")


# sns.set_style("white", {'grid.linestyle': '--'})

# fig, ax = plt.subplots(figsize=(10,8))

# sns.distplot(tf.Yards, kde=False, color="b", 
#             hist_kws={"linewidth": .9, 'edgecolor':'steelblue'})

# #########################ax.set_xlim(80,100)

# ## Remove the x-tick labels:  plt.xticks([])
# plt.yticks([])
# ## This method also hides the tick marks


# plt.title('\nOverall distribution of yards gained during an individual running play\n',fontsize=12)
# plt.xlabel('\nYards (yd) Gained -->\n')
# sns.despine(top=True, right=True, left=True, bottom=True)
# plt.tight_layout()
# ##################plt.axvline(x=22, color='maroon', linestyle="--", linewidth=.5)

# ###############plt.text(22.8, 114, r'Median: 22 carries', {'color': 'maroon', 'fontsize': 9})


# # sns.set_style("white", {'grid.linestyle': '--'})

# # # sns.set_style("ticks", {'grid.linestyle': '--'})
# # ##sns.set(style="white", palette="muted", color_codes=True)
# # ##sns.set(style="white", palette="muted", color_codes=True)

# # ##t2 = tf.groupby(['GameId','Team'])['PlayId'].count()
# # ##t2 = pd.DataFrame(t2)

# # fig, ax = plt.subplots(figsize=(9,8))


# # sns.distplot(tf.Yards, kde=False, color="b", 
# #             hist_kws={"linewidth": .9, 'edgecolor':'lightgrey'}, bins=24)


# # # #~~~~~~~~~~~ ax.set ~~~~~~~~~~~~~~~~
# # # ##ax.set_xlim(0, 6)
# # # ##ax.set_ylim(0, 6)
# # # ax.set_title('Average yards gained as the season progresses (week by week)\n')
# # # ax.set(ylabel='Yards Gained\n')
# # # ax.set(xlabel='\nWeek Number (in the season)')
# # # ax.yaxis.grid(True)   # Show the horizontal gridlines
# # # ax.xaxis.grid(False)  # Hide x-axis gridlines 
# # # # ax.yaxis.set_major_locator(plt.MultipleLocator(1))
# # # # ax.yaxis.set_minor_locator(plt.MultipleLocator(5))
# # # #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

# # ## Remove the x-tick labels:  plt.xticks([])
# # plt.yticks([])
# # ## This method also hides the tick marks
# # plt.title('\nDistribution of total number of run plays on a game basis (per team)\n',
# #           fontsize=12, loc="left")
# # plt.title('\nDistribution of total number of run plays on a game basis (per team)\n',fontsize=12, loc="left")
# # plt.xlabel('\nNumber of times the ball was run in the game\n')
# # sns.despine(top=True, right=True, left=True, bottom=True)
# # plt.tight_layout()
# plt.axvline(x=22, color='maroon', linestyle="--", linewidth=.5)

# plt.text(22.8, 114, r'Median: 22 carries', {'color': 'maroon', 'fontsize': 9})

# plt.tight_layout()
# plt.show();



# tf = df.query("NflIdRusher == NflId")


# sns.set_style("white", {'grid.linestyle': '--'})

# fig, ax = plt.subplots(figsize=(10,8))
# ax.set_xlim(-10,26)


# sns.distplot(tf.Yards, kde=False, color="b", bins=100,
#             hist_kws={"linewidth": .9, 'edgecolor':'grey'})

# #########################ax.set_xlim(80,100)

# ## Remove the x-tick labels:  plt.xticks([])
# plt.yticks([])
# ## This method also hides the tick marks


# plt.title('\nOverall distribution of yards gained during an individual running play\n',fontsize=12)
# plt.xlabel('\nYards (yd) Gained -->\n')
# sns.despine(top=True, right=True, left=True, bottom=True)
# plt.tight_layout()
# ##################plt.axvline(x=22, color='maroon', linestyle="--", linewidth=.5)

# ###############plt.text(22.8, 114, r'Median: 22 carries', {'color': 'maroon', 'fontsize': 9})
# plt.tight_layout()
# plt.show();

# # tf.Yards.describe()

tf = df.query("NflIdRusher == NflId")
sns.set_style("ticks", {'grid.linestyle': '--'})
######sns.set_style("ticks", {"xtick.major.size":1,"ytick.major.size":1})
flierprops = dict(markerfacecolor='0.75', 
                  markersize=1,
                  linestyle='none')

fig, ax = plt.subplots(figsize=(9,7))

ax.set_ylim(-7, 14)
ax.set_title('Yards Gained by Down\n', fontsize=12)

sns.boxplot(x='Down',
            y='Yards',
            data=tf,
            ax=ax,
            showfliers=False, 
            #color='blue'
            )
            #flierprops=flierprops)
    
#Completely hide tick markers...
# ax.yaxis.set_major_locator(plt.NullLocator())
# ax.xaxis.set_major_formatter(plt.NullFormatter())

ax.yaxis.grid(True)   # Show the horizontal gridlines
ax.xaxis.grid(False)  # Hide x-axis gridlines 

ax.yaxis.set_major_locator(plt.MultipleLocator(1))
ax.yaxis.set_minor_locator(plt.MultipleLocator(5))


# Add transparency to colors
for patch in ax.artists:
  r, g, b, a = patch.get_facecolor()
  patch.set_facecolor((r, g, b, .3))
    
    
##ax.legend(frameon=False)

# ax.set(xlabel=''common xlabel', ylabel='common ylabel', title='some title')

ax.set(xlabel='')
ax.set_xticklabels(['1st Down', '2nd Down', '3rd Down', '4th Down'])
plt.tight_layout(); plt.show();
# #value swirl
# YDS_by_down = tf.groupby("Down")['Yards'].size()
# total_run_plays = YDS_by_down.sum()
# df_ydsbydown = pd.DataFrame( {'Down':YDS_by_down.index, 'Count':YDS_by_down.values}).sort_values('Count', ascending=False)

# s = df.style.background_gradient(cmap=cm)
# # zebra = df_ydsbydown.style.set_caption('Top 10 Percentage of plays by Personnel (top 10):').background_gradient(cmap=cm)
# # display(HTML(df_ydsbydown.to_html(index=False)))

# #s = df.style.background_gradient(cmap=cm)
# df_ydsbydown.style.set_caption('Count of plays by Personnel:').background_gradient(cmap=cm)

# temp101 = pd.DataFrame(tf.DefensePersonnel.value_counts())
# temp101.index.name = 'Down'
# temp101.columns=['Play Count']
# temp101.reset_index('')
# cm = sns.light_palette("green", as_cmap=True)
# #s = df.style.background_gradient(cmap=cm)
# temp101.style.set_caption('Count of plays by Personnel:').background_gradient(cmap=cm)



# VANILLA SWIRL
YDS_by_down = tf.groupby("Down")['Yards'].size()
total_run_plays = YDS_by_down.sum()
df_ydsbydown = pd.DataFrame( {'Down':YDS_by_down.index, 'Count':YDS_by_down.values}).sort_values('Count', ascending=False)
df_ydsbydown.set_index('Down', drop=True, inplace=True)
# s = df.style.background_gradient(cmap=cm)
# df_ydsbydown.style.set_caption('Play count per Down:').background_gradient(cmap=cm)
df_ydsbydown['Percentage']=round(df_ydsbydown.Count/total_run_plays*100, 2)
cm = sns.light_palette("green", as_cmap=True)
df_ydsbydown.style.set_caption('PLAY COUNT PER DOWN:').background_gradient(cmap=cm)


# print(df_ydsbydown)
# s = df.style.background_gradient(cmap=cm)
# # zebra = df_ydsbydown.style.set_caption('Top 10 Percentage of plays by Personnel (top 10):').background_gradient(cmap=cm)
# # display(HTML(df_ydsbydown.to_html(index=False)))
# #s = df.style.background_gradient(cmap=cm)
# df_ydsbydown.style.set_caption('Count of plays by Personnel:').background_gradient(cmap=cm)

# temp107 = pd.DataFrame(round(tf.DefensePersonnel.value_counts(normalize=True) * 100,2)).head(10)
# temp107.index.name = 'DefensePersonnel'
# temp107.columns=['Play Percentage']
# cm = sns.light_palette("green", as_cmap=True)
# #s = df.style.background_gradient(cmap=cm)
# temp107.style.set_caption('Top 10 Percentage of plays by Personnel (top 10):').background_gradient(cmap=cm)
# sns.boxplot(x='Down',
#             y='Yards',
#             data=tf,
#             ax=ax,
#             showfliers=False, 
#             #color='blue'
#             )
tf = df.query("NflIdRusher == NflId")
flierprops = dict(markerfacecolor='0.75', 
                  markersize=1,
                  linestyle='none')
fig, ax = plt.subplots(figsize=(9,7))
ax.set_ylim(-7, 17)
ax.set_title('Yards Gained by Game Quarter\n\n', fontsize=12)

sns.boxplot(x='Quarter',
            y='Yards',
            data=tf,
            ax=ax,
            showfliers=False , 
            #color='blue'
            )
            #flierprops=flierprops)
    
#Completely hide tick markers...
# ax.yaxis.set_major_locator(plt.NullLocator())
# ax.xaxis.set_major_formatter(plt.NullFormatter())

ax.yaxis.grid(True)   # Show the horizontal gridlines
ax.xaxis.grid(False)  # Hide x-axis gridlines 

ax.yaxis.set_major_locator(plt.MultipleLocator(1))
ax.yaxis.set_minor_locator(plt.MultipleLocator(5))

# Add transparency to colors
for patch in ax.artists:
  r, g, b, a = patch.get_facecolor()
  patch.set_facecolor((r, g, b, .3))
    
# ax.set(xlabel=''common xlabel', ylabel='common ylabel', title='some title')
ax.set(xlabel='')
ax.set_xticklabels(['1Q', '2Q', '3Q', '4Q', '5Q'])
plt.tight_layout(); plt.show();

dff = tf[tf.DefendersInTheBox>2]
dff.DefendersInTheBox = dff.DefendersInTheBox.astype('int')

flierprops = dict(markerfacecolor='0.75', 
                  markersize=1,
                  linestyle='none')

fig, ax = plt.subplots(figsize=(9,7))
ax.set_ylim(-7, 23)
ax.set_title('Yards Gained vs number of Defenders in the box\n\n', fontsize=12)
sns.boxplot(x='DefendersInTheBox',
            y='Yards',
            data=dff,
            ax=ax,
            showfliers=False , 
            #color='blue'
            )
            #flierprops=flierprops)
#Completely hide tick markers...
# ax.yaxis.set_major_locator(plt.NullLocator())
# ax.xaxis.set_major_formatter(plt.NullFormatter())

ax.yaxis.grid(True)   # Show the horizontal gridlines
ax.xaxis.grid(False)  # Hide x-axis gridlines 

ax.yaxis.set_major_locator(plt.MultipleLocator(1))
ax.yaxis.set_minor_locator(plt.MultipleLocator(5))

# Add transparency to colors
for patch in ax.artists:
  r, g, b, a = patch.get_facecolor()
  patch.set_facecolor((r, g, b, .3))
    
# ax.set(xlabel=''common xlabel', ylabel='common ylabel', title='some title')
ax.set(xlabel="\nNumber of defensive players in the 'Box'\n\n")
# ax.set_xticklabels(['1Q', '2Q', '3Q', '4Q', '5Q'])

# plt.title('My subtitle',fontsize=16)
# plt.suptitle('My title',fontsize=24, x=0, y=1,ha="left")
# plt.text(x=4.7, y=4.7, s='Sepal Length vs Width', fontsize=16, weight='bold')
# plt.text(x=4.7, y=4.6, s='The size of each point corresponds to sepal width', fontsize=8, alpha=0.75)

plt.tight_layout()
plt.show();


temp101 = pd.DataFrame(tf.DefensePersonnel.value_counts())
temp101.index.name = 'DefensePersonnel'
temp101.columns=['Play Count']

cm = sns.light_palette("green", as_cmap=True)

#s = df.style.background_gradient(cmap=cm)

temp101.style.set_caption('Count of plays by Personnel:').background_gradient(cmap=cm)


# tf.DefensePersonnel.value_counts()
# 4 DL, 2 LB, 5 DB          6358
# 4 DL, 3 LB, 4 DB          6205
# 3 DL, 4 LB, 4 DB          3656
# 2 DL, 4 LB, 5 DB          2588
# 3 DL, 3 LB, 5 DB          2222
# 2 DL, 3 LB, 6 DB           529
# 4 DL, 1 LB, 6 DB           418
# 4 DL, 4 LB, 3 DB           237
# 3 DL, 2 LB, 6 DB           193
# 5 DL, 2 LB, 4 DB           161
# 5 DL, 3 LB, 3 DB           108
# 1 DL, 4 LB, 6 DB            65
# 3 DL, 5 LB, 3 DB            64
# 6 DL, 4 LB, 1 DB            56
# 5 DL, 4 LB, 2 DB            53
# 6 DL, 3 LB, 2 DB            47
# 5 DL, 1 LB, 5 DB            41
# 6 DL, 2 LB, 3 DB            32
# 1 DL, 5 LB, 5 DB            31
# 2 DL, 5 LB, 4 DB            22
# 1 DL, 3 LB, 7 DB            13
# 2 DL, 2 LB, 7 DB            13
# 3 DL, 1 LB, 7 DB            12
# 5 DL, 5 LB, 1 DB             7
# 5 DL, 3 LB, 2 DB, 1 OL       7
# 0 DL, 5 LB, 6 DB             6
# 4 DL, 5 LB, 2 DB             5
# 0 DL, 4 LB, 7 DB             4
# 4 DL, 0 LB, 7 DB             3
# 2 DL, 4 LB, 4 DB, 1 OL       3
# 5 DL, 4 LB, 1 DB, 1 OL       3
# 4 DL, 6 LB, 1 DB             2
# 0 DL, 6 LB, 5 DB             2
# 6 DL, 1 LB, 4 DB             1
# 3 DL, 4 LB, 3 DB, 1 OL       1
# 4 DL, 5 LB, 1 DB, 1 OL       1
# 1 DL, 2 LB, 8 DB             1
# 7 DL, 2 LB, 2 DB             1


# tf.DefensePersonnel.sort_values().unique()
# array(['0 DL, 4 LB, 7 DB', '0 DL, 5 LB, 6 DB', '0 DL, 6 LB, 5 DB',
#        '1 DL, 2 LB, 8 DB', '1 DL, 3 LB, 7 DB', '1 DL, 4 LB, 6 DB',
#        '1 DL, 5 LB, 5 DB', '2 DL, 2 LB, 7 DB', '2 DL, 3 LB, 6 DB',
#        '2 DL, 4 LB, 4 DB, 1 OL', '2 DL, 4 LB, 5 DB', '2 DL, 5 LB, 4 DB',
#        '3 DL, 1 LB, 7 DB', '3 DL, 2 LB, 6 DB', '3 DL, 3 LB, 5 DB',
#        '3 DL, 4 LB, 3 DB, 1 OL', '3 DL, 4 LB, 4 DB', '3 DL, 5 LB, 3 DB',
#        '4 DL, 0 LB, 7 DB', '4 DL, 1 LB, 6 DB', '4 DL, 2 LB, 5 DB',
#        '4 DL, 3 LB, 4 DB', '4 DL, 4 LB, 3 DB', '4 DL, 5 LB, 1 DB, 1 OL',
#        '4 DL, 5 LB, 2 DB', '4 DL, 6 LB, 1 DB', '5 DL, 1 LB, 5 DB',
#        '5 DL, 2 LB, 4 DB', '5 DL, 3 LB, 2 DB, 1 OL', '5 DL, 3 LB, 3 DB',
#        '5 DL, 4 LB, 1 DB, 1 OL', '5 DL, 4 LB, 2 DB', '5 DL, 5 LB, 1 DB',
#        '6 DL, 1 LB, 4 DB', '6 DL, 2 LB, 3 DB', '6 DL, 3 LB, 2 DB',
#        '6 DL, 4 LB, 1 DB', '7 DL, 2 LB, 2 DB'], dtype=object)

temp107 = pd.DataFrame(round(tf.DefensePersonnel.value_counts(normalize=True) * 100,2)).head(10)
temp107.index.name = 'DefensePersonnel'
temp107.columns=['Percentage']
cm = sns.light_palette("green", as_cmap=True)

#s = df.style.background_gradient(cmap=cm)
temp107.style.set_caption('Top 10 Percentage of plays by Defensive Personnel Grouping:').background_gradient(cmap=cm)

sns.set_style("ticks", {'grid.linestyle': '--'})

pers = tf
dff = pers 

flierprops = dict(markerfacecolor='0.2', 
                  markersize=1,
                  linestyle='none')

fig, ax = plt.subplots(figsize=(9,12))
ax.set_ylim(-7, 22)
ax.set_title('\nAverage yards gained by Defensive Personnel Schema\n', fontsize=12)



# sns.boxplot(y='DefensePersonnel',
#             x='Yards',
#             data=dff,
#             ax=ax,
#             showfliers=False , 
#             #color='blue'
#             )

sns.boxplot(y=dff['DefensePersonnel'].sort_values(ascending=False),
            x=dff['Yards'],
            ax=ax,
            showfliers=False ,
            linewidth=.8
            #color='blue'
            )


            #flierprops=flierprops)
#Completely hide tick markers...
# ax.yaxis.set_major_locator(plt.NullLocator())
# ax.xaxis.set_major_formatter(plt.NullFormatter())

ax.yaxis.grid(False)   # Show the horizontal gridlines
ax.xaxis.grid(True)  # Hide x-axis gridlines 

ax.xaxis.set_major_locator(plt.MultipleLocator(1))
#ax.yaxis.set_minor_locator(plt.MultipleLocator(5))

# Add transparency to colors
for patch in ax.artists:
  r, g, b, a = patch.get_facecolor()
  patch.set_facecolor((r, g, b, .3))
    
# ax.set(xlabel=''common xlabel', ylabel='common ylabel', title='some title')
ax.set(xlabel="\nYards Gained\n")

# plt.title('My subtitle',fontsize=16)
# plt.suptitle('My title',fontsize=24, x=0, y=1,ha="left")
# plt.text(x=4.7, y=4.7, s='Sepal Length vs Width', fontsize=16, weight='bold')
# plt.text(x=4.7, y=4.6, s='The size of each point corresponds to sepal width', fontsize=8, alpha=0.75)

# ax.spines['top'].set_linewidth(0)  
# ax.spines['left'].set_linewidth(.3)  
# ax.spines['right'].set_linewidth(0)  
# ax.spines['bottom'].set_linewidth(.3) 
sns.despine(top=True, right=True, left=True, bottom=True)
ax.xaxis.set_ticks_position('none') 

# ax.fill_between(t, upper_bound, X, facecolor='blue', alpha=0.5)
# plt.axhspan(9,10)  #horizontal shading
# plt.axvspan(9,10)  #horizontal shading

#ax.text(15,78, "#1", ha='center')

ax.text(15,17.3, '#1',
        verticalalignment='bottom', horizontalalignment='right',
        color='green', fontsize=12)

ax.text(15,16.3, '#2',
        verticalalignment='bottom', horizontalalignment='right',
        color='green', fontsize=11)

ax.text(15,21.3, '#3',
        verticalalignment='bottom', horizontalalignment='right',
        color='green', fontsize=11)

ax.text(15,24.3, '#5',
        verticalalignment='bottom', horizontalalignment='right',
        color='green', fontsize=11)

ax.text(15,27.3, '#4',
        verticalalignment='bottom', horizontalalignment='right',
        color='green', fontsize=11)

ax.text(9,2, '6 guys on the line',
        verticalalignment='bottom', horizontalalignment='right',
        color='green', fontsize=11)


ax.text(0,.2, 'line of scrimmage >',
        verticalalignment='bottom', horizontalalignment='right',
        color='blue', fontsize=9)


#-----more control-----#
ax.grid(linestyle='--', 
        linewidth='0.3', 
        color='lightgray', 
        alpha=0.8,
        axis='x'
       )

plt.axvline(0, 0,1, linewidth=.4, color="blue", linestyle="--")

plt.tight_layout()
plt.show();


# pers = tf
# dff = pers 

# # sns.boxplot(y=dff['DefensePersonnel'].sort_values(ascending=False),
#             x=dff['Yards'],
#             ax=ax,
#             showfliers=False ,
#             linewidth=.8
#             #color='blue'
#             )

# ax.yaxis.grid(False)   # Show the horizontal gridlines
# ax.xaxis.grid(True)  # Hide x-axis gridlines 

# ax.xaxis.set_major_locator(plt.MultipleLocator(1))
# #ax.yaxis.set_minor_locator(plt.MultipleLocator(5))

# # Add transparency to colors
# for patch in ax.artists:
#   r, g, b, a = patch.get_facecolor()
#   patch.set_facecolor((r, g, b, .3))
    
# # ax.set(xlabel=''common xlabel', ylabel='common ylabel', title='some title')
# ax.set(xlabel="\nYards Gained\n")

# sns.despine(top=True, right=True, left=True, bottom=True)
# ax.xaxis.set_ticks_position('none') 

# # ax.fill_between(t, upper_bound, X, facecolor='blue', alpha=0.5)
# # plt.axhspan(9,10)  #horizontal shading
# # plt.axvspan(9,10)  #horizontal shading

# #ax.text(15,78, "#1", ha='center')

# #-----more control-----#
# ax.grid(linestyle='--', 
#         linewidth='0.3', 
#         color='lightgray', 
#         alpha=0.8,
#         axis='x'
#        )

# plt.axvline(0, 0,1, linewidth=.4, color="blue", linestyle="--")
# plt.tight_layout()
# plt.show();




# from bokeh.plotting import figure, output_file, show

# p = figure(plot_width=800, plot_height=700,
#            title = '\nYards by Distance\n',
#            x_axis_label ='Distance to Go\n',
#            y_axis_label ='Yards\n')

# p.circle(dff.Distance,
#          dff['Yards'],
#          size=5, 
#          color="navy", 
#          alpha=0.5)


# show(p); 






# from numpy import linspace
# from scipy.stats.kde import gaussian_kde
# from bokeh.io import output_file, show
# from bokeh.models import ColumnDataSource, FixedTicker, PrintfTickFormatter
# from bokeh.plotting import figure
# from bokeh.sampledata.perceptions import probly
# import colorcet as cc


# def ridge(category, data, scale=20):
#     return list(zip([category]*len(data), scale*data))

# cats = list(reversed(probly.keys()))

# palette = [cc.rainbow[i*15] for i in range(17)]

# x = linspace(-20,110, 500)

# source = ColumnDataSource(data=dict(x=x))

# p = figure(y_range=cats, plot_width=700, x_range=(-5, 105), toolbar_location=None)

# for i, cat in enumerate(reversed(cats)):
#     pdf = gaussian_kde(probly[cat])
#     y = ridge(cat, pdf(x))
#     source.add(y, cat)
#     p.patch('x', cat, color=palette[i], alpha=0.6, line_color="black", source=source)

# p.outline_line_color = None
# p.background_fill_color = "#efefef"

# p.xaxis.ticker = FixedTicker(ticks=list(range(0, 101, 10)))
# p.xaxis.formatter = PrintfTickFormatter(format="%d%%")

# p.ygrid.grid_line_color = None
# p.xgrid.grid_line_color = "#dddddd"
# p.xgrid.ticker = p.xaxis[0].ticker

# p.axis.minor_tick_line_color = None
# p.axis.major_tick_line_color = None
# p.axis.axis_line_color = None

# p.y_range.range_padding = 0.12

# show(p); 







pers = tf
dff = pers 


# sns.boxplot(y=dff['DefensePersonnel'].sort_values(ascending=False),
#             x=dff['Yards'],
#             ax=ax,
#             showfliers=False ,
#             linewidth=.8
#             #color='blue'
#             )

# ax.yaxis.grid(False)   # Show the horizontal gridlines
# ax.xaxis.grid(True)  # Hide x-axis gridlines 

# ax.xaxis.set_major_locator(plt.MultipleLocator(1))
# #ax.yaxis.set_minor_locator(plt.MultipleLocator(5))

# # Add transparency to colors
# for patch in ax.artists:
#   r, g, b, a = patch.get_facecolor()
#   patch.set_facecolor((r, g, b, .3))
    
# # ax.set(xlabel=''common xlabel', ylabel='common ylabel', title='some title')
# ax.set(xlabel="\nYards Gained\n")

# sns.despine(top=True, right=True, left=True, bottom=True)
# ax.xaxis.set_ticks_position('none') 

# # ax.fill_between(t, upper_bound, X, facecolor='blue', alpha=0.5)
# # plt.axhspan(9,10)  #horizontal shading
# # plt.axvspan(9,10)  #horizontal shading

# #ax.text(15,78, "#1", ha='center')

# #-----more control-----#
# ax.grid(linestyle='--', 
#         linewidth='0.3', 
#         color='lightgray', 
#         alpha=0.8,
#         axis='x'
#        )

# plt.axvline(0, 0,1, linewidth=.4, color="blue", linestyle="--")
# plt.tight_layout()
# plt.show();



from bokeh.plotting import figure, output_file, show

p = figure(plot_width=800, plot_height=700,
           title = '\nYards by Rusher Weight\n',
           x_axis_label ='Rusher Weight (in lbs)\n',
           y_axis_label ='Yards\n')

p.circle(dff.PlayerWeight,
         dff['Yards'],
         size=5, 
         color="navy", 
         alpha=0.5)


show(p); 


#MDK

# # sns.set(style="white", palette="muted", color_codes=True)
# sns.set_style("ticks", {'grid.linestyle': '--'})


# t = tf[['Week', "Yards"]].groupby('Week').mean().sort_values(by = "Yards")
# ###sns.set_style("white", {'grid.linestyle': '--'})
# fig, ax = plt.subplots(figsize=(8,7))

# sns.barplot(x=t.index,
#             y=t.Yards,
#             ax=ax, 
#             linewidth=.5, 
#             facecolor=(1, 1, 1, 0),
#             errcolor=".2", 
#             edgecolor=".2")

        
# # for patch in ax.artists:
# #     r, g, b, a = patch.get_facecolor()
# #     patch.set_facecolor((r, g, b, .1))
    
# #~~~~~~~~~~~ ax.set ~~~~~~~~~~~~~~~~
# ##ax.set_xlim(0, 6)
# ax.set_ylim(2, 6)
# ax.set_title('Average yards gained per play as the season progresses (week by week)\n', fontsize=12)
# ax.set(ylabel='Yards Gained\n')
# ax.set(xlabel='\nWeek Number (in the season)')
# ax.yaxis.grid(True)   # Show the horizontal gridlines
# ax.xaxis.grid(False)  # Hide x-axis gridlines 
# ax.yaxis.set_major_locator(plt.MultipleLocator(1))
# ax.yaxis.set_minor_locator(plt.MultipleLocator(5))
# #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ 

# plt.tight_layout()
# plt.show();


# clrs = ['grey' if (x < max(values)) else 'green' for x in values ]
# sns.barplot(x=labels, y=values, palette=clrs) # color=clrs)
# #Rotate x-labels 
# plt.xticks(rotation=40)

sns.set(style="white", palette="muted", color_codes=True)

#sns.set_style("ticks", {'grid.linestyle': '--'})
# # this may not work right
# sns.set_style({'grid.linestyle': '--'}, )

t = tf[['Week', "Yards"]].groupby('Week').mean().sort_values(by = "Yards")
###sns.set_style("white", {'grid.linestyle': '--'})

fig, ax = plt.subplots(figsize=(9,6))

specific_colors=['grey']*17
specific_colors[8]='#ffbf00'
specific_colors[5]='#169016'

#print(specific_colors)

#sns.set_color_codes('pastel')             
sns.barplot(x=t.index,
            y=t.Yards,
            ax=ax, 
            linewidth=.2, 
            #color='red'
            #facecolor='#888888',
            #facecolor=(1, 1, 1, 0),
            #facecolor='specific_colors',
            #errcolor=".2",
            edgecolor="black",
            palette=specific_colors)

#~~~~~~~~~~~ ax.set ~~~~~~~~~~~~~~~~
##ax.set_xlim(0, 6)
ax.set_ylim(0, 5.5)
ax.set_title('\nOverall Average yards gained per play as the season progresses (week by week)\n\n', fontsize=11)
# ax.set(ylabel='Yards Gained\n', rotation='horizontal')
ax.set(xlabel='\nWeek Number (in the season)')
ax.yaxis.grid(True)   # Show the horizontal gridlines
ax.xaxis.grid(False)  # Hide x-axis gridlines 
ax.yaxis.set_major_locator(plt.MultipleLocator(1))
ax.yaxis.set_minor_locator(plt.MultipleLocator(5))
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ 

#-----more control-----#
ax.grid(linestyle='--', 
        linewidth='0.7', 
        color='lightgray', 
        alpha=0.9,
        axis='y'
       )

# Don't allow the axis to be on top of your data
# ax.set_axisbelow(True)

for patch in ax.artists:
    r, g, b, a = patch.get_facecolor()
    patch.set_facecolor((r, g, b, .3))
ax.spines['top'].set_linewidth(0)  
ax.spines['left'].set_linewidth(.3)  
ax.spines['right'].set_linewidth(0)  
ax.spines['bottom'].set_linewidth(.3) 

plt.ylabel("YDS\n", fontsize=11, rotation=90)

plt.tight_layout()
plt.show()



#----------------------------------------------------------------------------




#https://seaborn.pydata.org/generated/seaborn.lineplot.html
t = tf[['Week', "Yards"]].groupby('Week').mean().sort_values(by = "Yards")
t['WeekInSeason']= t.index
t.reset_index(drop=True, inplace=True)
starter= t.loc[0,'Yards']
t['gain']=t.Yards/starter
t['gainpct']=round(100*(t.gain-1), 3)


fig, ax = plt.subplots(figsize=(9.5,5))


sns.lineplot(x="WeekInSeason", y="gainpct", data=t, 
            color='grey', 
            ax=ax,
            markers=True, marker='o', 
            #palette=specific_colors, 
            dashes=True) 

ax.set_title('\nPercent Gain in the average running yards gained per play (week by week)\n\n', fontsize=11)

# ax.xaxis.set_major_locator(plt.MultipleLocator(13))
# ax.yaxis.set_minor_locator(plt.MultipleLocator(5))

ax.xaxis.set_major_locator(plt.MultipleLocator(1))
ax.xaxis.set_minor_locator(plt.MultipleLocator(1))
ax.set(ylabel='Gain in average YDS per carry (in %)\n')

ax.set(xlabel='\nWeek Number (in the season)')
ax.yaxis.grid(True)   # Show the horizontal gridlines
ax.xaxis.grid(False)  # Hide x-axis gridlines 


ax.spines['top'].set_linewidth(0)  
ax.spines['left'].set_linewidth(.3)  
ax.spines['right'].set_linewidth(0)  
ax.spines['bottom'].set_linewidth(.3); 

plt.tight_layout()
plt.show(); 


sns.set_style("white", {'grid.linestyle': '--'})

# sns.set_style("ticks", {'grid.linestyle': '--'})
##sns.set(style="white", palette="muted", color_codes=True)
##sns.set(style="white", palette="muted", color_codes=True)

t2 = tf.groupby(['GameId','Team'])['PlayId'].count()
t2 = pd.DataFrame(t2)
fig, ax = plt.subplots(figsize=(9,7))

sns.distplot(t2.PlayId, kde=False, color="b", 
            hist_kws={"linewidth": .9, 'edgecolor':'black'}, bins=24)


# #~~~~~~~~~~~ ax.set ~~~~~~~~~~~~~~~~
# ##ax.set_xlim(0, 6)
# ##ax.set_ylim(0, 6)
# ax.set_title('Average yards gained as the season progresses (week by week)\n')
# ax.set(ylabel='Yards Gained\n')
# ax.set(xlabel='\nWeek Number (in the season)')
# ax.yaxis.grid(True)   # Show the horizontal gridlines
# ax.xaxis.grid(False)  # Hide x-axis gridlines 
# # ax.yaxis.set_major_locator(plt.MultipleLocator(1))
# # ax.yaxis.set_minor_locator(plt.MultipleLocator(5))
# #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

## Remove the x-tick labels:  plt.xticks([])
plt.yticks([])
## This method also hides the tick marks
plt.title('\nDistribution of total number of run plays on a game basis (per team)\n',
          fontsize=12, loc="left")
plt.title('\nDistribution of total number of run plays on a game basis (per team)\n',fontsize=12, loc="left")
plt.xlabel('\nNumber of times a team ran the ball in the game\n', fontsize=9)
sns.despine(top=True, right=True, left=True, bottom=True)
plt.axvline(x=22, color='maroon', linestyle="--", linewidth=.5)

plt.text(22.8, 114, r'Median: 22 carries', {'color': 'maroon', 'fontsize': 9})
plt.gca().xaxis.set_major_locator(plt.MultipleLocator(5))


plt.tight_layout()
plt.show();

#number_plays_2018 = bbb.groupby(['GameId'], as_index=False).agg({'PlayId': 'nunique'})
number_plays_2018_perteam = bbb.groupby(['GameId', 'Team'], as_index=False).agg({'PlayId': 'nunique'})

sns.set_style("white", {'grid.linestyle': '--'})
fig, ax = plt.subplots(figsize=(7,7))

#Completely hide tick markers...
# ax.yaxis.set_major_locator(plt.NullLocator())
# ax.xaxis.set_major_formatter(plt.NullFormatter())

#ax.yaxis.grid(True)   # Show the horizontal gridlines
ax.xaxis.grid(True)  # Hide x-axis gridlines 

ax.xaxis.set_major_locator(plt.MultipleLocator(5))
#ax.yaxis.set_minor_locator(plt.MultipleLocator(5))


sns.swarmplot(number_plays_2018_perteam.PlayId, color="b", ax=ax)
sns.despine(top=True, right=True, left=True, bottom=True)

plt.ylabel('The Number of Teams that ran the x-axis play count value\n', fontsize=10)

plt.xlabel('\nTotal Run Plays by a Team in an entire game', fontsize=10)
plt.title('\n2018 Season: Number of Run Plays Distribution by Team\n',fontsize=12, loc="left")


# - - - - - - - - 
plt.tight_layout()
plt.show();


# Add transparency to colors
# for patch in ax.artists:
#   r, g, b, a = patch.get_facecolor()
#   patch.set_facecolor((r, g, b, .3))


plt.style.use('dark_background')


df04 = tf.groupby('PossessionTeam')['Yards'].agg(sum).sort_values(ascending=True)
df04 = pd.DataFrame(df04)
df04['group'] = df04.index

my_range=range(1,33)

fig, ax = plt.subplots(figsize=(9,9))

# Create a color if the group is "B"
##my_color=np.where(df04['group']=='NE', 'orange', 'skyblue')

##my_color=np.where(df04[  ('group'=='NE') | ('group'=='NO')  ], 'orange', 'skyblue')

my_color=np.where( (df04.group == 'NE') | (df04.group == 'NO') | (df04.group == 'LA') , 'orange', 'skyblue')

##movies[(movies.duration >= 200) | (movies.genre == 'Drama')]
##df04[(df04.group == 'NE') | (df04.group == 'NO') ]
##(movies.duration >= 200) & (movies.genre == 'Drama')

my_size=np.where(df04['group']=='B', 70, 30)
 
plt.hlines(y=my_range, xmin=0, xmax=df04['Yards'], color=my_color, alpha=0.4)

plt.scatter(df04.Yards, my_range, color=my_color, s=my_size, alpha=1)
 
# Add title and exis names
plt.yticks(my_range, df04.group)
plt.title("\nTotal Rushing Yards per Team \n(over the course of two NFL seasons)\n\n", loc='left', fontsize=12)
plt.xlabel('\n Total Rushing Yards', fontsize=10)
plt.ylabel('')
##############plt.ylabel('NFL\nTeam\n')

ax.spines['top'].set_linewidth(.3)  
ax.spines['left'].set_linewidth(.3)  
ax.spines['right'].set_linewidth(.3)  
ax.spines['bottom'].set_linewidth(.3)  


plt.text(0, 33.3, r'Top Three:  LA Rams, New England Patriots, and New Orleans Saints absolutely dominating the rushing game...', {'color': 'white', 'fontsize': 8.5})
sns.despine(top=True, right=True, left=True, bottom=True)

plt.text(4005, 2, '<-- I call these icicles', {'color': 'white', 'fontsize': 8})

plt.axvline(x=3500, color='lightgrey', ymin = .01, ymax=.82, linestyle="--", linewidth=.4)
plt.axvline(x=4000, color='lightgrey', ymin = .01, ymax=.9, linestyle="--", linewidth=.4)
plt.axvline(x=3000, color='lightgrey', ymin = .01, ymax=.43, linestyle="--", linewidth=.4)
plt.axvline(x=2500, color='lightgrey', ymin = .01, ymax=.07, linestyle="--", linewidth=.4)

plt.tight_layout()
plt.show();

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
sns.set_style("white", {'grid.linestyle': '--'})
# sns.set_style("ticks", {'grid.linestyle': '--'})
##sns.set(style="white", palette="muted", color_codes=True)
##sns.set(style="white", palette="muted", color_codes=True)
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
speed = bbb.groupby(['DisplayName'])['S'].agg('max').sort_values(ascending=True)
speed = pd.DataFrame(speed)
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
fig, ax = plt.subplots(figsize=(9,7))
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
sns.distplot(speed, kde=False, color="m", 
 hist_kws={"linewidth": .9, 'edgecolor':'lightgrey'}, bins=38)
sns.despine(top=True, right=True, left=True, bottom=True)
ax.xaxis.set_major_locator(plt.MultipleLocator(1))
ax.set_yticklabels([])
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
plt.title('\nDistribution of running speed for all players in the 2017/2018 seasons (yds/s)\n',
           fontsize=12, loc="center")
ax.set(xlabel="\nRunner Speed (yds/sec)\n")
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
plt.tight_layout()
plt.show();
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~




# # #~~~~~~~~~~~ ax.set ~~~~~~~~~~~~~~~~
# # ##ax.set_xlim(0, 6)
# # ##ax.set_ylim(0, 6)
# # ax.set_title('Average yards gained as the season progresses (week by week)\n')
# # ax.set(ylabel='Yards Gained\n')
# # ax.set(xlabel='\nWeek Number (in the season)')
# # ax.yaxis.grid(True)   # Show the horizontal gridlines
# # ax.xaxis.grid(False)  # Hide x-axis gridlines 
# # # ax.yaxis.set_major_locator(plt.MultipleLocator(1))
# # # ax.yaxis.set_minor_locator(plt.MultipleLocator(5))
# # #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

# ## Remove the x-tick labels:  plt.xticks([])
# plt.yticks([])
# ## This method also hides the tick marks
# plt.title('\nDistribution of total number of run plays on a game basis (per team)\n',
#           fontsize=12, loc="left")
# plt.title('\nDistribution of total number of run plays on a game basis (per team)\n',fontsize=12, loc="left")
# plt.xlabel('\nNumber of times the ball was run in the game\n')
# plt.tight_layout()
# plt.axvline(x=22, color='maroon', linestyle="--", linewidth=.5)

# plt.text(22.8, 114, r'Median: 22 carries', {'color': 'maroon', 'fontsize': 9})
# plt.gca().xaxis.set_major_locator(plt.MultipleLocator(5))


# plt.tight_layout()
# plt.show()


# ttt = bbb[bbb.Position.isin(['RB','WR','HB'])]

# ttt['kg']=ttt["PlayerWeight"] * 0.45359237
# ttt['Force_Newtons']=ttt['kg'] * ttt['A'] * 0.9144
# tips = ttt[['Force_Newtons', 'Yards']]

# sns.scatterplot(x="Force_Newtons", y="Yards", data=tips, s=1, ax=ax, color='r', markers='o', edgecolor='r')
# #sns.lmplot(x="Force_Newtons", y="Yards", data=tips)

# plt.title('Correlation between Yards Gained and Player Kinetic Force',fontsize=12)
# plt.suptitle('Kinetic Force',fontsize=13, x=0, y=1,ha="left")
# ##plt.text(x=4.7, y=14.7, s='Sepal Length vs Width', fontsize=10, weight='bold')

# ax.set(xlabel="\nPlayer Kinetic Force\n")
# ax.set(ylabel="Yards Gained\n")

# sns.despine(top=True, right=True, left=True, bottom=True)


# # dff = tf[tf.DefendersInTheBox>2]
# # dff.DefendersInTheBox = dff.DefendersInTheBox.astype('int')

# # flierprops = dict(markerfacecolor='0.75', 
# #                   markersize=1,
# #                   linestyle='none')

# # fig, ax = plt.subplots(figsize=(9,7))
# # ax.set_ylim(-7, 23)
# # ax.set_title('Yards Gained vs number of Defenders in the box\n\n', fontsize=12)
# # sns.boxplot(x='DefendersInTheBox',
# #             y='Yards',
# #             data=dff,
# #             ax=ax,
# #             showfliers=False , 
# #             #color='blue'
# #             )
# #             #flierprops=flierprops)
# # #Completely hide tick markers...
# # ax.yaxis.set_major_locator(plt.NullLocator())
# # ax.xaxis.set_major_formatter(plt.NullFormatter())

# # ax.yaxis.grid(False)   # Show the horizontal gridlines
# # ax.xaxis.grid(False)  # Hide x-axis gridlines 

# # ax.yaxis.set_minor_locator(plt.MultipleLocator(5))

# # # Add transparency to colors
# # for patch in ax.artists:
# #   r, g, b, a = patch.get_facecolor()
# #   patch.set_facecolor((r, g, b, .3))
    
# # # ax.set(xlabel=''common xlabel', ylabel='common ylabel', title='some title')
# # ax.set(xlabel="\nNumber of defensive players in the 'Box'\n\n")
# # # ax.set_xticklabels(['1Q', '2Q', '3Q', '4Q', '5Q'])

# # # plt.title('My subtitle',fontsize=16)
# # # plt.suptitle('My title',fontsize=24, x=0, y=1,ha="left")
# # # plt.text(x=4.7, y=4.7, s='Sepal Length vs Width', fontsize=16, weight='bold')
# # # plt.text(x=4.7, y=4.6, s='The size of each point corresponds to sepal width', fontsize=8, alpha=0.75)

# # plt.tight_layout()
# # plt.show();


# # PLOT THE RAW RUNNER WEIGHT ? 

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
sns.set_style("white", {'grid.linestyle': '--'})
# sns.set_style("ticks", {'grid.linestyle': '--'})
##sns.set(style="white", palette="muted", color_codes=True)
##sns.set(style="white", palette="muted", color_codes=True)
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
speed = bbb.groupby(['DisplayName'])['S'].agg('max').sort_values(ascending=True)
speed = pd.DataFrame(speed)
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
fig, ax = plt.subplots(figsize=(9,7))
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
sns.distplot(speed*2.04545, kde=False, color="orange", 
 hist_kws={"linewidth": .8, 'edgecolor':'black'}, bins=38)
sns.despine(top=True, right=True, left=True, bottom=True)
ax.xaxis.set_major_locator(plt.MultipleLocator(1))
ax.set_yticklabels([])
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
plt.title('\nDistribution of running speed for all players in the 2017/2018 seasons (mph)\n',
           fontsize=12, loc="center")
ax.set(xlabel="\nRunner Speed (miles per hour mph)\n")
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
plt.tight_layout()
plt.show();
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~




# # #~~~~~~~~~~~ ax.set ~~~~~~~~~~~~~~~~
# # ##ax.set_xlim(0, 6)
# # ##ax.set_ylim(0, 6)
# # ax.set_title('Average yards gained as the season progresses (week by week)\n')
# # ax.set(ylabel='Yards Gained\n')
# # ax.set(xlabel='\nWeek Number (in the season)')
# # ax.yaxis.grid(True)   # Show the horizontal gridlines
# # ax.xaxis.grid(False)  # Hide x-axis gridlines 
# # # ax.yaxis.set_major_locator(plt.MultipleLocator(1))
# # # ax.yaxis.set_minor_locator(plt.MultipleLocator(5))
# # #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

# ## Remove the x-tick labels:  plt.xticks([])
# plt.yticks([])
# ## This method also hides the tick marks
# plt.title('\nDistribution of total number of run plays on a game basis (per team)\n',
#           fontsize=12, loc="left")
# plt.title('\nDistribution of total number of run plays on a game basis (per team)\n',fontsize=12, loc="left")
# plt.xlabel('\nNumber of times the ball was run in the game\n')
# plt.tight_layout()
# plt.axvline(x=22, color='maroon', linestyle="--", linewidth=.5)

# plt.text(22.8, 114, r'Median: 22 carries', {'color': 'maroon', 'fontsize': 9})
# plt.gca().xaxis.set_major_locator(plt.MultipleLocator(5))


# plt.tight_layout()
# plt.show()


# ttt = bbb[bbb.Position.isin(['RB','WR','HB'])]

# ttt['kg']=ttt["PlayerWeight"] * 0.45359237
# ttt['Force_Newtons']=ttt['kg'] * ttt['A'] * 0.9144
# tips = ttt[['Force_Newtons', 'Yards']]

# sns.scatterplot(x="Force_Newtons", y="Yards", data=tips, s=1, ax=ax, color='r', markers='o', edgecolor='r')
# #sns.lmplot(x="Force_Newtons", y="Yards", data=tips)

# plt.title('Correlation between Yards Gained and Player Kinetic Force',fontsize=12)
# plt.suptitle('Kinetic Force',fontsize=13, x=0, y=1,ha="left")
# ##plt.text(x=4.7, y=14.7, s='Sepal Length vs Width', fontsize=10, weight='bold')

# ax.set(xlabel="\nPlayer Kinetic Force\n")
# ax.set(ylabel="Yards Gained\n")

# sns.despine(top=True, right=True, left=True, bottom=True)


# # dff = tf[tf.DefendersInTheBox>2]
# # dff.DefendersInTheBox = dff.DefendersInTheBox.astype('int')

# # flierprops = dict(markerfacecolor='0.75', 
# #                   markersize=1,
# #                   linestyle='none')

# # fig, ax = plt.subplots(figsize=(9,7))
# # ax.set_ylim(-7, 23)
# # ax.set_title('Yards Gained vs number of Defenders in the box\n\n', fontsize=12)
# # sns.boxplot(x='DefendersInTheBox',
# #             y='Yards',
# #             data=dff,
# #             ax=ax,
# #             showfliers=False , 
# #             #color='blue'
# #             )
# #             #flierprops=flierprops)
# # #Completely hide tick markers...
# # ax.yaxis.set_major_locator(plt.NullLocator())
# # ax.xaxis.set_major_formatter(plt.NullFormatter())

# # ax.yaxis.grid(False)   # Show the horizontal gridlines
# # ax.xaxis.grid(False)  # Hide x-axis gridlines 

# # ax.yaxis.set_minor_locator(plt.MultipleLocator(5))

# # # Add transparency to colors
# # for patch in ax.artists:
# #   r, g, b, a = patch.get_facecolor()
# #   patch.set_facecolor((r, g, b, .3))
    
# # # ax.set(xlabel=''common xlabel', ylabel='common ylabel', title='some title')
# # ax.set(xlabel="\nNumber of defensive players in the 'Box'\n\n")
# # # ax.set_xticklabels(['1Q', '2Q', '3Q', '4Q', '5Q'])

# # # plt.title('My subtitle',fontsize=16)
# # # plt.suptitle('My title',fontsize=24, x=0, y=1,ha="left")
# # # plt.text(x=4.7, y=4.7, s='Sepal Length vs Width', fontsize=16, weight='bold')
# # # plt.text(x=4.7, y=4.6, s='The size of each point corresponds to sepal width', fontsize=8, alpha=0.75)

# # plt.tight_layout()
# # plt.show();


# # PLOT THE RAW RUNNER WEIGHT ? 
# bbb.OffenseFormation.value_counts()
# SINGLEBACK    4920
# SHOTGUN       3626
# I_FORM        2058
# PISTOL         340
# JUMBO          244
# WILDCAT         66
# EMPTY           15

my_x = bbb.groupby('OffenseFormation')['Yards'].mean().sort_values(ascending=False).values
my_y = bbb.groupby('OffenseFormation')['Yards'].mean().index

## original !  deleting for sec sns.set(style="white", palette="muted", color_codes=True)
#sns.set(style="white", palette="muted", color_codes=True)
# sns.set_style("ticks", {'grid.linestyle': '--'})
sns.set(style="white", palette="muted", color_codes=True)

#sns.set_style("ticks", {'grid.linestyle': '--'})
# # this may not work right
# sns.set_style({'grid.linestyle': '--'}, )

##t = tf[['Week', "Yards"]].groupby('Week').mean().sort_values(by = "Yards")
###sns.set_style("white", {'grid.linestyle': '--'})


fig, ax = plt.subplots(figsize=(9,7))

sns.barplot(x=my_y,
            y=my_x,
            ax=ax, 
            linewidth=.2, 
            edgecolor="black")

    
#~~~~~~~~~~~ ax.set ~~~~~~~~~~~~~~~~
##ax.set_xlim(0, 6)
##ax.set_ylim(2, 6)
ax.set_title('\n2018: Avg YDS gained per playOffense Formations\n', fontsize=12)
# ax.set(ylabel='Yards Gained\n', rotation='horizontal')
ax.set(xlabel='\nOffense Formations')
ax.yaxis.grid(True)   # Show the horizontal gridlines
ax.xaxis.grid(False)  # Hide x-axis gridlines 
ax.yaxis.set_major_locator(plt.MultipleLocator(1))
#ax.yaxis.set_minor_locator(plt.MultipleLocator(5))
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ 

#  I am the author of all of this work:  Tom Bresee (this is my notebook)

#-----more control-----#
ax.grid(linestyle='--', 
        linewidth='0.9', 
        color='lightgray', 
        alpha=0.9,
        axis='y'
       )

# ax.set_axisbelow(True)

plt.ylabel("Avg YDS gained per Play\n", rotation=90)
sns.despine(top=True, right=True, left=True, bottom=True)


# Add transparency to colors
for patch in ax.artists:
  r, g, b, a = patch.get_facecolor()
  patch.set_facecolor((r, g, b, .3))
    
plt.tight_layout()
plt.show();


aaa = gold
aaa['IsRunner'] = aaa.NflId == aaa.NflIdRusher
bbb = aaa[aaa.IsRunner & (aaa.Season == 2018)]


fig, ax = plt.subplots(figsize=(10,6))
ax.set_xlim(150,380)
ax.set_title('2018 Season: Player Weight distribution (Runners vs Non-Runners)\n\n', fontsize=12)

sns.kdeplot(bbb.PlayerWeight, shade=True, color="orange", ax=ax)
sns.kdeplot(aaa[~aaa.IsRunner & (aaa.Season == 2018)].PlayerWeight, shade=True, color='blue', ax=ax)

ax.xaxis.set_major_locator(plt.MultipleLocator(10))
ax.xaxis.set_minor_locator(plt.MultipleLocator(5))

# Add transparency to colors
for patch in ax.artists:
  r, g, b, a = patch.get_facecolor()
  patch.set_facecolor((r, g, b, .3))
    
    
####plt.text(x=4.7, y=4.7, s='Sepal Length vs Width', fontsize=16, weight='bold')
# plt.text(x=4.7, y=4.6, s='The size of each point corresponds to sepal width', fontsize=8, alpha=0.75)
sns.despine(top=True, right=True, left=True, bottom=True)

# Turn off tick labels
ax.set_yticklabels([])
#ax.set_xticklabels([])


ax.set(xlabel="\nPlayer Weight\n\n")
plt.legend(title='Category:  Ran the ball, or did not ever run the ball', loc='upper right', labels=['Runners', 'Non-Runners'])
plt.tight_layout()
plt.show();

#
#Creating a playah profile, as a reference df:
#
#
player_profile=aaa.loc[:,['DisplayName','Position','NflId' 'PlayerBirthDate', 'PlayerWeight', 'PlayerCollegeName']].drop_duplicates()
player_profile_2018=aaa[aaa.Season==2018]
player_profile_2018 = player_profile_2018.loc[: ,['DisplayName','Position','NflId' 'PlayerBirthDate', 'PlayerWeight', 'PlayerCollegeName'] ].drop_duplicates()
#
#
# len(player_profile)
# len(player_profile_2018)
#
#
#
player_profile_2018["kg"] = player_profile_2018["PlayerWeight"] * 0.45359237
#
#
##player_profile_2018.PlayerCollegeName.value_counts()
#
#
z = player_profile_2018.groupby('Position')['PlayerWeight'].agg(['min', 'median', 'mean', 'max']).round(1).sort_values(by=['median'], 
                                                                                                                   ascending=False)
z['Avg Mass (kg)'] = (z['mean'] * 0.45359237).round(1)
z

# ####dfv = gold.loc[:,['NflId', 'DisplayName', 'PlayerBirthDate', 'PlayerWeight', 'PlayerHeight']].drop_duplicates()
# #
# #
# #
# #Plot
# sns.distplot(bbb, kde=False, color="b", 
#             hist_kws={"linewidth": .9, 'edgecolor':'lightgrey'}, bins=24)
# #
# #
# #
# #
# fig, ax = plt.subplots(figsize=(12, 8))
# #
# ax.set_xlim(150,380)
# ax.set_title('2018 Season: Player Weight distribution (Runners vs Non-Runners)\n\n', fontsize=12)


# sns.kdeplot(bbb.PlayerWeight, shade=True, color="orange", ax=ax)
# sns.kdeplot(aaa[~aaa.IsRunner & (aaa.Season == 2018)].PlayerWeight, shade=True, color='blue', ax=ax)

# ax.xaxis.set_major_locator(plt.MultipleLocator(10))
# ax.xaxis.set_minor_locator(plt.MultipleLocator(5))

# # Add transparency to colors
# for patch in ax.artists:
#   r, g, b, a = patch.get_facecolor()
#   patch.set_facecolor((r, g, b, .3))
    
    
# ####plt.text(x=4.7, y=4.7, s='Sepal Length vs Width', fontsize=16, weight='bold')
# # plt.text(x=4.7, y=4.6, s='The size of each point corresponds to sepal width', fontsize=8, alpha=0.75)
# sns.despine(top=True, right=True, left=True, bottom=True)

# # Turn off tick labels
# ax.set_yticklabels([])
# #ax.set_xticklabels([])


# ax.set(xlabel="\nPlayer Weight\n\n")
# plt.legend(title='Category:  Ran the ball, or did not ever run the ball', loc='upper right', labels=['Runners', 'Non-Runners'])
# plt.tight_layout()
# plt.show();
# bbb.Position.value_counts()
# RB    10476
# WR      372
# HB      327
# QB       40
# FB       35
# TE       16
# CB        3
# DE        1
# DT        1
#---------------------------
#10476/11271=93%
#---------------------------

from scipy import constants
plt.style.use('dark_background')
fig, ax = plt.subplots(figsize=(10,10))

#Specifically only using runners like RB, HB, and WR...
ttt = bbb[bbb.Position.isin(['RB','WR','HB'])]


# the kg column will be the true 'mass' of the body
# convert weight to kg and then divide by g to get the true mass 
ttt['kg']=ttt["PlayerWeight"] * 0.45359237 / scipy.constants.g


# the momentum is just mass (in kg) X speed in m/s (so convert from yards/sec to mps)
ttt['True Momentum']=ttt['kg'] * ttt['S'] * 0.9144 
tips = ttt[['True Momentum', 'Yards']]

sns.scatterplot(x="True Momentum", y="Yards", data=tips, s=4, ax=ax, color='cyan', markers='.', edgecolors='cyan', alpha=.8)
##sns.lmplot(x="Force_Newtons", y="Yards", data=tips, facecolor='cyan', edgecolors='cyan')

plt.title('Correlation between Yards Gained and Player Momentum\n',fontsize=11)
plt.suptitle('Kinetic Momentum',fontsize=10, x=0, y=1,ha="left")
##plt.text(x=4.7, y=14.7, s='Sepal Length vs Width', fontsize=10, weight='bold')

ax.set(xlabel="Player Kinetic Momentum $\Rightarrow$\n")
ax.set(ylabel="Yards Gained  $\Rightarrow$\n")

sns.despine(top=True, right=True, left=True, bottom=True)


# dff = tf[tf.DefendersInTheBox>2]
# dff.DefendersInTheBox = dff.DefendersInTheBox.astype('int')

# flierprops = dict(markerfacecolor='0.75', 
#                   markersize=1,
#                   linestyle='none')

# fig, ax = plt.subplots(figsize=(9,7))
# ax.set_ylim(-7, 23)
# ax.set_title('Yards Gained vs number of Defenders in the box\n\n', fontsize=12)
# sns.boxplot(x='DefendersInTheBox',
#             y='Yards',
#             data=dff,
#             ax=ax,
#             showfliers=False , 
#             #color='blue'
#             )
#             #flierprops=flierprops)
# #Completely hide tick markers...
# ax.yaxis.set_major_locator(plt.NullLocator())
# ax.xaxis.set_major_formatter(plt.NullFormatter())

# ax.yaxis.grid(False)   # Show the horizontal gridlines
# ax.xaxis.grid(False)  # Hide x-axis gridlines 

# ax.yaxis.set_major_locator(plt.MultipleLocator(1))
# ax.yaxis.set_minor_locator(plt.MultipleLocator(5))

# # Add transparency to colors
# for patch in ax.artists:
#   r, g, b, a = patch.get_facecolor()
#   patch.set_facecolor((r, g, b, .3))
    
# # ax.set(xlabel=''common xlabel', ylabel='common ylabel', title='some title')
# ax.set(xlabel="\nNumber of defensive players in the 'Box'\n\n")
# # ax.set_xticklabels(['1Q', '2Q', '3Q', '4Q', '5Q'])

# # plt.title('My subtitle',fontsize=16)
# # plt.suptitle('My title',fontsize=24, x=0, y=1,ha="left")
# # plt.text(x=4.7, y=4.7, s='Sepal Length vs Width', fontsize=16, weight='bold')
# # plt.text(x=4.7, y=4.6, s='The size of each point corresponds to sepal width', fontsize=8, alpha=0.75)

# plt.tight_layout()
# plt.show();

plt.xticks([])
#plt.yticks([])


# sns.relplot(x="True Momentum", y="Yards", data=tips, s=4, ax=ax, color='cyan', markers='.', edgecolors='cyan', alpha=.4)
# ##sns.lmplot(x="Force_Newtons", y="Yards", data=tips, facecolor='cyan', edgecolors='cyan')

plt.tight_layout()
plt.show();


plt.style.use('dark_background')
fig, ax = plt.subplots(figsize=(10,10))

#Specifically only using runners like RB, HB, and WR...
ttt = bbb[bbb.Position.isin(['RB','WR','HB'])]

# the kg column will be the true 'mass' of the body
# convert weight to kg and then divide by g to get the true mass 
ttt['kg']=ttt["PlayerWeight"] * 0.45359237 / scipy.constants.g


# the momentum is just mass (in kg) X speed in m/s (so convert from yards/sec to mps)
ttt['True Momentum']=ttt['kg'] * ttt['S'] * 0.9144 
tips = ttt[['True Momentum', 'Yards']]

sns.scatterplot(x="True Momentum", y="Yards", data=tips, s=4, ax=ax, color='cyan', markers='.', edgecolors='cyan', alpha=.8)
##sns.lmplot(x="Force_Newtons", y="Yards", data=tips, facecolor='cyan', edgecolors='cyan')

plt.title('Correlation between Yards Gained beyond 6 and Player Momentum\n',fontsize=11)
plt.suptitle('Kinetic Momentum',fontsize=10, x=0, y=1,ha="left")
##plt.text(x=4.7, y=14.7, s='Sepal Length vs Width', fontsize=10, weight='bold')

ax.set(xlabel="Player Kinetic Momentum $\Rightarrow$\n")
ax.set(ylabel="Yards Gained  $\Rightarrow$\n")

sns.despine(top=True, right=True, left=True, bottom=True)


# dff = tf[tf.DefendersInTheBox>2]
# dff.DefendersInTheBox = dff.DefendersInTheBox.astype('int')

# flierprops = dict(markerfacecolor='0.75', 
#                   markersize=1,
#                   linestyle='none')

# fig, ax = plt.subplots(figsize=(9,7))
ax.set_ylim(6,100)
# ax.set_title('Yards Gained vs number of Defenders in the box\n\n', fontsize=12)
# sns.boxplot(x='DefendersInTheBox',
#             y='Yards',
#             data=dff,
#             ax=ax,
#             showfliers=False , 
#             #color='blue'
#             )
#             #flierprops=flierprops)
# #Completely hide tick markers...
# ax.yaxis.set_major_locator(plt.NullLocator())
# ax.xaxis.set_major_formatter(plt.NullFormatter())

# ax.yaxis.grid(False)   # Show the horizontal gridlines
# ax.xaxis.grid(False)  # Hide x-axis gridlines 

# ax.yaxis.set_major_locator(plt.MultipleLocator(1))
# ax.yaxis.set_minor_locator(plt.MultipleLocator(5))

# # Add transparency to colors
# for patch in ax.artists:
#   r, g, b, a = patch.get_facecolor()
#   patch.set_facecolor((r, g, b, .3))
    
# # ax.set(xlabel=''common xlabel', ylabel='common ylabel', title='some title')
# ax.set(xlabel="\nNumber of defensive players in the 'Box'\n\n")
# # ax.set_xticklabels(['1Q', '2Q', '3Q', '4Q', '5Q'])

# # plt.title('My subtitle',fontsize=16)
# # plt.suptitle('My title',fontsize=24, x=0, y=1,ha="left")
# # plt.text(x=4.7, y=4.7, s='Sepal Length vs Width', fontsize=16, weight='bold')
# # plt.text(x=4.7, y=4.6, s='The size of each point corresponds to sepal width', fontsize=8, alpha=0.75)

# plt.tight_layout()
# plt.show();

plt.xticks([])
#plt.yticks([])

plt.tight_layout()
plt.show();

plt.style.use('dark_background')
fig, ax = plt.subplots(figsize=(10,10))

ttt = bbb[bbb.Position.isin(['RB','WR','HB'])]


# true mass in kg 
ttt['kg']=ttt["PlayerWeight"] * 0.45359237 / scipy.constants.g

# F = ma 
ttt['Force_Newtons']=ttt['kg'] * ttt['A'] * 0.9144
tips = ttt[['Force_Newtons', 'Yards']]

sns.scatterplot(x="Force_Newtons", y="Yards", data=tips, s=1, ax=ax, color='r', markers='o', edgecolor='r')
#sns.lmplot(x="Force_Newtons", y="Yards", data=tips)

plt.title('\nCorrelation between Yards Gained and Player Kinetic Force\n',fontsize=11)
plt.suptitle('Kinetic Force',fontsize=10, x=0, y=1,ha="left")
##plt.text(x=4.7, y=14.7, s='Sepal Length vs Width', fontsize=10, weight='bold')

ax.set(xlabel="Player Kinetic Force$\Rightarrow$\n")
ax.set(ylabel="Yards Gained $\Rightarrow$\n")
sns.despine(top=True, right=True, left=True, bottom=True)



# dff = tf[tf.DefendersInTheBox>2]
# dff.DefendersInTheBox = dff.DefendersInTheBox.astype('int')

# flierprops = dict(markerfacecolor='0.75', 
#                   markersize=1,
#                   linestyle='none')

# fig, ax = plt.subplots(figsize=(9,7))
# ax.set_ylim(-7, 23)
# ax.set_title('Yards Gained vs number of Defenders in the box\n\n', fontsize=12)
# sns.boxplot(x='DefendersInTheBox',
#             y='Yards',
#             data=dff,
#             ax=ax,
#             showfliers=False , 
#             #color='blue'
#             )
#             #flierprops=flierprops)
# #Completely hide tick markers...
# ax.yaxis.set_major_locator(plt.NullLocator())
# ax.xaxis.set_major_formatter(plt.NullFormatter())

# ax.yaxis.grid(False)   # Show the horizontal gridlines
# ax.xaxis.grid(False)  # Hide x-axis gridlines 

# ax.yaxis.set_major_locator(plt.MultipleLocator(1))
# ax.yaxis.set_minor_locator(plt.MultipleLocator(5))

# # Add transparency to colors
# for patch in ax.artists:
#   r, g, b, a = patch.get_facecolor()
#   patch.set_facecolor((r, g, b, .3))
    
# # ax.set(xlabel=''common xlabel', ylabel='common ylabel', title='some title')
# ax.set(xlabel="\nNumber of defensive players in the 'Box'\n\n")
# # ax.set_xticklabels(['1Q', '2Q', '3Q', '4Q', '5Q'])

# # plt.title('My subtitle',fontsize=16)
# # plt.suptitle('My title',fontsize=24, x=0, y=1,ha="left")
# # plt.text(x=4.7, y=4.7, s='Sepal Length vs Width', fontsize=16, weight='bold')
# # plt.text(x=4.7, y=4.6, s='The size of each point corresponds to sepal width', fontsize=8, alpha=0.75)

# plt.tight_layout()
# plt.show();
plt.xticks([])

plt.tight_layout()
plt.show();

from IPython.display import YouTubeVideo
# The Hill
# Video credit: Beastly67.
YouTubeVideo('dqmxWZ8Rbwc')
#
#
#  YR2.count()  -  all the value counts for each col 
#
#  df['Churn'].value_counts(normalize=True)
#
#
# len(df.NflId.unique())                    1788
# len(df[df.Season==2017].NflId.unique())   1783
# len(df[df.Season==2018].NflId.unique())   2231
# delta between these numbers:              448
#
#
#   len(df[df.Season==2017.DisplayName.unique())   .  1780        1788 
#   len(df[df.Season==2018].DisplayName.unique())  .  1782        1783
#   len(YRS.DisplayName.unique())                  .  2230.448delta
#
#
#    509762 total rows of data
#    509762/22 = 23,171 rows of true runner data 
#    23,171 split between 2018 and 2017 for about 11k each 
#    1696 + practice players = total allowed in the NFL per season
#    1856 if you allow practice players 
#    Remove these columns uniformly
#  killed_columns=['xyz','etc']
#
#
# YRS = dontbreak[dontbreak.NflId==dontbreak.NflIdRusher].copy()
# YR1 = YRS[YRS.Season==2017]
# YR2 = YRS[YRS.Season==2018]
#
#len(YRS)==len(YR1)+len(YR2)
#
#
# df_play.drop('Yards', axis=1)
#
#
#---------------------------------
def drop_these_columns(your_df,your_list):
    #KILL KOLUMNS
    your_df.drop(your_list,axis=1,inplace=True)
#---------------------------------
#
#
#
#
allcolumns = """
['GameId',
 'PlayId',
 'Team',
 'X',
 'Y',
 'S',
 'A',
 'Dis',
 'Orientation',
 'Dir',
 'NflId',
 'DisplayName',
 'JerseyNumber',
 'Season',
 'YardLine',
 'Quarter',
 'GameClock',
 'PossessionTeam',
 'Down',
 'Distance',
 'FieldPosition',
 'HomeScoreBeforePlay',
 'VisitorScoreBeforePlay',
 'NflIdRusher',
 'OffenseFormation',
 'OffensePersonnel',
 'DefendersInTheBox',
 'DefensePersonnel',
 'PlayDirection',
 'TimeHandoff',
 'TimeSnap',
 'Yards',
 'PlayerHeight',
 'PlayerWeight',
 'PlayerBirthDate',
 'PlayerCollegeName',
 'Position',
 'HomeTeamAbbr',
 'VisitorTeamAbbr',
 'Week',
 'Stadium',
 'Location',
 'StadiumType',
 'Turf',
 'GameWeather',
 'Temperature',
 'Humidity',
 'WindSpeed',
 'WindDirection',
 'IsRunner']
"""
#
# def common_elements(list1, list2): 
#     return [element for element in list1 if element in list2]
#
# z1 = df[df.Season==2017].NflId.unique()
# z2 = df[df.Season==2018].NflId.unique()
# len(common_elements(z1,z2))
#
#
# len(YR1.columns)
# len(YR2.columns)
#---------------------------------
kill=['WindSpeed','WindDirection','StadiumType','Temperature','GameWeather']
drop_these_columns(YR1,kill)
drop_these_columns(YR2, kill)
drop_these_columns(YRS,kill)
#---------------------------------
# len(YR1.columns)
# len(YR2.columns)
#
#
#YR2.PlayDirection.value_counts()
#
#------------------------------------------------------------------------------------------
#  EVENTUALLY REMOVE THE YARDS, THE NFLID AND THE NFLIDRUSHER AS THEY WILL NOT BE NEEDED...
#     df.drop(['Yards'], axis=1, inplace=True)
#     df.drop(['NflId', 'NflIdRusher'], axis=1, inplace=True)
#
#
#
# to strip a particular columns   -    df[0] = df[0].str.strip()
#YRS.count()
#YRS.nunique()
# for i in YRS.columns:
#     print(YRS[i].value_counts())
#YRS.nunique()
#
#
#

# #  Create master missing count and percentage chart for YRS: 
# missing_values_count = YRS.isnull().sum()
# missing=pd.DataFrame(missing_values_count[missing_values_count != 0])
# missing.columns=['MissingCount']
# missing['% Missing'] = round(missing.MissingCount/23171*100,2)
# missing = missing.sort_values(by='% Missing', ascending=False)
# missing.index.name='Both Years Combined'
# missing


# #  Create master missing count and percentage chart for YRS: 
# missing_values_count17 = YR1.isnull().sum()
# missing17=pd.DataFrame(missing_values_count17[missing_values_count17 != 0])
# missing17.columns=['MissingCount']
# missing17['% Missing'] = round(missing17.MissingCount/23171*100,2)
# missing17 = missing17.sort_values(by='% Missing', ascending=False)
# missing17.index.name='2017'
# missing17

# #  Create master missing count and percentage chart for YRS: 
# missing_values_count18 = YR2.isnull().sum()
# missing18=pd.DataFrame(missing_values_count18[missing_values_count18 != 0])
# missing18.columns=['MissingCount']
# missing18['% Missing'] = round(missing18.MissingCount/23171*100,2)
# missing18 = missing18.sort_values(by='% Missing', ascending=False)
# missing18.index.name='2018'
# missing18

# def showmeaplay():
#     display(df.iloc[0:22:,0:20])
#     print("")
#     display(df.iloc[0:22:,21:33])
#     print("")
#     display(df.iloc[0:22:,34:49])
# with pd.option_context('display.max_rows', None, 'display.max_columns', None):
#     showmeaplay()


def find(player):
    plyr = dontbreak.loc[:,['DisplayName',
                                      'Position',
                                      'JerseyNumber',
                                      'PlayerWeight',
                                      'PlayerHeight',
                                      'PlayerBirthDate',
                                      'NflId',
                                      'PlayerCollegeName']].drop_duplicates()
    output = plyr[plyr.DisplayName==str(player)]
    output.index.name='Player Profile'
    #output.reset_index()
    #print(output.columns)
    #display(HTML(output.T.to_html(index=False)))
    return output.T
    # ADD:  PLAYER AGE

person_I_want_to_show = 'Tom Brady'  # enter whatever name you want here ... 
find(person_I_want_to_show).style.set_properties(**{'text-align': 'center'}).set_table_styles([dict(selector='th', props=[('text-align', 'right')])])

# # N = 9
# # y = np.linspace(-2, 2, N)
# # x = y**2
# # source = ColumnDataSource(dict(y=y, right=x,))

# # p  = Plot(title=None, plot_width=300, plot_height=300,
# #      min_border=0, toolbar_location=None)

# # glyph = HBar(y="y", right="right", left=0, height=0.5, fill_color="#b3de69")

# # p.add_glyph(source, glyph)

# # xaxis = LinearAxis()

# # p.add_layout(xaxis, 'below')

# # yaxis = LinearAxis()
# # p.add_layout(yaxis, 'left')

# # p.add_layout(Grid(dimension=0, ticker=xaxis.ticker))
# # p.add_layout(Grid(dimension=1, ticker=yaxis.ticker))
# # curdoc().add_root(p)
# # output_notebook(hide_banner=True)
# # show(p); 

# my_data = df[['PlayerCollegeName','NflId', 'DisplayName']].drop_duplicates().copy()

# college_attended = my_data["PlayerCollegeName"].value_counts()

# df_cc = pd.DataFrame({'CollegeName':college_attended.index, 'Count':college_attended.values}).sort_values("Count", ascending = False)

# #df_cc.Count.astype('int', inplace=True)

# df_cc = df_cc[df_cc.CollegeName != 'Louisiana State']

# df_cc.at[42,'Count']=51
# #df_cc[df_cc.CollegeName=='LSU']['Count']

# df_cc.sort_values('Count', )


# pd.set_option('display.max_rows', 500)

# #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# p = figure(
#   y_range=df_cc.CollegeName[:40],          
#   title = '\nNumber of players that attended Colleges Attended - Player Count\n',
#   x_axis_label ='# of NFL players that attended the college prior\n',
#   plot_width=500,
#   plot_height=700,
#   tools="", toolbar_location=None)
#   #min_border=0
# #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~    
# p.hbar(
#     y=df_cc.CollegeName[:40],              
#     right=df_cc.Count[:40],   
#     left=0,
#     height=0.4,
#     color='orange',
#     fill_alpha=0.4
# )
# #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# p.xaxis.major_tick_line_color = None  # turn off x-axis major ticks
# p.xaxis.minor_tick_line_color = None  # turn off x-axis minor ticks
# p.yaxis.major_tick_line_color = None  # turn off y-axis major ticks
# p.yaxis.minor_tick_line_color = None  # turn off y-axis minor ticks
# #p.xaxis.major_label_text_font_size = '0pt'  # preferred method for removing tick labels
# #p.yaxis.major_label_text_font_size = '0pt'  # preferred method for removing tick labels
# #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# ###readmore:  https://medium.com/@deallen7/visualizing-data-with-pythons-bokeh-package-310315d830bb
# output_notebook(hide_banner=True)
# #show(p)


# #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# # SUPERHBAR:  i started learning bokeh two days ago, so this sucks 
# # To draw horizontal bars by specifying a (center) y-coordinate, height, and left and right
# # endpoints, use the hbar() glyph function:
# #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# my_data = df[['PlayerCollegeName','NflId', 'DisplayName']].drop_duplicates().copy()

# college_attended = my_data["PlayerCollegeName"].value_counts()

# df_cc = pd.DataFrame({'CollegeName':college_attended.index, 'Count':college_attended.values}).sort_values("Count", ascending = False)

# #df_cc.Count.astype('int', inplace=True)

# df_cc = df_cc[df_cc.CollegeName != 'Louisiana State']

# df_cc.at[42,'Count']=51

# #df_cc[df_cc.CollegeName=='LSU']['Count']

# df_cc.sort_values('Count',ascending=False, inplace=True)

# #pd.set_option('display.max_rows', 500)
# df_cc.index = df_cc.index + 1


# #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# mysource = ColumnDataSource(df_cc)
# #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# p = figure(
#   y_range=df_cc.CollegeName[:50],    # I need to enter the SAME thing here as y points, i find that odd 
#   # wait:  can i set this as the range, but not below ? ? ? 
#   # i think caegorical just list in a list the categories here 
#   title = '\nNFL Player Count by College Attended\n',
#   x_axis_label ='# of NFL players that attended the college prior\n',
#   plot_width=700,
#   plot_height=800,
#   tools="hover",       # or tools="" 
#   toolbar_location=None,   
#   #background_fill_color="#efe8e2")
#   #min_border=0))
# )
# #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~    
# p.hbar(
#     y='CollegeName',  # center of your y coordinate launcher, 40 points as def above ... 
#     left=0, # or left=20, etc
#     right='Count',    # right is 40 points... 
#     height=0.8,
#     alpha=.6,
#     #color='orange',    #color=Spectral3  #color=Blues8,   
#     #background_fill_color="#efe8e2", 
#     #     fill_color=Blues8,
#     #     fill_alpha=0.4, 
#     source = mysource, 
#     line_color='blue'   # line_coolor='red'
# ) 
# #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# # TITLE: 
# # p.title.text = 'Current frame:'
# # p.title.text_color = TEXT_COLOR
# # p.title.text_font = TEXT_FONT
# p.title.text_font_size = '11pt'
# # p.title.text_font_style = 'normal'
# #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# # AXES: 
# # p.xaxis.major_tick_line_color = None  # turn off x-axis major ticks
# # p.xaxis.minor_tick_line_color = None  # turn off x-axis minor ticks
# # p.yaxis.major_tick_line_color = None  # turn off y-axis major ticks
# # p.yaxis.minor_tick_line_color = None  # turn off y-axis minor ticks
# # p.xaxis.axis_line_color = None    # or 'red'
# # p.yaxis.axis_line_color = GRID_COLOR 
# #
# # X-TICKS:
# # p.xaxis[0].ticker = FixedTicker(ticks=[0, 1])
# # p.xaxis.major_tick_line_color = GRID_COLOR
# # p.xaxis.major_label_text_font_size = '7pt'
# # p.xaxis.major_label_text_font = TEXT_FONT
# # p.xaxis.major_label_text_color = None   #TEXT_COLOR
# #
# # Y-TICKS:
# # p.yaxis[0].ticker = FixedTicker(ticks=np.arange(1, len(labels) + 1, 1).tolist())
# # p.yaxis.major_label_text_font_size = '0pt'
# p.yaxis.major_tick_line_color = None
# p.axis.minor_tick_line_color = None  # turn off y-axis minor ticks

# # p.xaxis.major_label_text_font_size = '0pt'  # preferred method for removing tick labels
# # p.yaxis.major_label_text_font_size = '0pt'  # preferred method for removing tick labels
# p.yaxis.major_tick_line_color = None  # turn off y-axis major ticks
# p.yaxis.minor_tick_line_color = None  # turn off y-axis minor ticks
# #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# # GRID:
# # p.xgrid.grid_line_color = None
# p.ygrid.grid_line_color = None   
# #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# # # LEGENDgend.location = 'top_left'
# # p.legend.orientation='vertical'
# # p.legend.location='top_right'
# # p.legend.label_text_font_size='10px'
# #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# ### NOTES here> 
# #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# # HOVER:
# #     hover.names = ['bars']
# #     hover.tooltips = [
# #         ('Event', '@label'),
# #         ('Probability', '@pretty_value')]
# #
# hover = HoverTool()
# #p.select(HoverTool).tooltips = [("x1","@x1"), ("x2","@x2")]
# #
# # hover.tooltips = [
# #         ('Event', '@label')
# #         #('Probability', '@pretty_value'),
# #     ]
# # hover.tooltips = [
# #     ("Total:", "@Count")
# #     #("x1", "@x1"),
# #     #("Totals", "@TONS_HE High Explosive / @TONS_IC Incendiary / @TONS_FRAG Fragmentation")
# #     ]
# ###########################hover.mode = 'vline'
# #????curdoc().add_root(p)
# # hover.tooltips = """
# #     <div>
# #         <br>
# #         <h4>@CollegeName:</h4>
# #         <div><strong>Count: &ensp; </strong>@Count</div>
# #     </div>
# # """
# hover.tooltips = [
#     ("College Name:", "@CollegeName"),
#     ("Ranking by Count", "$index"),
#     ("Number of gradutes that entered the NFL:", "@Count"),
# ]
# #<div><strong>HP: </strong>@Horsepower</div>       
# p.add_tools(hover)
# #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# output_notebook(hide_banner=True)
# show(p); 
# #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# # hover.tooltips = [
# #     ("index", "$index"),
# #     ("(x,y)", "($x, $y)"),
# #     ("radius", "@radius"),
# #     ("fill color", "$color[hex, swatch]:fill_color"),
# #     ("foo", "@foo"),
# #     ("bar", "@bar"),
# # ]




# ### older:
# #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# #HBAR
# #    need y and 'right' (i.e. x) values
# #To draw horizontal bars by specifying a (center) y-coordinate, height, and left and right endpoints, use the hbar() glyph function:
# #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


# my_data = df[['PlayerCollegeName','NflId', 'DisplayName']].drop_duplicates().copy()

# college_attended = my_data["PlayerCollegeName"].value_counts()

# df_cc = pd.DataFrame({'CollegeName':college_attended.index, 'Count':college_attended.values}).sort_values("Count", ascending = False)

# df_cc = df_cc[df_cc.CollegeName != 'Louisiana State']

# df_cc.at[42,'Count']=51

# #df_cc[df_cc.CollegeName=='LSU']['Count']

# df_cc.sort_values('Count', ascending=False)

# pd.set_option('display.max_rows', 500)


# #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# mysource = ColumnDataSource(df_cc)
# #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# p = figure(
#   y_range=df_cc.CollegeName[:50],    # I need to enter the SAME thing here as y points, i find that odd        
#   title = '\nNumber of players that attended Colleges Attended - Player Count\n',
#   x_axis_label ='# of NFL players that attended the college prior\n',
#   plot_width=500,
#   plot_height=700,
#   tools="", toolbar_location=None)
#   #min_border=0
# #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~    
# p.hbar(
#     y=df_cc.CollegeName[:40],  # center of your y coordinate launcher, 40 points... 
#     left=0, # or left=20, etc
#     right=df_cc.Count[:40],    # right is 40 points... 
#     height=0.4,
#     color='orange',
#     fill_alpha=0.4,)
#     #source = mysource)
# #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# # Axes: 
# p.xaxis.major_tick_line_color = None  # turn off x-axis major ticks
# p.xaxis.minor_tick_line_color = None  # turn off x-axis minor ticks
# p.yaxis.major_tick_line_color = None  # turn off y-axis major ticks
# p.yaxis.minor_tick_line_color = None  # turn off y-axis minor ticks
# #p.xaxis.major_label_text_font_size = '0pt'  # preferred method for removing tick labels
# #p.yaxis.major_label_text_font_size = '0pt'  # preferred method for removing tick labels
# # Grid: 
# p.xgrid.grid_line_color = None
# p.ygrid.grid_line_color = None   
# #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# ### NOTES here: 
# #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# output_notebook(hide_banner=True)
# show(p); 
# #~~~~~~