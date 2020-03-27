
## load libraries
from plotly.offline import init_notebook_mode, iplot
from plotly.subplots import make_subplots
from IPython.display import display, HTML
import matplotlib.pyplot as plt
from collections import Counter 
import plotly.graph_objs as go
from plotly import tools
import seaborn as sns
import pandas as pd 
import numpy as np 
import plotly 
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from eli5.sklearn import PermutationImportance
from sklearn.metrics import accuracy_score
import eli5


init_notebook_mode(connected=True)

## load data and remove (students, not-employed)
df = pd.read_csv("../input/kaggle-survey-2019/multiple_choice_responses.csv", low_memory = False)
df = df[~df['Q5'].isin(["Student", "Not employed"])]
## function to identify value counts of a column responses presented in multiple other columns
def summate_columns(col, xaxis_def=None):
    xaxis, yaxis1 = [], []
    for i in range(1, 20):
        if col+'_Part_'+str(i) not in df:
            break
        
        doc = dict(df[col+'_Part_'+str(i)].value_counts())
        if len(doc) > 0:
            key = list(doc.keys())[0]
        else:
            continue 
        xaxis.append(key)
        
        if key in doc:
            yaxis1.append(doc[key])
        else:
            yaxis1.append(0)
            
    if xaxis_def != None:
        xaxis = xaxis_def
    
    xaxis = xaxis[::-1]
    xaxis = [_.split("(")[0]+"    " for _ in xaxis]
    yaxis1 = [x*100/len(df) for x in yaxis1][::-1]

    temp_df = pd.DataFrame()
    temp_df['xaxis'] = xaxis
    temp_df['yaxis'] = yaxis1
    temp_df = temp_df.sort_values("yaxis")
    temp_df['colors'] = ['#fad46b']*len(temp_df)
    colors = np.array(temp_df['colors'])
    colors[-3], colors[-2], colors[-1] = "#97c6e8", "#97c6e8", "#97c6e8"
    colors[-4] = "red"
    
    annotations = []
    for yd, xd in zip(xaxis, yaxis1):
        annotations.append(dict(xref='x', yref='y',
                            x=xd+2, y=yd,
                            text=str(round(xd, 1)) + '%',
                            font=dict(family='Arial', size=14, color='black'),
                            showarrow=False))
    
    fig = go.Figure(data=[go.Bar(name='', y=temp_df['xaxis'], x=temp_df['yaxis'], orientation='h', marker=dict(color=colors, opacity=0.6))], 
                    layout = go.Layout(title="Sources of Learning Data Science", plot_bgcolor='#fff', paper_bgcolor='#fff', margin=dict(l=100), 
                                       width=900, height=500, legend=dict(orientation="h", x=0.1, y=1.1)))
    fig.update_layout(barmode='group')
    
    annotations += [go.layout.Annotation(x=40, y=9, xref="x", yref="y", text="(Most Popular)", showarrow=False, arrowhead=7, ax=0, ay=-40)]
    
    fig.update_layout(annotations=annotations)
    fig.update_xaxes(showgrid=False, zeroline=False, title="% of respondents")
    fig.update_yaxes(showgrid=False, zeroline=False)
    fig.add_shape(go.layout.Shape(type="rect", xref="x", yref="paper", x0=22, x1=48.6, y0=0.75, y1=1, fillcolor="gray", opacity=0.2, layer="below", line_width=0))
    fig.show()
    
summate_columns("Q13")
d = {
        'Gain More Knowledge': 69.7,
        'Better Job Roles': 34.5,
        'Personal Goal': 26.2,
        'Career Change': 41.8,
        'Better Salary': 43.5,
}

xx = ["Personal Goal", "Better Salary", "Gain More Knowledge", "Better Job Roles", "Career Change"]
xx = [_ + "<br>(" +str(d[_])+ "%)" for _ in xx]
yy = [""]*len(d)
zz = [13.7, 63, 100, 40, 56]
cc = ['red', 'green', 'purple', 'orange', "blue"] 

trace1 = go.Scatter(x = xx, y = [""]*len(d), mode='markers', name="", marker=dict(color=cc, opacity=0.4, size = zz))
layout = go.Layout(barmode='stack', height=300, margin=dict(l=100), title='Reasons for pursuing University Degrees',
                   legend = dict(orientation="h", x=0.1, y=1.15), plot_bgcolor='#fff', paper_bgcolor='#fff', 
                   showlegend=False)

fig = go.Figure(data=[trace1], layout=layout)
iplot(fig)
university_data = """CMU Tepper	67575	24
UC Berkeley	66150	20
Georgia Tech	9900	23
Illinois-Urbana	19200	17.5
South California	55886	24
Wisconsin	30600	17.5
Boston Metropolitan	34400	24
Johns Hopkins	53740	17.5
Pennsylvania State	29250	17.5
Northwestern SPS	53148	18
Rutgers	23136	24
UCD Dublin	7575	36
Texas A&M	65000	24
Arizona State	39622	15
Northeastern	34200	24
Ohio	35250	20
Rice	129000	24
Indiana Bloomington	15172	18
Notre Dame	48000	21
IIT Chicago	30985	16
Syracuse	54000	24
California Riverside	24990	13
Iowa State	24000	21
Oregon State	23445	17.5
Missouri	34000	24
CUNY SPS	13200	17.5
American	54186	15
Oklahoma	26103	14
Colorado	39450	24
Oklahoma State	33990	24
Baker College	17100	17.5
Bay Path	29340	24
Bellevue	20700	17.5
Blanchardstown	2320	24
Capella	33360	12
Central Connecticut	21681	17.5
Colorado Technical	28080	24
DePaul	43160	17.5
DS TechInstitute	7900	9
Dakota State	13320	10
Elmhurst College	25350	24
Full Sail	28008	12
La Salle	26250	20
Lewis	26235	24
Maryville St. Louis	22950	17.5
Phoenix	29316	17.5
Regis	25560	17.5
Saint Mary's College	31946	24
South Dakota State	13320	12
Saint Joseph's	17520	24
Southern Methodist	57084	28
Southern New Hampshire	22572	15
Slippery Rock	16269	10
Alabama Huntsville	21810	12
Maryland College	24984	17.5
Villanova	43400	24
West Virginia	30690	17.5
Northcentral	30530	23
Edinburgh Napier	9060	33
Drexel	56925	24
Merrimack College	28320	16
Varna Free	5391	12
Johnson & Wales	23744	24
Kent State	21524	12"""

university_data = university_data.split("\n")
udf = pd.DataFrame()
udf['name'] = [_.split("	")[0] for _ in university_data]
udf['tution'] = [float(_.split("	")[1]) for _ in university_data]
udf['months'] = [float(_.split("	")[2]) for _ in university_data]

udf1 = udf[udf['months'] > 6]
udf1 = udf1[udf1['months'] < 25]
udf1 = udf1[udf1['tution'] < 100000]
udf1['name1'] = udf1.apply(lambda x : "" if x['months'] in [9, 12, 17.5, 24] else x['name'], axis = 1)

fig = go.Figure([go.Bar(x=udf['name'], y=udf['tution'], orientation="v", name="Tution Fee", marker=dict(color="orange", opacity=0.5), ),
                 go.Scatter(x=udf['name'], name="Median Household Income", y=[59000]*len(udf), mode="lines", marker=dict(color="red"), line=dict(dash='dash')) ])
fig.update_layout(title="Tution Fee : University Degrees in Data Science", plot_bgcolor='#fff', paper_bgcolor='#fff', legend=dict(orientation="h", x=0.1, y=1.1), 
                 xaxis = dict(tickangle = 45), height=500)
fig.show()

fig = go.Figure([go.Bar(x=udf['name'], y=udf['months'], orientation="v", marker=dict(color="#3498d5", opacity=0.5, line=dict(color="#3498d5")), name="Duration"),
                 go.Scatter(x=udf['name'], y=[12]*len(udf), mode="lines", marker=dict(color="red"), line=dict(dash='dash'), name="1 Yr"), 
                 go.Scatter(x=udf['name'], y=[18]*len(udf), mode="lines", marker=dict(color="blue"), line=dict(dash='dash'), name="1.5 Yr"), 
                 go.Scatter(x=udf['name'], y=[24]*len(udf), mode="lines", marker=dict(color="orange"), line=dict(dash='dash'), name="2 Yr"), 
                ])
fig.update_layout(title="Duration : University Degrees in Data Science", plot_bgcolor='#fff', paper_bgcolor='#fff', legend=dict(orientation="h", x=0.1, y=1.1),
                   xaxis = dict(tickangle = 45), height=400)
fig.show()
key1 = "University Courses (resulting in a university degree)"
df1 = df[df['Q13_Part_10'] == key1]
df2 = df[df['Q13_Part_10'] != key1]

nations = ["United States of America", "Canada", "Brazil", "Mexico", 
           "Germany", "Spain", "France", "Italy", 
           "India", "Japan", "China", "South Korea"]
nation_map = {"United States of America" : "USA", "United Kingdom of Great Britain and Northern Ireland" : "UK"}
plt.figure(figsize=(12,12))

vals = []
for j in range(len(nations)):
    country = nations[j]
    country_df = df[df['Q3'] == country]
    ddf1 = country_df[country_df['Q13_Part_10'] == key1]
    ddf2 = country_df[country_df['Q13_Part_10'] != key1]
    plt.subplot(4, 4, j+1)
    
    if j < 4:
        colors = ["#ff8ce0", "#60cfe6"]
    elif j < 8:
        colors = ["#ff8ce0", "#89e8a2"]
    else:
        colors = ["#ff8ce0", "#827ec4"]
    
    vals.append(len(ddf1) / (len(ddf1) + len(ddf2)))    
    plt.pie([len(ddf1), len(ddf2)],
            labels=["With Degree", "No Degree"],
            autopct="%1.0f%%", 
            colors=colors,
            wedgeprops={"linewidth":5,"edgecolor":"white"})
    if country in nation_map:
        country = nation_map[country]
    plt.title(r"$\bf{" + country + "}$")
col_yes = "#fad46b"
col_no = "#97c6e8"

def compute_stats(df, col, xaxis=None):
    agg_df = df[col].value_counts().to_frame()
    total = sum(agg_df[col])
    agg_df['percent'] = agg_df[col].apply(lambda x : 100*x / total)
    agg_df = agg_df.reset_index().rename(columns = {col: "count", 'index' : col})
    agg_doc = {}
    if xaxis != None:
        for _ in xaxis:
            try:
                agg_doc[_] = agg_df[agg_df[col] == _]['percent'].iloc(0)[0]
            except:
                agg_doc[_] = 0
    return agg_doc

def plot_ver_bars(c, ht=500, annot = True):
    dxf1 = df1[df1['Q5'].isin(["Data Scientist"])]
    dxf2 = df2[df2['Q5'].isin(["Data Scientist"])]

    count_df1 = dxf1[dxf1["Q3"].isin([c])]
    count_df2 = dxf2[dxf2["Q3"].isin([c])]
    col = "Q10"
    xaxis = ["$0-999", "1,000-1,999", "2,000-2,999", "3,000-3,999", "4,000-4,999", "5,000-7,499", "7,500-9,999", "10,000-14,999", "15,000-19,999", "20,000-24,999", "25,000-29,999", "30,000-39,999", "40,000-49,999", "50,000-59,999", "60,000-69,999", "70,000-79,999", "80,000-89,999", "90,000-99,999", "100,000-124,999", "125,000-149,999", "150,000-199,999", "200,000-249,999", "250,000-299,999", "300,000-500,000", "> $500,000"]
    t1_doc = compute_stats(count_df1, col, xaxis=xaxis)
    t2_doc = compute_stats(count_df2, col, xaxis=xaxis)

    ## plot the bar chart
    xaxis1 = ["$0K", "1 - 1K", "2 - 2K", "3 - 3K", "4 - 4K", "5 - 7.5K", "7.5 - 10K", "10 - 15K", "15 - 20K", "20 - 25K", "25 - 30K", "30 - 40K", "40 - 50K", "50 - 60K", "60 - 70K", "70 - 80K", "80 - 90K", "90 - 100K", "100 - 125K", "125 - 150K", "150 - 200K", "200 - 250K", "250 - 300K", "300 - 500K", "> $500K"]
    fig = go.Figure(data=[
                go.Bar(name='Without University Degree', x=xaxis1[7:], y=[t2_doc[_] for _ in xaxis][7:], marker=dict(color=col_no, opacity=0.8)),
                go.Bar(name='With University Degree', x=xaxis1[7:], y=[t1_doc[_] for _ in xaxis][7:], marker=dict(color=col_yes, opacity=0.8))])
    fig.update_layout(barmode='group', title="Data Scientists in <b>"+c+"</b>", 
                      yaxis = dict(title="% of respondents"), xaxis = dict(title="US$"), height=ht,
                      legend=dict(orientation="h", x=0.01, y=1.1), plot_bgcolor='#fff', paper_bgcolor='#fff')
                                 
    fig.update_xaxes(showgrid=False, zeroline=False)
    fig.update_yaxes(showgrid=False, zeroline=False)
    if annot == True:
        fig.update_layout(annotations=[go.layout.Annotation(x=15.5, y=13, xref="x", yref="y", 
                                                       text="More % of Individuals <br> Without University Degrees <br>Earning > $150K", 
                                                       showarrow=True, arrowhead=7, ax=0, ay=-40), 
                                 go.layout.Annotation(x=11, y=24, xref="x", yref="y", 
                                                       text="With University Degrees <br> Earning More", 
                                                       showarrow=True, arrowhead=7, ax=0, ay=-40)])
        fig.add_shape(go.layout.Shape(
                    type="rect",
                    xref="x",
                    yref="paper",
                    x0=12.5,
                    x1=17.5,
                    y0=0.03,
                    y1=1,
                    fillcolor="red",
                    opacity=0.1,
                    layer="below",
                    line_width=0))
        fig.add_shape(go.layout.Shape(
                    type="rect",
                    xref="x",
                    yref="paper",
                    x0=10.5,
                    x1=12.4,
                    y0=0.5,
                    y1=1,
                    fillcolor="green",
                    opacity=0.1,
                    layer="below",
                    line_width=0))
    
    
    fig.show()
    return count_df1, count_df2

nations = ["United States of America", "Germany"]
c1, c2 = plot_ver_bars(nations[0])
salaries = ["100,000-124,999", "125,000-149,999", "150,000-199,999", "200,000-249,999", "300,000-500,000"]
group1 = c1[c1["Q10"].isin(salaries)]
group2 = c2[c2["Q10"].isin(salaries)]

vc1 = group1['Q15'].value_counts().to_frame().reset_index()
vc1['percent'] = vc1['Q15'].apply(lambda x : 100*x / sum(vc1['Q15']))

vc2 = group2['Q15'].value_counts().to_frame().reset_index()
vc2['percent'] = vc2['Q15'].apply(lambda x : 100*x / sum(vc2['Q15']))

order = ["< 1 years", "1-2 years", "3-5 years", "5-10 years", "10-20 years", "20+ years"]
doc1, doc2 = {}, {}
for _, r in vc1.iterrows():
    doc1[r['index']] = r['percent']
for _, r in vc2.iterrows():
    doc2[r['index']] = r['percent']

trace1 = go.Bar(x=order, y=[doc1[_] for _ in order], name="", orientation="v", marker=dict(color=col_yes, opacity=0.8))
trace2 = go.Bar(x=order, y=[doc2[_] for _ in order], name="", orientation="v", marker=dict(color=col_no, opacity=0.8))

vc1 = group1['Q1'].value_counts().to_frame().reset_index()
vc1['percent'] = vc1['Q1'].apply(lambda x : 100*x / sum(vc1['Q1']))

vc2 = group2['Q1'].value_counts().to_frame().reset_index()
vc2['percent'] = vc2['Q1'].apply(lambda x : 100*x / sum(vc2['Q1']))

order = ['22-24', '25-29', '30-34', '35-39', '40-44', '45-49', '50-54', '55-59']
doc1, doc2 = {}, {}
for _, r in vc1.iterrows():
    doc1[r['index']] = r['percent']
for _, r in vc2.iterrows():
    doc2[r['index']] = r['percent']

trace3 = go.Scatter(x=order, y=[doc1[_] for _ in order], name="", mode="lines+markers", marker=dict(color=col_yes, opacity=0.8))
trace4 = go.Scatter(x=order, y=[doc2[_] for _ in order], name="", mode="lines+markers", marker=dict(color=col_no, opacity=0.8))


### ML Exp 

vc1 = group1['Q23'].value_counts().to_frame().reset_index()
vc1['percent'] = vc1['Q23'].apply(lambda x : 100*x / sum(vc1['Q23']))

vc2 = group2['Q23'].value_counts().to_frame().reset_index()
vc2['percent'] = vc2['Q23'].apply(lambda x : 100*x / sum(vc2['Q23']))

order = ['< 1 years', '1-2 years', '2-3 years', '3-4 years', '4-5 years', '5-10 years', '10-15 years', '15-20 years', '20+ years']
doc1, doc2 = {}, {}
for _, r in vc1.iterrows():
    doc1[r['index']] = r['percent']
for _, r in vc2.iterrows():
    doc2[r['index']] = r['percent']

trace5 = go.Bar(x=order, y=[doc1[_] if _ in doc1 else 0 for _ in order], name="With University Degree", orientation="v", marker=dict(color=col_yes, opacity=0.8))
trace6 = go.Bar(x=order, y=[doc2[_] if _ in doc1 else 0 for _ in order], name="Without University Degree", orientation="v", marker=dict(color=col_no, opacity=0.8))

fig = make_subplots(rows=1, cols=2, subplot_titles=("Machine Learning Experience (In Years)", "Coding Experience (In Years)"))
fig.add_trace(trace6, 1, 1)
fig.add_trace(trace5, 1, 1)
fig.add_trace(trace2, 1, 2)
fig.add_trace(trace1, 1, 2)
# fig.add_trace(trace4, 2, 1)
# fig.add_trace(trace3, 2, 1)
fig.update_layout(height=450, title="Data Scientists in USA earning >150K USD : Key Characteristics", plot_bgcolor='#fff', paper_bgcolor='#fff',
                  yaxis = dict(title="% of respondents"), yaxis3 = dict(title="% of respondents"), legend=dict(orientation="h", x=0.1, y=1.2))
fig.show()

fig = go.Figure(data=[trace4, trace3])
fig.update_layout(barmode='group', title="Age : Data Scientists earning >150K USD", showlegend=False, 
                  yaxis = dict(title="% of respondents"), xaxis = dict(title="US$"), height=350,
                  plot_bgcolor='#fff', paper_bgcolor='#fff')
fig.update_xaxes(showgrid=False, zeroline=False)
fig.update_yaxes(showgrid=False, zeroline=False)
fig.update_layout(annotations=[go.layout.Annotation(x=6, y=7.6, xref="x", yref="y", 
                                               text="Experienced people may not<br>need to get university degrees<br>in Data Science to earn more", 
                                               showarrow=True, arrowhead=7, ax=0, ay=-40)]) 
fig.show()
_, _ = plot_ver_bars("Canada", ht=300, annot = False)
_, _ = plot_ver_bars("Germany", ht=300, annot = False)
pd.options.mode.chained_assignment = None  # default='warn'
def diff_workroles(col, xaxis_def=None, get=False, sum_choices = False):
    xaxis, yaxis1, yaxis2 = [], [], []
    for i in range(1, 20):
        if col+'_Part_'+str(i) not in df1:
            break
            
        c1 = df1[df1["Q3"].isin(["United States of America"])]
        c2 = df2[df2["Q3"].isin(["United States of America"])]

        c1 = c1[c1['Q5'].isin(["Data Scientist"])]
        c2 = c2[c2['Q5'].isin(["Data Scientist"])]
        
        doc1 = dict(c1[col+'_Part_'+str(i)].value_counts())
        doc2 = dict(c2[col+'_Part_'+str(i)].value_counts())

        if len(doc1) > 0:
            key = list(doc1.keys())[0]
        elif len(doc2) > 0:
            key = list(doc2.keys())[0]
        else:
            continue 
        xaxis.append(key)
        
        if key in doc1:
            yaxis1.append(doc1[key])
        else:
            yaxis1.append(0)
        if key in doc2:
            yaxis2.append(doc2[key])
        else:
            yaxis2.append(0)
            
    if xaxis_def != None:
        xaxis = xaxis_def
    
    
    xaxis = xaxis[::-1]
    
    ln_c1 = len(c1)
    ln_c2 = len(c2)
    if sum_choices == True:
        ln_c1 = sum(yaxis1)
        ln_c2 = sum(yaxis2)
    
    yaxis1 = [x*100/ln_c1 for x in yaxis1][::-1]
    yaxis2 = [x*100/ln_c2 for x in yaxis2][::-1]
    if get == True:
        return xaxis, yaxis1, yaxis2
    
    tra1 = go.Bar(name='With Degree', y=xaxis, x=yaxis1, orientation='h', width=0.3, marker=dict(color=col_yes, opacity=0.8))
    tra2 = go.Bar(name='Without Degree', y=xaxis, x=yaxis2, orientation='h', width=0.3, marker=dict(color=col_no, opacity=0.8))
    return tra1, tra2

xaxis = ["Data Analysis / Exploration    ",
         "Build/Run Data Infrastructure    ",
         "Build Machine Learning Prototypes    ",
         "Build Machine Learning Services    ",
         "Experimentation and Improvements    ",
         "Research / SOTA Models    ",
         "None of these activities    ",
         "Other"]

x, y1, y2 = diff_workroles("Q9", xaxis, get=True)
trace1 = go.Bar(name='With Degree', y=x[2:], x=y1[2:], orientation='h', width=0.3, marker=dict(color=col_yes, opacity=0.8))
trace2 = go.Bar(name='Without Degree', y=x[2:], x=y2[2:], orientation='h', width=0.3, marker=dict(color=col_no, opacity=0.8))

x, y1, y2 = diff_workroles("Q9", xaxis, sum_choices = True, get=True)
trace5 = go.Bar(name='', y=x[2:], x=y1[2:], orientation='h', width=0.3, marker=dict(color=col_yes, opacity=0.5))
trace6 = go.Bar(name='', y=x[2:], x=y2[2:], orientation='h', width=0.3, marker=dict(color=col_no, opacity=0.5))


def summate(row, cols):
    count = 0
    for c in cols:
        if str(row[c]) == "nan":
            continue
        count += 1
    return count

def skill_variety(col, df):
    dfx = df[df["Q3"].isin(["United States of America"])]

    cols = []
    for i in range(1, 20):
        if col+'_Part_'+str(i) not in df:
            break
        doc = dict(dfx[col+'_Part_'+str(i)].value_counts())
        if any(_ in doc for _ in ["None of these activities are an important part of my role at work", "Other"]):
            continue
        cols.append(col+'_Part_'+str(i))            
    dfx[col+'_sum'] = dfx.apply(lambda x : summate(x, cols), axis=1)
    return dfx

order = [0,1,2,3,4,5,6]
doc1, doc2 = {}, {}
c1 = skill_variety("Q9", df1)
vc1 = c1['Q9_sum'].value_counts().to_frame().reset_index()
vc1['percent'] = vc1['Q9_sum'].apply(lambda x : 100*x / len(c1))
for _, r in vc1.iterrows():
    doc1[r['index']] = r['percent']

c2 = skill_variety("Q9", df2)
vc2 = c2['Q9_sum'].value_counts().to_frame().reset_index()
vc2['percent'] = vc2['Q9_sum'].apply(lambda x : 100*x / len(c2))
for _, r in vc2.iterrows():
    doc2[r['index']] = r['percent']

trace3 = go.Scatter(x=order, y=[doc1[_] for _ in order], name="With University Degrees", mode="markers+lines", marker=dict(color=col_yes, opacity=0.8))
trace4 = go.Scatter(x=order, y=[doc2[_] for _ in order], name="Without University Degrees", mode="markers+lines", marker=dict(color=col_no, opacity=0.8))

fig = make_subplots(rows=1, cols=2, 
                    subplot_titles=("A: % of respondents <br> with different responsibilities", 
                                    "B: % of times a <br> responsibility is selected"))
fig.add_trace(trace1, 1, 1)
fig.add_trace(trace2, 1, 1)
fig.add_trace(trace5, 1, 2)
fig.add_trace(trace6, 1, 2)
fig.update_layout(title="", plot_bgcolor='#fff', paper_bgcolor='#fff',
                  xaxis1 = dict(title="% of respondents"), 
                  xaxis2 = dict(title="% this responsibility is choosen"), yaxis2 = dict(showticklabels=False),
                  legend=dict(orientation="h", x=0.1, y=1.2))
fig.show()
fig = go.Figure([trace3, trace4])
fig.update_layout(title="Number of responsibilities of Data Scientists", plot_bgcolor='#fff', paper_bgcolor='#fff', height=500,
                  yaxis = dict(title="% of respondents", showgrid=False),xaxis = dict(title="Number of job responsibilities",showgrid=False), 
                  legend=dict(orientation="h", x=0.6, y=1.2))
fig.update_layout(annotations=[go.layout.Annotation(x=1, y=13, xref="x", yref="y", 
                                               text="Specialists : Involved in <br> only one activity", 
                                               showarrow=True, arrowhead=3, ax=0, ay=-60), 
                         go.layout.Annotation(x=6, y=10, xref="x", yref="y", 
                                               text="Generalists : Involved in <br> every activity", 
                                               showarrow=True, arrowhead=3, ax=0, ay=-60)])
fig.add_shape(go.layout.Shape(type="circle", xref="x", yref="paper", x0=0.8, x1=1.23, y0=0.1, y1=0.3, fillcolor="PaleTurquoise", opacity=0.7, layer='below', line_color="#34eb71", line_width=2))
fig.add_shape(go.layout.Shape(type="circle", xref="x", yref="paper", x0=6.2, x1=5.8, y0=0.01, y1=0.2, fillcolor="PaleTurquoise", opacity=0.7, layer='below', line_color="#34eb71", line_width=2))
fig.show()
def generate_bubb_diff(col, title):
    # x, y1, y2 = diff_workroles("Q9", xaxis, sum_choices = True, get=True)
    x, y1, y2 = diff_workroles(col, get = True, sum_choices = True)
    
    x = [_.split("(")[0].strip() for _ in x]
    x = x[2:]
    y1 = y1[2:]
    y2 = y2[2:]

    ss1 = y1 + y2 + [np.abs(a-b) for a,b in zip(y1, y2)]
    ss1 = [str(round(_, 1))+"%" for _ in y1+y2] + [round(a-b, 2) for a,b in zip(y1, y2)]
    
    diffs1, diffs2 = [], []
    for i in range(len(y1)):
        d = y1[i] - y2[i]
        if d > 0:
            diffs1.append(2.5+d)
            diffs2.append(3)
        else:
            diffs2.append(2.5-d)
            diffs1.append(3)
    ss = [_*diffs1[i] for i, _ in enumerate(y1)]
    ss += [_*diffs2[i] for i, _ in enumerate(y2)]
    ss += [np.abs(a-b)*10 for a,b in zip(y1, y2)]
    
    trace1 = go.Scatter(x = ["University Degree Holders"]*len(x)+["Self Made"]*len(x)+["Difference"]*len(x), y = x+x+x, mode='markers+text', textposition="middle right", text=ss1, name="", 
                        marker=dict(color=[col_yes]*len(x)+[col_no]*len(x), opacity=0.8, size = ss))
    layout = go.Layout(barmode='stack', margin=dict(l=200), height=900, title=title,
                       legend = dict(orientation="h", x=0.1, y=1.15), plot_bgcolor='#fff', paper_bgcolor='#fff', 
                       showlegend=False)

    fig = go.Figure(data=[trace1], layout=layout)
    iplot(fig)
    
generate_bubb_diff("Q24", title = "Do self made data scientists use different Machine Learning Models ?")
generate_bubb_diff("Q28", "Is there a difference in tools / techniques used by data scientists")
c1 = skill_variety("Q24", df1)
c2 = skill_variety("Q24", df2)

vc1 = c1['Q24_sum'].value_counts().to_frame().reset_index()
vc2 = c2['Q24_sum'].value_counts().to_frame().reset_index()

vc1['percent'] = vc1['Q24_sum'].apply(lambda x : 100*x / len(c1))
vc2['percent'] = vc2['Q24_sum'].apply(lambda x : 100*x / len(c2))

vc1 = vc1.sort_values("index")[1:]
vc2 = vc2.sort_values("index")[1:]

tr1 = go.Scatter(x=vc1['index'], y=vc1['percent'], name='with',fill='tozeroy', marker=dict(color=col_yes),showlegend=False, opacity=0.8)
tr2 = go.Scatter(x=vc2['index'], y=vc2['percent'], name='without',fill='tozeroy', marker=dict(color=col_no),showlegend=False, opacity=0.8)

c1 = skill_variety("Q28", df1)
c2 = skill_variety("Q28", df2)

vc1 = c1['Q28_sum'].value_counts().to_frame().reset_index()
vc2 = c2['Q28_sum'].value_counts().to_frame().reset_index()

vc1['percent'] = vc1['Q28_sum'].apply(lambda x : 100*x / len(c1))
vc2['percent'] = vc2['Q28_sum'].apply(lambda x : 100*x / len(c2))

vc1 = vc1.sort_values("index")[1:]
vc2 = vc2.sort_values("index")[1:]

tr3 = go.Scatter(x=vc1['index'], y=vc1['percent'], name='with',fill='tozeroy', marker=dict(color=col_yes),showlegend=False, opacity=0.8)
tr4 = go.Scatter(x=vc2['index'], y=vc2['percent'], name='without',fill='tozeroy', marker=dict(color=col_no),showlegend=False, opacity=0.8)

fig = make_subplots(rows=1, cols=2, subplot_titles=("# of ML Techniques", "# of ML Algorithms"))
fig.add_trace(tr1, 1, 1)
fig.add_trace(tr2, 1, 1)
fig.add_trace(tr3, 1, 2)
fig.add_trace(tr4, 1, 2)

fig.update_layout(height=400, yaxis1=dict(title="% respondents"), yaxis2=dict(title="% respondents"), showlegend = False, plot_bgcolor='#fff', paper_bgcolor='#fff',)
fig.show()
generate_bubb_diff("Q17", "Which notebooks are common among the two groups ? ")
generate_bubb_diff("Q16", "Which Editors / Tools are used by degree holders ?")
c1 = skill_variety("Q17", df1)
c2 = skill_variety("Q17", df2)

vc1 = c1['Q17_sum'].value_counts().to_frame().reset_index()
vc2 = c2['Q17_sum'].value_counts().to_frame().reset_index()

vc1['percent'] = vc1['Q17_sum'].apply(lambda x : 100*x / len(c1))
vc2['percent'] = vc2['Q17_sum'].apply(lambda x : 100*x / len(c2))

vc1 = vc1.sort_values("index")[1:]
vc2 = vc2.sort_values("index")[1:]

tr1 = go.Scatter(x=vc1['index'], y=vc1['percent'], name='with',fill='tozeroy', marker=dict(color=col_yes),showlegend=False, opacity=0.8)
tr2 = go.Scatter(x=vc2['index'], y=vc2['percent'], name='without',fill='tozeroy', marker=dict(color=col_no),showlegend=False, opacity=0.8)

c1 = skill_variety("Q16", df1)
c2 = skill_variety("Q16", df2)

vc1 = c1['Q16_sum'].value_counts().to_frame().reset_index()
vc2 = c2['Q16_sum'].value_counts().to_frame().reset_index()

vc1['percent'] = vc1['Q16_sum'].apply(lambda x : 100*x / len(c1))
vc2['percent'] = vc2['Q16_sum'].apply(lambda x : 100*x / len(c2))

vc1 = vc1.sort_values("index")[1:]
vc2 = vc2.sort_values("index")[1:]

tr3 = go.Scatter(x=vc1['index'], y=vc1['percent'], name='with',fill='tozeroy', marker=dict(color=col_yes),showlegend=False, opacity=0.8)
tr4 = go.Scatter(x=vc2['index'], y=vc2['percent'], name='without',fill='tozeroy', marker=dict(color=col_no),showlegend=False, opacity=0.8)
    
fig = make_subplots(rows=1, cols=2, subplot_titles=("# of Cloud Notebooks", "# of Platforms"))
fig.add_trace(tr1, 1, 1)
fig.add_trace(tr2, 1, 1)
fig.add_trace(tr3, 1, 2)
fig.add_trace(tr4, 1, 2)

fig.update_layout(height=400, yaxis1=dict(title="% respondents"), yaxis2=dict(title="% respondents"), showlegend = False, plot_bgcolor='#fff', paper_bgcolor='#fff',)
fig.show()
from IPython.core.display import display, HTML, Javascript
from collections import Counter 
import IPython.display
import json

naming_doc = {
              'Q25': 'ML Tools',
              'Q29': 'Cloud Computing Platforms',
              'Q30': 'Cloud Computing Products',
              'Q31': 'BigData / Analytics Products',
              'Q32': 'ML Products',
              'Q33': 'Automated ML Products',
              'Q34': 'Relational Database'}

results = {'name' : 'flare', "children" : [], "size":""}
for col in naming_doc:
    x, y1, y2 = diff_workroles(col, get = True)
    
    resp = {'name' : naming_doc[col], "children" : [], "size": ""}
    for k,v in zip(x, y1):
        if "none" in k.lower():
            continue
        k = str(k.strip()).split("(")[0]
        k = "\n".join(k.split())
        doc = {'name' : k, "size": round(v, 1)}
        resp['children'].append(doc)

    results['children'].append(resp)

with open('output.json', 'w') as outfile:  
    json.dump(results, outfile)

htmlt1 = """<!DOCTYPE html><meta charset="utf-8"><style>.node {cursor: pointer;}.node:hover {stroke: #fff;stroke-width: 1.5px;}.node--leaf {fill: white;}
.label {font: 11px "Helvetica Neue", Helvetica, Arial, sans-serif;text-anchor: middle;text-shadow: 0 1px 0 #fff, 1px 0 0 #fff, -1px 0 0 #fff, 0 -1px 0 #fff;}
.label,.node--root,.node--leaf {pointer-events: none;}</style><svg id="two" width="760" height="760"></svg>
"""
js_t1="""require.config({
    paths: {
        d3: "https://d3js.org/d3.v4.min"
     }
 });
require(["d3"], function(d3) {
var svg = d3.select("#two"),
    margin = 20,
    diameter = +svg.attr("width"),
    g = svg.append("g").attr("transform", "translate(" + diameter / 2 + "," + diameter / 2 + ")"),
    color = d3.scaleQuantize()
    .domain([-2,2])
    .range(["#fad46b", "#fad46b", "#fad46b", "#fad46b"]),
    
    pack = d3.pack().size([diameter - margin, diameter - margin]).padding(2);
d3.json("output.json", function(t, n) {
if (t) throw t;
var r, e = n = d3.hierarchy(n).sum(function(t) {
        return t.size
    }).sort(function(t, n) {
        return n.value - t.value
    }),
    a = pack(n).descendants(),
    i = g.selectAll("circle").data(a).enter().append("circle").attr("class", function(t) {
        return t.parent ? t.children ? "node" : "node node--leaf" : "node node--root"
    }).style("fill", function(t) {
        return t.children ? color(t.depth) : null
    }).on("click", function(t) {
        e !== t && (l(t), d3.event.stopPropagation())
    }),
    o = (g.selectAll("text").data(a).enter().append("text").attr("class", "label").style("fill-opacity", function(t) {
        return t.parent === n ? 1 : 0
    }).style("display", function(t) {
        return t.parent === n ? "inline" : "none"
    }).text(function(t) {
        return t.data.name + ": " + t.data.size + "%"
    }), g.selectAll("circle,text"));

function l(t) {
    e = t, d3.transition().duration(d3.event.altKey ? 7500 : 750).tween("zoom", function(t) {
        var n = d3.interpolateZoom(r, [e.x, e.y, 2 * e.r + margin]);
        return function(t) {
            c(n(t))
        }
    }).selectAll("text").filter(function(t) {
        return t.parent === e || "inline" === this.style.display
    }).style("fill-opacity", function(t) {
        return t.parent === e ? 1 : 0
    }).on("start", function(t) {
        t.parent === e && (this.style.display = "inline")
    }).on("end", function(t) {
        t.parent !== e && (this.style.display = "none")
    })
}

function c(n) {
    var e = diameter / n[2];
    r = n, o.attr("transform", function(t) {
        return "translate(" + (t.x - n[0]) * e + "," + (t.y - n[1]) * e + ")"
    }), i.attr("r", function(t) {
        return t.r * e
    })
}
svg.style("background", color(-1)).on("click", function() {
    l(n)
}), c([n.x, n.y, 2 * n.r + margin])
});
});"""

h = display(HTML("<i>Note: The following chart is interactive, Click on the Clusters to view more details</i>"))
h = display(HTML(htmlt1))
j = IPython.display.Javascript(js_t1)
IPython.display.display_javascript(j)
results = {'name' : 'flare', "children" : [], "size":""}
for col in naming_doc:
    x, y1, y2 = diff_workroles(col, get = True)
    resp = {'name' : naming_doc[col], "children" : [], "size": ""}
    for k,v in zip(x, y2):
        if "none" in k.lower():
            continue
        k = str(k.strip()).split("(")[0]
        k = "\n".join(k.split())
        doc = {'name' : k, "size": round(v, 1)}
        resp['children'].append(doc)
    results['children'].append(resp)

with open('output1.json', 'w') as outfile:  
    json.dump(results, outfile)

htmlt1 = """<svg id="three" width="760" height="760"></svg>"""
js_t1="""require(["d3"], function(d3) {
var svg1 = d3.select("#three"),
    margin = 20,
    diameter = +svg1.attr("width"),
    g = svg1.append("g").attr("transform", "translate(" + diameter / 2 + "," + diameter / 2 + ")"),
    color = d3.scaleQuantize()
    .domain([-2,2])
    .range(["#97c6e8", "#97c6e8", "#97c6e8", "#97c6e8"]),
    
    pack = d3.pack().size([diameter - margin, diameter - margin]).padding(2);


d3.json("output1.json", function(t, n) {
if (t) throw t;
var r, e = n = d3.hierarchy(n).sum(function(t) {
        return t.size
    }).sort(function(t, n) {
        return n.value - t.value
    }),
    a = pack(n).descendants(),
    i = g.selectAll("circle").data(a).enter().append("circle").attr("class", function(t) {
        return t.parent ? t.children ? "node" : "node node--leaf" : "node node--root"
    }).style("fill", function(t) {
        return t.children ? color(t.depth) : null
    }).on("click", function(t) {
        e !== t && (l(t), d3.event.stopPropagation())
    }),
    o = (g.selectAll("text").data(a).enter().append("text").attr("class", "label").style("fill-opacity", function(t) {
        return t.parent === n ? 1 : 0
    }).style("display", function(t) {
        return t.parent === n ? "inline" : "none"
    }).text(function(t) {
        return t.data.name + ": " + t.data.size + "%"
    }), g.selectAll("circle,text"));

function l(t) {
    e = t, d3.transition().duration(d3.event.altKey ? 7500 : 750).tween("zoom", function(t) {
        var n = d3.interpolateZoom(r, [e.x, e.y, 2 * e.r + margin]);
        return function(t) {
            c(n(t))
        }
    }).selectAll("text").filter(function(t) {
        return t.parent === e || "inline" === this.style.display
    }).style("fill-opacity", function(t) {
        return t.parent === e ? 1 : 0
    }).on("start", function(t) {
        t.parent === e && (this.style.display = "inline")
    }).on("end", function(t) {
        t.parent !== e && (this.style.display = "none")
    })
}

function c(n) {
    var e = diameter / n[2];
    r = n, o.attr("transform", function(t) {
        return "translate(" + (t.x - n[0]) * e + "," + (t.y - n[1]) * e + ")"
    }), i.attr("r", function(t) {
        return t.r * e
    })
}
svg1.style("background", color(-1)).on("click", function() {
    l(n)
}), c([n.x, n.y, 2 * n.r + margin])
});
});"""

h = display(HTML("<i>Note: The following chart is interactive, Click on the Clusters to view more details</i>"))
h = display(HTML(htmlt1))
j = IPython.display.Javascript(js_t1)
IPython.display.display_javascript(j)
np.random.seed(0)

cols = list(df.columns)
cols = [c for c in cols if "TEXT" not in c][1:]
tempdf = df[cols][1:].fillna(0)

## prepare the data
single_vars, multi_vars = [], []
for c in tempdf.columns:
    if len(tempdf[c].value_counts()) == 2:
        tempdf[c] = tempdf[c].apply(lambda x : 1 if x != 0 else x)
        single_vars.append(c)
    else:
        multi_vars.append (c)
one_hot_df = pd.get_dummies(tempdf[multi_vars])
one_hot_df = pd.concat([one_hot_df, tempdf[single_vars]], axis=1)

## train a simple RF
ignore_cols = ["Q13_Part_10"]#"Q11", "Q13", "Q10", "Q5", "Q4", "Q1", "Q2", "Q3"]
features = []
for x in one_hot_df.columns:
    if any(_ in x for _ in ignore_cols):
        continue
    features.append(x)

X = one_hot_df[features]
y = one_hot_df["Q13_Part_10"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 2)
model = RandomForestClassifier(n_estimators=50, random_state=0).fit(X_train, y_train)
y_pred = model.predict(X_test)
print ("Classifier Accuracy: ", accuracy_score(y_pred, y_test))
# Relative Feature Importance
feature_importances = pd.DataFrame(model.feature_importances_, index = X_train.columns, columns=['importance'])
feature_importances = feature_importances.sort_values('importance', ascending=False)

perm = PermutationImportance(model, random_state=1).fit(X_test, y_test)
pi_df = eli5.explain_weights_df(perm, feature_names = X_test.columns.tolist())

name_doc = {'Matplotlib' : 'Have used Matplotlib',
'Do research that advances the state of the art of machine learning' : 'Do research',
'Q7_20+' : 'Work with 20+ data scientists',
'Dense Neural Networks' : 'Have used Dense Neural Networks',
'Journal Publications' : 'Read Journal Publications',
'Google Colab' : 'Use Google Colab',
'Contextualized embeddings' : 'Used Contextualized embeddings',
'GPUs' : 'Used GPUs',
'R' : 'Used R',
'RStudio' : 'Used RStudio',
'Q1_25-29' : 'Aged 25-29',
'Q22_0' : 'Never Used TPU',
'AWS Elastic Compute Cloud' : 'Experience with AWS Elastic Compute Cloud',
'Q15_3-5 years' : 'Writing code for 3-5 years',
'Q14_Basic statistical software' : 'Have used Basic statistical software',
'Notepad++' : 'Used Notepad++',
'Spyder' : 'Used Spyder',
'Q15_1-2 years' : 'Writing Code for 1-2 years',
'Q14_Local development environments' : 'Used Local development environments',
'Q1_22-24' : 'Aged 22-24',
'Automated hyperparameter tuning' : 'Automated hyperparameter tuning',
'Q7_15-19' : 'Work with 15-19 data scientists',
'Plotly / Plotly Express' : 'Have used Plotly / Plotly Express',
'AWS DynamoDB' : 'Have used AWS DynamoDB',
'Oracle Cloud' : 'Have Used Oracle Cloud',
'AWS Elastic Beanstalk' : 'Have Used AWS Elastic Beanstalk',
'Transformer Networks' : 'Use Transformer Networks',
'AWS Athena' : 'Experience with AWS Athena'
}

colr_doc = {
        'Do research' : "#6f77e8",
        'Have used Dense Neural Networks' : "#6f77e8",
        'Read Journal Publications' : "#6f77e8",
        'Use Google Colab' : "#6f77e8",
        'Used Contextualized embeddings' : "#6f77e8",
        'Used GPUs' : "#6f77e8",
        
        'Used R' : "green",
        'Used RStudio' : "green",
        
        'Aged 25-29' : "red",
        'Aged 22-24' : "red",
}
    
def find_name(x):
    name = x
    if "part" in x.lower():
        name = df[x].value_counts().index[0].strip()
    name = name.split("(")[0].strip()
    if name in name_doc:
        name = name_doc[name]
    return name

pi_df = pi_df.sort_values("weight", ascending = False)
pi_df['name'] = pi_df["feature"].apply(lambda x : find_name(x))
pi_df = pi_df[~pi_df["name"].isin(["None", "Other"])]
pi_df['plot'] = pi_df['name'].apply(lambda x : 0 if any(_ in x for _ in ["Q4", "Q5", "Q19", "Q10"]) else 1)
pi_df = pi_df[pi_df['plot'] == 1]
pi_df['colr'] = pi_df['name'].apply(lambda x : colr_doc[x].strip() if x in colr_doc else col_yes)

data = [
    go.Bar(
        orientation = "h",
        y = [_+"    " for _ in list(pi_df.name[:25])][::-1],
        x = pi_df[:25].weight[::-1],
        marker = dict(
            opacity = 0.6,
            color = pi_df['colr'][:25][::-1]   ),
        name = 'expenses'
    )
]

layout = go.Layout(title="Key traits of individuals who prefer to go for university degrees", xaxis=dict(title="importance score"), 
                   height = 700, margin=dict(l=300), showlegend = False, plot_bgcolor='#fff', paper_bgcolor='#fff',)
fig = go.Figure(data=data, layout = layout)
fig.update_layout(annotations=[go.layout.Annotation(x=0.0015, y=19, xref="x", yref="y", 
                                               text="  More Focus on <br> Research / Deep Learning", 
                                               showarrow=True, arrowhead=3, ax=180,ay=0),
                               go.layout.Annotation(x=0.0015, y=17, xref="x", yref="y", 
                                               text="  More use of R <br> than Python", 
                                               showarrow=True, arrowhead=3, ax=180,ay=0),
                               go.layout.Annotation(x=0.0015, y=14, xref="x", yref="y", 
                                               text="  Young Individuals <br> (Aged 22 - 30)", 
                                               showarrow=True, arrowhead=3, ax=130,ay=0),
                               go.layout.Annotation(x=0.001, y=8, xref="x", yref="y", 
                                               text="  Others : <br> Involved in Coding (for atleast 1 year) <br> Not kaggle / coursera <br> Use basic softwares / tools", 
                                               showarrow=True, arrowhead=3, ax=130,ay=0)
                              ])
iplot(fig, filename='base-bar')
pfs = ["  University Courses", "  Self Made : Kaggle / Coursera"]
roles = ["Data Analysis/Exploration  ", "Build Infrastructures  ", "Build ML Services  ", "Build ML Prototypes  ", "Experimentation  ", "Research  "]
ds = ["Data Scientist"]
vals = [0.5657851779301338, 0.3480248122755468, 0.4975514201762977, 0.3382304929807378, 0.3813254978778975, 0.25334639242572643]
vals += [0.5208446362515413, 0.31336621454993836, 0.42824907521578296, 0.2953514180024661 ,0.3394327990135635, 0.20393341553637485]
# vals += [0.5001363512407963, 0.30160894464139626, 0.38396509408235613, 0.27188437414780475, 0.2877011180801745, 0.18271066266703026]
vals = [100*_ for _ in vals] 

fig = go.Figure(data=[go.Sankey(
    node = dict(
      pad = 50,
      thickness = 10,
      line = dict(color = "gray", width = 2),
      label =  pfs +  roles + ds,
      color = [col_yes, col_no] + ["#6ccc86"]*6
    ),
    link = dict(
      source = [0, 1]*len(roles) + [2,3,4,5,6,7]*len(ds),
      target = [2]*len(pfs) + [3]*len(pfs) + [4]*len(pfs) + [5]*len(pfs) + [6]*len(pfs) + [7]*len(pfs) + [8, 8, 8, 8, 8, 8],
      value = vals +  [73.5, 62.8, 71.4, 65.6, 67.4, 58.9]
  ))])

# print (df[df["Q5"] == "Data Scientist"]["Q9_Part_1"].value_counts().values[0])
# print (df[df["Q5"] == "Data Scientist"]["Q9_Part_2"].value_counts().values[0])
# print (df[df["Q5"] == "Data Scientist"]["Q9_Part_3"].value_counts().values[0])
# print (df[df["Q5"] == "Data Scientist"]["Q9_Part_4"].value_counts().values[0])
# print (df[df["Q5"] == "Data Scientist"]["Q9_Part_5"].value_counts().values[0])
# print (df[df["Q5"] == "Data Scientist"]["Q9_Part_6"].value_counts().values[0])

# a = [2554, 1393, 2323, 1697, 1884, 972]
# a = [_*100/sum(a) for _ in a]
# a

fig.update_layout(font=dict(size = 12, color = 'orange'),)
fig.show()