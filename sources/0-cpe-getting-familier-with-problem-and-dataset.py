
import numpy as np 
import pandas as pd 
import folium
from folium import plugins
from io import StringIO
import geopandas as gpd
from pprint import pprint 
from plotly.offline import init_notebook_mode, iplot
import plotly.graph_objs as go
import plotly.plotly as py
from plotly import tools
import plotly.figure_factory as ff
import matplotlib.pyplot as plt
import seaborn as sns
import os 
init_notebook_mode(connected=True)
depts = [f for f in os.listdir("../input/cpe-data/") if f.startswith("Dept")]
pprint(depts)
files = os.listdir("../input/cpe-data/Dept_23-00089/23-00089_ACS_data/")
files
basepath = "../input/cpe-data/Dept_23-00089/23-00089_ACS_data/23-00089_ACS_race-sex-age/"
rca_df = pd.read_csv(basepath + "ACS_15_5YR_DP05_with_ann.csv")
rca_df.head()
a_df = pd.read_csv(basepath + "ACS_15_5YR_DP05_metadata.csv")

# for j, y in a_df.iterrows():
#     if y['Id'].startswith("Estimate"):
#         print (y['GEO.id'], y['Id'])

a_df.head()
total_population = rca_df["HC01_VC03"][1:]

trace = go.Histogram(x=total_population, marker=dict(color='orange', opacity=0.6))
layout = dict(title="Total Population Distribution - Across the counties", margin=dict(l=200), width=800, height=400)
data = [trace]
fig = go.Figure(data=data, layout=layout)
iplot(fig)

male_pop = rca_df["HC01_VC04"][1:]
female_pop = rca_df["HC01_VC05"][1:]

trace1 = go.Histogram(x=male_pop, name="male population", marker=dict(color='blue', opacity=0.6))
trace2 = go.Histogram(x=female_pop, name="female population", marker=dict(color='pink', opacity=0.6))
layout = dict(title="Population Distribution Breakdown - Across the Census Tracts", margin=dict(l=200), width=800, height=400)
data = [trace1, trace2]
fig = go.Figure(data=data, layout=layout)
iplot(fig)
age_cols = []
names = []
for i in range(13):
    if i < 2:
        i = "0"+str(i+8)
        relcol = "HC01_VC" + str(i)
    else:
        relcol = "HC01_VC" + str(i+8)
    age_cols.append(relcol)
    name = a_df[a_df["GEO.id"] == relcol]["Id"].iloc(0)[0].replace("Estimate; SEX AND AGE - ","")
    names.append(name)

rca_df['GEO.display-label_cln'] = rca_df["GEO.display-label"].apply(lambda x : x.replace(", Marion County, Indiana", "").replace("Census Tract ", "CT: "))

traces = []
for i,agecol in enumerate(age_cols):
    x = rca_df["GEO.display-label_cln"][1:]
    y = rca_df[agecol][1:]
    trace = go.Bar(y=y, x=x, name=names[i])
    traces.append(trace)

tmp = pd.DataFrame()
vals = []
Geo = []
Col = []
for i,age_col in enumerate(age_cols):
    Geo += list(rca_df["GEO.display-label_cln"][1:].values)
    Col += list([names[i]]*len(rca_df[1:]))
    vals += list(rca_df[age_col][1:].values)

tmp['Geo'] = Geo
tmp['Col'] = Col
tmp['Val'] = vals
tmp['Val'] = tmp['Val'].astype(int)  * 0.01

data = [go.Scatter(x = tmp["Geo"], y = tmp["Col"], mode="markers", marker=dict(size=list(tmp["Val"].values)))]
layout = dict(title="Age Distribution by Census Tract - Marion County, Indiana", legend=dict(x=-0.1, y=1, orientation="h"), 
              margin=dict(l=150, b=100), height=600, barmode="stack")
fig = go.Figure(data=data, layout=layout)
iplot(fig)
trace1 = go.Histogram(x = rca_df["HC01_VC26"][1:], name="18+", marker=dict(opacity=0.4)) 
trace2 = go.Histogram(x = rca_df["HC01_VC27"][1:], name="21+", marker=dict(opacity=0.3)) 
trace3 = go.Histogram(x = rca_df["HC01_VC28"][1:], name="62+", marker=dict(opacity=0.4)) 
trace4 = go.Histogram(x = rca_df["HC01_VC29"][1:], name="65+", marker=dict(opacity=0.3)) 

titles = ["Age : 18+","Age : 21+","Age : 62+","Age : 65+",]
fig = tools.make_subplots(rows=2, cols=2, print_grid=False, subplot_titles=titles)
fig.append_trace(trace1, 1, 1);
fig.append_trace(trace2, 1, 2);
fig.append_trace(trace3, 2, 1);
fig.append_trace(trace4, 2, 2);
fig['layout'].update(height=600, title="Distribution of Age across the Census Tracts", showlegend=False);
iplot(fig, filename='simple-subplot');
single_race_df = rca_df[["HC01_VC49", "HC01_VC50", "HC01_VC51", "HC01_VC56", "HC01_VC64", "HC01_VC69"]][1:]
ops = [1, 0.85, 0.75, 0.65, 0.55, 0.45]
traces = []
for i, col in enumerate(single_race_df.columns):
    nm = a_df[a_df["GEO.id"] == col]["Id"].iloc(0)[0].replace("Estimate; RACE - One race - ", "")
    trace = go.Bar(x=rca_df["GEO.display-label_cln"][1:], y=single_race_df[col], name=nm, marker=dict(opacity=0.6))
    traces.append(trace)
layout = dict(barmode="stack", title="Population Breakdown by Race (Single)", margin=dict(b=100), height=600, legend=dict(x=-0.1, y=1, orientation="h"))
fig = go.Figure(data=traces, layout=layout)
iplot(fig)
traces = []
for i, col in enumerate(single_race_df.columns):
    nm = a_df[a_df["GEO.id"] == col]["Id"].iloc(0)[0].replace("Estimate; RACE - One race - ", "")
    if nm in ["White", "Black or African American"]:
        continue
    trace = go.Bar(x=rca_df["GEO.display-label_cln"][1:], y=single_race_df[col], name=nm, marker=dict(opacity=0.6))
    traces.append(trace)
layout = dict(barmode="stack", title="Population Breakdown by Race (Single)", margin=dict(b=100), height=400, legend=dict(x=-0.1, y=1, orientation="h"))
fig = go.Figure(data=traces, layout=layout)
iplot(fig)
basepath2 = "../input/cpe-data/Dept_23-00089/23-00089_ACS_data/23-00089_ACS_poverty/"
a_df = pd.read_csv(basepath2 + "ACS_15_5YR_S1701_metadata.csv")
# for j, y in a_df.iterrows():
#     if "Below poverty level; Estimate" in y['Id']:
#         print (y['GEO.id'], y['Id'])        
        
a_df.T.head()
pov_df = pd.read_csv(basepath2 + "ACS_15_5YR_S1701_with_ann.csv")[1:]
pov_df.head()

# pov_df[["HC02_EST_VC66", ""]]
# pov_df["HC02_EST_VC01"] = pov_df["HC02_EST_VC01"].astype(float)
# pov_df.sort_values("HC02_EST_VC01", ascending = False)["HC02_EST_VC01"]
age_bp = ["HC02_EST_VC04", "HC02_EST_VC05", "HC02_EST_VC08", "HC02_EST_VC09", "HC02_EST_VC11"]
pov_df[age_bp]

pov_df['GEO.display-label_cln'] = pov_df["GEO.display-label"].apply(lambda x : x.replace(", Marion County, Indiana", "").replace("Census Tract ", "CT: "))

names = ["Below 5", "5-17", "18-34", "34-64", "65+"]

vals = []
Geo = []
Col = []
tmp = pd.DataFrame()
for i,age_col in enumerate(age_bp):
    Geo += list(pov_df["GEO.display-label_cln"][1:].values)
    Col += list([names[i]]*len(pov_df[1:]))
    vals += list(pov_df[age_col][1:].values)

tmp['Geo'] = Geo
tmp['Col'] = Col
tmp['Val'] = vals
tmp['Val'] = tmp['Val'].astype(int)  * 0.025

geos = tmp.groupby("Geo").agg({"Val" : "sum"}).sort_values("Val", ascending = False)[:75].reset_index()['Geo']
tmp1 = tmp[tmp["Geo"].isin(geos)]
data = [go.Scatter(x = tmp1["Geo"], y = tmp1["Col"], mode="markers", marker=dict(color="red", size=list(tmp1["Val"].values)))]
layout = dict(title="Age Distribution by Census Tract - Marion County, Indiana", legend=dict(x=-0.1, y=1, orientation="h"), 
              margin=dict(l=150, b=100), height=600, barmode="stack")
fig = go.Figure(data=data, layout=layout)
iplot(fig)
basepath = "../input/cpe-data/Dept_23-00089/23-00089_ACS_data/23-00089_ACS_owner-occupied-housing/"
a_df = pd.read_csv(basepath + "ACS_15_5YR_S2502_metadata.csv")
# for i,val in a_df.iterrows():
#     if "Estimate" in val['Id']:
#         if "Owner-occupied" in val["Id"]:
#             print (val['GEO.id'], val["Id"])
a_df.T.head()    
basepath = "../input/cpe-data/Dept_23-00089/23-00089_ACS_data/23-00089_ACS_education-attainment/"
a_df = pd.read_csv(basepath + "ACS_15_5YR_S1501_metadata.csv")
a_df.T.head()
a_df = pd.read_csv(basepath + "ACS_15_5YR_S1501_with_ann.csv")
a_df.head()
path = "../input/cpe-data/Dept_35-00103/35-00103_UOF-OIS-P_prepped.csv"
incidents = pd.read_csv(path)
incidents.head()
incidents["SUBJECT_INJURY_TYPE"].value_counts()
incidents.shape[0]
kmap = folium.Map([35.22, -80.89], height=400, zoom_start=10, tiles='CartoDB dark_matter')
for j, rown in incidents[1:].iterrows():
    if str(rown["LOCATION_LONGITUDE"]) != "nan":
        lon = float(rown["LOCATION_LATITUDE"])
        lat = float(rown["LOCATION_LONGITUDE"])
        folium.CircleMarker([lon, lat], radius=5, color='red', fill=True).add_to(kmap)
kmap
imap = folium.Map([35.22, -80.89], height=400, zoom_start=10, tiles='CartoDB positron')
for j, rown in incidents[1:].iterrows():
    if str(rown["LOCATION_LONGITUDE"]) != "nan":
        lon = float(rown["LOCATION_LATITUDE"])
        lat = float(rown["LOCATION_LONGITUDE"])
        
        if rown["SUBJECT_RACE"] == "Black":
            col = "black"
        elif rown["SUBJECT_RACE"]== "White":
            col = "green"
        elif rown["SUBJECT_RACE"]== "Hispanic":
            col = "yellow"
        else:
            col = "red"
                
        folium.CircleMarker([lon, lat], radius=5, color=col, fill=True).add_to(imap)    
imap
imap = folium.Map([35.22, -80.89], height=400, zoom_start=10, tiles='CartoDB positron')
for j, rown in incidents[1:].iterrows():
    if str(rown["LOCATION_LONGITUDE"]) != "nan":
        lon = float(rown["LOCATION_LATITUDE"])
        lat = float(rown["LOCATION_LONGITUDE"])
        
        if rown["SUBJECT_GENDER"] == "Male":
            col = "blue"
        else:
            col = "red"
                
        folium.CircleMarker([lon, lat], radius=5, color=col, fill=True).add_to(imap)        
imap
imap = folium.Map([35.22, -80.89], height=400, zoom_start=10, tiles='CartoDB positron')
for j, rown in incidents[1:].iterrows():
    if str(rown["LOCATION_LONGITUDE"]) != "nan":
        lon = float(rown["LOCATION_LATITUDE"])
        lat = float(rown["LOCATION_LONGITUDE"])
        
        if rown["SUBJECT_INJURY_TYPE"] == "Non-Fatal Injury":
            col = "red"
        elif rown["SUBJECT_INJURY_TYPE"] == "Fatal Injury":
            col = "green"
        else:
            col = "blue"                
        folium.CircleMarker([lon, lat], radius=5, color=col, fill=True).add_to(imap)        
imap
p2 = """../input/cpe-data/Dept_35-00103/35-00103_Shapefiles/CMPD_Police_Division_Offices.shp"""
One = gpd.read_file(p2) 
for j, rown in One.iterrows():
    lon = float(str(rown["geometry"]).split()[1].replace("(",""))
    lat = float(str(rown["geometry"]).split()[2].replace(")",""))
    folium.CircleMarker([lat, lon], radius=5, color='blue', fill=True).add_to(kmap)
kmap
p1 = """../input/cpe-data/Dept_23-00089/23-00089_Shapefiles/Indianapolis_Police_Zones.shp"""
One = gpd.read_file(p1)  
One.head()
mapa = folium.Map([39.81, -86.26060805912148], height=400, zoom_start=10, tiles='CartoDB dark_matter',API_key='wrobstory.map-12345678')
folium.GeoJson(One).add_to(mapa)
mapa 
f, ax = plt.subplots(1, figsize=(10, 8))
One.plot(column="DISTRICT", ax=ax, cmap='Accent',legend=True);
plt.title("Districts : Indianapolis Police Zones")
plt.show()
f, ax = plt.subplots(1, figsize=(10, 8))
One.plot(column="JURISDCTN", ax=ax, cmap='Accent', legend=True);
plt.title("JuriDiction : Indianapolis Police Zones")
plt.show()
p3 = """../input/cpe-data/Dept_11-00091/11-00091_Shapefiles/boston_police_districts_f55.shp"""
One = gpd.read_file(p3)  
mapa = folium.Map([42.3, -71.0], height=400, zoom_start=10,  tiles='CartoDB dark_matter',API_key='wrobstory.map-12345678')
folium.GeoJson(One).add_to(mapa)
mapa 
p4 = """../input/cpe-data/Dept_37-00049/37-00049_Shapefiles/EPIC.shp"""
One = gpd.read_file(p4)  
mapa = folium.Map([32.7, -96.7],zoom_start=10, height=400, tiles='CartoDB dark_matter',API_key='wrobstory.map-12345678')
folium.GeoJson(One).add_to(mapa)
mapa 
p5 = "../input/cpe-data/Dept_37-00027/37-00027_UOF-P_2014-2016_prepped.csv"
dept_37_27_df = pd.read_csv(p5)[1:]
dept_37_27_df["INCIDENT_DATE"] = pd.to_datetime(dept_37_27_df["INCIDENT_DATE"]).astype(str)
dept_37_27_df["MonthDate"] = dept_37_27_df["INCIDENT_DATE"].apply(lambda x : x.split("-")[0] +'-'+ x.split("-")[1] + "-01")

tmp = dept_37_27_df.groupby("MonthDate").agg({"INCIDENT_REASON" : "count"}).reset_index()
tmp

import plotly.plotly as py
import plotly.graph_objs as go

trace1 = go.Scatter(x=tmp["MonthDate"], y=tmp.INCIDENT_REASON, name="Month wise Incidents")
# trace2 = go.Scatter(x=tmp["MonthDate"], y=tmp.INCIDENT_REASON)

data = [trace1]
layout = go.Layout(height=400, title="Incidents in Austin Texas")
fig = go.Figure(data, layout)
iplot(fig)
a = dept_37_27_df["SUBJECT_GENDER"].value_counts()
tr1 = go.Bar(x = a.index, y = a.values, name="Gender")

a = dept_37_27_df["INCIDENT_REASON"].value_counts()
tr2 = go.Bar(x = a.index, y = a.values, name="INCIDENT_REASON")

a = dept_37_27_df["SUBJECT_RACE"].value_counts()
tr3 = go.Bar(x = a.index, y = a.values, name="SUBJECT_RACE")


fig = tools.make_subplots(rows=1, cols=3, print_grid=False, subplot_titles=["Gender", "Incident Reason", "Subject Race"])
fig.append_trace(tr1, 1, 1);
fig.append_trace(tr2, 1, 2);
fig.append_trace(tr3, 1, 3);
fig['layout'].update(height=400, title="Austin Incidents Distribution", showlegend=False);
iplot(fig, filename='simple-subplot');
a = dept_37_27_df["REASON_FOR_FORCE"].value_counts()[:6]
tr1 = go.Bar(x = a.index, y = a.values, name="Gender")

a = dept_37_27_df["TYPE_OF_FORCE_USED1"].value_counts()[:8]
tr2 = go.Bar(x = a.index, y = a.values, name="INCIDENT_REASON")

fig = tools.make_subplots(rows=1, cols=2, print_grid=False, subplot_titles=["REASON_FOR_FORCE", "TYPE_OF_FORCE_USED1"])
fig.append_trace(tr1, 1, 1);
fig.append_trace(tr2, 1, 2);
fig['layout'].update(height=400, margin=dict(b=140), title="Austin Incidents Distribution", showlegend=False);
iplot(fig, filename='simple-subplot');
p5 = "../input/cpe-data/Dept_37-00027/37-00027_Shapefiles/APD_DIST.shp"
dept_37_27_shp = gpd.read_file(p5)
dept_37_27_shp.head()
f, ax = plt.subplots(1, figsize=(10, 12))
dept_37_27_shp.plot(column="SECTOR", ax=ax, cmap='Accent',legend=True);
plt.title("Sectors ")
plt.show()
f, ax = plt.subplots(1, figsize=(10, 12))
dept_37_27_shp.plot(column="PATROL_ARE", ax=ax, cmap='coolwarm',legend=True);
plt.title("Patrol Areas ")
plt.show()
from shapely.geometry import Point

## remove na
notna = dept_37_27_df[['LOCATION_LATITUDE','LOCATION_LONGITUDE']].dropna().index
dept_37_27_df = dept_37_27_df.iloc[notna].reset_index(drop=True)
dept_37_27_df['coordinates'] = (dept_37_27_df.apply(lambda x: Point(float(x['LOCATION_LONGITUDE']), float(x['LOCATION_LATITUDE'])), axis=1))
dept_37_27_gdf = gpd.GeoDataFrame(dept_37_27_df, geometry='coordinates')

# ## make the corrdinate system same
dept_37_27_gdf.crs = {'init' :'epsg:4326'}
dept_37_27_shp.crs = {'init' :'esri:102739'}
dept_37_27_shp = dept_37_27_shp.to_crs(epsg='4326')
## plot
f, ax = plt.subplots(1, figsize=(10, 12))
dept_37_27_shp.plot(ax=ax, column='PATROL_ARE', cmap = "gray", legend=True)
dept_37_27_gdf.plot(ax=ax, marker='*', color='red', markersize=10)
plt.title("Incident Locations and Patrol Areas ")
plt.show()
## plot
f, ax = plt.subplots(1, figsize=(10, 12))
dept_37_27_shp.plot(ax=ax, column='SECTOR', cmap = "Oranges", legend=True)
dept_37_27_gdf.plot(ax=ax, marker='*', color='Black', markersize=10)
plt.title("Incident Locations and Sectors ")
plt.show()