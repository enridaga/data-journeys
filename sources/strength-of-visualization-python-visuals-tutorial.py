
import pandas as pd  #pandas for using dataframe and reading csv 

import numpy as np   #numpy for vector operations and basic maths 

#import simplejson    #getting JSON in simplified format

import urllib        #for url stuff

#import gmaps       #for using google maps to visulalize places on maps

import re            #for processing regular expressions

import datetime      #for datetime operations

import calendar      #for calendar for datetime operations

import time          #to get the system time

import scipy         #for other dependancies

from sklearn.cluster import KMeans # for doing K-means clustering

from haversine import haversine # for calculating haversine distance

import math          #for basic maths operations

import seaborn as sns #for making plots

import matplotlib.pyplot as plt # for plotting

import os  # for os commands

from scipy.misc import imread, imresize, imsave  # for plots 

import plotly.plotly as py

import plotly.graph_objs as go

import plotly

from bokeh.palettes import Spectral4

from bokeh.plotting import figure, output_notebook, show

from IPython.display import HTML

from matplotlib.pyplot import *

from matplotlib import cm

from matplotlib import animation

import io

import base64

output_notebook()

plotly.offline.init_notebook_mode() # run at the start of every ipython notebook
s = time.time()

train_fr_1 = pd.read_csv('../input/new-york-city-taxi-with-osrm/fastest_routes_train_part_1.csv')

train_fr_2 = pd.read_csv('../input/new-york-city-taxi-with-osrm/fastest_routes_train_part_2.csv')

train_fr = pd.concat([train_fr_1, train_fr_2])

train_fr_new = train_fr[['id', 'total_distance', 'total_travel_time', 'number_of_steps']]

train_df = pd.read_csv('../input/new-york-city-taxi-with-osrm/train.csv')

train = pd.merge(train_df, train_fr_new, on = 'id', how = 'left')

train_df = train.copy()

end = time.time()

print("Time taken by above cell is {}.".format((end-s)))

train_df.head()
# checking if Ids are unique, 

start = time.time()

train_data = train_df.copy()

start = time.time()

print("Number of columns and rows and columns are {} and {} respectively.".format(train_data.shape[1], train_data.shape[0]))

if train_data.id.nunique() == train_data.shape[0]:

    print("Train ids are unique")

print("Number of Nulls - {}.".format(train_data.isnull().sum().sum()))

end = time.time()

print("Time taken by above cell is {}.".format(end-start))
### matplotlib inline

start = time.time()

sns.set(style="white", palette="muted", color_codes=True)

f, axes = plt.subplots(1, 1, figsize=(11, 7), sharex=True)

sns.despine(left=True)

sns.distplot(np.log(train_df['trip_duration'].values+1), axlabel = 'Log(trip_duration)', label = 'log(trip_duration)', bins = 50, color="r")

plt.setp(axes, yticks=[])

plt.tight_layout()

end = time.time()

print("Time taken by above cell is {}.".format((end-start)))

plt.show()
start = time.time()

sns.set(style="white", palette="muted", color_codes=True)

f, axes = plt.subplots(2,2,figsize=(10, 10), sharex=False, sharey = False)

sns.despine(left=True)

sns.distplot(train_df['pickup_latitude'].values, label = 'pickup_latitude',color="m",bins = 100, ax=axes[0,0])

sns.distplot(train_df['pickup_longitude'].values, label = 'pickup_longitude',color="m",bins =100, ax=axes[0,1])

sns.distplot(train_df['dropoff_latitude'].values, label = 'dropoff_latitude',color="m",bins =100, ax=axes[1, 0])

sns.distplot(train_df['dropoff_longitude'].values, label = 'dropoff_longitude',color="m",bins =100, ax=axes[1, 1])

plt.setp(axes, yticks=[])

plt.tight_layout()

end = time.time()

print("Time taken by above cell is {}.".format((end-start)))

plt.show()
start = time.time()

df = train_df.loc[(train_df.pickup_latitude > 40.6) & (train_df.pickup_latitude < 40.9)]

df = df.loc[(df.dropoff_latitude>40.6) & (df.dropoff_latitude < 40.9)]

df = df.loc[(df.dropoff_longitude > -74.05) & (df.dropoff_longitude < -73.7)]

df = df.loc[(df.pickup_longitude > -74.05) & (df.pickup_longitude < -73.7)]

train_data_new = df.copy()

sns.set(style="white", palette="muted", color_codes=True)

f, axes = plt.subplots(2,2,figsize=(12, 12), sharex=False, sharey = False)#

sns.despine(left=True)

sns.distplot(train_data_new['pickup_latitude'].values, label = 'pickup_latitude',color="m",bins = 100, ax=axes[0,0])

sns.distplot(train_data_new['pickup_longitude'].values, label = 'pickup_longitude',color="g",bins =100, ax=axes[0,1])

sns.distplot(train_data_new['dropoff_latitude'].values, label = 'dropoff_latitude',color="m",bins =100, ax=axes[1, 0])

sns.distplot(train_data_new['dropoff_longitude'].values, label = 'dropoff_longitude',color="g",bins =100, ax=axes[1, 1])

plt.setp(axes, yticks=[])

plt.tight_layout()

end = time.time()

print("Time taken by above cell is {}.".format((end-start)))

print(df.shape[0], train_data.shape[0])

plt.show()
start = time.time()

temp = train_data.copy()

train_data['pickup_datetime'] = pd.to_datetime(train_data.pickup_datetime)

train_data.loc[:, 'pick_date'] = train_data['pickup_datetime'].dt.date

train_data.head()



ts_v1 = pd.DataFrame(train_data.loc[train_data['vendor_id']==1].groupby('pick_date')['trip_duration'].mean())

ts_v1.reset_index(inplace = True)

ts_v2 = pd.DataFrame(train_data.loc[train_data.vendor_id==2].groupby('pick_date')['trip_duration'].mean())

ts_v2.reset_index(inplace = True)



from bokeh.palettes import Spectral4

from bokeh.plotting import figure, output_notebook, show

#from bokeh.sampledata.stocks import AAPL, IBM, MSFT, GOOG

output_notebook()

p = figure(plot_width=800, plot_height=250, x_axis_type="datetime")

p.title.text = 'Click on legend entries to hide the corresponding lines'



for data, name, color in zip([ts_v1, ts_v2], ["vendor 1", "vendor 2"], Spectral4):

    df = data

    p.line(df['pick_date'], df['trip_duration'], line_width=2, color=color, alpha=0.8, legend=name)



p.legend.location = "top_left"

p.legend.click_policy="hide"

show(p)

end = time.time()

train_data = temp

print("Time Taken by above cell is {}.".format(end - start))
start = time.time()

rgb = np.zeros((3000, 3500, 3), dtype=np.uint8)

rgb[..., 0] = 0

rgb[..., 1] = 0

rgb[..., 2] = 0

train_data_new['pick_lat_new'] = list(map(int, (train_data_new['pickup_latitude'] - (40.6000))*10000))

train_data_new['drop_lat_new'] = list(map(int, (train_data_new['dropoff_latitude'] - (40.6000))*10000))

train_data_new['pick_lon_new'] = list(map(int, (train_data_new['pickup_longitude'] - (-74.050))*10000))

train_data_new['drop_lon_new'] = list(map(int,(train_data_new['dropoff_longitude'] - (-74.050))*10000))



summary_plot = pd.DataFrame(train_data_new.groupby(['pick_lat_new', 'pick_lon_new'])['id'].count())



summary_plot.reset_index(inplace = True)

summary_plot.head(120)

lat_list = summary_plot['pick_lat_new'].unique()

for i in lat_list:

    lon_list = summary_plot.loc[summary_plot['pick_lat_new']==i]['pick_lon_new'].tolist()

    unit = summary_plot.loc[summary_plot['pick_lat_new']==i]['id'].tolist()

    for j in lon_list:

        a = unit[lon_list.index(j)]

        if (a//50) >0:

            rgb[i][j][0] = 255

            rgb[i,j, 1] = 0

            rgb[i,j, 2] = 255

        elif (a//10)>0:

            rgb[i,j, 0] = 0

            rgb[i,j, 1] = 255

            rgb[i,j, 2] = 0

        else:

            rgb[i,j, 0] = 255

            rgb[i,j, 1] = 0

            rgb[i,j, 2] = 0

fig, ax = plt.subplots(nrows=1,ncols=1,figsize=(14,20))

end = time.time()

print("Time taken by above cell is {}.".format((end-start)))

ax.imshow(rgb, cmap = 'hot')

ax.set_axis_off() 
start = time.time()

def haversine_(lat1, lng1, lat2, lng2):

    """function to calculate haversine distance between two co-ordinates"""

    lat1, lng1, lat2, lng2 = map(np.radians, (lat1, lng1, lat2, lng2))

    AVG_EARTH_RADIUS = 6371  # in km

    lat = lat2 - lat1

    lng = lng2 - lng1

    d = np.sin(lat * 0.5) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(lng * 0.5) ** 2

    h = 2 * AVG_EARTH_RADIUS * np.arcsin(np.sqrt(d))

    return(h)



def manhattan_distance_pd(lat1, lng1, lat2, lng2):

    """function to calculate manhatten distance between pick_drop"""

    a = haversine_(lat1, lng1, lat1, lng2)

    b = haversine_(lat1, lng1, lat2, lng1)

    return a + b



import math

def bearing_array(lat1, lng1, lat2, lng2):

    """ function was taken from beluga's notebook as this function works on array

    while my function used to work on individual elements and was noticably slow"""

    AVG_EARTH_RADIUS = 6371  # in km

    lng_delta_rad = np.radians(lng2 - lng1)

    lat1, lng1, lat2, lng2 = map(np.radians, (lat1, lng1, lat2, lng2))

    y = np.sin(lng_delta_rad) * np.cos(lat2)

    x = np.cos(lat1) * np.sin(lat2) - np.sin(lat1) * np.cos(lat2) * np.cos(lng_delta_rad)

    return np.degrees(np.arctan2(y, x))



end = time.time()

print("Time taken by above cell is {}.".format((end-start)))
start = time.time()

train_data = temp.copy()

train_data['pickup_datetime'] = pd.to_datetime(train_data.pickup_datetime)

train_data.loc[:, 'pick_month'] = train_data['pickup_datetime'].dt.month

train_data.loc[:, 'hour'] = train_data['pickup_datetime'].dt.hour

train_data.loc[:, 'week_of_year'] = train_data['pickup_datetime'].dt.weekofyear

train_data.loc[:, 'day_of_year'] = train_data['pickup_datetime'].dt.dayofyear

train_data.loc[:, 'day_of_week'] = train_data['pickup_datetime'].dt.dayofweek

train_data.loc[:,'hvsine_pick_drop'] = haversine_(train_data['pickup_latitude'].values, train_data['pickup_longitude'].values, train_data['dropoff_latitude'].values, train_data['dropoff_longitude'].values)

train_data.loc[:,'manhtn_pick_drop'] = manhattan_distance_pd(train_data['pickup_latitude'].values, train_data['pickup_longitude'].values, train_data['dropoff_latitude'].values, train_data['dropoff_longitude'].values)

train_data.loc[:,'bearing'] = bearing_array(train_data['pickup_latitude'].values, train_data['pickup_longitude'].values, train_data['dropoff_latitude'].values, train_data['dropoff_longitude'].values)



end = time.time()

print("Time taken by above cell is {}.".format(end-start))
start = time.time()

def color(hour):

    """function for color change in animation"""

    return(10*hour)



def Animation(hour, temp, rgb):

    """Function to generate return a pic of plotings"""

    #ax.clear()

    train_data_new = temp.loc[temp['hour'] == hour]

    start = time.time()

    rgb = np.zeros((3000, 3500, 3), dtype=np.uint8)

    rgb[..., 0] = 0

    rgb[..., 1] = 0

    rgb[..., 2] = 0

    train_data_new['pick_lat_new'] = list(map(int, (train_data_new['pickup_latitude'] - (40.6000))*10000))

    train_data_new['drop_lat_new'] = list(map(int, (train_data_new['dropoff_latitude'] - (40.6000))*10000))

    train_data_new['pick_lon_new'] = list(map(int, (train_data_new['pickup_longitude'] - (-74.050))*10000))

    train_data_new['drop_lon_new'] = list(map(int,(train_data_new['dropoff_longitude'] - (-74.050))*10000))



    summary_plot = pd.DataFrame(train_data_new.groupby(['pick_lat_new', 'pick_lon_new'])['id'].count())



    summary_plot.reset_index(inplace = True)

    summary_plot.head(120)

    lat_list = summary_plot['pick_lat_new'].unique()

    for i in lat_list:

        #print(i)

        lon_list = summary_plot.loc[summary_plot['pick_lat_new']==i]['pick_lon_new'].tolist()

        unit = summary_plot.loc[summary_plot['pick_lat_new']==i]['id'].tolist()

        for j in lon_list:

            #j = int(j)

            a = unit[lon_list.index(j)]

            #print(a)

            if (a//50) >0:

                rgb[i][j][0] = 255 - color(hour)

                rgb[i,j, 1] = 255 - color(hour)

                rgb[i,j, 2] = 0 + color(hour)

            elif (a//10)>0:

                rgb[i,j, 0] = 0 + color(hour)

                rgb[i,j, 1] = 255 - color(hour)

                rgb[i,j, 2] = 0 + color(hour)

            else:

                rgb[i,j, 0] = 255 - color(hour)

                rgb[i,j, 1] = 0 + color(hour)

                rgb[i,j, 2] = 0 + color(hour)

    #fig, ax = plt.subplots(nrows=1,ncols=1,figsize=(14,20))

    end = time.time()

    print("Time taken by above cell is {} for {}.".format((end-start), hour))

    return(rgb)

end = time.time()

print("Time taken by above cell is {}.".format(end -start))
start = time.time()

images_list=[]

train_data_new['pickup_datetime'] = pd.to_datetime(train_data_new.pickup_datetime)

train_data_new.loc[:, 'hour'] = train_data_new['pickup_datetime'].dt.hour



for i in list(range(0, 24)):

    im = Animation(i, train_data_new, rgb.copy())

    images_list.append(im)

end = time.time()

print("Time taken by above cell is {}.".format(end -start))
start = time.time()

def build_gif(imgs = images_list, show_gif=False, save_gif=True, title=''):

    """function to create a gif of heatmaps"""

    fig, ax = plt.subplots(nrows=1,ncols=1,figsize=(10,10))

    ax.set_axis_off()

    hr_range = list(range(0,24))

    def show_im(pairs):

        ax.clear()

        ax.set_title('Absolute Traffic - Hour ' + str(int(pairs[0])) + ':00')

        ax.imshow(pairs[1])

        ax.set_axis_off() 

    pairs = list(zip(hr_range, imgs))

    #ims = map(lambda x: (ax.imshow(x), ax.set_title(title)), imgs)

    im_ani = animation.FuncAnimation(fig, show_im, pairs,interval=500, repeat_delay=0, blit=False)

    plt.cla()

    if save_gif:

        im_ani.save('animation.gif', writer='imagemagick') #, writer='imagemagick'

    if show_gif:

        plt.show()

    return

end = time.time()

print("Time taken by above cell is {}".format(end-start))
start = time.time()

build_gif()

end = time.time()

print(end-start)
filename = 'animation.gif'

video = io.open(filename, 'r+b').read()

encoded = base64.b64encode(video)

HTML(data='''<img src="data:image/gif;base64,{0}" type="gif" />'''.format(encoded.decode('ascii')))
start = time.time()

summary_wdays_avg_duration = pd.DataFrame(train_data.groupby(['vendor_id','day_of_week'])['trip_duration'].mean())

summary_wdays_avg_duration.reset_index(inplace = True)

summary_wdays_avg_duration['unit']=1

sns.set(style="white", palette="muted", color_codes=True)

sns.set_context("poster")

sns.tsplot(data=summary_wdays_avg_duration, time="day_of_week", unit = "unit", condition="vendor_id", value="trip_duration")

sns.despine(bottom = False)

end = time.time()

print(end - start)

import seaborn as sns

sns.set(style="whitegrid", palette="pastel", color_codes=True)

sns.set_context("poster")

train_data2 = train_data.copy()

train_data2['trip_duration']= np.log(train_data['trip_duration'])

sns.violinplot(x="passenger_count", y="trip_duration", hue="vendor_id", data=train_data2, split=True,

               inner="quart",palette={1: "g", 2: "r"})



sns.despine(left=True)

print(df.shape[0])
start = time.time()

sns.set(style="ticks")

sns.set_context("poster")

sns.boxplot(x="day_of_week", y="trip_duration", hue="vendor_id", data=train_data, palette="PRGn")

plt.ylim(0, 6000)

sns.despine(offset=10, trim=True)

print(train_data.trip_duration.max())

end = time.time()

print("Time taken by above cell is {}.".format(end-start))
summary_hour_duration = pd.DataFrame(train_data.groupby(['day_of_week','hour'])['trip_duration'].mean())

summary_hour_duration.reset_index(inplace = True)

summary_hour_duration['unit']=1

sns.set(style="white", palette="muted", color_codes=False)

sns.set_context("poster")

sns.tsplot(data=summary_hour_duration, time="hour", unit = "unit", condition="day_of_week", value="trip_duration")

sns.despine(bottom = False)
start = time.time()

def assign_cluster(df, k):

    """function to assign clusters """

    df_pick = df[['pickup_longitude','pickup_latitude']]

    df_drop = df[['dropoff_longitude','dropoff_latitude']]

    """I am using initialization as from the output of

    k-means from my local machine to save time in this kernel"""

    init = np.array([[ -73.98737616,   40.72981533],

       [-121.93328857,   37.38933945],

       [ -73.78423222,   40.64711269],

       [ -73.9546417 ,   40.77377538],

       [ -66.84140269,   36.64537175],

       [ -73.87040541,   40.77016484],

       [ -73.97316185,   40.75814346],

       [ -73.98861094,   40.7527791 ],

       [ -72.80966949,   51.88108444],

       [ -76.99779701,   38.47370625],

       [ -73.96975298,   40.69089596],

       [ -74.00816622,   40.71414939],

       [ -66.97216034,   44.37194443],

       [ -61.33552933,   37.85105133],

       [ -73.98001393,   40.7783577 ],

       [ -72.00626526,   43.20296402],

       [ -73.07618713,   35.03469086],

       [ -73.95759366,   40.80316361],

       [ -79.20167796,   41.04752096],

       [ -74.00106031,   40.73867723]])

    k_means_pick = KMeans(n_clusters=k, init=init, n_init=1)

    k_means_pick.fit(df_pick)

    clust_pick = k_means_pick.labels_

    df['label_pick'] = clust_pick.tolist()

    df['label_drop'] = k_means_pick.predict(df_drop)

    return df, k_means_pick



end = time.time()

print("time taken by thie script by now is {}.".format(end-start))
start = time.time()

train_cl, k_means = assign_cluster(train_data, 20)  # make it 100 when extracting features 

centroid_pickups = pd.DataFrame(k_means.cluster_centers_, columns = ['centroid_pick_long', 'centroid_pick_lat'])

centroid_dropoff = pd.DataFrame(k_means.cluster_centers_, columns = ['centroid_drop_long', 'centroid_drop_lat'])

centroid_pickups['label_pick'] = centroid_pickups.index

centroid_dropoff['label_drop'] = centroid_dropoff.index

#centroid_pickups.head()

train_cl = pd.merge(train_cl, centroid_pickups, how='left', on=['label_pick'])

train_cl = pd.merge(train_cl, centroid_dropoff, how='left', on=['label_drop'])

#train_cl.head()

end = time.time()

print("Time taken in clustering is {}.".format(end - start))
start = time.time()

train_cl.loc[:,'hvsine_pick_cent_p'] = haversine_(train_cl['pickup_latitude'].values, train_cl['pickup_longitude'].values, train_cl['centroid_pick_lat'].values, train_cl['centroid_pick_long'].values)

train_cl.loc[:,'hvsine_drop_cent_d'] = haversine_(train_cl['dropoff_latitude'].values, train_cl['dropoff_longitude'].values, train_cl['centroid_drop_lat'].values, train_cl['centroid_drop_long'].values)

train_cl.loc[:,'hvsine_cent_p_cent_d'] = haversine_(train_cl['centroid_pick_lat'].values, train_cl['centroid_pick_long'].values, train_cl['centroid_drop_lat'].values, train_cl['centroid_drop_long'].values)

train_cl.loc[:,'manhtn_pick_cent_p'] = manhattan_distance_pd(train_cl['pickup_latitude'].values, train_cl['pickup_longitude'].values, train_cl['centroid_pick_lat'].values, train_cl['centroid_pick_long'].values)

train_cl.loc[:,'manhtn_drop_cent_d'] = manhattan_distance_pd(train_cl['dropoff_latitude'].values, train_cl['dropoff_longitude'].values, train_cl['centroid_drop_lat'].values, train_cl['centroid_drop_long'].values)

train_cl.loc[:,'manhtn_cent_p_cent_d'] = manhattan_distance_pd(train_cl['centroid_pick_lat'].values, train_cl['centroid_pick_long'].values, train_cl['centroid_drop_lat'].values, train_cl['centroid_drop_long'].values)



train_cl.loc[:,'bearing_pick_cent_p'] = bearing_array(train_cl['pickup_latitude'].values, train_cl['pickup_longitude'].values, train_cl['centroid_pick_lat'].values, train_cl['centroid_pick_long'].values)

train_cl.loc[:,'bearing_drop_cent_p'] = bearing_array(train_cl['dropoff_latitude'].values, train_cl['dropoff_longitude'].values, train_cl['centroid_drop_lat'].values, train_cl['centroid_drop_long'].values)

train_cl.loc[:,'bearing_cent_p_cent_d'] = bearing_array(train_cl['centroid_pick_lat'].values, train_cl['centroid_pick_long'].values, train_cl['centroid_drop_lat'].values, train_cl['centroid_drop_long'].values)

train_cl['speed_hvsn'] = train_cl.hvsine_pick_drop/train_cl.total_travel_time

train_cl['speed_manhtn'] = train_cl.manhtn_pick_drop/train_cl.total_travel_time

end = time.time()

print("Time Taken by above cell is {}.".format(end-start))

train_cl.head()
start = time.time()

def cluster_summary(sum_df):

    """function to calculate summary of given list of clusters """

    #agg_func = {'trip_duration':'mean','label_drop':'count','bearing':'mean','id':'count'} # that's how you use agg function with groupby

    summary_avg_time = pd.DataFrame(sum_df.groupby('label_pick')['trip_duration'].mean())

    summary_avg_time.reset_index(inplace = True)

    summary_pref_clus = pd.DataFrame(sum_df.groupby(['label_pick', 'label_drop'])['id'].count())

    summary_pref_clus = summary_pref_clus.reset_index()

    summary_pref_clus = summary_pref_clus.loc[summary_pref_clus.groupby('label_pick')['id'].idxmax()]

    summary =pd.merge(summary_avg_time, summary_pref_clus, how = 'left', on = 'label_pick')

    summary = summary.rename(columns={'trip_duration':'avg_triptime'})

    return summary

end = time.time()

print("Time Taken by above cell is {}.".format(end-start))
import folium

def show_fmaps(train_data, path=1):

    """function to generate map and add the pick up and drop coordinates

    1. Path = 1 : Join pickup (blue) and drop(red) using a straight line

    """

    full_data = train_data

    summary_full_data = pd.DataFrame(full_data.groupby('label_pick')['id'].count())

    summary_full_data.reset_index(inplace = True)

    summary_full_data = summary_full_data.loc[summary_full_data['id']>70000]

    map_1 = folium.Map(location=[40.767937, -73.982155], zoom_start=10,tiles='Stamen Toner') # manually added centre

    new_df = train_data.loc[train_data['label_pick'].isin(summary_full_data.label_pick.tolist())].sample(50)

    new_df.reset_index(inplace = True, drop = True)

    for i in range(new_df.shape[0]):

        pick_long = new_df.loc[new_df.index ==i]['pickup_longitude'].values[0]

        pick_lat = new_df.loc[new_df.index ==i]['pickup_latitude'].values[0]

        dest_long = new_df.loc[new_df.index ==i]['dropoff_longitude'].values[0]

        dest_lat = new_df.loc[new_df.index ==i]['dropoff_latitude'].values[0]

        folium.Marker([pick_lat, pick_long]).add_to(map_1)

        folium.Marker([dest_lat, dest_long]).add_to(map_1)

    return map_1
def clusters_map(clus_data, full_data, tile = 'OpenStreetMap', sig = 0, zoom = 12, circle = 0, radius_ = 30):

    """ function to plot clusters on map"""

    map_1 = folium.Map(location=[40.767937, -73.982155], zoom_start=zoom,tiles= tile) # 'Mapbox' 'Stamen Toner'

    summary_full_data = pd.DataFrame(full_data.groupby('label_pick')['id'].count())

    summary_full_data.reset_index(inplace = True)

    if sig == 1:

        summary_full_data = summary_full_data.loc[summary_full_data['id']>70000]

    sig_cluster = summary_full_data['label_pick'].tolist()

    clus_summary = cluster_summary(full_data)

    for i in sig_cluster:

        pick_long = clus_data.loc[clus_data.index ==i]['centroid_pick_long'].values[0]

        pick_lat = clus_data.loc[clus_data.index ==i]['centroid_pick_lat'].values[0]

        clus_no = clus_data.loc[clus_data.index ==i]['label_pick'].values[0]

        most_visited_clus = clus_summary.loc[clus_summary['label_pick']==i]['label_drop'].values[0]

        avg_triptime = clus_summary.loc[clus_summary['label_pick']==i]['avg_triptime'].values[0]

        pop = 'cluster = '+str(clus_no)+' & most visited cluster = ' +str(most_visited_clus) +' & avg triptime from this cluster =' + str(avg_triptime)

        if circle == 1:

            folium.CircleMarker(location=[pick_lat, pick_long], radius=radius_,

                    color='#F08080',

                    fill_color='#3186cc', popup=pop).add_to(map_1)

        folium.Marker([pick_lat, pick_long], popup=pop).add_to(map_1)

    return map_1
osm = show_fmaps(train_data, path=1)

osm
clus_map = clusters_map(centroid_pickups, train_cl, sig =0, zoom =3.2, circle =1, tile = 'Stamen Terrain')

clus_map
clus_map_sig = clusters_map(centroid_pickups, train_cl, sig =1, circle =1)

clus_map_sig
from pandas.tools.plotting import parallel_coordinates

parallel_coordinates(train_data.sample(1200)[['vendor_id','day_of_week', 'passenger_count', 'pick_month','label_pick', 'hour']], 'vendor_id', colormap='rainbow')

plt.show()
test_df = pd.read_csv('../input/nyc-taxi-trip-duration/test.csv')

test_fr = pd.read_csv('../input/new-york-city-taxi-with-osrm/fastest_routes_test.csv')

test_fr_new = test_fr[['id', 'total_distance', 'total_travel_time', 'number_of_steps']]

test_df = pd.merge(test_df, test_fr_new, on = 'id', how = 'left')

test_df.head()
start = time.time()

test_data = test_df.copy()

test_data['pickup_datetime'] = pd.to_datetime(test_data.pickup_datetime)

test_data.loc[:, 'pick_month'] = test_data['pickup_datetime'].dt.month

test_data.loc[:, 'hour'] = test_data['pickup_datetime'].dt.hour

test_data.loc[:, 'week_of_year'] = test_data['pickup_datetime'].dt.weekofyear

test_data.loc[:, 'day_of_year'] = test_data['pickup_datetime'].dt.dayofyear

test_data.loc[:, 'day_of_week'] = test_data['pickup_datetime'].dt.dayofweek

end = time.time()

print("Time taken by above cell is {}.".format(end-start))
strat = time.time()

test_data.loc[:,'hvsine_pick_drop'] = haversine_(test_data['pickup_latitude'].values, test_data['pickup_longitude'].values, test_data['dropoff_latitude'].values, test_data['dropoff_longitude'].values)

test_data.loc[:,'manhtn_pick_drop'] = manhattan_distance_pd(test_data['pickup_latitude'].values, test_data['pickup_longitude'].values, test_data['dropoff_latitude'].values, test_data['dropoff_longitude'].values)

test_data.loc[:,'bearing'] = bearing_array(test_data['pickup_latitude'].values, test_data['pickup_longitude'].values, test_data['dropoff_latitude'].values, test_data['dropoff_longitude'].values)

end = time.time()

print("Time taken by above cell is {}.".format(end-strat))
start = time.time()

test_data['label_pick'] = k_means.predict(test_data[['pickup_longitude','pickup_latitude']])

test_data['label_drop'] = k_means.predict(test_data[['dropoff_longitude','dropoff_latitude']])

test_cl = pd.merge(test_data, centroid_pickups, how='left', on=['label_pick'])

test_cl = pd.merge(test_cl, centroid_dropoff, how='left', on=['label_drop'])

#test_cl.head()

end = time.time()

print("Time Taken by above cell is {}.".format(end-start))
start = time.time()

test_cl.loc[:,'hvsine_pick_cent_p'] = haversine_(test_cl['pickup_latitude'].values, test_cl['pickup_longitude'].values, test_cl['centroid_pick_lat'].values, test_cl['centroid_pick_long'].values)

test_cl.loc[:,'hvsine_drop_cent_d'] = haversine_(test_cl['dropoff_latitude'].values, test_cl['dropoff_longitude'].values, test_cl['centroid_drop_lat'].values, test_cl['centroid_drop_long'].values)

test_cl.loc[:,'hvsine_cent_p_cent_d'] = haversine_(test_cl['centroid_pick_lat'].values, test_cl['centroid_pick_long'].values, test_cl['centroid_drop_lat'].values, test_cl['centroid_drop_long'].values)

test_cl.loc[:,'manhtn_pick_cent_p'] = manhattan_distance_pd(test_cl['pickup_latitude'].values, test_cl['pickup_longitude'].values, test_cl['centroid_pick_lat'].values, test_cl['centroid_pick_long'].values)

test_cl.loc[:,'manhtn_drop_cent_d'] = manhattan_distance_pd(test_cl['dropoff_latitude'].values, test_cl['dropoff_longitude'].values, test_cl['centroid_drop_lat'].values, test_cl['centroid_drop_long'].values)

test_cl.loc[:,'manhtn_cent_p_cent_d'] = manhattan_distance_pd(test_cl['centroid_pick_lat'].values, test_cl['centroid_pick_long'].values, test_cl['centroid_drop_lat'].values, test_cl['centroid_drop_long'].values)



test_cl.loc[:,'bearing_pick_cent_p'] = bearing_array(test_cl['pickup_latitude'].values, test_cl['pickup_longitude'].values, test_cl['centroid_pick_lat'].values, test_cl['centroid_pick_long'].values)

test_cl.loc[:,'bearing_drop_cent_p'] = bearing_array(test_cl['dropoff_latitude'].values, test_cl['dropoff_longitude'].values, test_cl['centroid_drop_lat'].values, test_cl['centroid_drop_long'].values)

test_cl.loc[:,'bearing_cent_p_cent_d'] = bearing_array(test_cl['centroid_pick_lat'].values, test_cl['centroid_pick_long'].values, test_cl['centroid_drop_lat'].values, test_cl['centroid_drop_long'].values)

test_cl['speed_hvsn'] = test_cl.hvsine_pick_drop/test_cl.total_travel_time

test_cl['speed_manhtn'] = test_cl.manhtn_pick_drop/test_cl.total_travel_time

end = time.time()

print("Time Taken by above cell is {}.".format(end-start))
test_cl.head()
import xgboost as xgb

from sklearn.model_selection import train_test_split

from sklearn.decomposition import PCA

from sklearn.cluster import MiniBatchKMeans

import warnings
# Lets Add PCA features in the model, reference Beluga's PCA

train = train_cl

test = test_cl

start = time.time()

coords = np.vstack((train[['pickup_latitude', 'pickup_longitude']].values,

                    train[['dropoff_latitude', 'dropoff_longitude']].values,

                    test[['pickup_latitude', 'pickup_longitude']].values,

                    test[['dropoff_latitude', 'dropoff_longitude']].values))



pca = PCA().fit(coords)

train['pickup_pca0'] = pca.transform(train[['pickup_latitude', 'pickup_longitude']])[:, 0]

train['pickup_pca1'] = pca.transform(train[['pickup_latitude', 'pickup_longitude']])[:, 1]

train['dropoff_pca0'] = pca.transform(train[['dropoff_latitude', 'dropoff_longitude']])[:, 0]

train['dropoff_pca1'] = pca.transform(train[['dropoff_latitude', 'dropoff_longitude']])[:, 1]

test['pickup_pca0'] = pca.transform(test[['pickup_latitude', 'pickup_longitude']])[:, 0]

test['pickup_pca1'] = pca.transform(test[['pickup_latitude', 'pickup_longitude']])[:, 1]

test['dropoff_pca0'] = pca.transform(test[['dropoff_latitude', 'dropoff_longitude']])[:, 0]

test['dropoff_pca1'] = pca.transform(test[['dropoff_latitude', 'dropoff_longitude']])[:, 1]

end = time.time()

print("Time Taken by above cell is {}.".format(end - start))
train['store_and_fwd_flag_int'] = np.where(train['store_and_fwd_flag']=='N', 0, 1)

test['store_and_fwd_flag_int'] = np.where(test['store_and_fwd_flag']=='N', 0, 1)
feature_names = list(train.columns)

print("Difference of features in train and test are {}".format(np.setdiff1d(train.columns, test.columns)))

print("")

do_not_use_for_training = ['pick_date','id', 'pickup_datetime', 'dropoff_datetime', 'trip_duration', 'store_and_fwd_flag']

feature_names = [f for f in train.columns if f not in do_not_use_for_training]

print("We will be using following features for training {}.".format(feature_names))

print("")

print("Total number of features are {}.".format(len(feature_names)))

y = np.log(train['trip_duration'].values + 1)

start = time.time()

Xtr, Xv, ytr, yv = train_test_split(train[feature_names].values, y, test_size=0.2, random_state=1987)

dtrain = xgb.DMatrix(Xtr, label=ytr)

dvalid = xgb.DMatrix(Xv, label=yv)

dtest = xgb.DMatrix(test[feature_names].values)

watchlist = [(dtrain, 'train'), (dvalid, 'valid')]



xgb_pars = {'min_child_weight': 50, 'eta': 0.3, 'colsample_bytree': 0.3, 'max_depth': 10,

            'subsample': 0.8, 'lambda': 1., 'nthread': -1, 'booster' : 'gbtree', 'silent': 1,

            'eval_metric': 'rmse', 'objective': 'reg:linear'}



# You could try to train with more epoch

model = xgb.train(xgb_pars, dtrain, 15, watchlist, early_stopping_rounds=2,

                  maximize=False, verbose_eval=1)

end = time.time()

print("Time taken by above cell is {}.".format(end - start))

print('Modeling RMSLE %.5f' % model.best_score)
weather = pd.read_csv('../input/weather-data-in-new-york-city-2016/weather_data_nyc_centralpark_2016.csv')

weather.head()
from ggplot import *

weather.date = pd.to_datetime(weather.date)

weather['day_of_year']= weather.date.dt.dayofyear

p = ggplot(aes(x='date'),data=weather) + geom_line(aes(y='minimum temperature', colour = "blue")) + geom_line(aes(y='maximum temerature', colour = "red"))

p + geom_point(aes(y='minimum temperature', colour = "blue")) #+ stat_smooth(colour='yellow', span=0.2)
import matplotlib.pyplot as plt

### matplotlib inline

weather['precipitation'].unique()

weather['precipitation'] = np.where(weather['precipitation']=='T', '0.00',weather['precipitation'])

weather['precipitation'] = list(map(float, weather['precipitation']))

weather['snow fall'] = np.where(weather['snow fall']=='T', '0.00',weather['snow fall'])

weather['snow fall'] = list(map(float, weather['snow fall']))

weather['snow depth'] = np.where(weather['snow depth']=='T', '0.00',weather['snow depth'])

weather['snow depth'] = list(map(float, weather['snow depth']))

import plotly.plotly as py

import plotly.graph_objs as go

import plotly

random_x = weather['date'].values

random_y0 = weather['precipitation']

random_y1 = weather['snow fall']

random_y2 = weather['snow depth']



# Create traces

trace0 = go.Scatter(

    x = random_x,

    y = random_y0,

    mode = 'markers',

    name = 'precipitation'

)

trace1 = go.Scatter(

    x = random_x,

    y = random_y1,

    mode = 'markers',

    name = 'snow fall'

)

trace2 = go.Scatter(

    x = random_x,

    y = random_y2,

    mode = 'markers',

    name = 'snow depth'

)



data = [trace0, trace1, trace2]

plotly.offline.iplot(data, filename='scatter-mode')
def freq_turn(step_dir):

    """function to create dummy for turn type"""

    from collections import Counter

    step_dir_new = step_dir.split("|")

    a_list = Counter(step_dir_new).most_common()

    path = {}

    for i in range(len(a_list)):

        path.update({a_list[i]})

    a = 0

    b = 0

    c = 0

    if 'straight' in (path.keys()):

        a = path['straight']

        #print(a)

    if 'left' in (path.keys()):

        b = path['left']

        #print(b)

    if 'right' in (path.keys()):

        c = path['right']

        #print(c)

    return a,b,c
start = time.time()

train_fr['straight']= 0

train_fr['left'] =0

train_fr['right'] = 0

train_fr['straight'], train_fr['left'], train_fr['right'] = zip(*train_fr['step_direction'].map(freq_turn))

end = time.time()

print("Time Taken by above cell is {}.".format(end - start))

#train_fr.head()
train_fr_new = train_fr[['id','straight','left','right']]

train = pd.merge(train, train_fr_new, on = 'id', how = 'left')

#train = pd.merge(train, weather, on= 'date', how = 'left')

print(len(train.columns))

#train.columns
train['pickup_datetime'] = pd.to_datetime(train['pickup_datetime'])

train['date'] = train['pickup_datetime'].dt.date

train.head()



train['date'] = pd.to_datetime(train['date'])

train = pd.merge(train, weather[['date','minimum temperature', 'precipitation', 'snow fall', 'snow depth']], on= 'date', how = 'left')

train.shape[0]



train.loc[:,'hvsine_pick_cent_d'] = haversine_(train['pickup_latitude'].values, train['pickup_longitude'].values, train['centroid_drop_lat'].values, train['centroid_drop_long'].values)

train.loc[:,'hvsine_drop_cent_p'] = haversine_(train['dropoff_latitude'].values, train['dropoff_longitude'].values, train['centroid_pick_lat'].values, train['centroid_pick_long'].values)



test.loc[:,'hvsine_pick_cent_d'] = haversine_(test['pickup_latitude'].values, test['pickup_longitude'].values, test['centroid_drop_lat'].values, test['centroid_drop_long'].values)

test.loc[:,'hvsine_drop_cent_p'] = haversine_(test['dropoff_latitude'].values, test['dropoff_longitude'].values, test['centroid_pick_lat'].values, test['centroid_pick_long'].values)



print("shape of train_features is {}.".format(len(train.columns)))
start = time.time()

temp = train[['hvsine_drop_cent_p', 'hvsine_pick_cent_d', 'hvsine_drop_cent_d', 'hvsine_pick_cent_p', 'hvsine_pick_drop', 'hvsine_cent_p_cent_d', 'total_distance']]

temp.total_distance.dropna(inplace = True)

print("total number of Nulls {}.".format(temp.total_distance.isnull().sum()))

# Lets take distance of pick---cent_p---cent_d---drop as distance_pick_cp_cd_drop

# Lets take distance of pick---cent_d---drop as distance_pick_cd_drop

# Lets take distance of pick---cent_p---drop as distance_pick_cp_drop

# Lets take distance of pick--drop as total_distance 

temp['distance_pick_cp_cd_drop'] = temp['hvsine_pick_cent_p'] + temp['hvsine_cent_p_cent_d'] + temp['hvsine_drop_cent_d']

temp['distance_pick_cd_drop'] = temp['hvsine_pick_cent_d'] + temp['hvsine_drop_cent_d']

temp['distance_pick_cp_drop'] = temp['hvsine_pick_cent_p'] + temp['hvsine_drop_cent_p']

temp['total_distance'] = np.floor(temp['total_distance']/1000)

temp['distance_pick_cp_drop'] = np.floor(temp['distance_pick_cp_drop'])

temp['distance_pick_cd_drop'] = np.floor(temp['distance_pick_cd_drop'])

temp['distance_pick_cp_cd_drop'] = np.floor(temp['distance_pick_cp_cd_drop'])

#temp.head()

temp1 = temp.copy()

temp = temp1.sample(100000)

aggregation = {'distance_pick_cp_cd_drop':'count', 'distance_pick_cd_drop':'count', 'distance_pick_cp_drop':'count', 'total_distance':'count'}

temp2 = pd.DataFrame(temp.groupby('total_distance').agg(aggregation))

X_plot = np.linspace(0, temp.total_distance.max(), temp.shape[0])

temp2.rename(columns={'total_distance':'count'}, inplace = True)

temp2.reset_index(inplace = True)

temp2.total_distance = map(int, temp2.total_distance)

temp = temp.sample(100000)

X_plot = temp.total_distance.unique()

a = np.histogram(temp[['total_distance']].values, range(0,95))

N = temp.shape[0]

data = []

trace1 = go.Scatter(x=np.histogram(temp[['total_distance']].values, range(0,95))[1], y=np.histogram(temp[['total_distance']].values, range(0,95))[0],

                    mode='lines', fill='tozeroy',

                    line=dict(color='black', width=2), 

                    name='Total_distance_OSRM')

data.append(trace1)



for kernel in ['distance_pick_cp_cd_drop', 'distance_pick_cd_drop', 'distance_pick_cp_drop']:

    trace2 = go.Scatter(x=np.histogram(temp[['total_distance']].values, range(0,95))[1], y=np.histogram(temp[[kernel]].values, range(0,95))[0],

                        mode='lines', 

                        line=dict(width=2, dash='dash'), 

                        name=kernel)

    data.append(trace2)



layout=go.Layout(annotations=[dict(x=6, y=0.38, showarrow=False,

                                   text="N={0} points".format(N)),

                                   ],

                 xaxis=dict(zeroline=False), hovermode='closest')

fig = go.Figure(data=data, layout=layout)



plotly.offline.iplot(fig)  # last 150+ lines of code just to make a beautiful histogram ;) haha
Holoviews = False

if Holoviews:

    """Holoview is heavy, it stops to notebook from getting loaded,so

    run this cell if you are running it cell-by-cell mode ;)"""

    import holoviews as hv

    import bokeh

    hv.extension('bokeh', width=95)

    from holoviews.operation import gridmatrix

    temp4 = train.sample(10000)

    temp4['total_distance'] = np.floor(temp4['total_distance']//1000)

    temp4 = temp4[['passenger_count','total_distance', 'total_travel_time', 'number_of_steps', 'hour', 'trip_duration']]

    temp4 = temp4.rename(columns = {'total_distance':'OSRM_total_distance', 'total_travel_time':'OSRM_total_travel_time'})



    ds = hv.Dataset(temp4)

    grouped_by_passenger_count = ds.groupby('passenger_count', container_type=hv.NdOverlay)

    grid = gridmatrix(grouped_by_passenger_count, diagonal_type=hv.Scatter)

    plot_opts = dict(tools=['hover', 'box_select'], bgcolor='#efe8e2')

    style = dict(fill_alpha=0.2, size=4)

    grid({'Scatter': {'plot': plot_opts, 'style': style}})

sns.set(style="white")



# Generate a large random dataset

temp3 = train.copy()



# Compute the correlation matrix

corr = temp3.corr()



# Generate a mask for the upper triangle

mask = np.zeros_like(corr, dtype=np.bool)

mask[np.triu_indices_from(mask)] = True



# Set up the matplotlib figure

f, ax = plt.subplots(figsize=(15, 13))



# Generate a custom diverging colormap

cmap = sns.diverging_palette(220, 10, as_cmap=True)



# Draw the heatmap with the mask and correct aspect ratio

sns.heatmap(corr, mask=mask, cmap=cmap, vmax=.3, center=0,

            square=True, linewidths=.5, cbar_kws={"shrink": .5})
start = time.time()

test_fr['straight']= 0

test_fr['left'] =0

test_fr['right'] = 0

test_fr['straight'], test_fr['left'], test_fr['right'] = zip(*test_fr['step_direction'].map(freq_turn))

end = time.time()

print("Time Taken by above cell is {}.".format(end - start))

#test_fr.head()
test_fr_new = test_fr[['id','straight','left','right']]

test = pd.merge(test, test_fr_new, on = 'id', how = 'left')

print(len(test.columns))

#test.columns
test['pickup_datetime'] = pd.to_datetime(test['pickup_datetime'])

test['date'] = test['pickup_datetime'].dt.date

test['date'] = pd.to_datetime(test['date'])
test= pd.merge(test, weather[['date','minimum temperature', 'precipitation', 'snow fall', 'snow depth']], on= 'date', how = 'left')

feature_names = list(train.columns)

print("Difference of features in train and test are {}".format(np.setdiff1d(train.columns, test.columns)))

print("")

do_not_use_for_training = ['pick_date','id', 'pickup_datetime', 'dropoff_datetime', 'trip_duration', 'store_and_fwd_flag', 'date']

feature_names = [f for f in train.columns if f not in do_not_use_for_training]

print("We will be using following features for training {}.".format(feature_names))

print("")

print("Total number of features are {}.".format(len(feature_names)))
y = np.log(train['trip_duration'].values + 1)

Xtr, Xv, ytr, yv = train_test_split(train[feature_names].values, y, test_size=0.2, random_state=1987)

dtrain = xgb.DMatrix(Xtr, label=ytr)

dvalid = xgb.DMatrix(Xv, label=yv)

dtest = xgb.DMatrix(test[feature_names].values)

watchlist = [(dtrain, 'train'), (dvalid, 'valid')]



start = time.time()

xgb_par = {'min_child_weight': 20, 'eta': 0.05, 'colsample_bytree': 0.5, 'max_depth': 15,

            'subsample': 0.9, 'lambda': 2.0, 'nthread': -1, 'booster' : 'gbtree', 'silent': 1,

            'eval_metric': 'rmse', 'objective': 'reg:linear'}



model_1 = xgb.train(xgb_par, dtrain, 10, watchlist, early_stopping_rounds=4, maximize=False, verbose_eval=1)

print('Modeling RMSLE %.5f' % model.best_score)

end = time.time()

print("Time taken in training is {}.".format(end - start))
print('Modeling RMSLE %.5f' % model_1.best_score)

end = time.time()

print("Time taken in training is {}.".format(end - start))

start = time.time()

yvalid = model_1.predict(dvalid)

ytest = model_1.predict(dtest)

end = time.time()

print("Time taken in prediction is {}.".format(end - start))
# Lets check how the distribution of test and vaidation set looks like ...

start = time.time()

fig, ax = plt.subplots(nrows=2, sharex=True, sharey=True)

sns.distplot(yvalid, ax=ax[0], color='blue', label='Validation')

sns.distplot(ytest, ax=ax[1], color='green', label='Test')

ax[0].legend(loc=0)

ax[1].legend(loc=0)

plt.show()

end = time.time()

print("Time taken by above cell is {}.".format((end-start)))
start = time.time()

if test.shape[0] == ytest.shape[0]:

    print('Test shape OK.') 

test['trip_duration'] = np.exp(ytest) - 1

test[['id', 'trip_duration']].to_csv('mahesh_xgb_submission.csv', index=False)

end = time.time()

print("Time taken in training is {}.".format(end - start))
class maps:



	def __init__(self, centerLat, centerLng, zoom ):

		self.center = (float(centerLat),float(centerLng))

		self.zoom = int(zoom)

		self.grids = None

		self.paths = []

		self.points = []

		self.radpoints = []

		self.gridsetting = None

		self.coloricon = 'http://chart.apis.google.com/chart?cht=mm&chs=12x16&chco=FFFFFF,XXXXXX,000000&ext=.png'



	def setgrids(self,slat,elat,latin,slng,elng,lngin):

		self.gridsetting = [slat,elat,latin,slng,elng,lngin]



	def addpoint(self, lat, lng, color = '#FF0000'):

		self.points.append((lat,lng,color[1:]))



	#def addpointcoord(self, coord):

	#	self.points.append((coord[0],coord[1]))



	def addradpoint(self, lat,lng,rad,color = '#0000FF'):

		self.radpoints.append((lat,lng,rad,color))



	def addpath(self,path,color = '#FF0000'):

		path.append(color)

		self.paths.append(path)

	

	#create the html file which inlcude one google map and all points and paths

	def draw(self, htmlfile):

		f = open(htmlfile,'w')

		f.write('<html>\n')

		f.write('<head>\n')

		f.write('<meta name="viewport" content="initial-scale=1.0, user-scalable=no" />\n')

		f.write('<meta http-equiv="content-type" content="text/html; charset=UTF-8"/>\n')

		f.write('<title>Google Maps - pygmaps </title>\n')

		f.write('<script type="text/javascript" src="http://maps.google.com/maps/api/js?sensor=false"></script>\n')

		f.write('<script type="text/javascript">\n')

		f.write('\tfunction initialize() {\n')

		self.drawmap(f)

		self.drawgrids(f)

		self.drawpoints(f)

		self.drawradpoints(f)

		self.drawpaths(f,self.paths)

		f.write('\t}\n')

		f.write('</script>\n')

		f.write('</head>\n')

		f.write('<body style="margin:0px; padding:0px;" onload="initialize()">\n')

		f.write('\t<div id="map_canvas" style="width: 100%; height: 100%;"></div>\n')

		f.write('</body>\n')

		f.write('</html>\n')

		f.close()



	def drawgrids(self, f):

		if self.gridsetting == None:

			return

		slat = self.gridsetting[0]

		elat = self.gridsetting[1]

		latin = self.gridsetting[2]

		slng = self.gridsetting[3]

		elng = self.gridsetting[4]

		lngin = self.gridsetting[5]

		self.grids = []



		r = [slat+float(x)*latin for x in range(0, int((elat-slat)/latin))]

		for lat in r:

			self.grids.append([(lat+latin/2.0,slng+lngin/2.0),(lat+latin/2.0,elng+lngin/2.0)])



		r = [slng+float(x)*lngin for x in range(0, int((elng-slng)/lngin))]

		for lng in r:

			self.grids.append([(slat+latin/2.0,lng+lngin/2.0),(elat+latin/2.0,lng+lngin/2.0)])

		

		for line in self.grids:

			self.drawPolyline(f,line,strokeColor = "#000000")

	def drawpoints(self,f):

		for point in  self.points:

			self.drawpoint(f,point[0],point[1],point[2])



	def drawradpoints(self, f):

		for rpoint in self.radpoints:

			path = self.getcycle(rpoint[0:3])

			self.drawPolygon(f,path,strokeColor = rpoint[3])



	def getcycle(self,rpoint):

		cycle = []

		lat = rpoint[0]

		lng = rpoint[1]

		rad = rpoint[2] #unit: meter

		d = (rad/1000.0)/6378.8;

		lat1 = (math.pi/180.0)* lat

		lng1 = (math.pi/180.0)* lng



		r = [x*30 for x in range(12)]

		for a in r:

			tc = (math.pi/180.0)*a;

			y = math.asin(math.sin(lat1)*math.cos(d)+math.cos(lat1)*math.sin(d)*math.cos(tc))

			dlng = math.atan2(math.sin(tc)*math.sin(d)*math.cos(lat1),math.cos(d)-math.sin(lat1)*math.sin(y))

			x = ((lng1-dlng+math.pi) % (2.0*math.pi)) - math.pi 

			cycle.append( ( float(y*(180.0/math.pi)),float(x*(180.0/math.pi)) ) )

		return cycle



	def drawpaths(self, f, paths):

		for path in paths:

			#print path

			self.drawPolyline(f,path[:-1], strokeColor = path[-1])



	#############################################

	# # # # # # Low level Map Drawing # # # # # # 

	#############################################

	def drawmap(self, f):

		f.write('\t\tvar centerlatlng = new google.maps.LatLng(%f, %f);\n' % (self.center[0],self.center[1]))

		f.write('\t\tvar myOptions = {\n')

		f.write('\t\t\tzoom: %d,\n' % (self.zoom))

		f.write('\t\t\tcenter: centerlatlng,\n')

		f.write('\t\t\tmapTypeId: google.maps.MapTypeId.ROADMAP\n')

		f.write('\t\t};\n')

		f.write('\t\tvar map = new google.maps.Map(document.getElementById("map_canvas"), myOptions);\n')

		f.write('\n')







	def drawpoint(self,f,lat,lon,color):

		f.write('\t\tvar latlng = new google.maps.LatLng(%f, %f);\n'%(lat,lon))

		f.write('\t\tvar img = new google.maps.MarkerImage(\'%s\');\n' % (self.coloricon.replace('XXXXXX',color)))

		f.write('\t\tvar marker = new google.maps.Marker({\n')

		f.write('\t\ttitle: "no implimentation",\n')

		f.write('\t\ticon: img,\n')

		f.write('\t\tposition: latlng\n')

		f.write('\t\t});\n')

		f.write('\t\tmarker.setMap(map);\n')

		f.write('\n')

		

	def drawPolyline(self,f,path,\

			clickable = False, \

			geodesic = True,\

			strokeColor = "#FF0000",\

			strokeOpacity = 1.0,\

			strokeWeight = 2

			):

		f.write('var PolylineCoordinates = [\n')

		for coordinate in path:

			f.write('new google.maps.LatLng(%f, %f),\n' % (coordinate[0],coordinate[1]))

		f.write('];\n')

		f.write('\n')



		f.write('var Path = new google.maps.Polyline({\n')

		f.write('clickable: %s,\n' % (str(clickable).lower()))

		f.write('geodesic: %s,\n' % (str(geodesic).lower()))

		f.write('path: PolylineCoordinates,\n')

		f.write('strokeColor: "%s",\n' %(strokeColor))

		f.write('strokeOpacity: %f,\n' % (strokeOpacity))

		f.write('strokeWeight: %d\n' % (strokeWeight))

		f.write('});\n')

		f.write('\n')

		f.write('Path.setMap(map);\n')

		f.write('\n\n')



	def drawPolygon(self,f,path,\

			clickable = False, \

			geodesic = True,\

			fillColor = "#000000",\

			fillOpacity = 0.0,\

			strokeColor = "#FF0000",\

			strokeOpacity = 1.0,\

			strokeWeight = 1

			):

		f.write('var coords = [\n')

		for coordinate in path:

			f.write('new google.maps.LatLng(%f, %f),\n' % (coordinate[0],coordinate[1]))

		f.write('];\n')

		f.write('\n')



		f.write('var polygon = new google.maps.Polygon({\n')

		f.write('clickable: %s,\n' % (str(clickable).lower()))

		f.write('geodesic: %s,\n' % (str(geodesic).lower()))

		f.write('fillColor: "%s",\n' %(fillColor))

		f.write('fillOpacity: %f,\n' % (fillOpacity))

		f.write('paths: coords,\n')

		f.write('strokeColor: "%s",\n' %(strokeColor))

		f.write('strokeOpacity: %f,\n' % (strokeOpacity))

		f.write('strokeWeight: %d\n' % (strokeWeight))

		f.write('});\n')

		f.write('\n')

		f.write('polygon.setMap(map);\n')

		f.write('\n\n')





def show_gmaps(train_data, path):

    """function to generate map and add the pick up and drop coordinates

    1. Path = 1 : Join pickup (blue) and drop(red) using a straight line

    """

    mymap = maps(40.767937, -73.982155, 12) # manually added centre

    for i in range(train_data.shape[0]):

        pick_long = train_data.loc[train_data.index ==i]['pickup_longitude'].values[0]

        pick_lat = train_data.loc[train_data.index ==i]['pickup_latitude'].values[0]

        dest_long = train_data.loc[train_data.index ==i]['dropoff_longitude'].values[0]

        dest_lat = train_data.loc[train_data.index ==i]['dropoff_latitude'].values[0]

        mymap.addpoint(pick_lat, pick_long, "#FF0000")

        #mymap.addradpoint(dest_lat, dest_long, 50, "#00FF00")

        #mymap.getcycle([dest_lat, dest_long, 0.01])

        mymap.addpoint(dest_lat, dest_long, "#0000FF")

        #if path == 1:

        path = [(pick_lat, pick_long),(dest_lat, dest_long)]

        mymap.addpath(path,"#000000")

        if i%1000 == 0:

            print(i, dest_lat, dest_long) #time.time(),

    mymap.draw('./Google_map_showing_trips.txt')

    return 



end = time.time()

print("time taken by thie script by now is {}.".format(end-start))
# Lets visulalize a sample of 200 trips from all data

train_sample_vis = train_data.loc[np.random.randint(1458644, size =200)]

train_sample_vis.reset_index(drop = True, inplace = True)

show_gmaps(train_sample_vis, 1)

#print( os.listdir('../input/'))



from IPython.display import IFrame, HTML, display

IFrame(HTML('../kaggle/working/Google_map_showing_trips.txt'), width=1000, height=500)





end = time.time()

print("time taken by thie script by now is {}.".format(end-start))





df_cluster4 = train_cl
start = time.time()

centroid_drops = centroid_pickups.rename(columns={'centroid_pick_long':'centroid_drop_long', 'centroid_pick_lat':'centroid_drop_lat','label_pick':'label_drop'})

centroid_drops.head()

clus5 = df_cluster4 # just to be safe side so store it

df_cluster4 = pd.merge(df_cluster4, centroid_drops, how='left', on=['label_drop'])

df_cluster4.head()



end = time.time()

print("time taken by thie script by now is {}.".format(end-start))
### matplotlib inline

def bar_plot(x_var, y_var):

    """function to show barplot between two variables"""

    objects = x_var

    performance = y_var

    plt.bar(objects, performance, align='center', alpha=0.5)

    plt.xlabel('class')

    plt.ylabel('frequency')

    plt.title('barplot') 

    plt.show()

    return 0



end = time.time()

print("time taken by thie script by now is {}.".format(end-start))
def show_cluster_gmaps(train_df, cluster_df, points_per_cluster, cluster_list, path, name):

    """function to vasualize cluster 

    1. train_df - df containg lat-long

    2. cluster_df - df containing cluster centroid

    3. points_per_cluster = number od coordniates shown per cluster

    4. cluster_list - list of clusters you want to show on map

    5. path - if 1, show paths to cluster centroid"""

    cluster_df_new = cluster_df[cluster_df['label_pick'].isin(cluster_list)]

    init_lat = cluster_df_new.centroid_pick_lat.mean()

    init_long = cluster_df_new.centroid_pick_long.mean()

    print(init_lat, init_long)

    mymap = maps(init_lat,init_long, 12) # manually added centre

    cluster_df_new = cluster_df_new.reset_index(drop = True)

    #print(cluster_df_new)

    for i in range(cluster_df_new.shape[0]):

        pick_long = cluster_df_new.loc[cluster_df_new.index ==i]['centroid_pick_long'].values[0]

        pick_lat = cluster_df_new.loc[cluster_df_new.index ==i]['centroid_pick_lat'].values[0]

        mymap.addpoint(pick_lat, pick_long, "#FF0000")

        mymap.addradpoint(pick_lat, pick_long, 750, "#0000FF")

        #mymap.getcycle([pick_lat, pick_long, 0.01])

    mymap.draw('./cluster_map_'+name+'.txt')

    return



end = time.time()

print("time taken by thie script by now is {}.".format(end-start))
cluster_list = range(0,20)

#summary_significant_clusters.tolist()

#print(cluster_list)

show_cluster_gmaps(train_data.head(100), centroid_pickups, 30, cluster_list, 1, 'all clusters')



from IPython.display import IFrame

IFrame('cluster_map_all clusters.txt', width=1000, height=500)



end = time.time()

print("time taken by thie script by now is {}.".format(end-start))
summary_clusters_time = pd.DataFrame(df_cluster4.groupby('label_pick')['trip_duration'].count())

summary_significant_clusters = summary_clusters_time.loc[summary_clusters_time['trip_duration']>50000].index.values

summary_significant_clusters

cluster_list = summary_significant_clusters.tolist()



print("list of clusters with 50k pick ups are - {}.".format(cluster_list))

show_cluster_gmaps(train_data.head(100), centroid_pickups, 30, cluster_list, 1, 'significant')



from IPython.display import IFrame

IFrame('cluster_map_significant.txt', width=1000, height=500)



end = time.time()

print("time taken by thie script by now is {}.".format(end-start))
index_rand = np.random.randint(len(cluster_list),size =4)

print(cluster_list, len(cluster_list))

cluster_list4 = [cluster_list[x] for x in index_rand]

print(cluster_list4)



end = time.time()

print("time taken by thie script by now is {}.".format(end-start))
# cluster visualization - 

def cluster_visualization(train_df, cluster_df, points_per_cluster, cluster_list, path_mode):

    """function to vasualize cluster 

    1. train_df - df containg lat-long

    2. cluster_df - df containing cluster centroid

    3. points_per_cluster = number od coordniates shown per cluster

    4. cluster_list - list of clusters you want to show on map - only two clusters

    5. path - if 1, show paths to cluster centroid"""

    cluster_df_new = cluster_df[cluster_df['label_pick'].isin(cluster_list)]

    init_lat = cluster_df_new.centroid_pick_lat.mean()

    init_long = cluster_df_new.centroid_pick_long.mean()

    print(init_lat, init_long)

    train_df_new = train_df[train_df['label_pick'].isin(cluster_list)]

    train_df_new = train_df_new.reset_index(drop = True)

    sample_list = np.random.randint(train_df_new.shape[0], size = 100) # INITIAL 50

    sample_df = train_df_new.loc[sample_list]

    sample_df = sample_df.reset_index(drop = True)

    #print(sample_df.head())

    

    mymap = maps(init_lat,init_long, 13.5) # manually added centre #INITIAL - 12

    cluster_df_new = cluster_df_new.reset_index(drop = True)

    #print(cluster_df_new)

    for i in range(cluster_df_new.shape[0]):

        pick_long = cluster_df_new.loc[cluster_df_new.index ==i]['centroid_pick_long'].values[0]

        pick_lat = cluster_df_new.loc[cluster_df_new.index ==i]['centroid_pick_lat'].values[0]

        mymap.addpoint(pick_lat, pick_long, "#FF0000")

        mymap.addradpoint(pick_lat, pick_long, 750, "#0000FF")

        sample_df_clus = sample_df.loc[sample_df['label_pick'] == cluster_df_new.loc[i]['label_pick']]

        sample_df_clus = sample_df_clus.reset_index(drop = True)

        for j in range(sample_df_clus.shape[0]):

            sample_lat = sample_df_clus.loc[j]['pickup_latitude']

            sample_long = sample_df_clus.loc[j]['pickup_longitude']

            mymap.addpoint(sample_lat, sample_long, "#FF0000")

            path = [(pick_lat, pick_long),(sample_lat, sample_long)]

            mymap.addpath(path,"#000000")

    if path_mode ==1:

        for k in range(cluster_df_new.shape[0]):

            ln1 = cluster_df_new.loc[cluster_df_new.index ==k]['centroid_pick_long'].values[0]

            lt1 = cluster_df_new.loc[cluster_df_new.index ==k]['centroid_pick_lat'].values[0]

            for l in range(cluster_df_new.shape[0]):

                if k!=l:

                    ln2 = cluster_df_new.loc[cluster_df_new.index ==l]['centroid_pick_long'].values[0]

                    lt2 = cluster_df_new.loc[cluster_df_new.index ==l]['centroid_pick_lat'].values[0]

                    path_c = [(lt1, ln1),(lt2, ln2)]

                    mymap.addpath(path_c,"#000000")

                    #print('path added')

    mymap.draw('./multi_clusters_.txt')

    return



end = time.time()

print("time taken by thie script by now is {}.".format(end-start))
cluster_list = cluster_list4#[0, 13, 26, 39] #INITIAL 9,23, 21,23, 33- 23,

print(cluster_list4)

#summary_significant_clusters.tolist()

#[ 0,  3,  7,  9, 12, 13, 15, 19, 21, 23, 25, 28, 33, 39, 41, 42]

print("clusters which are getting shown on Google maps are - {}.".format(cluster_list))

cluster_visualization(df_cluster4, centroid_pickups, 30, cluster_list, 1)



from IPython.display import IFrame

IFrame('multi_clusters_.txt', width=1000, height=500)



end = time.time()

print("time taken by thie script by now is {}.".format(end-start))
# creating a df containing all such combinations

df_clus_pick_dest = pd.DataFrame(columns=('pick_long', 'pick_lat', 'pick_label','drop_long', 'drop_lat', 'drop_label'))

list_vars =[]

k = centroid_pickups.shape[0]

for i in range(0,centroid_pickups.shape[0]):

    for j in range(0,centroid_pickups.shape[0]):

        if i !=j:

            pick_long = centroid_pickups.loc[i]['centroid_pick_long']

            pick_lat = centroid_pickups.loc[i]['centroid_pick_lat']

            pick_label = centroid_pickups.loc[i]['label_pick']

            drop_long = centroid_pickups.loc[j]['centroid_pick_long']

            drop_lat = centroid_pickups.loc[j]['centroid_pick_lat']

            drop_label = centroid_pickups.loc[j]['label_pick']

            list_data = [pick_long, pick_lat, pick_label, drop_long, drop_lat, drop_label]

            df_clus_pick_dest.loc[i*k+j] = list_data

            

print(df_clus_pick_dest.shape[0])           

df_clus_pick_dest.head()



end = time.time()

print("time taken by thie script by now is {}.".format(end-start))
#df_clus_pick_dest['data_from_google']=''

global count_hhh 

count_hhh = 0

def google_maps_query(row):

    """

    function to use google api on source and destination co-ordinates 

    returns following - 

    1. origin address 

    2. destination address

    3. duration ~ duration from distance matrix api

    4. distance of shrtest path

    """

    orig_lat = row['pick_lat']

    orig_lng = row['pick_long']

    dest_lat = 	row['drop_lat']

    dest_lng =  row['drop_long']

    url = """http://maps.googleapis.com/maps/api/distancematrix/json?origins=%s,%s"""%(orig_lat, orig_lng)+  \

    """&destinations=%s,%s&mode=driving&language=en-EN&sensor=false"""% (dest_lat, dest_lng)

    result= simplejson.load(urllib.urlopen(url))

    global count_hhh

    count_hhh = count_hhh + 1

    if count_hhh % 198 ==0:

        print(count_hhh//197, time.time())

    return result



def origin_address(row):

    return row['data_from_google']['origin_addresses']



def destination_address(row):

    return row['data_from_google']['destination_addresses']



def gmaps_duration(row):

    """function to extract the duration of ride"""

    if row['data_from_google']['status'] =='OK':

        if len(row['data_from_google']['rows'][0]['elements'][0].keys())==3:

            query_result = row['data_from_google']['rows'][0]['elements'][0]['duration']['value']

        else:

            query_result = np.nan

    else:

        query_result = np.nan

    return(query_result)



def gmaps_distance(row):

    """function to extract the dstance of ride"""

    if row['data_from_google']['status'] =='OK':

        if len(row['data_from_google']['rows'][0]['elements'][0].keys())==3:

            query_result = row['data_from_google']['rows'][0]['elements'][0]['distance']['value']

        else:

            query_result = np.nan

    else:

        query_result = np.nan

    return(query_result)



end = time.time()

print("time taken by thie script by now is {}.".format(end-start))
print("Thanks for reading my kernel, I hope you enjoyed it!!")