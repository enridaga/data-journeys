
#peak at the folder

### ls -al ../input/
#basics

import numpy as np

import pandas as pd



#misc

import gc

import time

import warnings





#viz

import matplotlib.pyplot as plt

import seaborn as sns 

import matplotlib.gridspec as gridspec 



# graph viz

import plotly.offline as pyo



from plotly.graph_objs import *

import plotly.graph_objs as go



#map section

import imageio

import folium

import folium.plugins as plugins

from mpl_toolkits.basemap import Basemap



#settings

start_time=time.time()

color = sns.color_palette()

sns.set_style("dark")

warnings.filterwarnings("ignore")

pyo.init_notebook_mode()



### matplotlib inline
#importing everything

business=pd.read_csv("../input/yelp_business.csv")

business_attributes=pd.read_csv("../input/yelp_business_attributes.csv")

business_hours=pd.read_csv("../input/yelp_business_hours.csv")

check_in=pd.read_csv("../input/yelp_checkin.csv")

reviews=pd.read_csv("../input/yelp_review.csv")

tip=pd.read_csv("../input/yelp_tip.csv")

user=pd.read_csv("../input/yelp_user.csv")

end_time=time.time()

print("Took",end_time-start_time,"s")
#take a peak

business.head()
#Get the distribution of the ratings

x=business['stars'].value_counts()

x=x.sort_index()

#plot

plt.figure(figsize=(8,4))

ax= sns.barplot(x.index, x.values, alpha=0.8)

plt.title("Star Rating Distribution")

plt.ylabel('# of businesses', fontsize=12)

plt.xlabel('Star Ratings ', fontsize=12)



#adding the text labels

rects = ax.patches

labels = x.values

for rect, label in zip(rects, labels):

    height = rect.get_height()

    ax.text(rect.get_x() + rect.get_width()/2, height + 5, label, ha='center', va='bottom')



plt.show()
#basic basemap of the world

plt.figure(1, figsize=(15,6))

# use ortho projection for the globe type version

m1=Basemap(projection='ortho',lat_0=20,lon_0=-50)



# hex codes from google maps color pallete = http://www.color-hex.com/color-palette/9261

#add continents

m1.fillcontinents(color='#bbdaa4',lake_color='#4a80f5') 

# add the oceans

m1.drawmapboundary(fill_color='#4a80f5')                

# Draw the boundaries of the countires

m1.drawcountries(linewidth=0.1, color="black")



#Add the scatter points to indicate the locations of the businesses

mxy = m1(business["longitude"].tolist(), business["latitude"].tolist())

m1.scatter(mxy[0], mxy[1], s=3, c="orange", lw=3, alpha=1, zorder=5)

plt.title("World-wide Yelp Reviews")

plt.show()
# Sample it down to only the North America region 

lon_min, lon_max = -132.714844, -59.589844

lat_min, lat_max = 13.976715,56.395664



#create the selector

idx_NA = (business["longitude"]>lon_min) &\

            (business["longitude"]<lon_max) &\

            (business["latitude"]>lat_min) &\

            (business["latitude"]<lat_max)

#apply the selector to subset

NA_business=business[idx_NA]



#initiate the figure

plt.figure(figsize=(12,6))

m2 = Basemap(projection='merc',

             llcrnrlat=lat_min,

             urcrnrlat=lat_max,

             llcrnrlon=lon_min,

             urcrnrlon=lon_max,

             lat_ts=35,

             resolution='i')



m2.fillcontinents(color='#191919',lake_color='#000000') # dark grey land, black lakes

m2.drawmapboundary(fill_color='#000000')                # black background

m2.drawcountries(linewidth=0.1, color="w")              # thin white line for country borders



# Plot the data

mxy = m2(NA_business["longitude"].tolist(), NA_business["latitude"].tolist())

m2.scatter(mxy[0], mxy[1], s=5, c="#1292db", lw=0, alpha=0.05, zorder=5)



plt.title("North America Region")

# Sample it down to only the Eurozone + Britain :p 

lon_min, lon_max = -8.613281,16.699219

lat_min, lat_max = 40.488737,59.204064



#create the selector

idx_euro = (business["longitude"]>lon_min) &\

            (business["longitude"]<lon_max) &\

            (business["latitude"]>lat_min) &\

            (business["latitude"]<lat_max)

#apply the selector to subset

euro_business=business[idx_euro]



#initiate the figure

plt.figure(figsize=(12,6))

m3 = Basemap(projection='merc',

             llcrnrlat=lat_min,

             urcrnrlat=lat_max,

             llcrnrlon=lon_min,

             urcrnrlon=lon_max,

             lat_ts=35,

             resolution='i')



m3.fillcontinents(color='#191919',lake_color='#000000') # dark grey land, black lakes

m3.drawmapboundary(fill_color='#000000')                # black background

m3.drawcountries(linewidth=0.1, color="w")              # thin white line for country borders



# Plot the data

mxy = m3(euro_business["longitude"].tolist(), euro_business["latitude"].tolist())

m3.scatter(mxy[0], mxy[1], s=5, c="#1292db", lw=0, alpha=0.05, zorder=5)



plt.title("Europe Region")

plt.show()
#Get the distribution of the ratings

x=business['city'].value_counts()

x=x.sort_values(ascending=False)

x=x.iloc[0:20]

plt.figure(figsize=(16,4))

ax = sns.barplot(x.index, x.values, alpha=0.8,color=color[3])

plt.title("Which city has the most reviews?")

locs, labels = plt.xticks()

plt.setp(labels, rotation=45)

plt.ylabel('# businesses', fontsize=12)

plt.xlabel('City', fontsize=12)



#adding the text labels

rects = ax.patches

labels = x.values

for rect, label in zip(rects, labels):

    height = rect.get_height()

    ax.text(rect.get_x() + rect.get_width()/2, height + 5, label, ha='center', va='bottom')



plt.show()
#get all ratings data

rating_data=business[['latitude','longitude','stars','review_count']]

# Creating a custom column popularity using stars*no_of_reviews

rating_data['popularity']=rating_data['stars']*rating_data['review_count']

f, (ax1, ax2) = plt.subplots(1, 2, figsize=(15,7))



#a random point inside vegas

lat = 36.207430

lon = -115.268460

#some adjustments to get the right pic

lon_min, lon_max = lon-0.3,lon+0.5

lat_min, lat_max = lat-0.4,lat+0.5

#subset for vegas

ratings_data_vegas=rating_data[(rating_data["longitude"]>lon_min) &\

                    (rating_data["longitude"]<lon_max) &\

                    (rating_data["latitude"]>lat_min) &\

                    (rating_data["latitude"]<lat_max)]



#Facet scatter plot

ratings_data_vegas.plot(kind='scatter', x='longitude', y='latitude',

                color='yellow', 

                s=.02, alpha=.6, subplots=True, ax=ax1)

ax1.set_title("Las Vegas")

ax1.set_facecolor('black')



#a random point inside pheonix

lat = 33.435463

lon = -112.006989

#some adjustments to get the right pic

lon_min, lon_max = lon-0.3,lon+0.5

lat_min, lat_max = lat-0.4,lat+0.5

#subset for pheonix

ratings_data_pheonix=rating_data[(rating_data["longitude"]>lon_min) &\

                    (rating_data["longitude"]<lon_max) &\

                    (rating_data["latitude"]>lat_min) &\

                    (rating_data["latitude"]<lat_max)]

#plot pheonix

ratings_data_pheonix.plot(kind='scatter', x='longitude', y='latitude',

                color='yellow', 

                s=.02, alpha=.6, subplots=True, ax=ax2)

ax2.set_title("Pheonix")

ax2.set_facecolor('black')

f.show()
f, (ax1, ax2) = plt.subplots(1, 2, figsize=(15,7))

#a random point inside Stuttgart 

lat = 48.7757200

lon = 9.187950

#some adjustments to get the right pic

lon_min, lon_max = lon-0.05,lon+0.05

lat_min, lat_max = lat-0.05,lat+0.05

#subset for stuttgart

ratings_data_stuttgart=rating_data[(rating_data["longitude"]>lon_min) &\

                    (rating_data["longitude"]<lon_max) &\

                    (rating_data["latitude"]>lat_min) &\

                    (rating_data["latitude"]<lat_max)]



#Facet scatter plot

ratings_data_stuttgart.plot(kind='scatter', x='longitude', y='latitude',

                color='yellow', 

                s=1, alpha=.6, subplots=True, ax=ax1)

ax1.set_title("Stuttgart")

ax1.set_facecolor('black')

#a random point inside edinburgh

lat = 55.947871

lon = -3.194255

#some adjustments to get the right pic

lon_min, lon_max = lon-0.03,lon+0.03

lat_min, lat_max = lat-0.01,lat+0.01

#subset for pheonix

ratings_data_edinburgh=rating_data[(rating_data["longitude"]>lon_min) &\

                    (rating_data["longitude"]<lon_max) &\

                    (rating_data["latitude"]>lat_min) &\

                    (rating_data["latitude"]<lat_max)]

#plot pheonix

ratings_data_edinburgh.plot(kind='scatter', x='longitude', y='latitude',

                color='yellow', 

                s=1, alpha=.6, subplots=True, ax=ax2)

ax2.set_title("Edinburgh")

ax2.set_facecolor('black')

f.show()
data=[]

#rearranging data to suit the format needed for folium

stars_list=list(rating_data['stars'].unique())

for star in stars_list:

    subset=ratings_data_vegas[ratings_data_vegas['stars']==star]

    data.append(subset[['latitude','longitude']].values.tolist())

#initialize at vegas

lat = 36.127430

lon = -115.138460

zoom_start=11

print("                     Vegas Review heatmap Animation ")



# basic map

m = folium.Map(location=[lat, lon], tiles="OpenStreetMap", zoom_start=zoom_start)

#inprovising the Heatmapwith time plugin to show variations across star ratings 

hm = plugins.HeatMapWithTime(data,max_opacity=0.3,auto_play=True,display_index=True,radius=7)

hm.add_to(m)

m
end_time=time.time()

print("Took",end_time-start_time,"s")
user_agg=reviews.groupby('user_id').agg({'review_id':['count'],'date':['min','max'],

                                'useful':['sum'],'funny':['sum'],'cool':['sum'],

                               'stars':['mean']})
user_agg=user_agg.sort_values([('review_id','count')],ascending=False)

print("          Top 10 Users in Yelp")

user_agg.head(10)
#Stalking the top user:

top_user_reviews=reviews[reviews['user_id']=='CxDOIDnH8gp9KXzpBHJYXw']

# Get locations of the places he/she has reviewed

top_user_locs=pd.merge(top_user_reviews,business,on='business_id')

# UNique list of dates from the user's reviews

date_list=list(top_user_locs['date'].unique())

#rearranging data to suit the format needed for folium

data=[]

for date in date_list:

    subset=top_user_locs[top_user_locs['date']==date]

    df=subset[['latitude','longitude','date']]

    data.append(subset[['latitude','longitude']].values.tolist())
#initialize at ontario,Canada (the location our top user is from)

lat = 43.860397

lon = -79.303184

zoom_start=9

print("                     Stalking the top User    ")

# basic map

m = folium.Map(location=[lat, lon], tiles="Cartodb Positron", zoom_start=zoom_start)



hm = plugins.HeatMapWithTime(data,max_opacity=0.3,auto_play=True,display_index=True)

hm.add_to(m)

m
# Cap max reviews to 30 for better visuals

user_agg[('review_id','count')].loc[user_agg[('review_id','count')]>30] = 30

plt.figure(figsize=(12,5))

plt.suptitle("User Deep dive",fontsize=20)

gridspec.GridSpec(1,2)

plt.subplot2grid((1,2),(0,0))

#Cumulative Distribution

ax=sns.kdeplot(user_agg[('review_id','count')],shade=True,color='r')

plt.title("How many reviews does an average user give?",fontsize=15)

plt.xlabel('# of reviews given', fontsize=12)

plt.ylabel('# of users', fontsize=12)



#Cumulative Distribution

plt.subplot2grid((1,2),(0,1))

sns.distplot(user_agg[('review_id','count')],

             kde_kws=dict(cumulative=True))

plt.title("Cumulative dist. of user reviews",fontsize=15)

plt.ylabel('Cumulative perc. of users', fontsize=12)

plt.xlabel('# of reviews given', fontsize=12)



plt.show()

end_time=time.time()

print("Took",end_time-start_time,"s")
# What are the popular business categories?

business_cats=' '.join(business['categories'])



cats=pd.DataFrame(business_cats.split(';'),columns=['category'])

x=cats.category.value_counts()

print("There are ",len(x)," different types/categories of Businesses in Yelp!")

#prep for chart

x=x.sort_values(ascending=False)

x=x.iloc[0:20]



#chart

plt.figure(figsize=(16,4))

ax = sns.barplot(x.index, x.values, alpha=0.8)#,color=color[5])

plt.title("What are the top categories?",fontsize=25)

locs, labels = plt.xticks()

plt.setp(labels, rotation=80)

plt.ylabel('# businesses', fontsize=12)

plt.xlabel('Category', fontsize=12)



#adding the text labels

rects = ax.patches

labels = x.values

for rect, label in zip(rects, labels):

    height = rect.get_height()

    ax.text(rect.get_x() + rect.get_width()/2, height + 5, label, ha='center', va='bottom')



plt.show()
#subset users who have atleast one friend

subset_users=user[user['friends']!='None']

#user has given atleast 10 reviews

subset_users=subset_users[subset_users['review_count']>=10]

subset_users=subset_users.sort_values('review_count',ascending=False)

subset_users.head()
#lets try to create our own network from the list of friends and users

import networkx as nx
subset_users=subset_users[['user_id','friends']]

# Trying to convert the data into a format that can be imported into Network X



#Experimenting with the first user

friend_list=list(subset_users.iloc[0].friends.split(','))

friend_frame=pd.DataFrame(friend_list,columns=['freind'])

friend_frame['user_id']=subset_users.iloc[0].user_id

friend_frame.head()
# There might be a better more efficient way of doing this! If you do know, plz let me know in the comments!



#Scaling to all the users

#network_data=pd.DataFrame(columns=['friends','user_id'])

for i in range(0,subset_users.shape[0]):

    #make the friend list

    friend_list=list(subset_users.iloc[i].friends.split(','))

    #Convert the friends list into a column

    friend_frame=pd.DataFrame(friend_list,columns=['friends'])

    # Broadcasting the user-id column with the friends col

    friend_frame['user_id']=subset_users.iloc[i].user_id

    if i==2000:

        #Stopping at 2k due to time contraints,remove this section to see all connections

        break

    if i==0:

        network_data=friend_frame

    else:

        network_data=network_data.append(friend_frame)
network_data=network_data.reset_index(drop=True)

#checks

network_data.tail()
#changing the column name to suit nx import

network_data.columns=['source','target']

# Considering each (user_id,friend) pair as an edge of a graph, constructing the graph

graph=nx.from_pandas_edgelist(network_data)

# logging time

end_time=time.time()

print("Took",end_time-start_time,"s")
#credits https://www.kaggle.com/crailtap/basic-network-analysis-tutorial

#basic info

nx.info(graph)
#check density

nx.density(graph)    

# we get a very sparse(not dense) graph. Maybe if we used all the

# nodes then we might get a more dense graph
nx.draw(graph, pos=nx.fruchterman_reingold_layout(graph))
# logging time

end_time=time.time()

print("Took",end_time-start_time,"s")
# to be continued