


import numpy as np 

import pandas as pd 



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))
orders = pd.read_csv('../input/orders.csv')

orders.head()
prior = pd.read_csv('../input/order_products__prior.csv')

prior.head()
train = pd.read_csv('../input/order_products__train.csv')

train.head()
##Due to the number of rows I have to reduce the set of prior data to publish the kernel 

##comment this if you execute it on your local machine

prior = prior[0:300000]

order_prior = pd.merge(prior,orders,on=['order_id','order_id'])

order_prior = order_prior.sort_values(by=['user_id','order_id'])

order_prior.head()
products = pd.read_csv('../input/products.csv')

products.head()
aisles = pd.read_csv('../input/aisles.csv')

aisles.head()
print(aisles.shape)
_mt = pd.merge(prior,products, on = ['product_id','product_id'])

_mt = pd.merge(_mt,orders,on=['order_id','order_id'])

mt = pd.merge(_mt,aisles,on=['aisle_id','aisle_id'])

mt.head(10)
mt['product_name'].value_counts()[0:10]
len(mt['product_name'].unique())
prior.shape
len(mt['aisle'].unique())
mt['aisle'].value_counts()[0:10]
cust_prod = pd.crosstab(mt['user_id'], mt['aisle'])

cust_prod.head(10)
cust_prod.shape
from sklearn.decomposition import PCA

pca = PCA(n_components=6)

pca.fit(cust_prod)

pca_samples = pca.transform(cust_prod)

ps = pd.DataFrame(pca_samples)

ps.head()
from matplotlib import pyplot as plt

from mpl_toolkits.mplot3d import Axes3D

from mpl_toolkits.mplot3d import proj3d

tocluster = pd.DataFrame(ps[[4,1]])

print (tocluster.shape)

print (tocluster.head())



fig = plt.figure(figsize=(8,8))

plt.plot(tocluster[4], tocluster[1], 'o', markersize=2, color='blue', alpha=0.5, label='class1')



plt.xlabel('x_values')

plt.ylabel('y_values')

plt.legend()

plt.show()
from sklearn.cluster import KMeans

from sklearn.metrics import silhouette_score



clusterer = KMeans(n_clusters=4,random_state=42).fit(tocluster)

centers = clusterer.cluster_centers_

c_preds = clusterer.predict(tocluster)

print(centers)
print (c_preds[0:100])
import matplotlib

fig = plt.figure(figsize=(8,8))

colors = ['orange','blue','purple','green']

colored = [colors[k] for k in c_preds]

print (colored[0:10])

plt.scatter(tocluster[4],tocluster[1],  color = colored)

for ci,c in enumerate(centers):

    plt.plot(c[0], c[1], 'o', markersize=8, color='red', alpha=0.9, label=''+str(ci))



plt.xlabel('x_values')

plt.ylabel('y_values')

plt.legend()

plt.show()
clust_prod = cust_prod.copy()

clust_prod['cluster'] = c_preds



clust_prod.head(10)
print (clust_prod.shape)

f,arr = plt.subplots(2,2,sharex=True,figsize=(15,15))



c1_count = len(clust_prod[clust_prod['cluster']==0])



c0 = clust_prod[clust_prod['cluster']==0].drop('cluster',axis=1).mean()

arr[0,0].bar(range(len(clust_prod.drop('cluster',axis=1).columns)),c0)

c1 = clust_prod[clust_prod['cluster']==1].drop('cluster',axis=1).mean()

arr[0,1].bar(range(len(clust_prod.drop('cluster',axis=1).columns)),c1)

c2 = clust_prod[clust_prod['cluster']==2].drop('cluster',axis=1).mean()

arr[1,0].bar(range(len(clust_prod.drop('cluster',axis=1).columns)),c2)

c3 = clust_prod[clust_prod['cluster']==3].drop('cluster',axis=1).mean()

arr[1,1].bar(range(len(clust_prod.drop('cluster',axis=1).columns)),c3)

plt.show()

c0.sort_values(ascending=False)[0:10]
c1.sort_values(ascending=False)[0:10]
c2.sort_values(ascending=False)[0:10]
c3.sort_values(ascending=False)[0:10]
from IPython.display import display, HTML

cluster_means = [[c0['fresh fruits'],c0['fresh vegetables'],c0['packaged vegetables fruits'], c0['yogurt'], c0['packaged cheese'], c0['milk'],c0['water seltzer sparkling water'],c0['chips pretzels']],

                 [c1['fresh fruits'],c1['fresh vegetables'],c1['packaged vegetables fruits'], c1['yogurt'], c1['packaged cheese'], c1['milk'],c1['water seltzer sparkling water'],c1['chips pretzels']],

                 [c2['fresh fruits'],c2['fresh vegetables'],c2['packaged vegetables fruits'], c2['yogurt'], c2['packaged cheese'], c2['milk'],c2['water seltzer sparkling water'],c2['chips pretzels']],

                 [c3['fresh fruits'],c3['fresh vegetables'],c3['packaged vegetables fruits'], c3['yogurt'], c3['packaged cheese'], c3['milk'],c3['water seltzer sparkling water'],c3['chips pretzels']]]

cluster_means = pd.DataFrame(cluster_means, columns = ['fresh fruits','fresh vegetables','packaged vegetables fruits','yogurt','packaged cheese','milk','water seltzer sparkling water','chips pretzels'])

HTML(cluster_means.to_html())
cluster_perc = cluster_means.iloc[:, :].apply(lambda x: (x / x.sum())*100,axis=1)

HTML(cluster_perc.to_html())
c0.sort_values(ascending=False)[10:15]
c1.sort_values(ascending=False)[10:15]
c2.sort_values(ascending=False)[10:15]
c3.sort_values(ascending=False)[10:15]