
import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from sklearn.decomposition import PCA # Principal Component Analysis module

from sklearn.cluster import KMeans # KMeans clustering 

import matplotlib.pyplot as plt # Python defacto plotting library

import seaborn as sns # More snazzy plotting library

### matplotlib inline 
movie = pd.read_csv('../input/movie_metadata.csv') # reads the csv and creates the dataframe called movie

movie.head()
str_list = [] # empty list to contain columns with strings (words)

for colname, colvalue in movie.iteritems():

    if type(colvalue[1]) == str:

         str_list.append(colname)

# Get to the numeric columns by inversion            

num_list = movie.columns.difference(str_list)         
movie_num = movie[num_list]

#del movie # Get rid of movie df as we won't need it now

movie_num.head()
movie_num = movie_num.fillna(value=0, axis=1)
X = movie_num.values

# Data Normalization

from sklearn.preprocessing import StandardScaler

X_std = StandardScaler().fit_transform(X)
movie.plot(y= 'imdb_score', x ='duration',kind='hexbin',gridsize=35, sharex=False, colormap='cubehelix', title='Hexbin of Imdb_Score and Duration',figsize=(12,8))

movie.plot(y= 'imdb_score', x ='gross',kind='hexbin',gridsize=45, sharex=False, colormap='cubehelix', title='Hexbin of Imdb_Score and Gross',figsize=(12,8))
# Set up the matplotlib figure

f, ax = plt.subplots(figsize=(12, 10))

plt.title('Pearson Correlation of Movie Features')

# Draw the heatmap using seaborn

sns.heatmap(movie_num.astype(float).corr(),linewidths=0.25,vmax=1.0, square=True, cmap="YlGnBu", linecolor='black', annot=True)
# Calculating Eigenvectors and eigenvalues of Cov matirx

mean_vec = np.mean(X_std, axis=0)

cov_mat = np.cov(X_std.T)

eig_vals, eig_vecs = np.linalg.eig(cov_mat)
# Create a list of (eigenvalue, eigenvector) tuples

eig_pairs = [ (np.abs(eig_vals[i]),eig_vecs[:,i]) for i in range(len(eig_vals))]



# Sort from high to low

eig_pairs.sort(key = lambda x: x[0], reverse= True)



# Calculation of Explained Variance from the eigenvalues

tot = sum(eig_vals)

var_exp = [(i/tot)*100 for i in sorted(eig_vals, reverse=True)] # Individual explained variance

cum_var_exp = np.cumsum(var_exp) # Cumulative explained variance
# PLOT OUT THE EXPLAINED VARIANCES SUPERIMPOSED 

plt.figure(figsize=(10, 5))

plt.bar(range(16), var_exp, alpha=0.3333, align='center', label='individual explained variance', color = 'g')

plt.step(range(16), cum_var_exp, where='mid',label='cumulative explained variance')

plt.ylabel('Explained variance ratio')

plt.xlabel('Principal components')

plt.legend(loc='best')

plt.show()
pca = PCA(n_components=9)

x_9d = pca.fit_transform(X_std)
plt.figure(figsize = (9,7))

plt.scatter(x_9d[:,0],x_9d[:,1], c='goldenrod',alpha=0.5)

plt.ylim(-10,30)

plt.show()
# Set a 3 KMeans clustering

kmeans = KMeans(n_clusters=3)

# Compute cluster centers and predict cluster indices

X_clustered = kmeans.fit_predict(x_9d)



# Define our own color map

LABEL_COLOR_MAP = {0 : 'r',1 : 'g',2 : 'b'}

label_color = [LABEL_COLOR_MAP[l] for l in X_clustered]



# Plot the scatter digram

plt.figure(figsize = (7,7))

plt.scatter(x_9d[:,0],x_9d[:,2], c= label_color, alpha=0.5) 

plt.show()
# Create a temp dataframe from our PCA projection data "x_9d"

df = pd.DataFrame(x_9d)

df = df[[0,1,2]] # only want to visualise relationships between first 3 projections

df['X_cluster'] = X_clustered
# Call Seaborn's pairplot to visualize our KMeans clustering on the PCA projected data

sns.pairplot(df, hue='X_cluster', palette= 'Dark2', diag_kind='kde',size=1.85)