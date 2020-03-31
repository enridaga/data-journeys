
import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

from skimage.io import imread, imshow

import cv2



### matplotlib inline

import plotly.offline as py

py.init_notebook_mode(connected=True)

import plotly.graph_objs as go

import plotly.tools as tls



from subprocess import check_output

print(check_output(["ls", "../input/train"]).decode("utf8"))
from glob import glob

basepath = '../input/train/'



all_cervix_images = []



for path in sorted(glob(basepath + "*")):

    cervix_type = path.split("/")[-1]

    cervix_images = sorted(glob(basepath + cervix_type + "/*"))

    all_cervix_images = all_cervix_images + cervix_images



all_cervix_images = pd.DataFrame({'imagepath': all_cervix_images})

all_cervix_images['filetype'] = all_cervix_images.apply(lambda row: row.imagepath.split(".")[-1], axis=1)

all_cervix_images['type'] = all_cervix_images.apply(lambda row: row.imagepath.split("/")[-2], axis=1)

all_cervix_images.head()
print('We have a total of {} images in the whole dataset'.format(all_cervix_images.shape[0]))

type_aggregation = all_cervix_images.groupby(['type', 'filetype']).agg('count')

type_aggregation_p = type_aggregation.apply(lambda row: 1.0*row['imagepath']/all_cervix_images.shape[0], axis=1)



fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(10, 8))



type_aggregation.plot.barh(ax=axes[0])

axes[0].set_xlabel("image count")

type_aggregation_p.plot.barh(ax=axes[1])

axes[1].set_xlabel("training size fraction")
fig = plt.figure(figsize=(12,8))



i = 1

for t in all_cervix_images['type'].unique():

    ax = fig.add_subplot(1,3,i)

    i+=1

    f = all_cervix_images[all_cervix_images['type'] == t]['imagepath'].values[0]

    plt.imshow(plt.imread(f))

    plt.title('sample for cervix {}'.format(t))
from collections import defaultdict



images = defaultdict(list)



for t in all_cervix_images['type'].unique():

    sample_counter = 0

    for _, row in all_cervix_images[all_cervix_images['type'] == t].iterrows():

        #print('reading image {}'.format(row.imagepath))

        try:

            img = imread(row.imagepath)

            sample_counter +=1

            images[t].append(img)

        except:

            print('image read failed for {}'.format(row.imagepath))

        if sample_counter > 35:

            break
dfs = []

for t in all_cervix_images['type'].unique():

    t_ = pd.DataFrame(

        {

            'nrows': list(map(lambda i: i.shape[0], images[t])),

            'ncols': list(map(lambda i: i.shape[1], images[t])),

            'nchans': list(map(lambda i: i.shape[2], images[t])),

            'type': t

        }

    )

    dfs.append(t_)



shapes_df = pd.concat(dfs, axis=0)

shapes_df_grouped = shapes_df.groupby(by=['nchans', 'ncols', 'nrows', 'type']).size().reset_index().sort_values(['type', 0], ascending=False)

shapes_df_grouped
shapes_df_grouped['size_with_type'] = shapes_df_grouped.apply(lambda row: '{}-{}-{}'.format(row.ncols, row.nrows, row.type), axis=1)

shapes_df_grouped = shapes_df_grouped.set_index(shapes_df_grouped['size_with_type'].values)

shapes_df_grouped['count'] = shapes_df_grouped[[0]]



plt.figure(figsize=(10,8))

#shapes_df_grouped['count'].plot.barh(figsize=(10,8))

sns.barplot(x="count", y="size_with_type", data=shapes_df_grouped)
def transform_image(img, rescaled_dim, to_gray=False):

    resized = cv2.resize(img, (rescaled_dim, rescaled_dim), cv2.INTER_LINEAR)



    if to_gray:

        resized = cv2.cvtColor(resized, cv2.COLOR_RGB2GRAY).astype('float')

    else:

        resized = resized.astype('float')



    normalized = cv2.normalize(resized, None, 0.0, 1.0, cv2.NORM_MINMAX)

    timg = normalized.reshape(1, np.prod(normalized.shape))



    return timg/np.linalg.norm(timg)



rescaled_dim = 100



all_images = []

all_image_types = []



for t in all_cervix_images['type'].unique():

    all_images = all_images + images[t]

    all_image_types = all_image_types + len(images[t])*[t]



# - normalize each uint8 image to the value interval [0, 1] as float image

# - rgb to gray

# - downsample image to rescaled_dim X rescaled_dim

# - L2 norm of each sample = 1

gray_all_images_as_vecs = [transform_image(img, rescaled_dim) for img in all_images]



gray_imgs_mat = np.array(gray_all_images_as_vecs).squeeze()

all_image_types = np.array(all_image_types)

gray_imgs_mat.shape, all_image_types.shape
from sklearn.manifold import TSNE

tsne = TSNE(

    n_components=3,

    init='random', # pca

    random_state=101,

    method='barnes_hut',

    n_iter=500,

    verbose=2

).fit_transform(gray_imgs_mat)
from sklearn import preprocessing



trace1 = go.Scatter3d(

    x=tsne[:,0],

    y=tsne[:,1],

    z=tsne[:,2],

    mode='markers',

    marker=dict(

        sizemode='diameter',

        color = preprocessing.LabelEncoder().fit_transform(all_image_types),

        colorscale = 'Portland',

        colorbar = dict(title = 'cervix types'),

        line=dict(color='rgb(255, 255, 255)'),

        opacity=0.9

    )

)



data=[trace1]

layout=dict(height=800, width=800, title='3D embedding of images')

fig=dict(data=data, layout=layout)

py.iplot(fig, filename='3DBubble')
for t in all_cervix_images['type'].unique():

    tsne_t = tsne[np.where(all_image_types == t), :][0]

    plt.scatter(tsne_t[:, 0], tsne_t[:, 1])

plt.legend(all_cervix_images['type'].unique())
from matplotlib.offsetbox import OffsetImage, AnnotationBbox

def imscatter(x, y, images, ax=None, zoom=0.01):

    ax = plt.gca()

    images = [OffsetImage(image, zoom=zoom) for image in images]

    artists = []

    for x0, y0, im0 in zip(x, y, images):

        ab = AnnotationBbox(im0, (x0, y0), xycoords='data', frameon=False)

        artists.append(ax.add_artist(ab))

    ax.update_datalim(np.column_stack([x, y]))

    ax.autoscale()

    #return artists



nimgs = 60

plt.figure(figsize=(10,8))

imscatter(tsne[0:nimgs,0], tsne[0:nimgs,1], all_images[0:nimgs])
pal = sns.color_palette("hls", 3)

sns.palplot(pal)
from scipy.spatial.distance import pdist, squareform



sq_dists = squareform(pdist(gray_imgs_mat))



all_image_types = list(all_image_types)



d = {

    'Type_1': pal[0],

    'Type_2': pal[1],

    'Type_3': pal[2]

}



# translate each sample to its color

colors = list(map(lambda t: d[t], all_image_types))



sns.clustermap(

    sq_dists,

    figsize=(12,12),

    row_colors=colors, col_colors=colors,

    cmap=plt.get_cmap('viridis')

)


mask = np.zeros_like(sq_dists, dtype=np.bool)

mask[np.triu_indices_from(mask)] = True



plt.figure(figsize=(12,12))

sns.heatmap(sq_dists, cmap=plt.get_cmap('viridis'), square=True, mask=mask)
# upper triangle of matrix set to np.nan

sq_dists[np.triu_indices_from(mask)] = np.nan

sq_dists[0, 0] = np.nan



fig = plt.figure(figsize=(12,8))

# maximally dissimilar image

ax = fig.add_subplot(1,3,1)

maximally_dissimilar_image_idx = np.nanargmax(np.nanmean(sq_dists, axis=1))

plt.imshow(all_images[maximally_dissimilar_image_idx])

plt.title('maximally dissimilar')



# maximally similar image

ax = fig.add_subplot(1,3,2)

maximally_similar_image_idx = np.nanargmin(np.nanmean(sq_dists, axis=1))

plt.imshow(all_images[maximally_similar_image_idx])

plt.title('maximally similar')



# now compute the mean image

ax = fig.add_subplot(1,3,3)

mean_img = gray_imgs_mat.mean(axis=0).reshape(rescaled_dim, rescaled_dim, 3)

plt.imshow(cv2.normalize(mean_img, None, 0.0, 1.0, cv2.NORM_MINMAX))

plt.title('mean image')
from sklearn.preprocessing import LabelEncoder

from sklearn.preprocessing import Normalizer

y = LabelEncoder().fit_transform(all_image_types).reshape(-1)

X = gray_imgs_mat # no need for normalizing, we already did this earlier Normalizer().fit_transform(gray_imgs_mat)

X.shape, y.shape
from sklearn.linear_model import LogisticRegression

from sklearn.model_selection import GridSearchCV, train_test_split



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)



X_train.shape, X_test.shape, y_train.shape, y_test.shape
y_train, y_test
clf = LogisticRegression()

grid = {

    'C': [1e-9, 1e-6, 1e-3, 1e0],

    'penalty': ['l1', 'l2']

}

cv = GridSearchCV(clf, grid, scoring='neg_log_loss', n_jobs=-1, verbose=1)

cv.fit(X_train, y_train)
for i in range(1, len(cv.cv_results_['params'])+1):

    rank = cv.cv_results_['rank_test_score'][i-1]

    s = cv.cv_results_['mean_test_score'][i-1]

    sd = cv.cv_results_['std_test_score'][i-1]

    params = cv.cv_results_['params'][i-1]

    print("{0}. Mean validation neg log loss: {1:.6f} (std: {2:.6f}) - {3}".format(

        rank,

        s,

        sd,

        params

    ))
y_test_hat_p = cv.predict_proba(X_test)
plt.figure(figsize=(12,8))

sns.distplot(y_test_hat_p[:,0], color='red')

sns.distplot(y_test_hat_p[:,1], color='blue')

sns.distplot(y_test_hat_p[:,2], color='green')

dfy = pd.DataFrame({'0': y_test_hat_p[:,0], '1': y_test_hat_p[:,1], '2': y_test_hat_p[:,2]})

sns.pairplot(dfy)
from sklearn.metrics import confusion_matrix



y_test_hat = cv.predict(X_test)



data = [

    go.Heatmap(

        z=confusion_matrix(y_test, y_test_hat),

        x=[0, 1, 2],

        y=[0, 1, 2],

        colorscale='Viridis',

        text = True ,

        opacity = 1.0

    )

]



layout = go.Layout(

    title='Test Confusion matrix',

    xaxis = dict(ticks='', nticks=36),

    yaxis = dict(ticks='' ),

    width = 900, height = 700,

    

)





fig = go.Figure(data=data, layout=layout)

py.iplot(fig, filename='labelled-heatmap')
from sklearn.metrics import confusion_matrix



y_train_hat = cv.predict(X_train)



data = [

    go.Heatmap(

        z=confusion_matrix(y_train, y_train_hat),

        x=[0, 1, 2],

        y=[0, 1, 2],

        colorscale='Viridis',

        text = True ,

        opacity = 1.0

    )

]



layout = go.Layout(

    title='Training Confusion matrix',

    xaxis = dict(ticks='', nticks=36),

    yaxis = dict(ticks='' ),

    width = 900, height = 700,

    

)





fig = go.Figure(data=data, layout=layout)

py.iplot(fig, filename='labelled-heatmap')
