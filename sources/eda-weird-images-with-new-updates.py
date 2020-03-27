
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np 
import pandas as pd 
import gc
import os
import PIL

from scipy import stats
from multiprocessing import Pool
from PIL import ImageOps, ImageFilter
from tqdm import tqdm
from wordcloud import WordCloud

tqdm.pandas()
df_train = pd.read_csv('../input/train-file-with-labels-and-meta-data/weird_images_w_labels.csv')
train_path = '../input/imet-2019-fgvc6/train/'
label_df = pd.read_csv('../input/imet-2019-fgvc6/labels.csv')

print('Files loaded!')
plt.figure(figsize=(14,6))
plt.subplot(121)
sns.distplot(df_train['width'],kde=False, label='Width')
sns.distplot(df_train['height'], kde=False, label='Height')
plt.legend()
plt.title('Image Dimension Histogram', fontsize=15)

plt.subplot(122)
sns.kdeplot(df_train['width'], label='Width')
sns.kdeplot(df_train['height'], label='Height')
plt.legend()
plt.title('Image Dimension KDE Plot', fontsize=15)

plt.tight_layout()
plt.show()
df_train[['width','height']].sort_values(by='width',ascending=False).head()
df_train[['width','height']].sort_values(by='height',ascending=False).head()
weird_height_id = [v for v in df_train.sort_values(by='height',ascending=False).head(20)['id'].values]
weird_width_id = [v for v in df_train.sort_values(by='width',ascending=False).head(20)['id'].values]
plt.figure(figsize=(12,10))

for num, img_id in enumerate(weird_height_id):
    img = PIL.Image.open(f'{train_path}{img_id}.png')
    plt.subplot(1,20,num + 1)
    plt.imshow(img)
    plt.axis('off')
    
plt.suptitle('Images with HUGE Height', fontsize=20)
plt.show()
plt.figure(figsize=(12,10))

for num, img_id in enumerate(weird_width_id):
    img = PIL.Image.open(f'{train_path}{img_id}.png')
    plt.subplot(20,1,num + 1)
    plt.imshow(img)
    plt.axis('off')
    
plt.suptitle('Images with HUGE Width', fontsize=20)
plt.show()
img = PIL.Image.open(f'{train_path}{weird_height_id[0]}.png')

w_resized = int(img.size[0] * 300 / img.size[1])
resized = img.resize((w_resized ,300))
pad_width = 300 - w_resized
padding = (pad_width // 2, 0, pad_width-(pad_width//2), 0)
resized_w_pad = ImageOps.expand(resized, padding)

resized_wo_pad = img.resize(size=(300,300))
plt.figure(figsize=(12,8))

plt.subplot(131)
plt.imshow(img)
plt.axis('off')
plt.title('Original Image',fontsize=15)

plt.subplot(132)
plt.imshow(resized_wo_pad)
plt.axis('off')
plt.title('A bent flat head screw?',fontsize=15)

plt.subplot(133)
plt.imshow(resized_w_pad)
plt.axis('off')
plt.title('Padded Image',fontsize=15)

plt.show()
label_names = label_df['attribute_name'].values

num_labels = np.zeros((df_train.shape[0],))
train_labels = np.zeros((df_train.shape[0], len(label_names)))

for row_index, row in enumerate(df_train['attribute_ids']):
    num_labels[row_index] = len(row.split())    
    for label in row.split():
        train_labels[row_index, int(label)] = 1
culture, tag, unknown = 0, 0, 0

for l in label_names:
    if l[:3] == 'cul':
        culture += 1
    elif l[:3] == 'tag':
        tag += 1
    else:
        unknown += 1
        
print(f'Culture : {culture}')
print(f'Tag     : {tag}')
print(f'Unknown : {unknown}')
print(f'Total   : {culture + tag + unknown}')
label_sum = np.sum(train_labels, axis=0)

culture_sequence = label_sum[:398].argsort()[::-1]
tag_sequence = label_sum[398:].argsort()[::-1]

culture_labels = [label_names[x][9:] for x in culture_sequence]
culture_counts = [label_sum[x] for x in culture_sequence]

tag_labels = [label_names[x + 398][5:] for x in tag_sequence]
tag_counts = [label_sum[x + 398] for x in tag_sequence]
culture_labels_dict = dict((l,1) for l in culture_labels)
tag_labels_dict = dict((l,1) for l in tag_labels)

culture_labels_dict['<CULTURE>'] = 50
tag_labels_dict['<TAG>'] = 50

culture_cloud = WordCloud(background_color='Black', colormap='Paired', width=1600, height=800, random_state=123).generate_from_frequencies(culture_labels_dict)
tag_cloud = WordCloud(background_color='Black', colormap='Paired', width=1600, height=800, random_state=123).generate_from_frequencies(tag_labels_dict)

plt.figure(figsize=(24,24))
plt.subplot(211)
plt.imshow(culture_cloud,interpolation='bilinear')
plt.axis('off')

plt.subplot(212)
plt.imshow(tag_cloud, interpolation='bilinear')
plt.axis('off')

plt.tight_layout()
plt.show()
plt.figure(figsize=(20,15))

plt.subplot(1,2,1)
ax1 = sns.barplot(y=culture_labels[:20], x=culture_counts[:20], orient="h")
plt.title('Label Counts by Culture (Top 20)',fontsize=15)
plt.xlim((0, max(culture_counts)*1.15))
plt.yticks(fontsize=15)

for p in ax1.patches:
    ax1.annotate(f'{int(p.get_width())}\n{p.get_width() * 100 / df_train.shape[0]:.2f}%',
                (p.get_width(), p.get_y() + p.get_height() / 2.), 
                ha='left', 
                va='center', 
                fontsize=12, 
                color='black',
                xytext=(7,0), 
                textcoords='offset points')

plt.subplot(1,2,2)    
ax2 = sns.barplot(y=tag_labels[:20], x=tag_counts[:20], orient="h")
plt.title('Label Counts by Tag (Top 20)',fontsize=15)
plt.xlim((0, max(tag_counts)*1.15))
plt.yticks(fontsize=15)

for p in ax2.patches:
    ax2.annotate(f'{int(p.get_width())}\n{p.get_width() * 100 / df_train.shape[0]:.2f}%',
                (p.get_width(), p.get_y() + p.get_height() / 2.), 
                ha='left', 
                va='center', 
                fontsize=12, 
                color='black',
                xytext=(7,0), 
                textcoords='offset points')

plt.tight_layout()
plt.show()
plt.figure(figsize=(20,8))

ax = sns.countplot(num_labels)
plt.xlabel('Number of Labels')
plt.title('Number of Labels per Image', fontsize=20)

for p in ax.patches:
    ax.annotate(f'{p.get_height() * 100 / df_train.shape[0]:.3f}%',
            (p.get_x() + p.get_width() / 2., p.get_height()), 
            ha='center', 
            va='center', 
            fontsize=11, 
            color='black',
            xytext=(0,7), 
            textcoords='offset points')
weird_img_index = np.nonzero(num_labels == 11)[0][0]

img_w_11_labels_path = df_train.iloc[weird_img_index,0]
img_labels = df_train.iloc[weird_img_index,1]

img = PIL.Image.open(f'{train_path}{img_w_11_labels_path}.png')

print('LABELS OF IMAGE\n', *[label_names[int(l)] for l in sorted([int(l) for l in img_labels.split()])],sep='\n')

plt.figure(figsize=(20,6))
plt.imshow(img)
plt.axis('off')
plt.show()
pal = ['red', 'green', 'blue']

plt.figure(figsize=(20,6))
plt.subplot(1,2,1)
sns.kdeplot(df_train['r_mean'], color=pal[0])
sns.kdeplot(df_train['g_mean'], color=pal[1])
sns.kdeplot(df_train['b_mean'], color=pal[2])
plt.ylabel('Density')
plt.xlabel('Mean Pixel Value')
plt.title('KDE Plot - Mean Pixel Value by Channel', fontsize=15)

plt.subplot(1,2,2)
sns.kdeplot(df_train['r_std'], color=pal[0])
sns.kdeplot(df_train['g_std'], color=pal[1])
sns.kdeplot(df_train['b_std'], color=pal[2])
plt.ylabel('Density')
plt.xlabel('Standard Deviation of Pixel Value')
plt.title('KDE Plot - Stdev Pixel Value by Channel', fontsize=15)

plt.show()
reddest = (df_train['r_mean'] - df_train['g_mean'] - df_train['b_mean'] + 255*2)
greenest = (df_train['g_mean'] - df_train['r_mean'] - df_train['b_mean'] + 255*2)
bluest = (df_train['b_mean'] - df_train['g_mean'] - df_train['r_mean'] + 255*2)

reddest_img_path = df_train.iloc[reddest.idxmax(),0]
greenest_img_path = df_train.iloc[greenest.idxmax(),0]
bluest_img_path = df_train.iloc[bluest.idxmax(),0]
reddest_im = PIL.Image.open(f'{train_path}{reddest_img_path}.png')
greenest_im = PIL.Image.open(f'{train_path}{greenest_img_path}.png')
bluest_im = PIL.Image.open(f'{train_path}{bluest_img_path}.png')

plt.figure(figsize=(20,6))
plt.subplot(131)
plt.imshow(reddest_im)
plt.axis('off')
plt.title('REDDEST glass-like object')

plt.subplot(132)
plt.imshow(greenest_im)
plt.axis('off')
plt.title('GREENEST piece of ancient writing')

plt.subplot(133)
plt.imshow(bluest_im)
plt.axis('off')
plt.title('BLUEST dark sky with some flags')

plt.show()
plt.figure(figsize=(20,4))

random_image_paths = df_train['id'].sample(n=3, random_state=123).values

for index, path in enumerate(random_image_paths):
    im = PIL.Image.open(f'{train_path}{path}.png')
    plt.subplot(1,6, index*2 + 1)
    plt.imshow(im)
    plt.axis('off')
    plt.title('Original')

    plt.subplot(1,6, index*2 + 2)
    plt.imshow(im.filter(ImageFilter.FIND_EDGES))
    plt.axis('off')
    plt.title('Edge Only')

plt.show()
pal = ['red', 'green', 'blue']

plt.figure(figsize=(20,6))
plt.subplot(1,2,1)
sns.kdeplot(df_train['r_edge_mean'], color=pal[0])
sns.kdeplot(df_train['g_edge_mean'], color=pal[1])
sns.kdeplot(df_train['b_edge_mean'], color=pal[2])
plt.ylabel('Density')
plt.xlabel('Mean Pixel Value')
plt.title('KDE Plot - Mean Pixel Value (Edge) by Channel', fontsize=15)

plt.subplot(1,2,2)
sns.kdeplot(df_train['r_edge_std'], color=pal[0])
sns.kdeplot(df_train['g_edge_std'], color=pal[1])
sns.kdeplot(df_train['b_edge_std'], color=pal[2])
plt.ylabel('Density')
plt.xlabel('Standard Deviation of Pixel Value')
plt.title('KDE Plot - Stdev Pixel Value (Edge) by Channel', fontsize=15)

plt.show()
edge_min, edge_median, edge_max = df_train['r_edge_mean'].min(), df_train['r_edge_mean'].median(), df_train['r_edge_mean'].max()

low_mean_edge = df_train.loc[df_train['r_edge_mean'] == edge_min, 'id'].values[0]
med_mean_edge = df_train.loc[df_train['r_edge_mean'] == edge_median, 'id'].values[0]
high_mean_edge = df_train.loc[df_train['r_edge_mean'] == edge_max, 'id'].values[0]

low_mean_edge_im = PIL.Image.open(f'{train_path}{low_mean_edge}.png')
med_mean_edge_im = PIL.Image.open(f'{train_path}{med_mean_edge}.png')
high_mean_edge_im = PIL.Image.open(f'{train_path}{high_mean_edge}.png')

plt.figure(figsize=(20,16))
plt.subplot(231)
plt.imshow(low_mean_edge_im)
plt.axis('off')
plt.title(f'Mean Edge = {edge_min:.2f} (Raw)')

plt.subplot(232)
plt.imshow(med_mean_edge_im)
plt.axis('off')
plt.title(f'Mean Edge = {edge_median:.2f} (Raw)')

plt.subplot(233)
plt.imshow(high_mean_edge_im)
plt.axis('off')
plt.title(f'Mean Edge = {edge_max:.2f} (Raw)')

plt.subplot(234)
plt.imshow(low_mean_edge_im.filter(ImageFilter.FIND_EDGES))
plt.axis('off')
plt.title(f'Mean Edge = {edge_min:.2f} (Edge)')

plt.subplot(235)
plt.imshow(med_mean_edge_im.filter(ImageFilter.FIND_EDGES))
plt.axis('off')
plt.title(f'Mean Edge = {edge_median:.2f} (Edge)')

plt.subplot(236)
plt.imshow(high_mean_edge_im.filter(ImageFilter.FIND_EDGES))
plt.axis('off')
plt.title(f'Mean Edge = {edge_max:.2f} (Edge)')

plt.show()
