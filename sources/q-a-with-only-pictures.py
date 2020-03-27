
import glob
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import pydicom
import numpy as np
import warnings
import multiprocessing
import os
from skimage import morphology
from skimage import feature
from skimage import measure
from skimage import util
from skimage import transform

warnings.filterwarnings('ignore')
sns.set_style('darkgrid')
sns.set_context('notebook', font_scale=1.2)
plt.rcParams['figure.figsize'] = [14, 8]
plt.rcParams['lines.linewidth'] = 2.5

# Get all data
tr = pd.read_csv('../input/stage_1_train_labels.csv')
tr['aspect_ratio'] = (tr['width']/tr['height'])
tr['area'] = tr['width'] * tr['height']

def get_info(patientId, root_dir='../input/stage_1_train_images/'):
    fn = os.path.join(root_dir, f'{patientId}.dcm')
    dcm_data = pydicom.read_file(fn)
    return {'age': dcm_data.PatientAge, 
            'gender': dcm_data.PatientSex, 
            'id': os.path.basename(fn).split('.')[0],
            'pixel_spacing': float(dcm_data.PixelSpacing[0]),
            'mean_black_pixels': np.mean(dcm_data.pixel_array == 0)}

patient_ids = list(tr.patientId.unique())
with multiprocessing.Pool(4) as pool:
    result = pool.map(get_info, patient_ids)
    
demo = pd.DataFrame(result)
demo['gender'] = demo['gender'].astype('category')
demo['age'] = demo['age'].astype(int)

tr = (tr.merge(demo, left_on='patientId', right_on='id', how='left')
        .drop(columns='id'))
boxes_per_patient = tr.groupby('patientId')['Target'].sum()

ax = (boxes_per_patient > 0).value_counts().plot.bar()
_ = ax.set_title('Are the classes imbalanced?')
_ = ax.set_xlabel('Has Pneumonia')
_ = ax.set_ylabel('Count')
_ = ax.xaxis.set_tick_params(rotation=0)
ax = boxes_per_patient.value_counts().plot.bar()
_ = ax.set_title('How many cases are there per image?')
_ = ax.set_xlabel('Number of cases')
_ = ax.xaxis.set_tick_params(rotation=0)
centers = (tr.dropna(subset=['x'])
           .assign(center_x=tr.x + tr.width / 2, center_y=tr.y + tr.height / 2))
ax = sns.jointplot("center_x", "center_y", data=centers, height=9, alpha=0.1)
_ = ax.fig.suptitle("Where is Pneumonia located?", y=1.01)
g = sns.FacetGrid(col='Target', hue='gender', 
                  data=tr.drop_duplicates(subset=['patientId']), 
                  height=9, palette=dict(F="red", M="blue"))
_ = g.map(sns.distplot, 'age', hist_kws={'alpha': 0.3}).add_legend()
_ = g.fig.suptitle("What is the age distribution by gender and target?", y=1.02, fontsize=20)
areas = tr.dropna(subset=['area'])
g = sns.FacetGrid(hue='gender', data=areas, height=9, palette=dict(F="red", M="blue"), aspect=1.4)
_ = g.map(sns.distplot, 'area', hist_kws={'alpha': 0.3}).add_legend()
_ = g.fig.suptitle('What are the areas of the bounding boxes by gender?', y=1.01)
pixel_vc = tr.drop_duplicates('patientId')['pixel_spacing'].value_counts()
ax = pixel_vc.iloc[:6].plot.bar()
_ = ax.set_xticklabels([f'{ps:.4f}' for ps in pixel_vc.index[:6]])
_ = ax.set_xlabel('Pixel Spacing')
_ = ax.set_ylabel('Count')
_ = ax.set_title('How is the pixel spacing distributed?', fontsize=20)
areas_with_count = areas.merge(pd.DataFrame(boxes_per_patient).rename(columns={'Target': 'bbox_count'}), 
                               on='patientId')
g = sns.FacetGrid(hue='bbox_count', data=areas_with_count, height=8, aspect=1.4)
_ = g.map(sns.distplot, 'area').add_legend()
_ = g.fig.suptitle("How are the bounding box areas distributed by the number of boxes?", y=1.01)
from sklearn.mixture import GaussianMixture
clf = GaussianMixture(n_components=2)
clf.fit(centers[['center_x', 'center_y']])
center_probs = clf.predict_proba(centers[['center_x', 'center_y']])
Z = -clf.score_samples(centers[['center_x', 'center_y']])
outliers = centers.iloc[Z > 17]
fig, ax = plt.subplots()
centers.plot.scatter('center_x', 'center_y', c=Z, alpha=0.5, cmap='viridis', ax=ax)
outliers.plot.scatter('center_x', 'center_y', c='red', marker='x', s=100, ax=ax)
_ = ax.set_title('Where are the outliers?', fontsize=18)
import matplotlib.patches as patches

def get_image(patientId, root_dir='../input/stage_1_train_images/'):
    fn = os.path.join(root_dir, f'{patientId}.dcm')
    dcm_data = pydicom.read_file(fn)
    return dcm_data.pixel_array

def draw_bbs(bbs, ax):
    for bb in bbs.itertuples():
        rect = patches.Rectangle(
            (bb.x, bb.y), bb.width, bb.height,
            linewidth=2, edgecolor='red', facecolor='none')
        ax.add_patch(rect)

def draw_image(img, bbs, ax):
    ax.imshow(img, cmap='gray')
    ax.grid(False)
    ax.set_xticks([])
    ax.set_yticks([])
    if bbs is not None:
        draw_bbs(bbs, ax)

outliers_15 = outliers.drop_duplicates(subset=['patientId']).iloc[:15]
fig, axes = plt.subplots(3, 5)
for row, ax in zip(outliers_15.itertuples(), axes.flatten()):
    img = get_image(row.patientId)
    bbs = tr.loc[tr.patientId == row.patientId, ['x', 'y', 'width', 'height']]
    draw_image(img, bbs, ax)
fig.tight_layout(pad=-0.5)
ax = sns.boxplot(tr.mean_black_pixels)
_ = ax.set_xlabel('Percentage of black pixels')
_ = ax.set_title('Are there images with mostly black pixels?')
high_black_pixel_patientIds = tr.loc[tr.mean_black_pixels > 0.55, 'patientId'].drop_duplicates()
fig, axes = plt.subplots(4, 5)
for i, (patient_id, ax) in enumerate(zip(high_black_pixel_patientIds, axes.flatten())):
    row = tr.loc[tr.patientId == patient_id]
    img = get_image(row.patientId.iloc[0])
    bbs = row[['x', 'y', 'width', 'height']]
    draw_image(img, bbs, ax)
fig.tight_layout(pad=-1)
high_white_pixel_patientIds = tr.loc[tr.mean_black_pixels < 0.000001, 'patientId'].drop_duplicates()
fig, axes = plt.subplots(4, 5)
for patient_id, ax in zip(high_white_pixel_patientIds, axes.flatten()):
    row = tr.loc[tr.patientId == patient_id]
    img = get_image(row.patientId.iloc[0])
    bbs = row[['x', 'y', 'width', 'height']]
    draw_image(img, bbs, ax)
fig.tight_layout(pad=-1)
high_black_pixel_images = np.empty(shape=(high_black_pixel_patientIds.shape[0], 1024, 1024))

for i, patient_id in enumerate(high_black_pixel_patientIds):
    row = tr.loc[tr.patientId == patient_id]
    img = get_image(row.patientId.iloc[0])
    high_black_pixel_images[i] = img 
    
high_black_pixel_contours = []
for img in high_black_pixel_images:
    img2 = feature.canny(img != 0)
    img2 = morphology.convex_hull_image(img2)
    c = measure.find_contours(img2, 0)[0]
    c = measure.approximate_polygon(c, 20)
    high_black_pixel_contours.append(c)

fig, axes = plt.subplots(4, 5)
contours = []
for c, img, ax in zip(high_black_pixel_contours, high_black_pixel_images, axes.flatten()):
    draw_image(img, None, ax)
    _ = ax.plot(c[:, 1], c[:, 0], '-b', linewidth=4)
fig.tight_layout(pad=-1)
def order_coordinates(coords):
    """Returns coordinates with order:
    (top left, top right, bottom right, bottom left)
    """
    coords = coords[:-1]
    output = np.empty((4, 2), dtype=np.float32)
    dists = coords[:, 1]**2 + coords[:, 0]**2
    ratios = coords[:, 1]/np.sqrt(dists)
    
    tl = coords[np.argmin(dists)]
    br = coords[np.argmax(dists)]
    
    tr = coords[np.argmax(ratios)]
    bl = coords[np.argmin(ratios)]
    
    output[0] = tl
    output[1] = tr
    output[2] = br
    output[3] = bl
    
    return output[:,::-1]

def _convert_bb(bb, tfm):
    x, y, w, h = bb.x, bb.y, bb.width, bb.height
    pts = np.array([
        [x, y],
        [x + w, y],
        [x + w, y + h],
        [x, y + h]
    ])
    new_pts = tfm.inverse(pts)
    pts_min = np.min(new_pts, axis=0)
    pts_max = np.max(new_pts, axis=0)
    
    x, y = pts_min
    w, h = pts_max - pts_min
    
    return np.array([x, y, w, h])

def convert_bbs(bboxs, tfm):
    output = np.empty_like(bboxs, dtype=np.float32)
    
    for i, bb in enumerate(bboxs.itertuples()):
        output[i] = _convert_bb(bb, tfm)
    
    return pd.DataFrame(output, columns=['x', 'y', 'width', 'height'])

fig, axes = plt.subplots(4, 2, figsize=(8, 10))

orig_coords = np.array([[0, 0], [1024, 0], [1024, 1024], [0, 1024]])
interesting_idices = [0, 2, 3, 17]

for i, (ax1, ax2) in zip(interesting_idices, axes):
    patient_id = high_black_pixel_patientIds.iloc[i]
    img = high_black_pixel_images[i]
    contour = high_black_pixel_contours[i]
    
    row = tr.loc[tr.patientId == patient_id]
    bbs = row[['x', 'y', 'width', 'height']]
    ordered_coors = order_coordinates(contour)
    tform = transform.estimate_transform('projective', orig_coords, ordered_coors)
    img_t = transform.warp(img, tform, output_shape=(1024, 1024))
    
    new_bbs = convert_bbs(bbs, tform)
    _ = draw_image(img, bbs, ax1)
    _ = draw_image(img_t, new_bbs, ax2)
    
fig.tight_layout(pad=-1)
ax = sns.distplot(tr['aspect_ratio'].dropna(), norm_hist=True)
_ = ax.set_title("What does the distribution of bounding aspect ratios look like?")
_ = ax.set_xlabel("Aspect Ratio")
aspect_ratios = tr['aspect_ratio'].dropna()
high_aspect_ratio_tr = (tr.iloc[aspect_ratios[aspect_ratios > aspect_ratios.quantile(q=0.99)].index]
                          .drop_duplicates(['patientId']))
fig, axes = plt.subplots(3, 5)
for row, ax in zip(high_aspect_ratio_tr.itertuples(), axes.flatten()):
    img = get_image(row.patientId)
    bbs = tr.loc[tr.patientId == row.patientId, ['x', 'y', 'width', 'height']]
    draw_image(img, bbs, ax)
fig.tight_layout(pad=-0.5)
g = sns.relplot(x='area', y='aspect_ratio', 
            data=tr.dropna(subset=['area', 'aspect_ratio']), 
            height=8, alpha=0.8, aspect=1.4,)
_ = g.fig.suptitle("Is there a relationship between the bounding box's aspect ratio and area?", y=1.005)