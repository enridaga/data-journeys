
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import imageio

from scipy import ndimage
from pathlib import Path

# Get image
im_id = '2bf5343f03'
im_dir = Path('../input/train/')
im_path = im_dir / 'images' / '{}.png'.format(im_id)
img = imageio.imread(im_path.as_posix())

# Get mask
im_dir = Path('../input/train/')
im_path = im_dir / 'masks' / '{}.png'.format(im_id)
target_mask = imageio.imread(im_path.as_posix())

# Fake prediction mask
pred_mask = ndimage.rotate(target_mask, 10, mode='constant', reshape=False, order=0)
pred_mask = ndimage.binary_dilation(pred_mask, iterations=1)

# Plot the objects
fig, axes = plt.subplots(1,3, figsize=(16,9))
axes[0].imshow(img)
axes[1].imshow(target_mask,cmap='hot')
axes[2].imshow(pred_mask, cmap='hot')

labels = ['Original', '"GroundTruth" Mask', '"Predicted" Mask']
for ind, ax in enumerate(axes):
    ax.set_title(labels[ind], fontsize=18)
    ax.axis('off')
A = target_mask
B = pred_mask
intersection = np.logical_and(A, B)
union = np.logical_or(A, B)

fig, axes = plt.subplots(1,4, figsize=(16,9))
axes[0].imshow(A, cmap='hot')
axes[0].annotate('npixels = {}'.format(np.sum(A>0)), 
                 xy=(10, 10), color='white', fontsize=16)
axes[1].imshow(B, cmap='hot')
axes[1].annotate('npixels = {}'.format(np.sum(B>0)), 
                 xy=(10, 10), color='white', fontsize=16)

axes[2].imshow(intersection, cmap='hot')
axes[2].annotate('npixels = {}'.format(np.sum(intersection>0)), 
                 xy=(10, 10), color='white', fontsize=16)

axes[3].imshow(union, cmap='hot')
axes[3].annotate('npixels = {}'.format(np.sum(union>0)), 
                 xy=(10, 10), color='white', fontsize=16)

labels = ['GroundTruth', 'Predicted', 'Intersection', 'Union']
for ind, ax in enumerate(axes):
    ax.set_title(labels[ind], fontsize=18)
    ax.axis('off')
def get_iou_vector(A, B, n):
    intersection = np.logical_and(A, B)
    union = np.logical_or(A, B)
    iou = np.sum(intersection > 0) / np.sum(union > 0)
    s = pd.Series(name=n)
    for thresh in np.arange(0.5,1,0.05):
        s[thresh] = iou > thresh
    return s

print('Does this IoU hit at each threshold?')
print(get_iou_vector(A, B, 'GT-P'))
ppt = get_iou_vector(A, B, 'precisions_per_thresholds')
print ("Average precision value for image `{}` is {}".format(im_id, ppt.mean()))
# `A` is the target mask
ppt = get_iou_vector(A, A, 'GT-P')
print('Does this IoU hit at each threshold?')
print(ppt)
print ("Average precision value for image `{}` is {}".format(im_id, ppt.mean()))