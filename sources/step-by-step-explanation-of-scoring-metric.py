
import matplotlib.pyplot as plt
import numpy as np
from scipy import ndimage
import pandas as pd
import imageio
from pathlib import Path

# Get image
im_id = '01d44a26f6680c42ba94c9bc6339228579a95d0e2695b149b7cc0c9592b21baf'
im_dir = Path('../input/stage1_train/{}'.format(im_id))
im_path = im_dir / 'images' / '{}.png'.format(im_id)
im = imageio.imread(im_path.as_posix())

# Get masks
targ_masks = []
for mask_path in im_dir.glob('masks/*.png'):
    targ = imageio.imread(mask_path.as_posix())
    targ_masks.append(targ)
targ_masks = np.stack(targ_masks)

# Make messed up masks
pred_masks = np.zeros(targ_masks.shape)
for ind, orig_mask in enumerate(targ_masks):
    aug_mask = ndimage.rotate(orig_mask, ind*1.5, 
                              mode='constant', reshape=False, order=0)
    pred_masks[ind] = ndimage.binary_dilation(aug_mask, iterations=1)

# Plot the objects
fig, axes = plt.subplots(1,3, figsize=(16,9))
axes[0].imshow(im)
axes[1].imshow(targ_masks.sum(axis=0),cmap='hot')
axes[2].imshow(pred_masks.sum(axis=0), cmap='hot')

labels = ['Original', '"GroundTruth" Masks', '"Predicted" Masks']
for ind, ax in enumerate(axes):
    ax.set_title(labels[ind], fontsize=18)
    ax.axis('off')
A = targ_masks[3]
B = pred_masks[3]
intersection = np.logical_and(A, B)
union = np.logical_or(A, B)

fig, axes = plt.subplots(1,4, figsize=(16,9))
axes[0].imshow(A, cmap='hot')
axes[0].annotate('npixels = {}'.format(np.sum(A>0)), 
                 xy=(114, 245), color='white', fontsize=16)
axes[1].imshow(B, cmap='hot')
axes[1].annotate('npixels = {}'.format(np.sum(B>0)), 
                 xy=(114, 245), color='white', fontsize=16)

axes[2].imshow(intersection, cmap='hot')
axes[2].annotate('npixels = {}'.format(np.sum(intersection>0)), 
                 xy=(114, 245), color='white', fontsize=16)

axes[3].imshow(union, cmap='hot')
axes[3].annotate('npixels = {}'.format(np.sum(union>0)), 
                 xy=(114, 245), color='white', fontsize=16)

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
df = pd.DataFrame()
for ind, gt_mask in enumerate(targ_masks):
    s = get_iou_vector(pred_masks[3], gt_mask, 'P3-GT{}'.format(ind))
    df = df.append(s)
print('Performance of Predicted Mask 3 vs. each Ground Truth mask across IoU thresholds')
print(df)
iou_vol = np.zeros([10, 7, 7])
for ii, pm in enumerate(pred_masks):
    for jj, gt in enumerate(targ_masks):
        s = get_iou_vector(pm, gt, 'P{}-GT{}'.format(ii,jj))
        iou_vol[:,ii,jj] = s.values

mask_labels = ['P{}'.format(x) for x in range(7)]
truth_labels = ['GT{}'.format(x) for x in range(7)]

hits50 = iou_vol[0]
hits75 = iou_vol[4]

fig, axes = plt.subplots(1,2, figsize=(10,9))

axes[0].imshow(hits50, cmap='hot')
axes[0].set_xticks(range(7))
axes[0].set_xticklabels(truth_labels, rotation=45, ha='right', fontsize=16)
axes[0].tick_params(left=False, bottom=False)
axes[0].set_yticks(range(7))
axes[0].set_yticklabels(mask_labels, fontsize=16)
axes[0].tick_params(left=False, bottom=False)
axes[0].set_title('Hit Matrix at $thresh=0.50$', fontsize=18)

axes[1].imshow(hits75, cmap='hot')
axes[1].set_xticks(range(7))
axes[1].set_xticklabels(truth_labels, rotation=45, ha='right', fontsize=16)
axes[1].tick_params(left=False, bottom=False)
axes[1].tick_params(left=False, bottom=False, labelleft=False)
axes[1].set_title('Hit Matrix at $thresh=0.75$', fontsize=18)

for ax in axes:
    # Minor ticks and turn grid on
    ax.set_xticks(np.arange(-.5, 7, 1), minor=True);
    ax.set_yticks(np.arange(-.5, 7, 1), minor=True);
    ax.grid(which='minor', color='w', linestyle='-', linewidth=1)

plt.tight_layout()
plt.show()
def iou_thresh_precision(iou_mat):
    tp = np.sum( iou_mat.sum(axis=1) > 0  )
    fp = np.sum( iou_mat.sum(axis=1) == 0 )
    fn = np.sum( iou_mat.sum(axis=0) == 0 )
    p = tp / (tp + fp + fn)
    return (tp, fp, fn, p)

for thresh, hits in [[0.5, hits50], [0.75, hits75]]:
    tp, fp, fn, p = iou_thresh_precision(hits)
    print('At a threshold of {:0.2f}...\n\tTP = {}\n\tFP = {}\n\tFN = {}\n\tp = {:0.3f}'.format(
                thresh, tp, fp, fn, p))
print('Precision values at each threshold:')
ps = []
for thresh, iou_mat in zip(np.arange(0.5, 1, 0.05), iou_vol):
    _,_,_,p = iou_thresh_precision(iou_mat)
    print('\tt({:0.2f}) = {:0.3f}'.format(thresh, p))
    ps.append(p)
print('Mean precision for image is: {:0.3f}'.format(np.mean(ps)))