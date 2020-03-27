
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

from IPython.display import HTML
HTML('<iframe width="800" height="500" src="https://www.youtube.com/embed/Kb_wzb7-rvE" frameborder="0" allow="accelerometer; autoplay; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>')
import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
#%matplotlib inline
import seaborn as sns
sns.set()

import pydicom

from os import listdir

from skimage.transform import resize
from imgaug import augmenters as iaa

from sklearn.model_selection import train_test_split

from keras.applications import ResNet50, VGG16
from keras.applications.resnet50 import preprocess_input as preprocess_resnet_50
from keras.applications.vgg16 import preprocess_input as preprocess_vgg_16

from keras.utils import Sequence

listdir("../input/rsna-intracranial-hemorrhage-detection/")
INPUT_PATH = "../input/rsna-intracranial-hemorrhage-detection/"
submission = pd.read_csv(INPUT_PATH + "stage_1_sample_submission.csv")
submission.head(10)
traindf = pd.read_csv(INPUT_PATH + "stage_1_train.csv")
traindf.head()
label = traindf.Label.values
traindf = traindf.ID.str.rsplit("_", n=1, expand=True)
traindf.loc[:, "label"] = label
traindf.head()
traindf = traindf.rename({0: "id", 1: "subtype"}, axis=1)
subtype_counts = traindf.groupby("subtype").label.value_counts().unstack()
subtype_counts = subtype_counts.loc[:, 1] / traindf.groupby("subtype").size() * 100
traindf.head()
testdf = submission.ID.str.rsplit("_", n=1, expand=True)
testdf = testdf.rename({0: "id", 1: "subtype"}, axis=1)
testdf.loc[:, "label"] = 0
testdf.head()
multi_target_count = traindf.groupby("id").label.sum()

fig, ax = plt.subplots(1,3,figsize=(20,5))

sns.countplot(traindf.label, ax=ax[0], palette="Reds")
ax[0].set_xlabel("Binary label")
ax[0].set_title("How often do we observe a positive label?");

sns.countplot(multi_target_count, ax=ax[1])
ax[1].set_xlabel("Numer of targets per image")
ax[1].set_ylabel("Frequency")
ax[1].set_title("Multi-Hot occurences")

sns.barplot(x=subtype_counts.index, y=subtype_counts.values, ax=ax[2], palette="Set2")
plt.xticks(rotation=45); 
ax[2].set_title("How much binary imbalance do we have?")
ax[2].set_ylabel("% of positive occurences (1)");

traindf.head()
traindf.id.nunique()
train_dir = INPUT_PATH + "stage_1_train_images/"
train_files = listdir(train_dir)
train_size = len(train_files)
train_size
test_dir = INPUT_PATH + "stage_1_test_images/"
test_files = listdir(test_dir)
test_size = len(test_files)
test_size
train_size/test_size
train_files[0:10]
subtypes = traindf.subtype.unique()
subtypes
dataset = pydicom.dcmread(train_dir + "ID_c5c23af94.dcm")
print(dataset)
HTML('<iframe width="560" height="315" src="https://www.youtube.com/embed/KZld-5W99cI" frameborder="0" allow="accelerometer; autoplay; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>')
fig, ax = plt.subplots(2,1,figsize=(20,10))
for file in train_files[0:10]:
    dataset = pydicom.dcmread(train_dir + file)
    image = dataset.pixel_array.flatten()
    rescaled_image = image * dataset.RescaleSlope + dataset.RescaleIntercept
    sns.distplot(image.flatten(), ax=ax[0]);
    sns.distplot(rescaled_image.flatten(), ax=ax[1])
ax[0].set_title("Raw pixel array distributions for 10 examples")
ax[1].set_title("HU unit distributions for 10 examples");
fig, ax = plt.subplots(4,10,figsize=(20,12))

for n in range(10):
    dataset = pydicom.dcmread(train_dir + train_files[n])
    image = dataset.pixel_array
    rescaled_image = image * dataset.RescaleSlope + dataset.RescaleIntercept
    mask2000 = np.where((rescaled_image <= -1500) & (rescaled_image > -2500), 1, 0)
    mask3000 = np.where(rescaled_image <= -2500, 1, 0)
    ax[0,n].imshow(rescaled_image)
    rescaled_image[rescaled_image < -1024] = -1024
    ax[1,n].imshow(mask2000)
    ax[2,n].imshow(mask3000)
    ax[3,n].imshow(rescaled_image)
    ax[0,n].grid(False)
    ax[1,n].grid(False)
    ax[2,n].grid(False)
    ax[3,n].grid(False)
ax[0,0].set_title("Rescaled image")
ax[1,0].set_title("Mask -2000")
ax[2,0].set_title("Mask -3000");
ax[3,0].set_title("Background to air");
pixelspacing_w = []
pixelspacing_h = []
spacing_filenames = []
for file in train_files[0:1000]:
    dataset = pydicom.dcmread(train_dir + file)
    spacing = dataset.PixelSpacing
    pixelspacing_w.append(spacing[0])
    pixelspacing_h.append(spacing[1])
    spacing_filenames.append(file)

fig, ax = plt.subplots(1,2,figsize=(20,5))
sns.distplot(pixelspacing_w, ax=ax[0], color="Limegreen", kde=False)
ax[0].set_title("Pixel spacing width \n distribution")
ax[0].set_ylabel("Frequency given 1000 images")
sns.distplot(pixelspacing_h, ax=ax[1], color="Mediumseagreen", kde=False)
ax[1].set_title("Pixel spacing height \n distribution");
ax[1].set_ylabel("Frequency given 1000 images");
min_file = spacing_filenames[np.argmin(pixelspacing_w)]
max_file = spacing_filenames[np.argmax(pixelspacing_w)]
def rescale_pixelarray(dataset):
    image = dataset.pixel_array
    rescaled_image = image * dataset.RescaleSlope + dataset.RescaleIntercept
    rescaled_image[rescaled_image < -1024] = -1024
    return rescaled_image
fig, ax = plt.subplots(1,2,figsize=(20,10))

dataset_min = pydicom.dcmread(train_dir + min_file)
image_min = rescale_pixelarray(dataset_min)

dataset_max = pydicom.dcmread(train_dir + max_file)
image_max = rescale_pixelarray(dataset_max)

ax[0].imshow(image_min, cmap="Spectral")
ax[0].set_title("Pixel spacing w: " + str(np.min(pixelspacing_w)))
ax[1].imshow(image_max, cmap="Spectral");
ax[1].set_title("Pixel spacing w: " + str(np.max(pixelspacing_w)))
ax[0].grid(False)
ax[1].grid(False)
np.min(pixelspacing_w) * 512
np.max(pixelspacing_w) * 512
def get_window_value(feature):
    if type(feature) == pydicom.multival.MultiValue:
        return np.int(feature[0])
    else:
        return np.int(feature)
window_widths = []
window_levels = []
spacing_filenames = []
for file in train_files[0:1000]:
    dataset = pydicom.dcmread(train_dir + file)
    win_width = get_window_value(dataset.WindowWidth)
    win_center = get_window_value(dataset.WindowCenter)
    window_widths.append(win_width)
    window_levels.append(win_center)
    spacing_filenames.append(file)
fig, ax = plt.subplots(2,2,figsize=(20,15))

sns.distplot(window_widths, kde=False, ax=ax[0,0], color="Tomato")
ax[0,0].set_title("Window width distribution \n of 1000 images")
ax[0,0].set_xlabel("Window width")
ax[0,0].set_ylabel("Frequency")

sns.distplot(window_levels, kde=False, ax=ax[0,1], color="Firebrick")
ax[0,1].set_title("Window level distribution \n of 1000 images")
ax[0,1].set_xlabel("Window level")
ax[0,1].set_ylabel("Frequency")

sns.distplot(np.log(window_widths), kde=False, ax=ax[1,0], color="Tomato")
ax[1,0].set_title("Log window width distribution \n of 1000 images")
ax[1,0].set_xlabel("Log window width")
ax[1,0].set_ylabel("Frequency")

sns.distplot(np.log(window_levels), kde=False, ax=ax[1,1], color="Firebrick")
ax[1,1].set_title("Log window level distribution \n of 1000 images")
ax[1,1].set_xlabel("Log window level")
ax[1,1].set_ylabel("Frequency");
doc_windows = pd.DataFrame(index=spacing_filenames, columns=["win_width", "win_level"])
doc_windows["win_width"] = window_widths
doc_windows["win_level"] = window_levels
doc_windows.head(20)
doc_windows[doc_windows.win_width==doc_windows.win_width.median()]
doc_windows.describe()
np.quantile(window_widths, 0.95)
np.quantile(window_levels, 0.95)
def set_manual_window(hu_image, custom_center, custom_width):
    min_value = custom_center - (custom_width/2)
    max_value = custom_center + (custom_width/2)
    hu_image[hu_image < min_value] = min_value
    hu_image[hu_image > max_value] = max_value
    return hu_image
fig, ax = plt.subplots(3,4,figsize=(20,15))

docs_dict = {"ID_352b300f9.dcm": {"width": 4000, "level": 600},
             "ID_7e7d7633a.dcm": {"width": 70, "level": 30},
             "ID_87e8b2528.dcm": {"width": 80, "level": 40}}
n = 0
for file in ["ID_352b300f9.dcm", "ID_7e7d7633a.dcm", "ID_87e8b2528.dcm"]:
    dataset = pydicom.dcmread(train_dir + file)
    pixelarray = dataset.pixel_array
    ax[n,0].imshow(pixelarray, cmap="Spectral")
    ax[n,0].grid(False)
    rescaled_image = rescale_pixelarray(dataset)
    ax[n,1].imshow(rescaled_image, cmap="Spectral")
    ax[n,1].grid(False)
    
    org_windowed_image = set_manual_window(rescaled_image, docs_dict[file]["level"], docs_dict[file]["width"])
    ax[n,2].imshow(org_windowed_image, cmap="Spectral")
    ax[n,2].grid(False)
    
    new_windowed_image = set_manual_window(rescaled_image, 40, 150)
    ax[n,3].imshow(new_windowed_image, cmap="Spectral")
    ax[n,3].grid(False)
    
    n+=1
num_rows = []
num_cols = []
spacing_filenames = []
for file in train_files[0:1000]:
    dataset = pydicom.dcmread(train_dir + file)
    num_rows.append(dataset.Rows)
    num_cols.append(dataset.Columns)
    spacing_filenames.append(file)

fig, ax = plt.subplots(1,2,figsize=(20,5))
sns.distplot(num_rows, ax=ax[0], color="Purple", kde=False)
ax[0].set_title("Number of rows \n distribution")
ax[0].set_ylabel("Frequency given 1000 train images")
sns.distplot(num_cols, ax=ax[1], color="Violet", kde=False)
ax[1].set_title("Number of columns \n distribution");
ax[1].set_ylabel("Frequency given 1000 train images");
num_rows = []
num_cols = []
spacing_filenames = []
for file in test_files[0:1000]:
    dataset = pydicom.dcmread(test_dir + file)
    num_rows.append(dataset.Rows)
    num_cols.append(dataset.Columns)
    spacing_filenames.append(file)

fig, ax = plt.subplots(1,2,figsize=(20,5))
sns.distplot(num_rows, ax=ax[0], color="Purple", kde=False)
ax[0].set_title("Number of rows \n distribution")
ax[0].set_ylabel("Frequency given 1000 test images")
sns.distplot(num_cols, ax=ax[1], color="Violet", kde=False)
ax[1].set_title("Number of columns \n distribution");
ax[1].set_ylabel("Frequency given 1000 test images");