
import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
import pydicom
import os

print('Loaded in libraries!')
TRAIN_IMG_PATH = "../input/rsna-intracranial-hemorrhage-detection/stage_1_train_images/"
TEST_IMG_PATH = "../input/rsna-intracranial-hemorrhage-detection/stage_1_test_images/"
BASE_PATH = '/kaggle/input/rsna-intracranial-hemorrhage-detection/'
TRAIN_DIR = 'stage_1_train_images/'
TEST_DIR = 'stage_1_test_images/'

train = pd.read_csv("../input/rsna-intracranial-hemorrhage-detection/stage_1_train.csv")
sub = pd.read_csv("../input/rsna-intracranial-hemorrhage-detection/stage_1_sample_submission.csv")
train_images = os.listdir("../input/rsna-intracranial-hemorrhage-detection/stage_1_train_images/")
test_images = os.listdir("../input/rsna-intracranial-hemorrhage-detection/stage_1_test_images/")

def window_image(img, window_center,window_width, intercept, slope, rescale=True):

    img = (img*slope +intercept)
    img_min = window_center - window_width//2
    img_max = window_center + window_width//2
    img[img<img_min] = img_min
    img[img>img_max] = img_max
    
    if rescale:
        # Extra rescaling to 0-1, not in the original notebook
        img = (img - img_min) / (img_max - img_min)
    
    return img
    
def get_first_of_dicom_field_as_int(x):
    #get x[0] as in int is x is a 'pydicom.multival.MultiValue', otherwise get int(x)
    if type(x) == pydicom.multival.MultiValue:
        return int(x[0])
    else:
        return int(x)

def get_windowing(data):
    dicom_fields = [data[('0028','1050')].value, #window center
                    data[('0028','1051')].value, #window width
                    data[('0028','1052')].value, #intercept
                    data[('0028','1053')].value] #slope
    return [get_first_of_dicom_field_as_int(x) for x in dicom_fields]

    
    
def view_images(images, title = '', aug = None):
    width = 5
    height = 2
    fig, axs = plt.subplots(height, width, figsize=(15,5))
    
    for im in range(0, height * width):
        data = pydicom.read_file(os.path.join(TRAIN_IMG_PATH,'ID_'+images[im]+ '.dcm'))
        image = data.pixel_array
        window_center , window_width, intercept, slope = get_windowing(data)
        image_windowed = window_image(image, window_center, window_width, intercept, slope)


        i = im // width
        j = im % width
        axs[i,j].imshow(image_windowed, cmap=plt.cm.bone) 
        axs[i,j].axis('off')
        
    plt.suptitle(title)
    plt.show()
#split out information to type and patientID

train['type'] = train['ID'].str.split("_", n = 3, expand = True)[2]
train['PatientID'] = train['ID'].str.split("_", n = 3, expand = True)[1]
train['filename'] = train['ID'].apply(lambda st: "ID_" + st.split('_')[1] + ".png")

sub['filename'] = sub['ID'].apply(lambda st: "ID_" + st.split('_')[1] + ".png")
sub['type'] = sub['ID'].apply(lambda st: st.split('_')[2])

view_images(train[(train['type'] == 'epidural') & (train['Label'] == 1)][:10].PatientID.values, title = 'Images with epidural')
train[(train['type'] == 'subdural') & (train['Label'] == 1)][:10]
###ID_6ef1c9a94 good subdural

case = os.path.join(TRAIN_IMG_PATH,'ID_9d9cc6b01.dcm')

data = pydicom.read_file(case)

#print(data)
window_center , window_width, intercept, slope = get_windowing(data)
img = pydicom.read_file(case).pixel_array

#displaying the image

fig, (ax1, ax2, ax3, ax4) = plt.subplots(4,1, sharex='col', figsize=(10,24), gridspec_kw={'hspace': 0.1, 'wspace': 0})

ax1.set_title('Default window')
im1 = ax1.imshow(img,  cmap=plt.cm.bone)

ax2.set_title('Brain window')
img2 = window_image(img, 40, 80, intercept, slope)
im2 = ax2.imshow(img2, cmap=plt.cm.bone)

ax3.set_title('Subdural window')
img3 = window_image(img, 80, 200, intercept, slope)
im3 = ax3.imshow(img3, cmap=plt.cm.bone)
ax3.annotate('', xy=(150, 380), xytext=(120, 430),
            arrowprops=dict(facecolor='red', shrink=0.05),
            )
ax3.annotate('', xy=(220, 430), xytext=(190, 500),
            arrowprops=dict(facecolor='red', shrink=0.05),
            )

ax4.set_title('Bone window')
img4 = window_image(img, 600, 2800, intercept, slope)
im4 = plt.imshow(img4, cmap=plt.cm.bone)

for ax in fig.axes:
    ax.axis("off")
    
plt.show()
print(case)