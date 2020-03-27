
from glob import glob
import os
import pandas as pd
import numpy as np
import re
from PIL import Image
import seaborn as sns
from random import randrange

#checnking the input files
print(os.listdir("../input/rsna-intracranial-hemorrhage-detection/"))
#reading all dcm files into train and text
train = sorted(glob("../input/rsna-intracranial-hemorrhage-detection/stage_1_train_images/*.dcm"))
test = sorted(glob("../input/rsna-intracranial-hemorrhage-detection/stage_1_test_images/*.dcm"))
print("train files: ", len(train))
print("test files: ", len(test))

pd.reset_option('max_colwidth')
train_df = pd.read_csv('../input/rsna-intracranial-hemorrhage-detection/stage_1_train.csv')
def window_image(img, window_center,window_width, intercept, slope):

    img = (img*slope +intercept)
    img_min = window_center - window_width//2
    img_max = window_center + window_width//2
    img[img<img_min] = img_min
    img[img>img_max] = img_max
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
    
    
    

import pydicom
import matplotlib.pyplot as plt
case = 5

data = pydicom.dcmread(train[case])

#print(data)
window_center , window_width, intercept, slope = get_windowing(data)


#displaying the image
img = pydicom.read_file(train[case]).pixel_array

img = window_image(img, window_center, window_width, intercept, slope)
plt.imshow(img, cmap=plt.cm.bone)
plt.grid(False)

print(data)
TRAIN_IMG_PATH = "../input/rsna-intracranial-hemorrhage-detection/stage_1_train_images/"
TEST_IMG_PATH = "../input/rsna-intracranial-hemorrhage-detection/stage_1_test_images/"

def view_images(images, title = '', aug = None):
    width = 5
    height = 2
    fig, axs = plt.subplots(height, width, figsize=(15,5))
    
    for im in range(0, height * width):
        data = pydicom.read_file(os.path.join(TRAIN_IMG_PATH,images[im]+ '.dcm'))
        image = data.pixel_array
        window_center , window_width, intercept, slope = get_windowing(data)
        image_windowed = window_image(image, window_center, window_width, intercept, slope)


        i = im // width
        j = im % width
        axs[i,j].imshow(image_windowed, cmap=plt.cm.bone) 
        axs[i,j].axis('off')
        
    plt.suptitle(title)
    plt.show()
train_df['image'] = train_df['ID'].str.slice(stop=12)
train_df['diagnosis'] = train_df['ID'].str.slice(start=13)

view_images(train_df[(train_df['diagnosis'] == 'epidural') & (train_df['Label'] == 1)][:10].image.values, title = 'Images with epidural')
view_images(train_df[(train_df['diagnosis'] == 'intraparenchymal') & (train_df['Label'] == 1)][:10].image.values, title = 'Images with intraparenchymal')
view_images(train_df[(train_df['diagnosis'] == 'intraventricular')& (train_df['Label'] == 1)][:10].image.values, title = 'Images with intraventricular')
view_images(train_df[(train_df['diagnosis'] == 'subarachnoid')& (train_df['Label'] == 1)][:10].image.values, title = 'Images with subarachnoid')
view_images(train_df[(train_df['diagnosis'] == 'subdural') & (train_df['Label'] == 1)][:10].image.values, title = 'Images with subarachnoid')