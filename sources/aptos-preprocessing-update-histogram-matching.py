
# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import math

import cv2
import matplotlib.pyplot as plt
import matplotlib
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
#for dirname, _, filenames in os.walk('/kaggle/input'):
    #for filename in filenames:
        #print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.
print(os.listdir('/kaggle/input/aptos2019-blindness-detection/'))
root = '/kaggle/input/aptos2019-blindness-detection/'
train_df = pd.read_csv(os.path.join(root, 'train.csv'))
train_df.shape
train_df.head(10)
train_df['diagnosis'].value_counts()
fig=plt.figure(figsize=(8, 5))
train_df['diagnosis'].value_counts().plot.bar()
#%%time



SEED = 125
fig = plt.figure(figsize=(25, 16))
img_list = []
img_size = []
# display 10 images from each class
#for class_id in sorted(train_y.unique()):
for class_id in [0, 1, 2, 3, 4]:
    for i, (idx, row) in enumerate(train_df.loc[train_df['diagnosis'] == class_id].sample(5, random_state=SEED).iterrows()):
        ax = fig.add_subplot(5, 5, class_id * 5 + i + 1, xticks=[], yticks=[])
        path = os.path.join(root, 'train_images', '{}.png'.format(row['id_code']))
        image = cv2.imread(path)
        img_size.append(image.shape)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        #image = cv2.resize(image, (IMG_SIZE, IMG_SIZE))
        img_list.append(row['id_code'])
        
        plt.imshow(image)
        ax.set_title('Label: %d ' % (class_id) )
IMAGE_SIZE = 256

def info_image(im):
    # Compute the center (cx, cy) and radius of the eye
    cy = im.shape[0]//2
    midline = im[cy,:]
    midline = np.where(midline>midline.mean()/3)[0]
    if len(midline)>im.shape[1]//2:
        x_start, x_end = np.min(midline), np.max(midline)
    else: # This actually rarely happens p~1/10000
        x_start, x_end = im.shape[1]//10, 9*im.shape[1]//10
    cx = (x_start + x_end)/2
    r = (x_end - x_start)/2
    return cx, cy, r

def resize_image(im, augmentation=False):
    # Crops, resizes and potentially augments the image to IMAGE_SIZE
    cx, cy, r = info_image(im)
    scaling = IMAGE_SIZE/(2*r)
    rotation = 0
    if augmentation:
        scaling *= 1 + 0.3 * (np.random.rand()-0.5)
        rotation = 360 * np.random.rand()
    M = cv2.getRotationMatrix2D((cx,cy), rotation, scaling)
    M[0,2] -= cx - IMAGE_SIZE/2
    M[1,2] -= cy - IMAGE_SIZE/2
    return cv2.warpAffine(im,M,(IMAGE_SIZE,IMAGE_SIZE)) # This is the most important line

def subtract_median_bg_image(im):
    k = np.max(im.shape)//20*2+1
    bg = cv2.medianBlur(im, k)
    return cv2.addWeighted (im, 4, bg, -4, 128)

#def subtract_gaussian_bg_image(im):
#    k = np.max(im.shape)/10
#    bg = cv2.GaussianBlur(im ,(0,0) ,k)
#    return cv2.addWeighted (im, 4, bg, -4, 128)
# To remove irregularities along the circular boundary of the image
PARAM = 96
def Radius_Reduction(img,PARAM):
    h,w,c=img.shape
    Frame=np.zeros((h,w,c),dtype=np.uint8)
    cv2.circle(Frame,(int(math.floor(w/2)),int(math.floor(h/2))),int(math.floor((h*PARAM)/float(2*100))), (255,255,255), -1)
    Frame1=cv2.cvtColor(Frame, cv2.COLOR_BGR2GRAY)
    img1 =cv2.bitwise_and(img,img,mask=Frame1)
    return img1
image = cv2.imread(os.path.join(root, 'train_images', '{}.png'.format(img_list[0])))
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
fig=plt.figure(figsize=(8, 8))
plt.imshow(image)
fig=plt.figure(figsize=(8, 8))
res_image = resize_image(image)
plt.imshow(res_image)
fig=plt.figure(figsize=(8, 8))
sub_med = subtract_median_bg_image(res_image)
plt.imshow(sub_med)
fig=plt.figure(figsize=(8, 8))
img_rad_red=Radius_Reduction(sub_med, PARAM)
plt.imshow(img_rad_red)
w=10
h=10
fig=plt.figure(figsize=(20, 20))
columns = 5
rows = 5
for i in range(1, columns*rows +1):
    #img = np.random.randint(10, size=(h,w))
    img = cv2.imread(os.path.join(root, 'train_images', '{}.png'.format(img_list[i-1])))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    res_image = resize_image(img)
    sub_med = subtract_median_bg_image(res_image)
    img_rad_red=Radius_Reduction(sub_med, PARAM)
    fig.add_subplot(rows, columns, i)
    plt.imshow(img_rad_red)
plt.show()
image = cv2.imread(os.path.join(root, 'train_images', '{}.png'.format(img_list[0])))
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
res_image = resize_image(image)

matplotlib.rc('figure', figsize=[7, 7])
plt.imshow(res_image)
def adjust_gamma(image, gamma):
    # build a lookup table mapping the pixel values [0, 255] to
    # their adjusted gamma values
    invGamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** invGamma) * 255
                for i in np.arange(0, 256)]).astype("uint8")
    
    # apply gamma correction using the lookup table
    return cv2.LUT(image, table)
adjusted = adjust_gamma(res_image, gamma=0.5)
adjusted_75 = adjust_gamma(res_image, gamma=0.75)
adjusted_15 = adjust_gamma(res_image, gamma=1.5)
adjusted_3 = adjust_gamma(res_image, gamma=2.5)
matplotlib.rc('figure', figsize=[15, 15])

fig, axarr = plt.subplots(2,2)
axarr[0,0].imshow(adjusted)
axarr[0,1].imshow(adjusted_75)
axarr[1,0].imshow(adjusted_15)
axarr[1,1].imshow(adjusted_3)
Parameter = 95

def Redius_Reduction(img, Parameter):
    h,w,c=img.shape
    Frame=np.zeros((h,w,c),dtype=np.uint8)
    cv2.circle(Frame,(int(math.floor(w/2)),int(math.floor(h/2))),int(math.floor((h*Parameter)/float(2*100))), (255,255,255), -1)
    Frame1=cv2.cvtColor(Frame, cv2.COLOR_BGR2GRAY)
    img1 =cv2.bitwise_and(img,img,mask=Frame1)
    return img1, Frame1
w=10
h=10
fig=plt.figure(figsize=(20, 20))
columns = 5
rows = 5
for i in range(1, columns*rows +1):
    #img = np.random.randint(10, size=(h,w))
    img = cv2.imread(os.path.join(root, 'train_images', '{}.png'.format(img_list[i-1])))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    res_image = resize_image(img)
    #sub_med = subtract_median_bg_image(res_image)
    img_rad_red, mask = Redius_Reduction(res_image, Parameter)
    
    
    fig.add_subplot(rows, columns, i)
    plt.imshow(img_rad_red)
    
plt.show()

w=10
h=10
fig=plt.figure(figsize=(20, 20))
columns = 5
rows = 5
for i in range(1, columns*rows +1):
    
    img = cv2.imread(os.path.join(root, 'train_images', '{}.png'.format(img_list[i-1])))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    res_image = resize_image(img)
    
    
    
    img_rad_red, mask = Redius_Reduction(img, Parameter)
    pixel_list = img_rad_red[:,:,1][np.where(mask == 255)].tolist()
    
    print('Median:', np.median(pixel_list))
    adjusted_img = None
    
    if (np.median(pixel_list) < 70):
        print('Inside')
        adjusted_img = adjust_gamma(res_image, gamma = 1.65)
        
    elif (np.median(pixel_list) > 180):
        adjusted_img = adjust_gamma(res_image, gamma = 0.75)
        
    
    else:
        
        adjusted_img = res_image
    
    fig.add_subplot(rows, columns, i)
    plt.imshow(adjusted_img)
    
plt.show()

#-----Converting image to LAB Color model----------------------------------- 
lab= cv2.cvtColor(res_image, cv2.COLOR_BGR2LAB)
matplotlib.rc('figure', figsize=[7, 7])
plt.imshow(lab)


#-----Splitting the LAB image to different channels-------------------------
l, a, b = cv2.split(lab)
#cv2.imshow('l_channel', l)
#cv2.imshow('a_channel', a)
#cv2.imshow('b_channel', b)

#-----Applying CLAHE to L-channel-------------------------------------------
clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
cl = clahe.apply(l)
matplotlib.rc('figure', figsize=[7, 7])
plt.imshow(cl)


#-----Merge the CLAHE enhanced L-channel with the a and b channel-----------
limg = cv2.merge((cl,a,b))
matplotlib.rc('figure', figsize=[7, 7])
plt.imshow(limg)


#-----Converting image from LAB Color model to RGB model--------------------
final = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
matplotlib.rc('figure', figsize=[7, 7])
plt.imshow(final)
w=10
h=10
fig=plt.figure(figsize=(20, 20))
columns = 5
rows = 5
for i in range(1, columns*rows +1):
    #img = np.random.randint(10, size=(h,w))
    img = cv2.imread(os.path.join(root, 'train_images', '{}.png'.format(img_list[i-1])))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    res_image = resize_image(img)
    lab= cv2.cvtColor(res_image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
    cl = clahe.apply(l)
    limg = cv2.merge((cl,a,b))
    final = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
    fig.add_subplot(rows, columns, i)
    plt.imshow(final)
plt.show()
def contrast_stretching(img):        
    rr, gg, bb = cv2.split(img)    
    imgray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)    
    im = imgray    
    ih, iw = imgray.shape    
    (minVal, maxVal, minLoc, maxLoc) = cv2.minMaxLoc(imgray)    
    for i in range(ih):        
        for j in range(iw):            
            im[i, j] = 255 * ((gg[i, j] - minVal) / (maxVal - minVal))        
    limg = cv2.merge((rr, im, bb))    
    return limg
contrast_image = contrast_stretching(res_image)

matplotlib.rc('figure', figsize=[15, 15])
fig, axarr = plt.subplots(1,2)
axarr[0].imshow(res_image)
axarr[1].imshow(contrast_image)
def histogram_normalization(image):    
    hist,bins = np.histogram(image.flatten(),256,[0,256])    
    cdf = hist.cumsum()   
    # cdf_normalized = cdf * hist.max()/ cdf.max()    
    cdf_m = np.ma.masked_equal(cdf,0)    
    cdf_m = (cdf_m - cdf_m.min())*255/(cdf_m.max()-cdf_m.min())    
    cdf = np.ma.filled(cdf_m,0).astype('uint8')     
    img2 = cdf[image]    
    return img2
hist_norm_image = histogram_normalization(res_image)

matplotlib.rc('figure', figsize=[15, 15])
fig, axarr = plt.subplots(1,2)
axarr[0].imshow(res_image)
axarr[1].imshow(hist_norm_image)
def matching(source,template):    
    oldshape = source.shape    
    source1 = source.ravel()    
    template1 = template.ravel()    
    s_values, bin_idx, s_counts = np.unique(source1, return_inverse=True,return_counts=True)    
    t_values, t_counts = np.unique(template1, return_counts=True)    
    s_quantiles = np.cumsum(s_counts).astype(np.float64)    
    s_quantiles /= s_quantiles[-1]    
    t_quantiles = np.cumsum(t_counts).astype(np.float64)    
    t_quantiles /= t_quantiles[-1]    
    interp_t_values = np.interp(s_quantiles, t_quantiles, t_values)    
    interp_t_values1=interp_t_values.astype(np.uint8)    
    sub=interp_t_values-interp_t_values1    
    interp_t_values1[sub>.5]+=1    
    match_v1=interp_t_values1[bin_idx].reshape(oldshape).astype(np.uint8)    
    #match_v2=Radius_Reduction(match_v1, PARAM)    
    return match_v1
def Histo_Specification(source, template):    
    #source_v5=Ychannel_Stretch(source_v4)    
    #source_v6 = cv2.resize(source,(IMAGE_SIZE, IMAGE_SIZE)) 
    #template_v6 = cv2.resize(template,(IMAGE_SIZE, IMAGE_SIZE)) 
    source_v6 = resize_image(source)
    template_v6 = resize_image(template)
    f=[]    
    for x in range(0,3):        
        f.append(matching(source_v6[:,:,x], template_v6[:,:,x]))    
    img = cv2.merge((f[0],f[1],f[2]))     
    return img
source = cv2.imread(os.path.join(root, 'train_images', '{}.png'.format(img_list[3])))
source = cv2.cvtColor(source, cv2.COLOR_BGR2RGB)
#source = contrast_stretching(source)
#source = histogram_normalization(source)
fig=plt.figure(figsize=(5, 5))
plt.imshow(source)
template = cv2.imread(os.path.join(root, 'train_images', '{}.png'.format(img_list[13])))
template = cv2.cvtColor(template, cv2.COLOR_BGR2RGB)
fig=plt.figure(figsize=(5, 5))
plt.imshow(template)
histo_matched_image = Histo_Specification(source, template)
fig=plt.figure(figsize=(5, 5))
plt.imshow(histo_matched_image)
w=10
h=10
fig=plt.figure(figsize=(20, 20))
columns = 5
rows = 5
for i in range(1, columns*rows +1):
    #img = np.random.randint(10, size=(h,w))
    img = cv2.imread(os.path.join(root, 'train_images', '{}.png'.format(img_list[i-1])))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    #res_image = resize_image(img)
    #sub_med = subtract_median_bg_image(res_image)
    #img_rad_red=Radius_Reduction(sub_med, PARAM)
    histo_matched_image = Histo_Specification(img, template)
    fig.add_subplot(rows, columns, i)
    plt.imshow(histo_matched_image)
plt.show()
def Root_Channel_SQUR(crop):    
    blu=crop[:,:,0].astype(np.int64)    
    gre=crop[:,:,1].astype(np.int64)    
    red=crop[:,:,2].astype(np.int64)      
    lll=(((blu**2)+(gre**2)+(red**2))/float(3))**0.5    
    lll=lll.astype(np.uint8)#1st version of image    
    return lll
Root_Channel_SQUR_image = Root_Channel_SQUR(res_image)

matplotlib.rc('figure', figsize=[15, 15])
fig, axarr = plt.subplots(1,2)
axarr[0].imshow(res_image)
axarr[1].imshow(Root_Channel_SQUR_image)
SEED = 125
fig = plt.figure(figsize=(25, 16))
img_list = []
img_size = []
# display 10 images from each class
#for class_id in sorted(train_y.unique()):
for class_id in [0, 1, 2, 3, 4]:
    for i, (idx, row) in enumerate(train_df.loc[train_df['diagnosis'] == class_id].sample(5, random_state=SEED).iterrows()):
        ax = fig.add_subplot(5, 5, class_id * 5 + i + 1, xticks=[], yticks=[])
        path = os.path.join(root, 'train_images', '{}.png'.format(row['id_code']))
        image = cv2.imread(path)
        img_size.append(image.shape)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        #image = cv2.resize(image, (IMG_SIZE, IMG_SIZE))
        img_list.append(row['id_code'])
        
        print(row['id_code'])
        
        
        plt.imshow(image)
        ax.set_title('Label: %d ' % (class_id) )
def Krish(crop):    
    Input=crop[:,:,1]    
    a,b=Input.shape    
    Kernel=np.zeros((3,3,8))#windows declearations(8 windows)    
    Kernel[:,:,0]=np.array([[5,5,5],[-3,0,-3],[-3,-3,-3]])     
    Kernel[:,:,1]=np.array([[-3,5,5],[-3,0,5],[-3,-3,-3]])    
    Kernel[:,:,2]=np.array([[-3,-3,5],[-3,0,5],[-3,-3,5]])    
    Kernel[:,:,3]=np.array([[-3,-3,-3],[-3,0,5],[-3,5,5]])    
    Kernel[:,:,4]=np.array([[-3,-3,-3],[-3,0,-3],[5,5,5]])    
    Kernel[:,:,5]=np.array([[-3,-3,-3],[5,0,-3],[5,5,-3]])    
    Kernel[:,:,6]=np.array([[5,-3,-3],[5,0,-3],[5,-3,-3]])    
    Kernel[:,:,7]=np.array([[5,5,-3],[5,0,-3],[-3,-3,-3]])    
    #Kernel=(1/float(15))*Kernel    
    #Convolution output    
    dst=np.zeros((a,b,8))    
    for x in range(0,8):        
        dst[:,:,x] = cv2.filter2D(Input,-1,Kernel[:,:,x])    
    Out=np.zeros((a,b))    
    for y in range(0,a-1):        
        for z in range(0,b-1):            
            Out[y,z]=max(dst[y,z,:])    
    Out=np.uint8(Out)            
    return Out


Krish_image = Krish(res_image)

matplotlib.rc('figure', figsize=[15, 15])
fig, axarr = plt.subplots(1,2)
axarr[0].imshow(res_image)
axarr[1].imshow(Krish_image)
def illumination_correction(image, factor):       
    hls= cv2.cvtColor(image, cv2.COLOR_BGR2HLS)    
    hh,ll,ss = cv2.split(hls)    
    final_v = cv2.medianBlur(ss, factor)    
    conv = cv2.merge((hh,final_v,ll))    
    fin = cv2.cvtColor(conv, cv2.COLOR_HSV2BGR)
    return fin
illum_image = illumination_correction(res_image, factor = 31)

matplotlib.rc('figure', figsize=[15, 15])
fig, axarr = plt.subplots(1,2)
axarr[0].imshow(res_image)
axarr[1].imshow(illum_image)
from skimage.feature import hog

def HOG_Image(crop):    
    image=crop[:,:,1]    
    fd, hog_image = hog(image, orientations=8, pixels_per_cell=(4, 4),cells_per_block=(2, 2), visualise=True)    
    return hog_image
image = cv2.imread(os.path.join(root, 'train_images', '{}.png'.format(img_list[0])))
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
res_image = resize_image(image)

hog_image = HOG_Image(res_image)
#matplotlib.rc('figure', figsize=[7, 7])
#plt.imshow(hog_image)

matplotlib.rc('figure', figsize=[15, 15])
fig, axarr = plt.subplots(1,2)
axarr[0].imshow(res_image)
axarr[1].imshow(hog_image)
def Tan_inv(crop):    
    gre=crop[:,:,1].astype(np.float64)    
    red=crop[:,:,2].astype(np.float64)        
    red[red==0]=0.000001    
    m=gre/red# n=((np.arctan(m))*180)/3.14#n=((np.arctan(m))*255)/3.14    
    n=np.arctan(m)#j=n.astype(np.uint8)    
    ij=(n*255)/3.14    
    j=ij.astype(np.uint8)    
    equ = cv2.equalizeHist(j)    
    return equ

 
Tan_inv_image = Tan_inv(res_image)

matplotlib.rc('figure', figsize=[15, 15])
fig, axarr = plt.subplots(1,2)
axarr[0].imshow(res_image)
axarr[1].imshow(Tan_inv_image)
def Cos_inv(crop):    
    blu=crop[:,:,0].astype(np.float64)    
    gre=crop[:,:,1].astype(np.float64)    
    red=crop[:,:,2].astype(np.float64)     
    l=((blu**2)+(gre**2)+(red**2))**0.5    
    l=l.astype(np.float64)    
    l[l==0]=0.0000000001    
    m=blu/l    
    Max=np.max(m)    
    Min=np.min(m)    
    j=((m-float(Min)/(float(Max)-float(Min)))*2)-1    
    n=((np.arccos(j))*255)/3.14    
    nm=n.astype(np.uint8)    
    equ1 = cv2.equalizeHist(nm)    
    return equ1
Cos_inv_image = Cos_inv(res_image)

matplotlib.rc('figure', figsize=[15, 15])
fig, axarr = plt.subplots(1,2)
axarr[0].imshow(res_image)
axarr[1].imshow(Cos_inv_image)
#In progress
'''
template = cv2.imread(os.path.join(root, ''))
template1 = cv2.imread(os.path.join(root, ''))
template2 = cv2.imread(os.path.join(root, ''))
template3 = cv2.imread(os.path.join(root, ''))
template4 = cv2.imread(os.path.join(root, ''))
template5 = cv2.imread(os.path.join(root, ''))
template6 = cv2.imread(os.path.join(root, ''))
template7 = cv2.imread(os.path.join(root, ''))

def matching(source, template):    
    oldshape = source.shape    
    source1 = source.ravel()    
    template1 = np.asarray(template)   
    s_values, bin_idx, s_counts = np.unique(source1, return_inverse=True,return_counts=True)    
    t_values, t_counts = np.unique(template1, return_counts=True)    
    s_quantiles = np.cumsum(s_counts).astype(np.float64)    
    s_quantiles /= s_quantiles[-1]    
    t_quantiles = np.cumsum(t_counts).astype(np.float64)    
    t_quantiles /= t_quantiles[-1]    
    interp_t_values = np.interp(s_quantiles, t_quantiles, t_values)    
    interp_t_values1=interp_t_values.astype(np.uint8)    
    sub=interp_t_values-interp_t_values1    
    interp_t_values1[sub>.5]+=1    
    match_v1=interp_t_values1[bin_idx].reshape(oldshape).astype(np.uint8)    
    #match_v2=Radius_Reduction(match_v1, PARAM)    
    return match_v1

templates_list = [template, template1, template2, template3, template4, template5, template6, template7]

r_l = []
g_l= []
b_l = []
for t in templates_list:
    t = cv2.resize(t, (IMAGE_SIZE, IMAGE_SIZE))
    r_l.append(list(t[:,:,0].ravel()))
    g_l.append(list(t[:,:,1].ravel()))
    b_l.append(list(t[:,:,2].ravel()))

r_list = [item for sublist in r_l for item in sublist]
g_list = [item for sublist in g_l for item in sublist]
b_list = [item for sublist in b_l for item in sublist]

r_list = list(set(r_list))
g_list = list(set(g_list))
b_list = list(set(b_list))

source = cv2.imread(os.path.join(root, ''))
source = cv2.resize(source, (IMAGE_SIZE, IMAGE_SIZE))

match_r = matching(source[:,:,0], r_list)
match_g = matching(source[:,:,1], g_list)
match_b = matching(source[:,:,2], b_list)

fin = cv2.merge((match_r, match_g, match_b))
'''