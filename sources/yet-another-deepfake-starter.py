
import numpy as np
import pandas as pd
import os
import matplotlib
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm import tqdm_notebook as tqdm
### matplotlib inline 
import cv2 as cv
from os.path import join
import argparse
import subprocess


DATA_FOLDER = '../input/deepfake-detection-challenge/'
TRAIN_SAMPLE_FOLDER = 'train_sample_videos/'
TEST_FOLDER = 'test_videos/'
DATA_PATH = os.path.join(DATA_FOLDER,TRAIN_SAMPLE_FOLDER)
os.makedirs('/kaggle/working/output', exist_ok=True)
os.makedirs('/kaggle/working/test_output', exist_ok=True)
OUTPUT_PATH = '/kaggle/working/output'
TEST_OUTPUT_PATH = '/kaggle/working/test_output/'
print(f"Train samples: {len(os.listdir(os.path.join(DATA_FOLDER, TRAIN_SAMPLE_FOLDER)))}")
print(f"Test samples: {len(os.listdir(os.path.join(DATA_FOLDER, TEST_FOLDER)))}")
SPLIT='00'
train_list = list(os.listdir(os.path.join(DATA_FOLDER, TRAIN_SAMPLE_FOLDER)))
ext_dict = []
for file in train_list:
    file_ext = file.split('.')[1]
    if (file_ext not in ext_dict):
        ext_dict.append(file_ext)
print(f"Extensions: {ext_dict}")      
for file_ext in ext_dict:
    print(f"Files with extension `{file_ext}`: {len([file for file in train_list if  file.endswith(file_ext)])}")
test_list = list(os.listdir(os.path.join(DATA_FOLDER, TEST_FOLDER)))
ext_dict = []
for file in test_list:
    file_ext = file.split('.')[1]
    if (file_ext not in ext_dict):
        ext_dict.append(file_ext)
print(f"Extensions: {ext_dict}")
for file_ext in ext_dict:
    print(f"Files with extension `{file_ext}`: {len([file for file in train_list if  file.endswith(file_ext)])}")
json_file = [file for file in train_list if  file.endswith('json')][0]
print(f"JSON file: {json_file}")
any('ccfoszqabv.mp4'  in item for item in os.listdir(DATA_PATH))
def get_meta_from_json(path):
    df = pd.read_json(os.path.join(DATA_FOLDER, path, json_file))
    df = df.T
    return df

meta_train_df = get_meta_from_json(TRAIN_SAMPLE_FOLDER)
meta_train_df.head(20)
def missing_data(data):
    total = data.isnull().sum()
    percent = (data.isnull().sum()/data.isnull().count()*100)
    tt = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
    types = []
    for col in data.columns:
        dtype = str(data[col].dtype)
        types.append(dtype)
    tt['Types'] = types
    return(np.transpose(tt))
missing_data(meta_train_df)
missing_data(meta_train_df.loc[meta_train_df.label=='REAL'])
def unique_values(data):
    total = data.count()
    tt = pd.DataFrame(total)
    tt.columns = ['Total']
    uniques = []
    for col in data.columns:
        unique = data[col].nunique()
        uniques.append(unique)
    tt['Uniques'] = uniques
    return(np.transpose(tt))
unique_values(meta_train_df)
def most_frequent_values(data):
    total = data.count()
    tt = pd.DataFrame(total)
    tt.columns = ['Total']
    items = []
    vals = []
    for col in data.columns:
        itm = data[col].value_counts().index[0]
        val = data[col].value_counts().values[0]
        items.append(itm)
        vals.append(val)
    tt['Most frequent item'] = items
    tt['Frequence'] = vals
    tt['Percent from total'] = np.round(vals / total * 100, 3)
    return(np.transpose(tt))
most_frequent_values(meta_train_df)
def plot_count(feature, title, df, size=1):
    f, ax = plt.subplots(1,1, figsize=(4*size,4))
    total = float(len(df))
    g = sns.countplot(df[feature], order = df[feature].value_counts().index[:20], palette='Set3')
    g.set_title("Number and percentage of {}".format(title))
    if(size > 2):
        plt.xticks(rotation=90, size=8)
    for p in ax.patches:
        height = p.get_height()
        ax.text(p.get_x()+p.get_width()/2.,
                height + 3,
                '{:1.2f}%'.format(100*height/total),
                ha="center") 
    plt.show()    
plot_count('split', 'split (train)', meta_train_df)
plot_count('label', 'label (train)', meta_train_df)
meta = np.array(list(meta_train_df.index))
storage = np.array([file for file in train_list if  file.endswith('mp4')])
print(f"Metadata: {meta.shape[0]}, Folder: {storage.shape[0]}")
print(f"Files in metadata and not in folder: {np.setdiff1d(meta,storage,assume_unique=False).shape[0]}")
print(f"Files in folder and not in metadata: {np.setdiff1d(storage,meta,assume_unique=False).shape[0]}")
fake_train_sample_video = list(meta_train_df.loc[meta_train_df.label=='FAKE'].sample(3).index)
fake_train_sample_video
def display_image_from_video(video_path):
    '''
    input: video_path - path for video
    process:
    1. perform a video capture from the video
    2. read the image
    3. display the image
    '''
    capture_image = cv.VideoCapture(video_path) 
    ret, frame = capture_image.read()
    fig = plt.figure(figsize=(10,10))
    ax = fig.add_subplot(111)
    frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
    ax.imshow(frame)
for video_file in fake_train_sample_video:
    display_image_from_video(os.path.join(DATA_FOLDER, TRAIN_SAMPLE_FOLDER, video_file))
real_train_sample_video = list(meta_train_df.loc[meta_train_df.label=='REAL'].sample(3).index)
real_train_sample_video
for video_file in real_train_sample_video:
    display_image_from_video(os.path.join(DATA_FOLDER, TRAIN_SAMPLE_FOLDER, video_file))
meta_train_df['original'].value_counts()[0:5]
def display_image_from_video_list(video_path_list, video_folder=TRAIN_SAMPLE_FOLDER):
    '''
    input: video_path_list - path for video
    process:
    0. for each video in the video path list
        1. perform a video capture from the video
        2. read the image
        3. display the image
    '''
    plt.figure()
    fig, ax = plt.subplots(2,3,figsize=(16,8))
    # we only show images extracted from the first 6 videos
    for i, video_file in enumerate(video_path_list[0:6]):
        video_path = os.path.join(DATA_FOLDER, video_folder,video_file)
        capture_image = cv.VideoCapture(video_path) 
        ret, frame = capture_image.read()
        frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
        ax[i//3, i%3].imshow(frame)
        ax[i//3, i%3].set_title(f"Video: {video_file}")
        ax[i//3, i%3].axis('on')
same_original_fake_train_sample_video = list(meta_train_df.loc[meta_train_df.original=='meawmsgiti.mp4'].index)
display_image_from_video_list(same_original_fake_train_sample_video)
same_original_fake_train_sample_video = list(meta_train_df.loc[meta_train_df.original=='atvmxvwyns.mp4'].index)
display_image_from_video_list(same_original_fake_train_sample_video)
same_original_fake_train_sample_video = list(meta_train_df.loc[meta_train_df.original=='qeumxirsme.mp4'].index)
display_image_from_video_list(same_original_fake_train_sample_video)
same_original_fake_train_sample_video = list(meta_train_df.loc[meta_train_df.original=='kgbkktcjxf.mp4'].index)
display_image_from_video_list(same_original_fake_train_sample_video)
test_videos = pd.DataFrame(list(os.listdir(os.path.join(DATA_FOLDER, TEST_FOLDER))), columns=['video'])
test_videos.head()
display_image_from_video(os.path.join(DATA_FOLDER, TEST_FOLDER, test_videos.iloc[0].video))
display_image_from_video_list(test_videos.sample(6).video, TEST_FOLDER)
import cv2 as cv2
#https://github.com/ondyari/FaceForensics
    
def extract_frames(data_path, output_path, method='cv2'):
    """Method to extract frames, either with ffmpeg or opencv. FFmpeg won't
    start from 0 so we would have to rename if we want to keep the filenames
    coherent."""
    os.makedirs(output_path, exist_ok=True)


    if method == 'ffmpeg':
        subprocess.check_output(
            'ffmpeg -i {} {}'.format(
                data_path, os.path.join(output_path, '%04d.png')),
            shell=True, stderr=subprocess.STDOUT)
    elif method == 'cv2':
        reader = cv2.VideoCapture(data_path)
        fps = int(reader.get(cv2.CAP_PROP_FPS))
        frame_num = 0
        while reader.isOpened():
            success, image = reader.read()
            if not success:
                break
            if frame_num%(100*fps) == 0 :    ## Take every 10 seconds of frame and export as image
                cv2.imwrite(join(output_path, '{:04d}.png'.format(frame_num)),
                            image)
            frame_num += 1
        reader.release()
    else:
        raise Exception('Wrong extract frames method: {}'.format(method))
def extract_method_videos(data_path, outpath, compression):
    """Extracts all videos of a specified method and compression in the
    FaceForensics++ file structure"""
    videos_path = data_path
    images_path = outpath
    for video in tqdm(os.listdir(videos_path)):
        image_folder = video.split('.')[0]
        extract_frames(join(videos_path, video),
                       join(images_path, image_folder))
# I commented this because I dont need full size image as output anymore 
#extract_method_videos(DATA_PATH,OUTPUT_PATH,'c0')
plt.rcParams['figure.figsize'] = [15, 10]
SAMPLE_REAL_VIDEO_PATH = os.path.join(DATA_FOLDER,TRAIN_SAMPLE_FOLDER)+'ccfoszqabv.mp4'
SAMPLE_FAKE_VIDEO_PATH = os.path.join(DATA_FOLDER,TRAIN_SAMPLE_FOLDER)+'acqfdwsrhi.mp4'
import cv2
import numpy as np
def showdenseflow(SAMPLE_VIDEO_PATH):
    cap = cv2.VideoCapture(SAMPLE_VIDEO_PATH)
    ret, frame1 = cap.read()
    prvs = cv2.cvtColor(frame1,cv2.COLOR_BGR2GRAY)
    hsv = np.zeros_like(frame1)
    hsv[...,1] = 255
    for i in range(50):
        ret, frame2 = cap.read()
        next = cv2.cvtColor(frame2,cv2.COLOR_BGR2GRAY)
        flow = cv2.calcOpticalFlowFarneback(prvs,next, None, 0.5, 3, 15, 5, 5, 1.2, 0)
        mag, ang = cv2.cartToPolar(flow[...,0], flow[...,1])
        hsv[...,0] = ang*180/np.pi/2
        hsv[...,2] = cv2.normalize(mag,None,0,255,cv2.NORM_MINMAX)
        rgb = cv2.cvtColor(hsv,cv2.COLOR_HSV2BGR)
        plt.axis("off")
        plt.imshow(rgb,interpolation='nearest', aspect='auto')
    cap.release()    
#showdenseflow(SAMPLE_REAL_VIDEO_PATH)
#showdenseflow(SAMPLE_FAKE_VIDEO_PATH)
### pip install ../input/mtcnn-package/mtcnn-0.1.0-py3-none-any.whl
import cv2
from mtcnn.mtcnn import MTCNN
detector = MTCNN()
fig, axs = plt.subplots(19, 1, figsize=(200, 200))
axs = np.array(axs)
axs = axs.reshape(-1)
i = 0
for fn in meta_train_df.index[:23]:
    label = meta_train_df.loc[fn]['label']
    orig = meta_train_df.loc[fn]['label']
    video_file = f'/kaggle/input/deepfake-detection-challenge/train_sample_videos/{fn}'
    ax = axs[i]
    cap = cv.VideoCapture(video_file)
    success, image = cap.read()
    image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
    face_locations =  detector.detect_faces(image)
    if len(face_locations) > 0:
        # Print first face
        for person in face_locations:
            bounding_box = person['box']
            keypoints = person['keypoints']
    
            cv2.rectangle(image,
                          (bounding_box[0], bounding_box[1]),
                          (bounding_box[0]+bounding_box[2], bounding_box[1] + bounding_box[3]),
                          (0,155,255),
                          2)
    
            cv2.circle(image,(keypoints['left_eye']), 2, (0,155,255), 2)
            cv2.circle(image,(keypoints['right_eye']), 2, (0,155,255), 2)
            cv2.circle(image,(keypoints['nose']), 2, (0,155,255), 2)
            cv2.circle(image,(keypoints['mouth_left']), 2, (0,155,255), 2)
            cv2.circle(image,(keypoints['mouth_right']), 2, (0,155,255), 2)
    #display resulting frame
        ax.imshow(image)
        ax.grid(False)
        ax.title.set_text(f'{fn} - {label}')
        ax.xaxis.set_visible(False)
        ax.yaxis.set_visible(False)
        i += 1
        if i>18 :
            break
plt.grid(False)
plt.show()
##Utility for cropping the faces via bounding box , cropping and padding
def crop(box,image):
    x0 = box[0]
    y0 = box[1]
    w= box[2]
    h= box[3]   
    return image[y0:y0+h , x0:x0+w, :]

def cropnpad(box,image,pad):
    x0 = box[0]
    if x0-pad < 0:
        x0 =pad
    y0 = box[1]
    if y0-pad <0 :
        y0 =pad
    w= box[2]
    h= box[3]   
    return image[y0-pad:y0+h+pad , x0-pad:x0+w+pad, :]


##00faces12frames dataset contains the data from 00 set . I downloaded locally cropped using MTCNN and uploaded here 


plt.rcParams['figure.figsize'] = [10, 5]
## It will generate 12 faces from each set of videos 
for fn in meta_train_df.index[:23]:
    label = meta_train_df.loc[fn]['label']
    orig = meta_train_df.loc[fn]['label']
    video_file = f'{DATA_FOLDER}{TRAIN_SAMPLE_FOLDER}{fn}'
    print(f"{fn.split('.')[0]} n {label}")
    count=0
    cap = cv2.VideoCapture(video_file)
    #cap.set(cv2.CAP_PROP_FRAME_COUNT, frame_seq-1)
    while cap.isOpened():      
        success, image = cap.read()
        if success :  
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            face_locations =  detector.detect_faces(image)
            if len(face_locations) > 0:
            # Print first face
                for person in face_locations:
                    i=0
                    bounding_box = person['box']
                    keypoints = person['keypoints']
                    print(bounding_box)
                    print(image.shape)
                    plt.imshow(cropnpad(bounding_box,image,0))           
                    #plt.imsave(f"{OUTPUT_PATH}{fn.split('.')[0]}-{label}-{str(i)}-{str(count)}.png",cropnpad(bounding_box,image,0),format='png')
                    plt.show()
                    i+=1
            count += 200 # i.e. at 25 fps, each video creates 12 images . I want to limit the output to 500 due to kaggle
            cap.set(1, count)                   
        else:
            cap.release()
            break
import shutil
shutil.make_archive('output.zip', 'zip', '/kaggle/working/output/')
label_list =[]
image_list =os.listdir('../input/train-images/01/')
np_image_list = np.array(os.listdir('../input/train-images/01/'))

for i in image_list:
    if 'REAL' in i:
        
        label_list.append(0)
    else:
        label_list.append(1)
        
np_label_list = np.array(label_list)        
train_df = pd.DataFrame()
train_df["ImageId"]=np_image_list
train_df["Label"]=np_label_list
train_df.head(20)
sub = pd.read_csv("/kaggle/input/deepfake-detection-challenge/sample_submission.csv")
sub.head()
### mkdir ./out
TEST_OUTPUT_PATH = './out'
### %time
for fn in sub.filename[:100]:
    video_file = f'{DATA_FOLDER}{TEST_FOLDER}{fn}'
    count=0
    cap = cv2.VideoCapture(video_file)
    print(video_file)
    #cap.set(cv2.CAP_PROP_FRAME_COUNT, frame_seq-1)
    while cap.isOpened():      
        success, image = cap.read()
        if success :  
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            face_locations =  detector.detect_faces(image)
            if len(face_locations) > 0:
            # Print first face
                for person in face_locations:
                    i=0
                    bounding_box = person['box']
                    keypoints = person['keypoints']
                    print(bounding_box)
                    print(image.shape)
                    plt.imshow(cropnpad(bounding_box,image,20))           
                    plt.imsave(f"{TEST_OUTPUT_PATH}{fn.split('.')[0]}-{label}-{str(i)}-{str(count)}.png",cropnpad(bounding_box,image,20),format='png')
                    plt.show()
                    i+=1
            count += 400 # i.e. at 25 fps, each video creates 12 images . I want to limit the output to 500 due to kaggle
            cap.set(1, count) 
            
        else:
            cap.release()
            break
            
import shutil
shutil.make_archive('test_images', 'zip', './out/')
import os
import argparse


import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import torchvision

class Meso4(nn.Module):
	"""
	Pytorch Implemention of Meso4
	Autor: Honggu Liu
	Date: July 4, 2019
	"""
	def __init__(self, num_classes=2):
		super(Meso4, self).__init__()
		self.num_classes = num_classes
		self.conv1 = nn.Conv2d(3, 8, 3, padding=1, bias=False)
		self.bn1 = nn.BatchNorm2d(8)
		self.relu = nn.ReLU(inplace=True)
		self.leakyrelu = nn.LeakyReLU(0.1)

		self.conv2 = nn.Conv2d(8, 8, 5, padding=2, bias=False)
		self.bn2 = nn.BatchNorm2d(16)
		self.conv3 = nn.Conv2d(8, 16, 5, padding=2, bias=False)
		self.conv4 = nn.Conv2d(16, 16, 5, padding=2, bias=False)
		self.maxpooling1 = nn.MaxPool2d(kernel_size=(2, 2))
		self.maxpooling2 = nn.MaxPool2d(kernel_size=(4, 4))
		#flatten: x = x.view(x.size(0), -1)
		self.dropout = nn.Dropout2d(0.5)
		self.fc1 = nn.Linear(16*8*8, 16)
		self.fc2 = nn.Linear(16, num_classes)

	def forward(self, input):
		x = self.conv1(input) #(8, 256, 256)
		x = self.relu(x)
		x = self.bn1(x)
		x = self.maxpooling1(x) #(8, 128, 128)

		x = self.conv2(x) #(8, 128, 128)
		x = self.relu(x)
		x = self.bn1(x)
		x = self.maxpooling1(x) #(8, 64, 64)

		x = self.conv3(x) #(16, 64, 64)
		x = self.relu(x)
		x = self.bn2(x)
		x = self.maxpooling1(x) #(16, 32, 32)

		x = self.conv4(x) #(16, 32, 32)
		x = self.relu(x)
		x = self.bn2(x)
		x = self.maxpooling2(x) #(16, 8, 8)

		x = x.view(x.size(0), -1) #(Batch, 16*8*8)
		x = self.dropout(x)
		x = self.fc1(x) #(Batch, 16)
		x = self.leakyrelu(x)
		x = self.dropout(x)
		x = self.fc2(x)

		return x


class MesoInception4(nn.Module):
	"""
	Pytorch Implemention of MesoInception4
	Author: Honggu Liu
	Date: July 7, 2019
	"""
	def __init__(self, num_classes=2):
		super(MesoInception4, self).__init__()
		self.num_classes = num_classes
		#InceptionLayer1
		self.Incption1_conv1 = nn.Conv2d(3, 1, 1, padding=0, bias=False)
		self.Incption1_conv2_1 = nn.Conv2d(3, 4, 1, padding=0, bias=False)
		self.Incption1_conv2_2 = nn.Conv2d(4, 4, 3, padding=1, bias=False)
		self.Incption1_conv3_1 = nn.Conv2d(3, 4, 1, padding=0, bias=False)
		self.Incption1_conv3_2 = nn.Conv2d(4, 4, 3, padding=2, dilation=2, bias=False)
		self.Incption1_conv4_1 = nn.Conv2d(3, 2, 1, padding=0, bias=False)
		self.Incption1_conv4_2 = nn.Conv2d(2, 2, 3, padding=3, dilation=3, bias=False)
		self.Incption1_bn = nn.BatchNorm2d(11)


		#InceptionLayer2
		self.Incption2_conv1 = nn.Conv2d(11, 2, 1, padding=0, bias=False)
		self.Incption2_conv2_1 = nn.Conv2d(11, 4, 1, padding=0, bias=False)
		self.Incption2_conv2_2 = nn.Conv2d(4, 4, 3, padding=1, bias=False)
		self.Incption2_conv3_1 = nn.Conv2d(11, 4, 1, padding=0, bias=False)
		self.Incption2_conv3_2 = nn.Conv2d(4, 4, 3, padding=2, dilation=2, bias=False)
		self.Incption2_conv4_1 = nn.Conv2d(11, 2, 1, padding=0, bias=False)
		self.Incption2_conv4_2 = nn.Conv2d(2, 2, 3, padding=3, dilation=3, bias=False)
		self.Incption2_bn = nn.BatchNorm2d(12)

		#Normal Layer
		self.conv1 = nn.Conv2d(12, 16, 5, padding=2, bias=False)
		self.relu = nn.ReLU(inplace=True)
		self.leakyrelu = nn.LeakyReLU(0.1)
		self.bn1 = nn.BatchNorm2d(16)
		self.maxpooling1 = nn.MaxPool2d(kernel_size=(2, 2))

		self.conv2 = nn.Conv2d(16, 16, 5, padding=2, bias=False)
		self.maxpooling2 = nn.MaxPool2d(kernel_size=(4, 4))

		self.dropout = nn.Dropout2d(0.5)
		self.fc1 = nn.Linear(16*8*8, 16)
		self.fc2 = nn.Linear(16, num_classes)


	#InceptionLayer
	def InceptionLayer1(self, input):
		x1 = self.Incption1_conv1(input)
		x2 = self.Incption1_conv2_1(input)
		x2 = self.Incption1_conv2_2(x2)
		x3 = self.Incption1_conv3_1(input)
		x3 = self.Incption1_conv3_2(x3)
		x4 = self.Incption1_conv4_1(input)
		x4 = self.Incption1_conv4_2(x4)
		y = torch.cat((x1, x2, x3, x4), 1)
		y = self.Incption1_bn(y)
		y = self.maxpooling1(y)

		return y

	def InceptionLayer2(self, input):
		x1 = self.Incption2_conv1(input)
		x2 = self.Incption2_conv2_1(input)
		x2 = self.Incption2_conv2_2(x2)
		x3 = self.Incption2_conv3_1(input)
		x3 = self.Incption2_conv3_2(x3)
		x4 = self.Incption2_conv4_1(input)
		x4 = self.Incption2_conv4_2(x4)
		y = torch.cat((x1, x2, x3, x4), 1)
		y = self.Incption2_bn(y)
		y = self.maxpooling1(y)

		return y

	def forward(self, input):
		x = self.InceptionLayer1(input) #(Batch, 11, 128, 128)
		x = self.InceptionLayer2(x) #(Batch, 12, 64, 64)

		x = self.conv1(x) #(Batch, 16, 64 ,64)
		x = self.relu(x)
		x = self.bn1(x)
		x = self.maxpooling1(x) #(Batch, 16, 32, 32)

		x = self.conv2(x) #(Batch, 16, 32, 32)
		x = self.relu(x)
		x = self.bn1(x)
		x = self.maxpooling2(x) #(Batch, 16, 8, 8)

		x = x.view(x.size(0), -1) #(Batch, 16*8*8)
		x = self.dropout(x)
		x = self.fc1(x) #(Batch, 16)
		x = self.leakyrelu(x)
		x = self.dropout(x)
		x = self.fc2(x)

		return x
from torchvision import transforms

mesonet_data_transforms = {
    'train': transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize([0.5]*3, [0.5]*3)
    ]),
    'val': transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize([0.5] * 3, [0.5] * 3)
    ]),
    'test': transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize([0.5] * 3, [0.5] * 3)
    ]),
}
import torch
import torch.nn as nn
import torchvision
from torch.utils.data import DataLoader
import torch.optim as optim
from torch.optim import lr_scheduler
import argparse
import os
import cv2
from torchvision import datasets, models, transforms

def main():
	args = parse.parse_args()
	name = args.name
	train_path = args.train_path
	val_path = args.val_path
	continue_train = args.continue_train
	epoches = args.epoches
	batch_size = args.batch_size
	model_name = args.model_name
	model_path = args.model_path
	output_path = os.path.join('./output', name)
	if not os.path.exists(output_path):
		os.mkdir(output_path)
	torch.backends.cudnn.benchmark=True

	#creat train and val dataloader
	train_dataset = torchvision.datasets.ImageFolder(train_path, transform=mesonet_data_transforms['train'])
	val_dataset = torchvision.datasets.ImageFolder(val_path, transform=mesonet_data_transforms['val'])
	train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=False, num_workers=8)
	val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=True, drop_last=False, num_workers=8)
	train_dataset_size = len(train_dataset)
	val_dataset_size = len(val_dataset)


	#Creat the model
	model = Meso4()
	if continue_train:
		model.load_state_dict(torch.load(model_path))
	model = model.cuda()
	criterion = nn.CrossEntropyLoss()
	#optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9, weight_decay=0.001)
	optimizer = optim.Adam(model.parameters(), lr=0.001, betas=(0.9, 0.999), eps=1e-08)
	scheduler = lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)

	#Train the model using multiple GPUs
	#model = nn.DataParallel(model)

	best_model_wts = model.state_dict()
	best_acc = 0.0
	iteration = 0
	for epoch in range(epoches):
		print('Epoch {}/{}'.format(epoch+1, epoches))
		print('-'*10)
		model=model.train()
		train_loss = 0.0
		train_corrects = 0.0
		val_loss = 0.0
		val_corrects = 0.0
		for (image, labels) in train_loader:
			iter_loss = 0.0
			iter_corrects = 0.0
			image = image.cuda()
			labels = labels.cuda()
			optimizer.zero_grad()
			outputs = model(image)
			_, preds = torch.max(outputs.data, 1)
			loss = criterion(outputs, labels)
			loss.backward()
			optimizer.step()
			iter_loss = loss.data.item()
			train_loss += iter_loss
			iter_corrects = torch.sum(preds == labels.data).to(torch.float32)
			train_corrects += iter_corrects
			iteration += 1
			if not (iteration % 20):
				print('iteration {} train loss: {:.4f} Acc: {:.4f}'.format(iteration, iter_loss / batch_size, iter_corrects / batch_size))
		epoch_loss = train_loss / train_dataset_size
		epoch_acc = train_corrects / train_dataset_size
		print('epoch train loss: {:.4f} Acc: {:.4f}'.format(epoch_loss, epoch_acc))

		model.eval()
		with torch.no_grad():
			for (image, labels) in val_loader:
				image = image.cuda()
				labels = labels.cuda()
				outputs = model(image)
				_, preds = torch.max(outputs.data, 1)
				loss = criterion(outputs, labels)
				val_loss += loss.data.item()
				val_corrects += torch.sum(preds == labels.data).to(torch.float32)
			epoch_loss = val_loss / val_dataset_size
			epoch_acc = val_corrects / val_dataset_size
			print('epoch val loss: {:.4f} Acc: {:.4f}'.format(epoch_loss, epoch_acc))
			if epoch_acc > best_acc:
				best_acc = epoch_acc
				best_model_wts = model.state_dict()
		scheduler.step()
		if not (epoch % 10):
		#Save the model trained with multiple gpu
		#torch.save(model.module.state_dict(), os.path.join(output_path, str(epoch) + '_' + model_name))
			torch.save(model.state_dict(), os.path.join(output_path, str(epoch) + '_' + model_name))
	print('Best val Acc: {:.4f}'.format(best_acc))
	model.load_state_dict(best_model_wts)
	#torch.save(model.module.state_dict(), os.path.join(output_path, "best.pkl"))
	torch.save(model.state_dict(), os.path.join(output_path, "best.pkl"))



if __name__ == '__main__':
	parse = argparse.ArgumentParser(
		formatter_class=argparse.ArgumentDefaultsHelpFormatter)
	parse.add_argument('--name', '-n', type=str, default='Mesonet')
	parse.add_argument('--train_path', '-tp' , type=str, default = '../input/train_images/')
	parse.add_argument('--val_path', '-vp' , type=str, default = '../input/train_images/')
	parse.add_argument('--batch_size', '-bz', type=int, default=64)
	parse.add_argument('--epoches', '-e', type=int, default='50')
	parse.add_argument('--model_name', '-mn', type=str, default='meso4.pkl')
	parse.add_argument('--continue_train', type=bool, default=False)
	parse.add_argument('--model_path', '-mp', type=str, default='./output/Mesonet/best.pkl')
	main()
submission = pd.read_csv("/kaggle/input/deepfake-detection-challenge/sample_submission.csv")
submission['label'] = 0.5
submission.to_csv('submission.csv', index=False)