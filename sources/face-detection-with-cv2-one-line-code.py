

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os
import cv2

import glob
# Any results you write to the current directory are saved as output.
import matplotlib.pyplot as plt
# Get all test videos
filenames = glob.glob('/kaggle/input/deepfake-detection-challenge/test_videos/*.mp4')
v_cap = cv2.VideoCapture('/kaggle/input/deepfake-detection-challenge/test_videos/adohdulfwb.mp4')
for j in range(1):
    success, vframe = v_cap.read()

vframe = cv2.cvtColor(vframe,cv2.COLOR_BGR2RGB)
plt.imshow(vframe)
face_cascade = cv2.CascadeClassifier('/kaggle/input/haarcascadefrontalfaces/haarcascade_frontalface_default.xml')
face_cascade.load('/kaggle/input/haarcascadefrontalfaces/haarcascade_frontalface_default.xml')
boxs = face_cascade.detectMultiScale(vframe)
for box in boxs:
    img = cv2.rectangle(vframe,(box[0],box[1]),(box[0]+box[2],box[1]+box[3]),(255,0,0),5)
plt.imshow(img)
face = vframe[boxs[0][1]:boxs[0][1]+boxs[0][3],boxs[0][0]:boxs[0][0]+boxs[0][2]]
plt.imshow(face)
