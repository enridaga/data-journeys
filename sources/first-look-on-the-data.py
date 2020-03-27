
# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import cv2
import matplotlib.pyplot as plt
import os

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.
def video_prop(reader):
    w = 0
    h = 0

    success, image = reader.read()
    h = image.shape[0]
    w = image.shape[1]
    nFrames = int(reader.get(cv2.CAP_PROP_FRAME_COUNT))
    return h, w, nFrames

def video_size_counter(path):
    video_sizes = dict()
    for dirname, _, filenames in os.walk(path):
        for filename in filenames:
            if filename.endswith('.mp4'):
                video_filename = os.path.join(dirname, filename)
                reader = cv2.VideoCapture(video_filename)
                h, w, nFrames = video_prop(reader)
                if (h, w, nFrames) in video_sizes.keys():
                    video_sizes[(h, w, nFrames)] += 1
                else:
                    video_sizes[(h, w, nFrames)] = 1
    return video_sizes
video_sizes_train = video_size_counter('/kaggle/input/deepfake-detection-challenge/train_sample_videos')
video_sizes_test = video_size_counter('/kaggle/input/deepfake-detection-challenge/test_videos')
print(video_sizes_train)
print(video_sizes_test)
sizes1 = set([k for k in video_sizes_train.keys()])
sizes2 = set([k for k in video_sizes_test.keys()])
sizes = sizes1.union(sizes2)
sizes_str = [str(s) for s in sizes]
y_pos = [3*i for i in range(len(sizes_str))]

n_accurance = []
for s in sizes:
    if s in video_sizes_train.keys():
        n_accurance.append(video_sizes_train[s])
    else:
        n_accurance.append(0)
fig = plt.figure(1)
plt.bar(y_pos, n_accurance, width=1)
plt.xticks(y_pos, sizes_str)
plt.title('Video sizes distribution over the sampled training data')
plt.show()

n_accurance = []
for s in sizes:
    if s in video_sizes_test.keys():
        n_accurance.append(video_sizes_test[s])
    else:
        n_accurance.append(0)
fig = plt.figure(2)
plt.bar(y_pos, n_accurance, width=1)
plt.xticks(y_pos, sizes_str)
plt.title('Video sizes distribution over the sampled test data')
plt.show()