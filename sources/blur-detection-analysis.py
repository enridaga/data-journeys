
# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory
import cv2
import os
import matplotlib.pyplot as plt
#%matplotlib inline
DATA_FILE = [os.path.join(dirname, filename) for dirname, _, filenames in os.walk('/kaggle/input') for filename in filenames]
_defocus = [image_path for image_path in DATA_FILE if "dataset/defocused_blurred/" in image_path][:5]
_motionblur= [image_path for image_path in DATA_FILE if "dataset/motion_blurred/" in image_path][:5]
_sharp = [image_path for image_path in DATA_FILE if "dataset/sharp/" in image_path][:5]
for image in _defocus:
    _array = cv2.imread(image)
    gray = cv2.cvtColor(_array, cv2.COLOR_BGR2GRAY)
    score_gray = cv2.Laplacian(gray, cv2.CV_64F).var()
    score = cv2.Laplacian(_array, cv2.CV_64F).var()
    print(score, score_gray)
    plt.imshow(_array)
    plt.show()
for image in _motionblur:
    _array = cv2.imread(image)
    gray = cv2.cvtColor(_array, cv2.COLOR_BGR2GRAY)
    score_gray = cv2.Laplacian(gray, cv2.CV_64F).var()
    score = cv2.Laplacian(_array, cv2.CV_64F).var()
    print(score, score_gray)
    plt.imshow(_array)
    plt.show()
for image in _sharp:
    _array = cv2.imread(image)
    gray = cv2.cvtColor(_array, cv2.COLOR_BGR2GRAY)
    score_gray = cv2.Laplacian(gray, cv2.CV_64F).var()
    score = cv2.Laplacian(_array, cv2.CV_64F).var()
    print(score, score_gray)
    plt.imshow(_array)
    plt.show()