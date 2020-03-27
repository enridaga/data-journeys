
# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os


# Any results you write to the current directory are saved as output.
submission = pd.read_csv("/kaggle/input/submit-dfdc/submit2.csv",names = ["filename", "label"])
submission_ori = pd.read_csv("/kaggle/input/deepfake-detection-challenge/sample_submission.csv")
submission = submission.sort_values('filename')
submission_ori['label'] = 1-submission['label'].astype(float)
submission_ori.to_csv('submission.csv', index=False)

submission_ori
