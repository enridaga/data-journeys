
#%matplotlib inline
import matplotlib.pyplot as plt
import random
import os
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import cv2
import os
print(os.listdir("../input"))
train = pd.read_csv('../input/train.csv')
train = train[ train['EncodedPixels'].notnull() ]
print( train.shape )
train.head()
train.tail()
def rle2mask(rle, imgshape):
    width = imgshape[0]
    height= imgshape[1]
    
    mask= np.zeros( width*height ).astype(np.uint8)
    
    array = np.asarray([int(x) for x in rle.split()])
    starts = array[0::2]
    lengths = array[1::2]

    current_position = 0
    for index, start in enumerate(starts):
        mask[int(start):int(start+lengths[index])] = 1
        current_position += lengths[index]
        
    return np.flipud( np.rot90( mask.reshape(height,width), k=1 ) )

fig=plt.figure(figsize=(20,100))
columns = 2
rows = 50
for i in range(1, 100+1):
    fig.add_subplot(rows, columns, i)
    
    fn = train['ImageId_ClassId'].iloc[i].split('_')[0]
    img = cv2.imread( '../input/train_images/'+fn )
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    mask = rle2mask( train['EncodedPixels'].iloc[i], img.shape  )
    img[mask==1,0] = 255
    
    plt.imshow(img)
plt.show()