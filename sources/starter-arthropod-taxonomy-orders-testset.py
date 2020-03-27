
import pandas as pd
import numpy as np
import json
import matplotlib.pyplot as plt
#%matplotlib inline

# Check the revision log
with open('/kaggle/input/arthropod-taxonomy-orders-object-detection-testset/ArTaxOr_TestSet/revision history.txt', 'r') as f:
    print(f.read())
pfile='/kaggle/input/arthropod-taxonomy-orders-object-detection-testset/ArTaxOr_TestSet/positives/annotations/ArTaxOr_TestSet.vott'
df=pd.DataFrame()
with open(pfile) as file:
    pdata=json.load(file)
    df=df.append(pd.DataFrame(list(pdata['assets'].values())), ignore_index=True)
df['path']=df['path'].str.replace('file:F:/','')
df.sample(5)
tags=pd.DataFrame(list(pdata['tags']))
pattern=r'[A-Z]'
labels=tags[tags.name.str.match(pattern)]
labels
import seaborn as sns

ps=np.zeros(len(df))
for i in range(len(df)):
    ps[i]=df['size'][i]['width'] * df['size'][i]['height']/1e6
sns.distplot(ps, bins=21,kde=False).set_title('Image resolution in Mpix (total {})'.format(len(df)));
import os

ARTAXOR_PATH='/kaggle/input/arthropod-taxonomy-orders-object-detection-testset/'

anno=pd.DataFrame(columns=['label', 'label_idx', 'xres', 'yres', 'height', 'width', 'left', 'top', 
                           'right', 'bottom', 'area', 'xcenter', 'ycenter', 'blurred', 
                           'occluded', 'truncated', 'file', 'id'])
for i in range(len(df)):
    p=df['path'][i].split('/')
    p='/'.join(p[:2])
    afile=ARTAXOR_PATH+p+'/annotations/'+df['id'][i]+'-asset.json'
    if os.path.isfile(afile):
        with open(afile) as file:
            adata=json.load(file)
        xres,yres=adata['asset']['size']['width'],adata['asset']['size']['height'] 
        for j in range(len(adata['regions'])):
            h=adata['regions'][j]['boundingBox']['height']/yres
            w=adata['regions'][j]['boundingBox']['width']/xres
            tags=adata['regions'][j]['tags']
            anno=anno.append({'label': tags[0],
                              'label_idx': labels[labels.name==tags[0]].index[0],
                              'xres': xres,
                              'yres': yres,
                              'height': h,
                              'width': w,                              
                              'left': adata['regions'][j]['boundingBox']['left']/xres,
                              'top': adata['regions'][j]['boundingBox']['top']/yres,
                              'right': adata['regions'][j]['boundingBox']['left']/xres+w,
                              'bottom': adata['regions'][j]['boundingBox']['top']/yres+h, 
                              'area': h*w,
                              'xcenter': adata['regions'][j]['boundingBox']['left']/xres+0.5*w,
                              'ycenter': adata['regions'][j]['boundingBox']['top']/yres+0.5*h,
                              'blurred': int(any(ele == '_blurred' for ele in tags)),
                              'occluded': int(any(ele == '_occluded' for ele in tags)),
                              'truncated': int(any(ele == '_truncated' for ele in tags)),
                              'file': adata['asset']['path'].replace('file:F:/',''),
                              'id': adata['asset']['id'],}, ignore_index=True)
anno.head()
graph=sns.countplot(data=anno, x='label')
graph.set_xticklabels(graph.get_xticklabels(),rotation=90)
for p in graph.patches:
    height = p.get_height()
    graph.text(p.get_x()+p.get_width()/2., height + 0.1,height ,ha="center")
labels.to_pickle('./testset_labels.pkl')
df.to_pickle('./testset_filelist.pkl')
anno.to_pickle('./testset_objects.pkl')
!ls -al *.pkl
import hashlib
from io import BytesIO
from PIL import Image, ImageFont, ImageDraw
import tensorflow as tf

# Fetch attribution string from image EXIF data
def get_attribution(file):
    with Image.open(file) as img:
        exif_data = img._getexif()
    s='Photo: unknown'
    if exif_data is not None:
        if 37510 in exif_data:
            if len(exif_data[37510]) > 0:
                s = exif_data[37510][8:].decode('ascii')
        if 315 in exif_data:
            if len(exif_data[315]) > 0:
                s = 'Photo: ' + exif_data[315]
    return s

def create_tf_example(imagedf, longest_edge=1024):
    fname = ARTAXOR_PATH+imagedf.file.iloc[0]
    filename=fname.split('/')[-1] # exclude path
    by = get_attribution(fname)
    img = Image.open(fname, "r")
    # resize image if larger that longest edge while keeping aspect ratio
    if max(img.size) > longest_edge:
        img.thumbnail((longest_edge, longest_edge), Image.ANTIALIAS)
    height = img.size[1] # Image height
    width = img.size[0] # Image width
    buf= BytesIO()
    img.save(buf, format= 'JPEG') # encode to jpeg in memory
    encoded_image_data= buf.getvalue()
    image_format = b'jpeg'
    source_id = filename.split('.')[0]
    license = 'CC BY-NC-SA 4.0'
    # A hash of the image is used in some frameworks
    key = hashlib.sha256(encoded_image_data).hexdigest()   
    # object bounding boxes 
    xmins = imagedf.left.values # List of normalized left x coordinates in bounding box (1 per box)
    xmaxs = imagedf.right.values # List of normalized right x coordinates in bounding box
    ymins = imagedf.top.values # List of normalized top y coordinates in bounding box (1 per box)
    ymaxs = imagedf.bottom.values # List of normalized bottom y coordinates in bounding box
    # List of string class name & id of bounding box (1 per box)
    object_cnt = len(imagedf)
    classes_text = []
    classes = []
    for i in range(object_cnt):
        classes_text.append(imagedf.label.iloc[i].encode())
        classes.append(1+imagedf.label_idx.iloc[i])
    # unused features from Open Image 
    depiction = np.zeros(object_cnt, dtype=int)
    group_of = np.zeros(object_cnt, dtype=int)
    occluded = imagedf.occluded.values #also Pascal VOC
    truncated = imagedf.truncated.values # also Pascal VOC
    # Pascal VOC
    view_text = []
    for i in range(object_cnt):
        view_text.append('frontal'.encode())
    difficult = np.zeros(object_cnt, dtype=int)

    tf_record = tf.train.Example(features=tf.train.Features(feature={
        'image/height': tf.train.Feature(int64_list=tf.train.Int64List(value=[height])),
        'image/width': tf.train.Feature(int64_list=tf.train.Int64List(value=[width])),
        'image/filename': tf.train.Feature(bytes_list=tf.train.BytesList(value=[filename.encode()])),
        'image/source_id': tf.train.Feature(bytes_list=tf.train.BytesList(value=[source_id.encode()])),
        'image/license': tf.train.Feature(bytes_list=tf.train.BytesList(value=[license.encode()])),
        'image/by': tf.train.Feature(bytes_list=tf.train.BytesList(value=[by.encode()])),
        'image/encoded': tf.train.Feature(bytes_list=tf.train.BytesList(value=[encoded_image_data])),
        'image/key/sha256': tf.train.Feature(bytes_list=tf.train.BytesList(value=[key.encode()])),
        'image/format': tf.train.Feature(bytes_list=tf.train.BytesList(value=[image_format])),
        'image/object/bbox/xmin': tf.train.Feature(float_list=tf.train.FloatList(value=xmins)),
        'image/object/bbox/xmax': tf.train.Feature(float_list=tf.train.FloatList(value=xmaxs)),
        'image/object/bbox/ymin': tf.train.Feature(float_list=tf.train.FloatList(value=ymins)),
        'image/object/bbox/ymax': tf.train.Feature(float_list=tf.train.FloatList(value=ymaxs)),
        'image/object/class/text': tf.train.Feature(bytes_list=tf.train.BytesList(value=classes_text)),
        'image/object/class/label': tf.train.Feature(int64_list=tf.train.Int64List(value=classes)),
        'image/object/depiction': tf.train.Feature(int64_list=tf.train.Int64List(value=depiction)),
        'image/object/group_of': tf.train.Feature(int64_list=tf.train.Int64List(value=group_of)),
        'image/object/occluded': tf.train.Feature(int64_list=tf.train.Int64List(value=occluded)),
        'image/object/truncated': tf.train.Feature(int64_list=tf.train.Int64List(value=truncated)),
        'image/object/difficult': tf.train.Feature(int64_list=tf.train.Int64List(value=difficult)),
        'image/object/view': tf.train.Feature(bytes_list=tf.train.BytesList(value=view_text))
    }))
    return tf_record
with tf.io.TFRecordWriter('ArTaxOr_TestSet.tfrecord') as writer:
    for i in range(len(df)):
        imagedf=anno[anno.file == df.path[i]]
        tfr=create_tf_example(imagedf)
        writer.write(tfr.SerializeToString())
!ls -al *.tfrecord
