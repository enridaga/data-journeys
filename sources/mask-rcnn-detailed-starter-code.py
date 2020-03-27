
# library imports
import numpy as np 
import pandas as pd
import random as rn
import cv2 as cv 
import os
import sys
from pathlib import Path

# neural network wizardry
import tensorflow as tf

# visuals
from matplotlib import pyplot as plt
import plotly.express as px
import plotly.graph_objects as go

# for reproducibility
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
tf.set_random_seed(RANDOM_SEED)
rn.seed(RANDOM_SEED)

# paths
img_train_folder = Path('/kaggle/input/train_images/')
img_test_folder = Path('/kaggle/input/test_images/')
# imagine a 3*3 image with a diagional line across
X = np.eye(3,3, dtype=np.uint8)
Y = np.eye(3,3, dtype=np.uint8)

# we change one pixel
X[1,1] = 0
print(X)
print('')
print(Y)
def dice_coefficient(X, y):
    
    # convert the pixel/mask matrix to a one-dimensional series
    predicted = X.flatten()
    truth = y.flatten()
    
    # our masks will consist of ones and zeros
    # summing the result of their product gives us the cross section
    overlap = np.sum(predicted * truth)
    total_surface_area = np.sum(predicted + truth)
    
    # passing our calculated values to the formula
    return 2 * overlap / total_surface_area
print(f'The dice coefficient for 1 wrongly labeled pixel in a 3*3 image is: {dice_coefficient(X, Y)}')
print('(2 * 2 overlapping "1" pixels / 5 total "1" surface area)')
# a more elaborate version of kaggle.com/paulorzp/rle-functions-run-lenght-encode-decode
# note that we will transpose the incoming array outside of the function, 
# as I find this a clearer illustration

def mask_to_rle(mask):
    """
    params:  mask - numpy array
    returns: run-length encoding string (pairs of start & length of encoding)
    """
    
    # turn a n-dimensional array into a 1-dimensional series of pixels
    # for example:
    #     [[1. 1. 0.]
    #      [0. 0. 0.]   --> [1. 1. 0. 0. 0. 0. 1. 0. 0.]
    #      [1. 0. 0.]]
    flat = mask.flatten()
    
    # we find consecutive sequences by overlaying the mask
    # on a version of itself that is displaced by 1 pixel
    # for that, we add some padding before slicing
    padded = np.concatenate([[0], flat, [0]])
    
    # this returns the indeces where the sliced arrays differ
    runs = np.where(padded[1:] != padded[:-1])[0] 
    # indexes start at 0, pixel numbers start at 1
    runs += 1

    # every uneven element represents the start of a new sequence
    # every even element is where the run comes to a stop
    # subtract the former from the latter to get the length of the run
    runs[1::2] -= runs[0::2]
 
    # convert the array to a string
    return ' '.join(str(x) for x in runs)
rle_example = mask_to_rle(X)
print(f'The run-length encoding for our example would be: "{rle_example}"')
def rle_to_mask(lre, shape=(1600,256)):
    '''
    params:  rle   - run-length encoding string (pairs of start & length of encoding)
             shape - (width,height) of numpy array to return 
    
    returns: numpy array with dimensions of shape parameter
    '''    
    # the incoming string is space-delimited
    runs = np.asarray([int(run) for run in lre.split(' ')])
    
    # we do the same operation with the even and uneven elements, but this time with addition
    runs[1::2] += runs[0::2]
    # pixel numbers start at 1, indexes start at 0
    runs -= 1
    
    # extract the starting and ending indeces at even and uneven intervals, respectively
    run_starts, run_ends = runs[0::2], runs[1::2]
    
    # build the mask
    h, w = shape
    mask = np.zeros(h*w, dtype=np.uint8)
    for start, end in zip(run_starts, run_ends):
        mask[start:end] = 1
    
    # transform the numpy array from flat to the original image shape
    return mask.reshape(shape)
print(f'The mask reconstructed from the run-length encoding ("{rle_example}") \
for our example would be:\n{rle_to_mask(rle_example, shape=(3,3))}')
# reading in the training set
data = pd.read_csv('/kaggle/input/train.csv')

# isolating the file name and class
data['ImageId'], data['ClassId'] = data.ImageId_ClassId.str.split('_', n=1).str
data['ClassId'] = data['ClassId'].astype(np.uint8)

# storing a list of images without defects for later use and testing
no_defects = data[data['EncodedPixels'].isna()] \
                [['ImageId']] \
                .drop_duplicates()

# adding the columns so we can append (a sample of) the dataset if need be, later
no_defects['EncodedPixels'] = ''
no_defects['ClassId'] = np.empty((len(no_defects), 0)).tolist()
no_defects['Distinct Defect Types'] = 0
no_defects.reset_index(inplace=True)
# keep only the images with labels
squashed = data.dropna(subset=['EncodedPixels'], axis='rows', inplace=True)

# squash multiple rows per image into a list
squashed = data[['ImageId', 'EncodedPixels', 'ClassId']] \
            .groupby('ImageId', as_index=False) \
            .agg(list) \

# count the amount of class labels per image
squashed['Distinct Defect Types'] = squashed.ClassId.apply(lambda x: len(x))

# display first ten to show new structure
squashed.head(10)
print(f"""The training set now consists of {len(squashed):,} distinct images,
for a total of {squashed["Distinct Defect Types"].sum():,} labeled mask instances.

Furthermore, we have kept a backup of {len(no_defects):,} images witout defects,
in case our model starts producing a lot of false positives and needs more training
on a background class.""")
""" use a consistent color palette per label throughout the notebook """
import colorlover as cl

# see: https://plot.ly/ipython-notebooks/color-scales/
colors = cl.scales['4']['qual']['Set3']
labels = np.array(range(1,5))

# combining into a dictionary
palette = dict(zip(labels, np.array(cl.to_numeric(colors))))
# we want counts & frequency of the labels
classes = data.groupby(by='ClassId', as_index=False) \
               .agg({'ImageId':'count'}) \
               .rename(columns={'ImageId':'Count'})

classes['Frequency'] = round(classes.Count / classes.Count.sum() * 100, 2) 
classes['Frequency'] = classes['Frequency'].astype(str) + '%'

# plotly for interactive graphs
fig = go.Figure(
    
    data=go.Bar(
        orientation='h',
        x=classes.Count,
        y=classes.ClassId,
        hovertext=classes.Frequency,
        text=classes.Count,
        textposition='auto',
        marker_color=colors),
    
    layout=go.Layout(
        title='Defect Type: Count & Frequency',
        showlegend=False,
        xaxis=go.layout.XAxis(showticklabels=False),
        yaxis=go.layout.YAxis(autorange='reversed'),
        width=750, height=400
    )
)

# display
fig.show()
# we want counts of the possible combinations of labels
permutations = pd.DataFrame(data=squashed.ClassId.astype(str).value_counts())

# and their frequency
permutations['Frequency'] = round(permutations.ClassId / permutations.ClassId.sum() * 100, 2)
permutations['Frequency'] = permutations['Frequency'].astype(str) + '%'

# plotly for interactive graphs
fig = go.Figure(
    
    data=go.Bar(
        orientation='h',
        x=permutations.ClassId,
        y=permutations.index,
        hovertext=permutations.Frequency,
        text=permutations.ClassId,
        textposition='auto'),
    
    layout=go.Layout(
        title='Count of Distinct Defect Combinations in Images',
        showlegend=False,
        xaxis=go.layout.XAxis(showticklabels=False),
        yaxis=go.layout.YAxis(autorange='reversed'),
        width=750, height=500
    )
)

# display
fig.show()
def build_mask(encodings, labels):
    """ takes a pair of lists of encodings and labels, 
        and turns them into a 3d numpy array of shape (256, 1600, 4) 
    """
    
    # initialise an empty numpy array 
    mask = np.zeros((256,1600,4), dtype=np.uint8)
   
    # building the masks
    for rle, label in zip(encodings, labels):
        
        # classes are [1, 2, 3, 4], corresponding indeces are [0, 1, 2, 3]
        index = label - 1
        
        # fit the mask into the correct layer
        # note we need to transpose the matrix to account for 
        # numpy and openCV handling width and height in reverse order 
        mask[:,:,index] = rle_to_mask(rle).T
    
    return mask
def mask_to_contours(image, mask_layer, color):
    """ converts a mask to contours using OpenCV and draws it on the image
    """

    # https://docs.opencv.org/4.1.0/d4/d73/tutorial_py_contours_begin.html
    contours, hierarchy = cv.findContours(mask_layer, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    image = cv.drawContours(image, contours, -1, color, 2)
        
    return image
def visualise_mask(file_name, mask):
    """ open an image and draws clear masks, so we don't lose sight of the 
        interesting features hiding underneath 
    """
    
    # reading in the image
    image = cv.imread(f'{img_train_folder}/{file_name}')

    # going through the 4 layers in the last dimension 
    # of our mask with shape (256, 1600, 4)
    for index in range(mask.shape[-1]):
        
        # indeces are [0, 1, 2, 3], corresponding classes are [1, 2, 3, 4]
        label = index + 1
        
        # add the contours, layer per layer 
        image = mask_to_contours(image, mask[:,:,index], color=palette[label])   
        
    return image
# the images we want to see
conditions = [
    squashed['ClassId'].astype(str)=='[1]',
    squashed['ClassId'].astype(str)=='[2]',
    squashed['ClassId'].astype(str)=='[3]',
    squashed['ClassId'].astype(str)=='[4]',
    squashed['Distinct Defect Types']==2,
    squashed['Distinct Defect Types']==3
]

# max 2 due to limited population of [squashed['Distinct Defect Types']==3]
# remove that condition if you wish to increase the sample size, 
# or add replace=True to the .sample() method
sample_size = 2

# looping over the different combinations of labels 
for condition in conditions:
    
    # isolate from dataset and draw a sample
    sample = squashed[condition].sample(sample_size) 
    
    # make a subplot+
    fig, axes = plt.subplots(sample_size, 1, figsize=(16, sample_size*3))
    fig.tight_layout()
    
    # looping over sample
    for i, (index, row) in enumerate(sample.iterrows()):
        
        # current ax
        ax = axes[i,]
        
        # build the mask
        mask = build_mask(encodings=row.EncodedPixels, labels=row.ClassId)

        # fetch the image and draw the contours
        image = visualise_mask(file_name=row.ImageId, mask=mask)
        
        # display
        ax.set_title(f'{row.ImageId}: {row.ClassId}')
        ax.axis('off')
        ax.imshow(image);
WORKING_DIR = '/kaggle/working'
LOGS_DIR = os.path.join(WORKING_DIR, "logs")
MASK_RCNN_DIR = os.path.join(WORKING_DIR, 'Mask_RCNN-master')
# !git clone https://www.github.com/matterport/Mask_RCNN.git 
# results in Commit Error (too many nested subdirectories)

""" Credit to Simon Walker, whose method helped me to 
    circumvent the commit error. Check out his kernel at 
    https://www.kaggle.com/srwalker101/mask-rcnn-model
"""
!pip install git+https://github.com/rteuwens/Mask_RCNN
from mrcnn.utils import Dataset
from mrcnn.config import Config
from mrcnn.model import MaskRCNN
class SeverstalConfig(Config):

    # Give the configuration a recognizable name
    NAME = "severstal"

    # We use a GPU with 12GB memory, which can fit two images.
    # Adjust down if you use a smaller GPU.
    IMAGES_PER_GPU = 2

    # Number of classes (including background)
    NUM_CLASSES = 1 + 4  # background + steel defects

    # Number of training steps per epoch
    STEPS_PER_EPOCH = 100

    # Skip detections with < 90% confidence
    DETECTION_MIN_CONFIDENCE = 0.9
    
    # Discard inferior model weights
    SAVE_BEST_ONLY = True
    
# instantiating 
severstal_config = SeverstalConfig()
# super class can be found here:
# https://github.com/matterport/Mask_RCNN/blob/v2.1/utils.py

class SeverstalDataset(Dataset):
    
    def __init__(self, dataframe):
        
        # https://rhettinger.wordpress.com/2011/05/26/super-considered-super/
        super().__init__(self)
        
        # needs to be in the format of our squashed df, 
        # i.e. image id and list of rle plus their respective label on a single row
        self.dataframe = dataframe
        
    def load_dataset(self, subset='train'):
        """ takes:
                - pandas df containing 
                    1) file names of our images 
                       (which we will append to the directory to find our images)
                    2) a list of rle for each image 
                       (which will be fed to our build_mask() 
                       function we also used in the eda section)         
            does:
                adds images to the dataset with the utils.Dataset's add_image() metho
        """
        
        # input hygiene
        assert subset in ['train', 'test'], f'"{subset}" is not a valid value.'
        img_folder = img_train_folder if subset=='train' else img_test_folder
        
        # add our four classes
        for i in range(1,5):
            self.add_class(source='', class_id=i, class_name=f'defect_{i}')
        
        # add the image to our utils.Dataset class
        for index, row in self.dataframe.iterrows():
            file_name = row.ImageId
            file_path = f'{img_folder}/{file_name}'
            
            assert os.path.isfile(file_path), 'File doesn\'t exist.'
            self.add_image(source='', 
                           image_id=file_name, 
                           path=file_path)
    
    def load_mask(self, image_id):
        """As found in: 
            https://github.com/matterport/Mask_RCNN/blob/master/samples/coco/coco.py
        
        Load instance masks for the given image
        
        This function converts the different mask format to one format
        in the form of a bitmap [height, width, instances]
        
        Returns:
            - masks    : A bool array of shape [height, width, instance count] with
                         one mask per instance
            - class_ids: a 1D array of class IDs of the instance masks
        """
        
        # find the image in the dataframe
        row = self.dataframe.iloc[image_id]
        
        # extract function arguments
        rle = row['EncodedPixels']
        labels = row['ClassId']
        
        # create our numpy array mask
        mask = build_mask(encodings=rle, labels=labels)
        
        # we're actually doing semantic segmentation, so our second return value is a bit awkward
        # we have one layer per class, rather than per instance... so it will always just be 
        # 1, 2, 3, 4. See the section on Data Shapes for the Labels.
        return mask.astype(np.bool), np.array([1, 2, 3, 4], dtype=np.int32)
from sklearn.model_selection import train_test_split

# stratified split to maintain the same class balance in both sets
train, validate = train_test_split(squashed, test_size=0.2, random_state=RANDOM_SEED)
print(train['ClassId'].astype(str).value_counts(normalize=True))
print('')
print(validate['ClassId'].astype(str).value_counts(normalize=True))
%%time

# instantiating training set
dataset_train = SeverstalDataset(dataframe=train)
dataset_train.load_dataset()
dataset_train.prepare()

# instantiating validation set
dataset_validate = SeverstalDataset(dataframe=validate)
dataset_validate.load_dataset()
dataset_validate.prepare()
!curl -LO https://github.com/matterport/Mask_RCNN/releases/download/v2.0/mask_rcnn_coco.h5
# configuration
config = tf.ConfigProto()
config.gpu_options.allow_growth = True

# session stuff
session = tf.Session(config=config)
session.run(tf.global_variables_initializer())
session.run(tf.local_variables_initializer())

# initialiazing model
model = MaskRCNN(mode='training', config=severstal_config, model_dir='modeldir')

# we will retrain starting with the coco weights
model.load_weights('mask_rcnn_coco.h5', 
                   by_name=True, 
                   exclude=['mrcnn_bbox_fc',
                            'mrcnn_class_logits', 
                            'mrcnn_mask',
                            'mrcnn_bbox'])
%%time 

# ignore UserWarnongs
import warnings
warnings.filterwarnings('ignore', category=UserWarning)

# training at last
model.train(dataset_train,
            dataset_validate,
            epochs=1,
            layers='heads',
            learning_rate=severstal_config.LEARNING_RATE)

history = model.keras_model.history.history