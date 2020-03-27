
"""Copy Keras pre-trained model files to work directory from:
https://www.kaggle.com/gaborfodor/keras-pretrained-models

Code from: https://www.kaggle.com/classtag/extract-avito-image-features-via-keras-vgg16/notebook
"""
import os

cache_dir = os.path.expanduser(os.path.join('~', '.keras'))
if not os.path.exists(cache_dir):
    os.makedirs(cache_dir)

# Create symbolic links for trained models.
# Thanks to Lem Lordje Ko for the idea
# https://www.kaggle.com/lemonkoala/pretrained-keras-models-symlinked-not-copied
models_symlink = os.path.join(cache_dir, 'models')
if not os.path.exists(models_symlink):
    os.symlink('/kaggle/input/keras-pretrained-models/', models_symlink)

images_dir = os.path.expanduser(os.path.join('~', 'avito_images'))
if not os.path.exists(images_dir):
    os.makedirs(images_dir)
"""Extract images from Avito's advertisement image zip archive.

Code adapted from: https://www.kaggle.com/classtag/extract-avito-image-features-via-keras-vgg16/notebook
"""
import zipfile

NUM_IMAGES_TO_EXTRACT = 1000

with zipfile.ZipFile('../input/avito-demand-prediction/train_jpg.zip', 'r') as train_zip:
    files_in_zip = sorted(train_zip.namelist())
    for idx, file in enumerate(files_in_zip[:NUM_IMAGES_TO_EXTRACT]):
        if file.endswith('.jpg'):
            train_zip.extract(file, path=file.split('/')[3])

!mv *.jpg/data/competition_files/train_jpg/* ~/avito_images
!rm -rf *.jpg
import os

import numpy as np
import pandas as pd
from keras.preprocessing import image
import keras.applications.resnet50 as resnet50
import keras.applications.xception as xception
import keras.applications.inception_v3 as inception_v3
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image

sns.set_style("whitegrid")
sns.set_context("notebook", font_scale=1.5, rc={"lines.linewidth": 2.5})

#%matplotlib inline
resnet_model = resnet50.ResNet50(weights='imagenet')
inception_model = inception_v3.InceptionV3(weights='imagenet')
xception_model = xception.Xception(weights='imagenet')
def image_classify(model, pak, img, top_n=3):
    """Classify image and return top matches."""
    target_size = (224, 224)
    if img.size != target_size:
        img = img.resize(target_size)
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = pak.preprocess_input(x)
    preds = model.predict(x)
    return pak.decode_predictions(preds, top=top_n)[0]


def plot_preds(img, preds_arr):
    """Plot image and its prediction."""
    sns.set_color_codes('pastel')
    f, axarr = plt.subplots(1, len(preds_arr) + 1, figsize=(20, 5))
    axarr[0].imshow(img)
    axarr[0].axis('off')
    for i in range(len(preds_arr)):
        _, x_label, y_label = zip(*(preds_arr[i][1]))
        plt.subplot(1, len(preds_arr) + 1, i + 2)
        ax = sns.barplot(np.array(y_label), np.array(x_label))
        plt.xlim(0, 1)
        ax.set()
        plt.xlabel(preds_arr[i][0])
    plt.show()


def classify_and_plot(image_path):
    """Classify an image with different models.
    Plot it and its predicitons.
    """
    img = Image.open(image_path)
    resnet_preds = image_classify(resnet_model, resnet50, img)
    xception_preds = image_classify(xception_model, xception, img)
    inception_preds = image_classify(inception_model, inception_v3, img)
    preds_arr = [('Resnet50', resnet_preds), ('xception', xception_preds), ('Inception', inception_preds)]
    plot_preds(img, preds_arr)
image_files = [x.path for x in os.scandir(images_dir)]
classify_and_plot(image_files[10])
classify_and_plot(image_files[11])
classify_and_plot(image_files[12])
classify_and_plot(image_files[13])
classify_and_plot(image_files[14])
classify_and_plot(image_files[15])
classify_and_plot(image_files[16])
def classify_inception(image_path):
    """Classify image and return top match."""
    img = Image.open(image_path)
    target_size = (224, 224)
    if img.size != target_size:
        img = img.resize(target_size)
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = inception_v3.preprocess_input(x)
    preds = inception_model.predict(x)
    return inception_v3.decode_predictions(preds, top=1)[0][0]

def image_id_from_path(path):
    return path.split('/')[3].split('.')[0]
train = pd.read_csv('../input/avito-demand-prediction/train.csv')
train['desc_len'] = train['description'].str.len()
train['title_len'] = train['title'].str.len()
plt.figure(figsize=(10, 10))
inception_conf = [[image_id_from_path(x), classify_inception(x)[2]] for x in image_files]
confidence = pd.DataFrame(inception_conf, columns=['image', 'image_confidence'])
df = confidence.merge(train, on='image')
corr = df[['image', 'image_confidence', 'deal_probability', 'desc_len', 'title_len']].corr()
sns.heatmap(corr, annot=True)
plt.xticks(rotation=30)
plt.yticks(rotation=30)
plt.title('Correlation Between Deal Probability and Strong Model Predictors')
plt.show()