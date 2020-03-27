
class KernelSettings:
    
    def __init__(self, fit_baseline=False,
                 fit_improved_baseline=True,
                 fit_improved_higher_batchsize=False,
                 fit_improved_without_dropout=False):
        self.fit_baseline = fit_baseline
        self.fit_improved_baseline = fit_improved_baseline
        self.fit_improved_higher_batchsize = fit_improved_higher_batchsize
        self.fit_improved_without_dropout = fit_improved_without_dropout
kernelsettings = KernelSettings(fit_baseline=False,
                                fit_improved_baseline=False,
                                fit_improved_higher_batchsize=False,
                                fit_improved_without_dropout=False)
use_dropout=True
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns
import matplotlib.pyplot as plt
#%matplotlib inline
from PIL import Image
from scipy.misc import imread

import tensorflow as tf
sns.set()

import os
print(os.listdir("../input"))

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

# Any results you write to the current directory are saved as output.
train_labels = pd.read_csv("../input/human-protein-atlas-image-classification/train.csv")
train_labels.head()
train_labels.shape[0]
test_path = "../input/human-protein-atlas-image-classification/test/"
submission = pd.read_csv("../input/human-protein-atlas-image-classification/sample_submission.csv")
submission.head()
test_names = submission.Id.values
print(len(test_names))
print(test_names[0])
label_names = {
    0:  "Nucleoplasm",  
    1:  "Nuclear membrane",   
    2:  "Nucleoli",   
    3:  "Nucleoli fibrillar center",   
    4:  "Nuclear speckles",
    5:  "Nuclear bodies",   
    6:  "Endoplasmic reticulum",   
    7:  "Golgi apparatus",   
    8:  "Peroxisomes",   
    9:  "Endosomes",   
    10:  "Lysosomes",   
    11:  "Intermediate filaments",   
    12:  "Actin filaments",   
    13:  "Focal adhesion sites",   
    14:  "Microtubules",   
    15:  "Microtubule ends",   
    16:  "Cytokinetic bridge",   
    17:  "Mitotic spindle",   
    18:  "Microtubule organizing center",   
    19:  "Centrosome",   
    20:  "Lipid droplets",   
    21:  "Plasma membrane",   
    22:  "Cell junctions",   
    23:  "Mitochondria",   
    24:  "Aggresome",   
    25:  "Cytosol",   
    26:  "Cytoplasmic bodies",   
    27:  "Rods & rings"
}

reverse_train_labels = dict((v,k) for k,v in label_names.items())

def fill_targets(row):
    row.Target = np.array(row.Target.split(" ")).astype(np.int)
    for num in row.Target:
        name = label_names[int(num)]
        row.loc[name] = 1
    return row
for key in label_names.keys():
    train_labels[label_names[key]] = 0
train_labels = train_labels.apply(fill_targets, axis=1)
train_labels.head()
test_labels = pd.DataFrame(data=test_names, columns=["Id"])
for col in train_labels.columns.values:
    if col != "Id":
        test_labels[col] = 0
test_labels.head(1)
target_counts = train_labels.drop(["Id", "Target"],axis=1).sum(axis=0).sort_values(ascending=False)
plt.figure(figsize=(15,15))
sns.barplot(y=target_counts.index.values, x=target_counts.values, order=target_counts.index)
train_labels["number_of_targets"] = train_labels.drop(["Id", "Target"],axis=1).sum(axis=1)
count_perc = np.round(100 * train_labels["number_of_targets"].value_counts() / train_labels.shape[0], 2)
plt.figure(figsize=(20,5))
sns.barplot(x=count_perc.index.values, y=count_perc.values, palette="Reds")
plt.xlabel("Number of targets per image")
plt.ylabel("% of train data")
plt.figure(figsize=(15,15))
sns.heatmap(train_labels[train_labels.number_of_targets>1].drop(
    ["Id", "Target", "number_of_targets"],axis=1
).corr(), cmap="RdYlBu", vmin=-1, vmax=1)
def find_counts(special_target, labels):
    counts = labels[labels[special_target] == 1].drop(
        ["Id", "Target", "number_of_targets"],axis=1
    ).sum(axis=0)
    counts = counts[counts > 0]
    counts = counts.sort_values()
    return counts
lyso_endo_counts = find_counts("Lysosomes", train_labels)

plt.figure(figsize=(10,3))
sns.barplot(x=lyso_endo_counts.index.values, y=lyso_endo_counts.values, palette="Blues")
plt.ylabel("Counts in train data")
rod_rings_counts = find_counts("Rods & rings", train_labels)
plt.figure(figsize=(15,3))
sns.barplot(x=rod_rings_counts.index.values, y=rod_rings_counts.values, palette="Greens")
plt.ylabel("Counts in train data")
peroxi_counts = find_counts("Peroxisomes", train_labels)

plt.figure(figsize=(15,3))
sns.barplot(x=peroxi_counts.index.values, y=peroxi_counts.values, palette="Reds")
plt.ylabel("Counts in train data")
tubeends_counts = find_counts("Microtubule ends", train_labels)

plt.figure(figsize=(15,3))
sns.barplot(x=tubeends_counts.index.values, y=tubeends_counts.values, palette="Purples")
plt.ylabel("Counts in train data")
nuclear_speckles_counts = find_counts("Nuclear speckles", train_labels)

plt.figure(figsize=(15,3))
sns.barplot(x=nuclear_speckles_counts.index.values, y=nuclear_speckles_counts.values, palette="Oranges")
plt.xticks(rotation="70")
plt.ylabel("Counts in train data");
from os import listdir

files = listdir("../input/human-protein-atlas-image-classification/train")
for n in range(10):
    print(files[n])
len(files) / 4 == train_labels.shape[0]
train_path = "../input/human-protein-atlas-image-classification/train/"
def load_image(basepath, image_id):
    images = np.zeros(shape=(4,512,512))
    images[0,:,:] = imread(basepath + image_id + "_green" + ".png")
    images[1,:,:] = imread(basepath + image_id + "_red" + ".png")
    images[2,:,:] = imread(basepath + image_id + "_blue" + ".png")
    images[3,:,:] = imread(basepath + image_id + "_yellow" + ".png")
    return images

def make_image_row(image, subax, title):
    subax[0].imshow(image[0], cmap="Greens")
    subax[1].imshow(image[1], cmap="Reds")
    subax[1].set_title("stained microtubules")
    subax[2].imshow(image[2], cmap="Blues")
    subax[2].set_title("stained nucleus")
    subax[3].imshow(image[3], cmap="Oranges")
    subax[3].set_title("stained endoplasmatic reticulum")
    subax[0].set_title(title)
    return subax

def make_title(file_id):
    file_targets = train_labels.loc[train_labels.Id==file_id, "Target"].values[0]
    title = " - "
    for n in file_targets:
        title += label_names[n] + " - "
    return title
class TargetGroupIterator:
    
    def __init__(self, target_names, batch_size, basepath):
        self.target_names = target_names
        self.target_list = [reverse_train_labels[key] for key in target_names]
        self.batch_shape = (batch_size, 4, 512, 512)
        self.basepath = basepath
    
    def find_matching_data_entries(self):
        train_labels["check_col"] = train_labels.Target.apply(
            lambda l: self.check_subset(l)
        )
        self.images_identifier = train_labels[train_labels.check_col==1].Id.values
        train_labels.drop("check_col", axis=1, inplace=True)
    
    def check_subset(self, targets):
        return np.where(set(targets).issubset(set(self.target_list)), 1, 0)
    
    def get_loader(self):
        filenames = []
        idx = 0
        images = np.zeros(self.batch_shape)
        for image_id in self.images_identifier:
            images[idx,:,:,:] = load_image(self.basepath, image_id)
            filenames.append(image_id)
            idx += 1
            if idx == self.batch_shape[0]:
                yield filenames, images
                filenames = []
                images = np.zeros(self.batch_shape)
                idx = 0
        if idx > 0:
            yield filenames, images
            
your_choice = ["Lysosomes", "Endosomes"]
your_batch_size = 20
imageloader = TargetGroupIterator(your_choice, your_batch_size, train_path)
imageloader.find_matching_data_entries()
iterator = imageloader.get_loader()
file_ids, images = next(iterator)

fig, ax = plt.subplots(len(file_ids),4,figsize=(20,5*len(file_ids)))
if ax.shape == (4,):
    ax = ax.reshape(1,-1)
for n in range(len(file_ids)):
    make_image_row(images[n], ax[n], make_title(file_ids[n]))
train_files = listdir("../input/human-protein-atlas-image-classification/train")
test_files = listdir("../input/human-protein-atlas-image-classification/test")
percentage = np.round(len(test_files) / len(train_files) * 100)

print("The test set size turns out to be {} % compared to the train set.".format(percentage))
from sklearn.model_selection import RepeatedKFold

splitter = RepeatedKFold(n_splits=3, n_repeats=1, random_state=0)
partitions = []

for train_idx, test_idx in splitter.split(train_labels.index.values):
    partition = {}
    partition["train"] = train_labels.Id.values[train_idx]
    partition["validation"] = train_labels.Id.values[test_idx]
    partitions.append(partition)
    print("TRAIN:", train_idx, "TEST:", test_idx)
    print("TRAIN:", len(train_idx), "TEST:", len(test_idx))
partitions[0]["train"][0:5]
class ModelParameter:
    
    def __init__(self, basepath,
                 num_classes=28,
                 image_rows=512,
                 image_cols=512,
                 batch_size=200,
                 n_channels=1,
                 row_scale_factor=4,
                 col_scale_factor=4,
                 shuffle=False,
                 n_epochs=1):
        self.basepath = basepath
        self.num_classes = num_classes
        self.image_rows = image_rows
        self.image_cols = image_cols
        self.batch_size = batch_size
        self.n_channels = n_channels
        self.shuffle = shuffle
        self.row_scale_factor = row_scale_factor
        self.col_scale_factor = col_scale_factor
        self.scaled_row_dim = np.int(self.image_rows / self.row_scale_factor)
        self.scaled_col_dim = np.int(self.image_cols / self.col_scale_factor)
        self.n_epochs = n_epochs
parameter = ModelParameter(train_path)
from skimage.transform import resize

class ImagePreprocessor:
    
    def __init__(self, modelparameter):
        self.parameter = modelparameter
        self.basepath = self.parameter.basepath
        self.scaled_row_dim = self.parameter.scaled_row_dim
        self.scaled_col_dim = self.parameter.scaled_col_dim
        self.n_channels = self.parameter.n_channels
    
    def preprocess(self, image):
        image = self.resize(image)
        image = self.reshape(image)
        image = self.normalize(image)
        return image
    
    def resize(self, image):
        image = resize(image, (self.scaled_row_dim, self.scaled_col_dim))
        return image
    
    def reshape(self, image):
        image = np.reshape(image, (image.shape[0], image.shape[1], self.n_channels))
        return image
    
    def normalize(self, image):
        image /= 255 
        return image
    
    def load_image(self, image_id):
        image = np.zeros(shape=(512,512,4))
        image[:,:,0] = imread(self.basepath + image_id + "_green" + ".png")
        image[:,:,1] = imread(self.basepath + image_id + "_blue" + ".png")
        image[:,:,2] = imread(self.basepath + image_id + "_red" + ".png")
        image[:,:,3] = imread(self.basepath + image_id + "_yellow" + ".png")
        return image[:,:,0:self.parameter.n_channels]
        
preprocessor = ImagePreprocessor(parameter)
example = images[0,0]
preprocessed = preprocessor.preprocess(example)
print(example.shape)
print(preprocessed.shape)

fig, ax = plt.subplots(1,2,figsize=(20,10))
ax[0].imshow(example, cmap="Greens")
ax[1].imshow(preprocessed.reshape(parameter.scaled_row_dim,parameter.scaled_col_dim), cmap="Greens")
import keras

class DataGenerator(keras.utils.Sequence):
    
    def __init__(self, list_IDs, labels, modelparameter, imagepreprocessor):
        self.current_epoch = 0
        self.params = modelparameter
        self.labels = labels
        self.list_IDs = list_IDs
        self.dim = (self.params.scaled_row_dim, self.params.scaled_col_dim)
        self.batch_size = self.params.batch_size
        self.n_channels = self.params.n_channels
        self.num_classes = self.params.num_classes
        self.shuffle = self.params.shuffle
        self.preprocessor = imagepreprocessor
        self.on_epoch_end()
    
    def on_epoch_end(self):
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle == True:
            np.random.shuffle(self.indexes, random_state=self.current_epoch)
            self.current_epoch += 1
    
    def get_targets_per_image(self, identifier):
        return self.labels.loc[self.labels.Id==identifier].drop(
                ["Id", "Target", "number_of_targets"], axis=1).values
            
    def __data_generation(self, list_IDs_temp):
        'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)
        # Initialization
        X = np.empty((self.batch_size, *self.dim, self.n_channels))
        y = np.empty((self.batch_size, self.num_classes), dtype=int)
        # Generate data
        for i, identifier in enumerate(list_IDs_temp):
            # Store sample
            image = self.preprocessor.load_image(identifier)
            image = self.preprocessor.preprocess(image)
            X[i] = image
            # Store class
            y[i] = self.get_targets_per_image(identifier)
        return X, y
    
    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.list_IDs) / self.batch_size))
    
    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]
        # Find list of IDs
        list_IDs_temp = [self.list_IDs[k] for k in indexes]
        # Generate data
        X, y = self.__data_generation(list_IDs_temp)
        return X, y
class PredictGenerator:
    
    def __init__(self, predict_Ids, imagepreprocessor, predict_path):
        self.preprocessor = imagepreprocessor
        self.preprocessor.basepath = predict_path
        self.identifiers = predict_Ids
    
    def predict(self, model):
        y = np.empty(shape=(len(self.identifiers), self.preprocessor.parameter.num_classes))
        for n in range(len(self.identifiers)):
            image = self.preprocessor.load_image(self.identifiers[n])
            image = self.preprocessor.preprocess(image)
            image = image.reshape((1, *image.shape))
            y[n] = model.predict(image)
        return y
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.losses import binary_crossentropy
from keras.optimizers import Adadelta
from keras.initializers import VarianceScaling


class BaseLineModel:
    
    def __init__(self, modelparameter):
        self.params = modelparameter
        self.num_classes = self.params.num_classes
        self.img_rows = self.params.scaled_row_dim
        self.img_cols = self.params.scaled_col_dim
        self.n_channels = self.params.n_channels
        self.input_shape = (self.img_rows, self.img_cols, self.n_channels)
        self.my_metrics = ['accuracy']
    
    def build_model(self):
        self.model = Sequential()
        self.model.add(Conv2D(16, kernel_size=(3, 3), activation='relu', input_shape=self.input_shape,
                             kernel_initializer=VarianceScaling(seed=0)))
        self.model.add(Conv2D(32, (3, 3), activation='relu',
                             kernel_initializer=VarianceScaling(seed=0)))
        self.model.add(MaxPooling2D(pool_size=(2, 2)))
        self.model.add(Dropout(0.25))
        self.model.add(Flatten())
        self.model.add(Dense(64, activation='relu',
                            kernel_initializer=VarianceScaling(seed=0),))
        self.model.add(Dropout(0.5))
        self.model.add(Dense(self.num_classes, activation='sigmoid'))
    
    def compile_model(self):
        self.model.compile(loss=keras.losses.binary_crossentropy,
              optimizer=keras.optimizers.Adadelta(),
              metrics=self.my_metrics)
    
    def set_generators(self, train_generator, validation_generator):
        self.training_generator = train_generator
        self.validation_generator = validation_generator
    
    def learn(self):
        return self.model.fit_generator(generator=self.training_generator,
                    validation_data=self.validation_generator,
                    epochs=self.params.n_epochs, 
                    use_multiprocessing=True,
                    workers=8)
    
    def score(self):
        return self.model.evaluate_generator(generator=self.validation_generator,
                                      use_multiprocessing=True, 
                                      workers=8)
    
    def predict(self, predict_generator):
        y = predict_generator.predict(self.model)
        return y
    
    def save(self, modeloutputpath):
        self.model.save(modeloutputpath)
    
    def load(self, modelinputpath):
        self.model = load_model(modelinputpath)
# Datasets
partition = partitions[0]
labels = train_labels

print("Number of samples in train: {}".format(len(partition["train"])))
print("Number of samples in validation: {}".format(len(partition["validation"])))
training_generator = DataGenerator(partition['train'], labels, parameter, preprocessor)
validation_generator = DataGenerator(partition['validation'], labels, parameter, preprocessor)
predict_generator = PredictGenerator(partition['validation'], preprocessor, train_path)
test_preprocessor = ImagePreprocessor(parameter)
submission_predict_generator = PredictGenerator(test_names, test_preprocessor, test_path)
# Run computation and store results as csv
target_names = train_labels.drop(["Target", "number_of_targets", "Id"], axis=1).columns

if kernelsettings.fit_baseline == True:
    model = BaseLineModel(parameter)
    model.build_model()
    model.compile_model()
    model.set_generators(training_generator, validation_generator)
    history = model.learn()
    
    proba_predictions = model.predict(predict_generator)
    baseline_proba_predictions = pd.DataFrame(index = partition['validation'],
                                              data=proba_predictions,
                                              columns=target_names)
    baseline_proba_predictions.to_csv("baseline_predictions.csv")
    baseline_losses = pd.DataFrame(history.history["loss"], columns=["train_loss"])
    baseline_losses["val_loss"] = history.history["val_loss"]
    baseline_losses.to_csv("baseline_losses.csv")
    
    
    submission_proba_predictions = model.predict(submission_predict_generator)
    baseline_labels = test_labels.copy()
    baseline_labels.loc[:, test_labels.drop(["Id", "Target"], axis=1).columns.values] = submission_proba_predictions
    baseline_labels.to_csv("baseline_submission_proba.csv")
# If you already have done a baseline fit once, 
# you can load predictions as csv and further fitting is not neccessary:
else:
    baseline_proba_predictions = pd.read_csv("../input/protein-atlas-eab-predictions/baseline_predictions.csv", index_col=0)
    baseline_losses = pd.read_csv("../input/protein-atlas-eab-predictions/baseline_losses.csv", index_col=0)
    baseline_labels = pd.read_csv("../input/protein-atlas-eab-predictions/baseline_submission_proba.csv", index_col=0)
validation_labels = train_labels.loc[train_labels.Id.isin(partition["validation"])].copy()
validation_labels.shape
baseline_proba_predictions.shape
from sklearn.metrics import accuracy_score as accuracy

y_true = validation_labels.drop(["Id", "Target", "number_of_targets"], axis=1).values
y_pred = np.where(baseline_proba_predictions.values > 0.5, 1, 0)

accuracy(y_true.flatten(), y_pred.flatten())
y_pred[0]
y_true[0]
proba_predictions = baseline_proba_predictions.values
hot_values = validation_labels.drop(["Id", "Target", "number_of_targets"], axis=1).values.flatten()
one_hot = (hot_values.sum()) / hot_values.shape[0] * 100
zero_hot = (hot_values.shape[0] - hot_values.sum()) / hot_values.shape[0] * 100

fig, ax = plt.subplots(1,2, figsize=(20,5))
sns.distplot(proba_predictions.flatten() * 100, color="DodgerBlue", ax=ax[0])
ax[0].set_xlabel("Probability in %")
ax[0].set_ylabel("Density")
ax[0].set_title("Predicted probabilities")
sns.barplot(x=["label = 0", "label = 1"], y=[zero_hot, one_hot], ax=ax[1])
ax[1].set_ylim([0,100])
ax[1].set_title("True target label count")
ax[1].set_ylabel("Percentage");
mean_predictions = np.mean(proba_predictions, axis=0)
std_predictions = np.std(proba_predictions, axis=0)
mean_targets = validation_labels.drop(["Id", "Target", "number_of_targets"], axis=1).mean()

labels = validation_labels.drop(["Id", "Target", "number_of_targets"], axis=1).columns.values

fig, ax = plt.subplots(1,2,figsize=(20,5))
sns.barplot(x=labels,
            y=mean_predictions,
            ax=ax[0])
ax[0].set_xticklabels(labels=labels,
                      rotation=90)
ax[0].set_ylabel("Mean predicted probability")
ax[0].set_title("Mean predicted probability per class over all samples")
sns.barplot(x=labels,
           y=std_predictions,
           ax=ax[1])
ax[1].set_xticklabels(labels=labels,
                      rotation=90)
ax[1].set_ylabel("Standard deviation")
ax[1].set_title("Standard deviation of predicted probability per class over all samples");
fig, ax = plt.subplots(1,1,figsize=(20,5))
sns.barplot(x=labels, y=mean_targets.values, ax=ax)
ax.set_xticklabels(labels=labels,
                      rotation=90)
ax.set_ylabel("Percentage of hot (1)")
ax.set_title("Percentage of hot counts (ones) per target class")
feature = "Cytosol"
plt.figure(figsize=(20,5))
sns.distplot(baseline_proba_predictions[feature].values[0:-10], color="Purple")
plt.xlabel("Predicted probabilites of {}".format(feature))
plt.ylabel("Density")
plt.xlim([0,1])
wishlist = ["Nucleoplasm", "Cytosol", "Plasma membrane"]
class ImprovedDataGenerator(DataGenerator):
    
    # in contrast to the base DataGenerator we add a target wishlist to init
    def __init__(self, list_IDs, labels, modelparameter, imagepreprocessor, target_wishlist):
        super().__init__(list_IDs, labels, modelparameter, imagepreprocessor)
        self.target_wishlist = target_wishlist
    
    def get_targets_per_image(self, identifier):
        return self.labels.loc[self.labels.Id==identifier][self.target_wishlist].values
import keras.backend as K

def base_f1(y_true, y_pred):
    y_pred = K.round(y_pred)
    tp = K.sum(K.cast(y_true*y_pred, 'float'), axis=0)
    tn = K.sum(K.cast((1-y_true)*(1-y_pred), 'float'), axis=0)
    fp = K.sum(K.cast((1-y_true)*y_pred, 'float'), axis=0)
    fn = K.sum(K.cast(y_true*(1-y_pred), 'float'), axis=0)

    p = tp / (tp + fp + K.epsilon())
    r = tp / (tp + fn + K.epsilon())

    f1 = 2*p*r / (p+r+K.epsilon())
    f1 = tf.where(tf.is_nan(f1), tf.zeros_like(f1), f1)
    return f1

def f1_min(y_true, y_pred):
    f1 = base_f1(y_true, y_pred)
    return K.min(f1)

def f1_max(y_true, y_pred):
    f1 = base_f1(y_true, y_pred)
    return K.max(f1)

def f1_mean(y_true, y_pred):
    f1 = base_f1(y_true, y_pred)
    return K.mean(f1)

def f1_std(y_true, y_pred):
    f1 = base_f1(y_true, y_pred)
    return K.std(f1)
baseline_losses.head()
class TrackHistory(keras.callbacks.Callback):
    
    def on_train_begin(self, logs={}):
        self.losses = []

    def on_batch_end(self, batch, logs={}):
        self.losses.append(logs.get('loss'))
class ImprovedModel(BaseLineModel):
    
    def __init__(self, modelparameter,
                 use_dropout,
                 my_metrics=[f1_mean, f1_std, f1_min, f1_max]):
        
        super().__init__(modelparameter)
        self.my_metrics = my_metrics
        self.use_dropout = use_dropout
        
    def learn(self):
        self.history = TrackHistory()
        return self.model.fit_generator(generator=self.training_generator,
                    validation_data=self.validation_generator,
                    epochs=self.params.n_epochs, 
                    use_multiprocessing=True,
                    workers=8,
                    callbacks = [self.history])
    
    def build_model(self):
        self.model = Sequential()
        self.model.add(Conv2D(16, kernel_size=(3, 3), activation='relu', input_shape=self.input_shape,
                             kernel_initializer=VarianceScaling(seed=0),))
        self.model.add(Conv2D(32, (3, 3), activation='relu',
                             kernel_initializer=VarianceScaling(seed=0),))
        self.model.add(MaxPooling2D(pool_size=(2, 2)))
        if self.use_dropout:
            self.model.add(Dropout(0.25))
        self.model.add(Flatten())
        self.model.add(Dense(64, activation='relu',
                            kernel_initializer=VarianceScaling(seed=0),))
        if self.use_dropout:
            self.model.add(Dropout(0.5))
        self.model.add(Dense(self.num_classes, activation='sigmoid'))
parameter = ModelParameter(train_path, num_classes=len(wishlist), n_epochs=5, batch_size=64)
preprocessor = ImagePreprocessor(parameter)
labels = train_labels
training_generator = ImprovedDataGenerator(partition['train'], labels,
                                           parameter, preprocessor, wishlist)
validation_generator = ImprovedDataGenerator(partition['validation'], labels,
                                             parameter, preprocessor, wishlist)
predict_generator = PredictGenerator(partition['validation'], preprocessor, train_path)
test_preprocessor = ImagePreprocessor(parameter)
submission_predict_generator = PredictGenerator(test_names, test_preprocessor, test_path)
# Run computation and store results as csv
if kernelsettings.fit_improved_baseline == True:
    model = ImprovedModel(parameter, use_dropout=use_dropout)
    model.build_model()
    model.compile_model()
    model.set_generators(training_generator, validation_generator)
    epoch_history = model.learn()
    proba_predictions = model.predict(predict_generator)
    #model.save("improved_model.h5")
    
    improved_proba_predictions = pd.DataFrame(proba_predictions, columns=wishlist)
    improved_proba_predictions.to_csv("improved_predictions.csv")
    improved_losses = pd.DataFrame(epoch_history.history["loss"], columns=["train_loss"])
    improved_losses["val_loss"] = epoch_history.history["val_loss"]
    improved_losses.to_csv("improved_losses.csv")
    improved_batch_losses = pd.DataFrame(model.history.losses, columns=["batch_losses"])
    improved_batch_losses.to_csv("improved_batch_losses.csv")
    
    improved_submission_proba_predictions = model.predict(submission_predict_generator)
    improved_test_labels = test_labels.copy()
    improved_test_labels.loc[:, wishlist] = improved_submission_proba_predictions
    improved_test_labels.to_csv("improved_submission_proba.csv")
# If you already have done a baseline fit once, 
# you can load predictions as csv and further fitting is not neccessary:
else:
    improved_proba_predictions = pd.read_csv("../input/protein-atlas-eab-predictions/improved_predictions.csv", index_col=0)
    improved_losses= pd.read_csv("../input/protein-atlas-eab-predictions/improved_losses.csv", index_col=0)
    improved_batch_losses = pd.read_csv("../input/protein-atlas-eab-predictions/improved_batch_losses.csv", index_col=0)
    improved_test_labels = pd.read_csv("../input/protein-atlas-eab-predictions/improved_submission_proba.csv",
                                      index_col=0)
fig, ax = plt.subplots(2,1,figsize=(20,13))
ax[0].plot(np.arange(1,6), improved_losses["train_loss"].values, 'r--o', label="train_loss")
ax[0].plot(np.arange(1,6), improved_losses["val_loss"].values, 'g--o', label="validation_loss")
ax[0].set_xlabel("Epoch")
ax[0].set_ylabel("Loss")
ax[0].set_title("Loss evolution per epoch")
ax[0].legend()
ax[1].plot(improved_batch_losses.batch_losses.values, 'r-+', label="train_batch_losses")
ax[1].set_xlabel("Number of update steps in total")
ax[1].set_ylabel("Train loss")
ax[1].set_title("Train loss evolution per batch");
fig, ax = plt.subplots(3,1,figsize=(25,15))
sns.distplot(improved_proba_predictions.values[:,0], color="Orange", ax=ax[0])
ax[0].set_xlabel("Predicted probabilites of {}".format(improved_proba_predictions.columns.values[0]))
ax[0].set_xlim([0,1])
sns.distplot(improved_proba_predictions.values[:,1], color="Purple", ax=ax[1])
ax[1].set_xlabel("Predicted probabilites of {}".format(improved_proba_predictions.columns.values[1]))
ax[1].set_xlim([0,1])
sns.distplot(improved_proba_predictions.values[:,2], color="Limegreen", ax=ax[2])
ax[2].set_xlabel("Predicted probabilites of {}".format(improved_proba_predictions.columns.values[2]))
ax[2].set_xlim([0,1]);
improved_test_labels.head(1)
improved_proba_predictions.set_index(partition["validation"], inplace=True)
improved_proba_predictions.head(5)
validation_labels.set_index("Id", inplace=True)
validation_labels.head(5)
fig, ax = plt.subplots(3,1,figsize=(20,18))
for n in range(len(wishlist)):
    sns.distplot(improved_proba_predictions.loc[validation_labels[wishlist[n]] == 1,
                                                wishlist[n]], color="Green", label="1-hot", ax=ax[n])
    sns.distplot(improved_proba_predictions.loc[validation_labels[wishlist[n]] == 0,
                                                wishlist[n]], color="Red", label="0-zero", ax=ax[n])
    ax[n].set_title(wishlist[n])
    ax[n].legend()
th_cytosol = 0.4
th_plasma_membrane = 0.2
th_nucleoplasm = 0.55
improved_submission = improved_test_labels.copy()
improved_submission.head(2)
improved_submission["Nucleoplasm"] = np.where(improved_test_labels["Nucleoplasm"] >= th_nucleoplasm, 1, 0)
improved_submission["Cytosol"] = np.where(improved_test_labels["Cytosol"] >= th_cytosol, 1, 0)
improved_submission["Plasma membrane"] = np.where(
    improved_test_labels["Plasma membrane"] >= th_plasma_membrane, 1, 0)

improved_submission.head(5)
def transform_to_target(row):
    target_list = []
    for col in validation_labels.drop(["Target", "number_of_targets"], axis=1).columns:
        if row[col] == 1:
            target_list.append(str(reverse_train_labels[col]))
    if len(target_list) == 0:
        return str(0)
    return " ".join(target_list)
improved_submission["Predicted"] = improved_submission.apply(lambda l: transform_to_target(l), axis=1)
submission = improved_submission.loc[:, ["Id", "Predicted"]]
submission.head()
submission.to_csv("improved_submission.csv", index=False)
parameter = ModelParameter(train_path, num_classes=len(wishlist), n_epochs=10, batch_size=128)
preprocessor = ImagePreprocessor(parameter)
labels = train_labels

training_generator = ImprovedDataGenerator(partition['train'], labels,
                                           parameter, preprocessor, wishlist)
validation_generator = ImprovedDataGenerator(partition['validation'], labels,
                                             parameter, preprocessor, wishlist)
predict_generator = PredictGenerator(partition['validation'], preprocessor, train_path)
# Run computation and store results as csv
if kernelsettings.fit_improved_higher_batchsize == True:
    model = ImprovedModel(parameter, use_dropout=True)
    model.build_model()
    model.compile_model()
    model.set_generators(training_generator, validation_generator)
    epoch_history = model.learn()
    proba_predictions = model.predict(predict_generator)
    #model.save("improved_model.h5")
    improved_proba_predictions = pd.DataFrame(proba_predictions, columns=wishlist)
    improved_proba_predictions.to_csv("improved_hbatch_predictions.csv")
    improved_losses = pd.DataFrame(epoch_history.history["loss"], columns=["train_loss"])
    improved_losses["val_loss"] = epoch_history.history["val_loss"]
    improved_losses.to_csv("improved_hbatch_losses.csv")
    improved_batch_losses = pd.DataFrame(model.history.losses, columns=["batch_losses"])
    improved_batch_losses.to_csv("improved_hbatch_batch_losses.csv")
# If you already have done a baseline fit once, 
# you can load predictions as csv and further fitting is not neccessary:
else:
    improved_proba_predictions = pd.read_csv(
        "../input/protein-atlas-eab-predictions/improved_hbatch_predictions.csv", index_col=0)
    improved_losses= pd.read_csv(
        "../input/protein-atlas-eab-predictions/improved_hbatch_losses.csv", index_col=0)
    improved_batch_losses = pd.read_csv("../input/protein-atlas-eab-predictions/improved_hbatch_batch_losses.csv", index_col=0)
fig, ax = plt.subplots(2,1,figsize=(20,13))
ax[0].plot(np.arange(1,11), improved_losses["train_loss"].values, 'r--o', label="train_loss")
ax[0].plot(np.arange(1,11), improved_losses["val_loss"].values, 'g--o', label="validation_loss")
ax[0].set_xlabel("Epoch")
ax[0].set_ylabel("Loss")
ax[0].set_title("Loss evolution per epoch")
ax[0].legend()
ax[1].plot(improved_batch_losses.batch_losses.values, 'r-+', label="train_batch_losses")
ax[1].set_xlabel("Number of update steps in total")
ax[1].set_ylabel("Train loss")
ax[1].set_title("Train loss evolution per batch");
fig, ax = plt.subplots(3,1,figsize=(25,15))
sns.distplot(improved_proba_predictions.values[:,0], color="Orange", ax=ax[0])
ax[0].set_xlabel("Predicted probabilites of {}".format(improved_proba_predictions.columns.values[0]))
ax[0].set_xlim([0,1])
sns.distplot(improved_proba_predictions.values[:,1], color="Purple", ax=ax[1])
ax[1].set_xlabel("Predicted probabilites of {}".format(improved_proba_predictions.columns.values[1]))
ax[1].set_xlim([0,1])
sns.distplot(improved_proba_predictions.values[:,2], color="Limegreen", ax=ax[2])
ax[2].set_xlabel("Predicted probabilites of {}".format(improved_proba_predictions.columns.values[2]))
ax[2].set_xlim([0,1]);
# Run computation and store results as csv
if kernelsettings.fit_improved_without_dropout == True:
    model = ImprovedModel(parameter, use_dropout=False)
    model.build_model()
    model.compile_model()
    model.set_generators(training_generator, validation_generator)
    epoch_history = model.learn()
    proba_predictions = model.predict(predict_generator)
    #model.save("improved_model.h5")
    improved_proba_predictions = pd.DataFrame(proba_predictions, columns=wishlist)
    improved_proba_predictions.to_csv("improved_nodropout_predictions.csv")
    improved_losses = pd.DataFrame(epoch_history.history["loss"], columns=["train_loss"])
    improved_losses["val_loss"] = epoch_history.history["val_loss"]
    improved_losses.to_csv("improved_nodropout_losses.csv")
    improved_batch_losses = pd.DataFrame(model.history.losses, columns=["batch_losses"])
    improved_batch_losses.to_csv("improved_nodropout_batch_losses.csv")
# If you already have done a baseline fit once, 
# you can load predictions as csv and further fitting is not neccessary:
else:
    improved_proba_predictions_no_dropout = pd.read_csv(
        "../input/protein-atlas-eab-predictions/improved_nodropout_predictions.csv", index_col=0)
    improved_losses_no_dropout= pd.read_csv(
        "../input/protein-atlas-eab-predictions/improved_nodropout_losses.csv", index_col=0)
    improved_batch_losses_no_dropout = pd.read_csv(
        "../input/protein-atlas-eab-predictions/improved_nodropout_batch_losses.csv", index_col=0)
fig, ax = plt.subplots(2,1,figsize=(20,13))
ax[0].plot(np.arange(1,11), improved_losses["train_loss"].values, 'r--o', label="train_loss_dropout")
ax[0].plot(np.arange(1,11), improved_losses_no_dropout["train_loss"].values, 'r-o', label="train_loss_no_dropout")
ax[0].plot(np.arange(1,11), improved_losses["val_loss"].values, 'g--o', label="validation_loss")
ax[0].plot(np.arange(1,11), improved_losses_no_dropout["val_loss"].values, 'g-o', label="validation_loss_no_dropout")
ax[0].set_xlabel("Epoch")
ax[0].set_ylabel("Loss")
ax[0].set_title("Loss evolution per epoch")
ax[0].legend()
ax[1].plot(improved_batch_losses.batch_losses.values[-800::], 'r-+', label="train_batch_losses_dropout")
ax[1].plot(improved_batch_losses_no_dropout.batch_losses.values[-800::], 'b-+',
           label="train_batch_losses_no_dropout")
ax[1].set_xlabel("Number of update steps in total")
ax[1].set_ylabel("Train loss")
ax[1].set_title("Train loss evolution per batch");
ax[1].legend();