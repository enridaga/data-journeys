
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
#%matplotlib inline
import seaborn as sns
sns.set()

import pydicom

from os import listdir

from skimage.transform import resize
from imgaug import augmenters as iaa

from sklearn.model_selection import train_test_split, StratifiedKFold

from keras.applications import ResNet50, VGG16
from keras.applications.resnet50 import preprocess_input as preprocess_resnet_50
from keras.applications.vgg16 import preprocess_input as preprocess_vgg_16
from keras.layers import GlobalAveragePooling2D, Dense, Activation, concatenate, Dropout
from keras.initializers import glorot_normal, he_normal
from keras.regularizers import l2
from keras.models import Model, load_model
from keras.utils import Sequence
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
from keras.optimizers import Adam


from tensorflow.nn import sigmoid_cross_entropy_with_logits
import tensorflow as tf

listdir("../input/")
#listdir("../input/rsna-ih-detection-baseline-models")
train_brute_force = False
train_anysubtype_network = False
MODELOUTPUT_PATH_BRUTE_FORCE = "bruteforce_best_model.hdf5"
MODELOUTPUT_PATH_ANYSUBTYPE = "anysubtype_best_model.hdf5"
brute_force_model_input = "../input/rsna-ih-detection-baseline-models/_bruteforce_best_model.hdf5"
brute_force_losses_path = "../input/rsna-ih-detection-baseline-models/brute_force_losses.csv"
any_subtype_model_input = "../input/rsna-ih-detection-baseline-models/anysubtype_best_model.hdf5"
def rescale_pixelarray(dataset):
    image = dataset.pixel_array
    rescaled_image = image * dataset.RescaleSlope + dataset.RescaleIntercept
    rescaled_image[rescaled_image < -1024] = -1024
    return rescaled_image

def set_manual_window(hu_image, min_value, max_value):
    hu_image[hu_image < min_value] = min_value
    hu_image[hu_image > max_value] = min_value #max_value
    return hu_image

class Preprocessor:    
    
    def __init__(self, path, backbone, hu_min_value, hu_max_value, augment=False):
        self.path = path
        self.backbone = backbone
        self.nn_input_shape = backbone["nn_input_shape"]
        self.hu_min_value = hu_min_value
        self.hu_max_value = hu_max_value
        self.augment = augment
        
    # 1. We need to load the dicom dataset
    def load_dicom_dataset(self, filename):
        dataset = pydicom.dcmread(self.path + filename)
        return dataset
    
    # 2. We need to rescale the pixelarray to Hounsfield units
    #    and we need to focus on our custom window:
    def get_hounsfield_window(self, dataset, min_value, max_value):
        try:
            hu_image = rescale_pixelarray(dataset)
            windowed_image = set_manual_window(hu_image, min_value, max_value)
        except ValueError:
            # set to level 
            windowed_image = min_value * np.ones((self.nn_input_shape[0], self.nn_input_shape[1]))
        return windowed_image
        
    
    # 3. Resize the image to the input shape of our CNN
    def resize(self, image):
        image = resize(image, self.nn_input_shape)
        return image
    
    # 4. If we like to augment our image, let's do it:
    def augment_img(self, image): 
        augment_img = iaa.Sequential([
            #iaa.Crop(keep_size=True, percent=(0.01, 0.05), sample_independently=False),
            #iaa.Affine(rotate=(-10, 10)),
            iaa.Fliplr(0.5)])
        image_aug = augment_img.augment_image(image)
        return image_aug
    
    def fill_channels(self, image):
        filled_image = np.stack((image,)*3, axis=-1)
        return filled_image
    
    def preprocess(self, identifier):
        filename = identifier +  ".dcm"
        dataset = self.load_dicom_dataset(filename)
        windowed_image = self.get_hounsfield_window(dataset, self.hu_min_value, self.hu_max_value)
        image = self.resize(windowed_image)
        if self.augment:
            image = self.augment_img(image)
        image = self.fill_channels(image)
        return image
    
    def normalize(self, image):
        return (image - self.hu_min_value)/(self.hu_max_value-self.hu_min_value) * 0.5
class DataLoader(Sequence):
    
    def __init__(self, dataframe,
                 preprocessor,
                 batch_size,
                 shuffle,
                 num_classes=6,
                 steps=None):
        
        self.preprocessor = preprocessor
        self.data_ids = dataframe.index.values
        self.dataframe = dataframe
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.input_shape = self.preprocessor.backbone["nn_input_shape"]
        self.preprocess_fun = self.preprocessor.backbone["preprocess_fun"]
        self.num_classes = num_classes
        self.current_epoch=0
        
        self.steps=steps
        if self.steps is not None:
            self.steps = np.round(self.steps/3) * 3
            self.undersample()
        
    def undersample(self):
        part = np.int(self.steps/3 * self.batch_size)
        zero_ids = np.random.choice(self.dataframe.loc[self.dataframe["any"] == 0].index.values, size=2*part, replace=False)
        hot_ids = np.random.choice(self.dataframe.loc[self.dataframe["any"] == 1].index.values, size=1*part, replace=False)
        self.data_ids = list(set(zero_ids).union(hot_ids))
        np.random.shuffle(self.data_ids)
        
    # defines the number of steps per epoch
    def __len__(self):
        if self.steps is None:
            return np.int(np.ceil(len(self.data_ids) / np.float(self.batch_size)))
        else:
            return 3*np.int(self.steps/3) 
    
    # at the end of an epoch: 
    def on_epoch_end(self):
        # if steps is None and shuffle is true:
        if self.steps is None:
            self.data_ids = self.dataframe.index.values
            if self.shuffle:
                np.random.shuffle(self.data_ids)
        else:
            self.undersample()
        self.current_epoch += 1
    
    # should return a batch of images
    def __getitem__(self, item):
        # select the ids of the current batch
        current_ids = self.data_ids[item*self.batch_size:(item+1)*self.batch_size]
        X, y = self.__generate_batch(current_ids)
        return X, y
    
    # collect the preprocessed images and targets of one batch
    def __generate_batch(self, current_ids):
        X = np.empty((self.batch_size, *self.input_shape, 3))
        y = np.empty((self.batch_size, self.num_classes))
        for idx, ident in enumerate(current_ids):
            # Store sample
            image = self.preprocessor.preprocess(ident)
            X[idx] = self.preprocessor.normalize(image)
            # Store class
            y[idx] = self.__get_target(ident)
        return X, y
    
    # extract the targets of one image id:
    def __get_target(self, ident):
        targets = self.dataframe.loc[ident].values
        return targets
INPUT_PATH = "../input/rsna-intracranial-hemorrhage-detection/"
train_dir = INPUT_PATH + "stage_1_train_images/"
test_dir = INPUT_PATH + "stage_1_test_images/"
submission = pd.read_csv(INPUT_PATH + "stage_1_sample_submission.csv")
submission.head(7)
traindf = pd.read_csv(INPUT_PATH + "stage_1_train.csv")
traindf.head()
label = traindf.Label.values
traindf = traindf.ID.str.rsplit("_", n=1, expand=True)
traindf.loc[:, "label"] = label
traindf = traindf.rename({0: "id", 1: "subtype"}, axis=1)
traindf.head()
testdf = submission.ID.str.rsplit("_", n=1, expand=True)
testdf = testdf.rename({0: "id", 1: "subtype"}, axis=1)
testdf.loc[:, "label"] = 0
testdf.head()
traindf = pd.pivot_table(traindf, index="id", columns="subtype", values="label")
traindf.head()
testdf = pd.pivot_table(testdf, index="id", columns="subtype", values="label")
testdf.head(1)
pretrained_models_path = "../input/keras-pretrained-models/"
listdir("../input/keras-pretrained-models/")
pretrained_models = {
    "resnet_50": {"weights": "resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5",
                  "nn_input_shape": (224,224),
                  "preprocess_fun": preprocess_resnet_50},
    "vgg16": {"weights": "vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5",
              "nn_input_shape": (224,224),
              "preprocess_fun": preprocess_vgg_16},
}
def resnet_50():
    weights_path = pretrained_models_path + pretrained_models["resnet_50"]["weights"]
    net = ResNet50(include_top=False, weights=weights_path)
    for layer in net.layers:
        layer.trainable = False
    return net

def vgg_16():
    weights_path = pretrained_models_path + pretrained_models["vgg_16"]["weights"]
    net = VGG16(include_top=False, weights=weights_path)
    for layer in net.layers:
        layer.trainable = False
    return net
class MyNetwork:
    
    def __init__(self,
                 model_fun,
                 loss_fun,
                 metrics_list,
                 train_generator,
                 dev_generator,
                 epochs,
                 num_classes=6,
                 checkpoint_path=MODELOUTPUT_PATH_BRUTE_FORCE):
        self.model_fun = model_fun
        self.loss_fun = loss_fun
        self.metrics_list = metrics_list
        self.train_generator = train_generator
        self.dev_generator = dev_generator
        self.epochs = epochs
        self.num_classes = num_classes
        self.checkpoint_path = checkpoint_path 
        self.checkpoint = ModelCheckpoint(filepath=self.checkpoint_path,
                                          mode="min",
                                          verbose=1,
                                          save_best_only=True,
                                          save_weights_only=True,
                                          period=1)
        self.reduce_lr = ReduceLROnPlateau(monitor='val_loss',
                                           factor=0.5,
                                           patience=2,
                                           min_lr=1e-8,
                                           mode="min")
        self.e_stopping = EarlyStopping(monitor="val_loss",
                                        min_delta=0.01,
                                        patience=5,
                                        mode="min",
                                        restore_best_weights=True)
        
    def build_model(self):
        base_model = self.model_fun()
        x = base_model.output
        x = GlobalAveragePooling2D()(x)
        x = Dropout(0.3)(x)
        x = Dense(100, activation="relu")(x)
        x = Dropout(0.3)(x)
        pred = Dense(self.num_classes,
                     kernel_initializer=he_normal(seed=11),
                     kernel_regularizer=l2(0.05),
                     bias_regularizer=l2(0.05), activation="sigmoid")(x)
        self.model = Model(inputs=base_model.input, outputs=pred)
    
    def compile_model(self):
        self.model.compile(optimizer=Adam(learning_rate=LR),
                           loss=self.loss_fun, 
                           metrics=self.metrics_list)
    
    def learn(self):
        return self.model.fit_generator(generator=self.train_generator,
                    validation_data=self.dev_generator,
                    epochs=self.epochs,
                    callbacks=[self.checkpoint, self.reduce_lr, self.e_stopping],
                    #use_multiprocessing=False,
                    workers=8)
    
    def load_weights(self, path):
        self.model.load_weights(path)
    
    def predict(self, test_generator):
        predictions = self.model.predict_generator(test_generator, workers=8)
        return predictions
split_seed = 1
kfold = StratifiedKFold(n_splits=5, random_state=split_seed).split(np.arange(traindf.shape[0]), traindf["any"].values)

train_idx, dev_idx = next(kfold)

train_data = traindf.iloc[train_idx]
dev_data = traindf.iloc[dev_idx]

#train_data, dev_data = train_test_split(traindf, test_size=0.1, stratify=traindf.values, random_state=split_seed)
print(train_data.shape)
print(dev_data.shape)
pos_perc_train = train_data.sum() / train_data.shape[0] * 100
pos_perc_dev = dev_data.sum() / dev_data.shape[0] * 100

fig, ax = plt.subplots(2,1,figsize=(20,14))
sns.barplot(x=pos_perc_train.index, y=pos_perc_train.values, palette="Set2", ax=ax[0]);
ax[0].set_title("Target distribution used for training data")
sns.barplot(x=pos_perc_dev.index, y=pos_perc_dev.values, palette="Set2", ax=ax[1]);
ax[1].set_title("Target distribution used for dev data");
def np_multilabel_loss(y_true, y_pred, class_weights=None):
    y_pred = np.where(y_pred > 1-(1e-07), 1-1e-07, y_pred)
    y_pred = np.where(y_pred < 1e-07, 1e-07, y_pred)
    single_class_cross_entropies = - np.mean(y_true * np.log(y_pred) + (1-y_true) * np.log(1-y_pred), axis=0)
    
    print(single_class_cross_entropies)
    if class_weights is None:
        loss = np.mean(single_class_cross_entropies)
    else:
        loss = np.sum(class_weights*single_class_cross_entropies)
    return loss

def get_raw_xentropies(y_true, y_pred):
    y_pred = tf.clip_by_value(y_pred, 1e-7, 1-1e-7)
    xentropies = y_true * tf.log(y_pred) + (1-y_true) * tf.log(1-y_pred)
    return -xentropies

# multilabel focal loss equals multilabel loss in case of alpha=0.5 and gamma=0 
def multilabel_focal_loss(class_weights=None, alpha=0.5, gamma=2):
    def mutlilabel_focal_loss_inner(y_true, y_pred):
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.cast(y_pred, tf.float32)
        
        xentropies = get_raw_xentropies(y_true, y_pred)

        # compute pred_t:
        y_t = tf.where(tf.equal(y_true,1), y_pred, 1.-y_pred)
        alpha_t = tf.where(tf.equal(y_true, 1), alpha * tf.ones_like(y_true), (1-alpha) * tf.ones_like(y_true))

        # compute focal loss contributions
        focal_loss_contributions =  tf.multiply(tf.multiply(tf.pow(1-y_t, gamma), xentropies), alpha_t) 

        # our focal loss contributions have shape (n_samples, s_classes), we need to reduce with mean over samples:
        focal_loss_per_class = tf.reduce_mean(focal_loss_contributions, axis=0)

        # compute the overall loss if class weights are None (equally weighted):
        if class_weights is None:
            focal_loss_result = tf.reduce_mean(focal_loss_per_class)
        else:
            # weight the single class losses and compute the overall loss
            weights = tf.constant(class_weights, dtype=tf.float32)
            focal_loss_result = tf.reduce_sum(tf.multiply(weights, focal_loss_per_class))
            
        return focal_loss_result
    return mutlilabel_focal_loss_inner
BACKBONE = "resnet_50"
BATCH_SIZE = 16
TEST_BATCH_SIZE = 5
MIN_VALUE = 0
MAX_VALUE = 90
STEPS = 50
EPOCHS = 20

LR = 0.0001
train_preprocessor = Preprocessor(path=train_dir,
                                  backbone=pretrained_models[BACKBONE],
                                  hu_min_value=MIN_VALUE,
                                  hu_max_value=MAX_VALUE,
                                  augment=True)

dev_preprocessor = Preprocessor(path=train_dir,
                                backbone=pretrained_models[BACKBONE],
                                hu_min_value=MIN_VALUE,
                                hu_max_value=MAX_VALUE,
                                augment=False)

test_preprocessor = Preprocessor(path=test_dir,
                                backbone=pretrained_models[BACKBONE],
                                hu_min_value=MIN_VALUE,
                                hu_max_value=MAX_VALUE,
                                augment=False)
fig, ax = plt.subplots(1,4,figsize=(20,20))


for m in range(4):
    example = train_data.index.values[m]
    title = [col for col in train_data.loc[example,:].index if train_data.loc[example, col]==1]
    if len(title) == 0:
        title="Healthy"
    preprocess_example = train_preprocessor.preprocess(example)
    ax[m].imshow(preprocess_example[:,:,2], cmap="Spectral")
    ax[m].grid(False)
    ax[m].set_title(title);
fig, ax = plt.subplots(2,1,figsize=(20,10))
sns.distplot(preprocess_example[:,:,2].flatten(), kde=False, ax=ax[0])
sns.distplot(train_preprocessor.normalize(preprocess_example)[:,:,2].flatten(), kde=False)
plt.title("Image distribution after normalisation");
train_preprocessor.normalize(preprocess_example[:,:,2]).min()
train_preprocessor.normalize(preprocess_example[:,:,2]).max()
train_dataloader = DataLoader(train_data,
                              train_preprocessor,
                              BATCH_SIZE,
                              shuffle=True,
                              steps=STEPS)

dev_dataloader = DataLoader(dev_data, 
                            dev_preprocessor,
                            BATCH_SIZE,
                            shuffle=True,
                            steps=STEPS)

test_dataloader = DataLoader(testdf, 
                             test_preprocessor,
                             TEST_BATCH_SIZE,
                             shuffle=False)
train_dataloader.__len__()
len(train_dataloader.data_ids)/BATCH_SIZE
train_data.loc[train_dataloader.data_ids]["any"].value_counts()
test_dataloader.__len__()
dev_dataloader.__len__()
my_class_weights = [0.5, 0.1, 0.1, 0.1, 0.1, 0.1]
def turn_pred_to_dataframe(data_df, pred):
    df = pd.DataFrame(pred, columns=data_df.columns, index=data_df.index)
    df = df.stack().reset_index()
    df.loc[:, "ID"] = df.id.str.cat(df.subtype, sep="_")
    df = df.drop(["id", "subtype"], axis=1)
    df = df.rename({0: "Label"}, axis=1)
    return df
if train_brute_force:
    model = MyNetwork(model_fun=resnet_50,
                      loss_fun="binary_crossentropy", #multilabel_focal_loss(class_weights=my_class_weights, alpha=0.5, gamma=0),
                      metrics_list=[multilabel_focal_loss(alpha=0.5, gamma=0)],
                      train_generator=train_dataloader,
                      dev_generator=dev_dataloader,
                      epochs=EPOCHS,
                      num_classes=6)
    model.build_model()
    model.compile_model()
    history = model.learn()
    
    print(history.history.keys())
    
    fig, ax = plt.subplots(2,1,figsize=(20,5))
    ax[0].plot(history.history["loss"], 'o-')
    ax[0].plot(history.history["val_loss"], 'o-')
    ax[1].plot(history.history["lr"], 'o-')
    
    #test_pred = model.predict(test_dataloader)[0:testdf.shape[0]]
    #dev_pred = model.predict(dev_dataloader)
    
    #test_pred_df = turn_pred_to_dataframe(testdf, test_pred)
    #dev_pred_df = turn_pred_to_dataframe(dev_data, dev_pred)
    
    #test_pred_df.to_csv("brute_force_test_pred.csv", index=False)
    #dev_pred_df.to_csv("brute_force_dev_pred.csv", index=False)
dev_proba = model.predict(dev_dataloader)
dev_proba.shape
plt.figure(figsize=(20,5))
for n in range(6):
    sns.distplot(dev_proba[:,n])
alpha_subtypes = 0.25 
gamma_subtypes = 2
class AnySubtypeNetwork(MyNetwork):
    
    def __init__(self,
                 model_fun,
                 loss_fun,
                 metrics_list,
                 train_generator,
                 dev_generator,
                 epochs,
                 num_subtype_classes=5,
                 checkpoint_path=MODELOUTPUT_PATH_ANYSUBTYPE):
        MyNetwork.__init__(self, 
                           model_fun=model_fun,
                           loss_fun=loss_fun,
                           metrics_list=metrics_list,
                           train_generator=train_generator,
                           dev_generator=dev_generator,
                           epochs=epochs,
                           num_classes=num_subtype_classes)
    
    def build_model(self):
        base_model = self.model_fun()
        x = base_model.output
        x = GlobalAveragePooling2D()(x)
        x = Dropout(0.5)(x)
        any_logits = Dense(1, kernel_initializer=he_normal(seed=11))(x)
        any_pred = Activation("sigmoid", name="any_predictions")(any_logits)
        x = concatenate([any_pred, x])
        sub_pred = Dense(self.num_classes,
                         name="subtype_pred",
                         kernel_initializer=he_normal(seed=12),
                         activation="sigmoid")(x) 
        self.model = Model(inputs=base_model.input, outputs=[any_pred, sub_pred])
    
    def compile_model(self):
        self.model.compile(optimizer=Adam(LR),
                           loss=['binary_crossentropy', multilabel_focal_loss(alpha=alpha_subtypes, gamma=gamma_subtypes)],
                           loss_weights = [1., 0.],
                           metrics=self.metrics_list)
class AnySubtypeDataLoader(Sequence):
    
    def __init__(self, dataframe,
                 preprocessor,
                 batch_size,
                 num_classes=5,
                 shuffle=False,
                 steps=None):
        self.preprocessor = preprocessor
        self.data_ids = dataframe.index.values
        self.dataframe = dataframe
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.input_shape = self.preprocessor.backbone["nn_input_shape"]
        self.preprocess_fun = self.preprocessor.backbone["preprocess_fun"]
        self.num_classes = num_classes
        self.current_epoch=0
        
        self.steps=steps
        if self.steps is not None:
            self.steps = np.round(self.steps/2) * 2
            self.undersample()
        
    def undersample(self):
        part = np.int(self.steps/2 * self.batch_size)
        zero_ids = np.random.choice(self.dataframe.loc[self.dataframe["any"] == 0].index.values, size=1*part, replace=False)
        hot_ids = np.random.choice(self.dataframe.loc[self.dataframe["any"] == 1].index.values, size=1*part, replace=False)
        self.data_ids = list(set(zero_ids).union(hot_ids))
        np.random.shuffle(self.data_ids)
    
    # defines the number of steps per epoch
    def __len__(self):
        if self.steps is None:
            return np.int(np.ceil(len(self.data_ids) / np.float(self.batch_size)))
        else:
            return 2*np.int(self.steps/2) 
    
    # at the end of an epoch: 
    def on_epoch_end(self):
        # if steps is None and shuffle is true:
        if self.steps is None:
            self.data_ids = self.dataframe.index.values
            if self.shuffle:
                np.random.shuffle(self.data_ids)
        else:
            self.undersample()
        self.current_epoch += 1
    
    # should return a batch of images
    def __getitem__(self, item):
        # select the ids of the current batch
        current_ids = self.data_ids[item*self.batch_size:(item+1)*self.batch_size]
        X, y_any, y_subtype = self.__generate_batch(current_ids)
        return X, [y_any, y_subtype]
    
    # collect the preprocessed images and targets of one batch
    def __generate_batch(self, current_ids):
        X = np.empty((self.batch_size, *self.input_shape, 3))
        y_subtype = np.empty((self.batch_size, self.num_classes))
        y_any = np.empty((self.batch_size, 1))
        for idx, ident in enumerate(current_ids):
            # Store sample
            image = self.preprocessor.preprocess(ident)
            X[idx] = self.preprocessor.normalize(image)
            # Store class
            y_any[idx], y_subtype[idx] = self.__get_target(ident)
        return X, y_any, y_subtype
    
    # extract the targets of one image id:
    def __get_target(self, ident):
        y_any = self.dataframe.loc[ident, "any"]
        y_subtype = self.dataframe.drop("any", axis=1).loc[ident].values
        return y_any, y_subtype
train_dataloader = AnySubtypeDataLoader(train_data,
                                        train_preprocessor,
                                        BATCH_SIZE,
                                        steps=STEPS)
dev_dataloader = AnySubtypeDataLoader(dev_data, 
                                      dev_preprocessor,
                                      BATCH_SIZE,
                                      steps=STEPS,
                                      shuffle=False)

test_dataloader = AnySubtypeDataLoader(testdf, 
                                       test_preprocessor,
                                       TEST_BATCH_SIZE,
                                       shuffle=False)
X, [y1, y2] = train_dataloader.__getitem__(0)
y1.shape
y2.shape
X.shape
y1[0]
y2[0]
plt.imshow(X[0,:,:,0])
train_dataloader.__len__()
len(train_dataloader.data_ids)
train_data.loc[train_dataloader.data_ids].sum() / train_data.loc[train_dataloader.data_ids].shape[0]
if train_anysubtype_network:
    model = AnySubtypeNetwork(model_fun=resnet_50,
                              loss_fun=None,
                              metrics_list={"any_predictions":"binary_crossentropy",
                                            "subtype_pred": multilabel_focal_loss(alpha=0.5, gamma=0)},
                              train_generator=train_dataloader,
                              dev_generator=dev_dataloader,
                              epochs=50) 
    model.build_model()
    model.compile_model()
    history = model.learn()
    
    print(history.history.keys())
    
    fig, ax = plt.subplots(2,1,figsize=(20,10))
    ax[0].plot(history.history["loss"], 'o-')
    ax[0].plot(history.history["val_loss"], 'o-')
    ax[1].plot(history.history["lr"], 'o-')
else:
    model = AnySubtypeNetwork(model_fun=resnet_50,
                              loss_fun=None,
                              metrics_list={"any_predictions":"binary_crossentropy",
                                            "subtype_pred": multilabel_focal_loss(alpha=0.5, gamma=0)},
                              train_generator=train_dataloader,
                              dev_generator=dev_dataloader,
                              epochs=50) 
    model.build_model()
    #model.load_weights(any_subtype_model_input)
dev_proba_any, dev_proba_subtype = model.predict(dev_dataloader)
train_proba_any, train_proba_subtype = model.predict(train_dataloader)
dev_proba_any.shape
dev_proba_subtype.shape
if train_anysubtype_network:
    fig, ax = plt.subplots(2,1, figsize=(20,10))
    sns.distplot(dev_proba_any[:,0], ax=ax[0], color="Purple")
    sns.distplot(dev_proba_subtype[:,0], ax=ax[1])
    sns.distplot(dev_proba_subtype[:,1], ax=ax[1])
    sns.distplot(dev_proba_subtype[:,2], ax=ax[1])
    sns.distplot(dev_proba_subtype[:,3], ax=ax[1])
    sns.distplot(dev_proba_subtype[:,4], ax=ax[1])
    ax[0].set_title("Predicted probability of hemorrhage occurence in dev batch")
    ax[1].set_title("Predicted probability of hemorrhage subtypes in dev batch")
    
    fig, ax = plt.subplots(2,1, figsize=(20,10))
    sns.distplot(train_proba_any[:,0], ax=ax[0], color="Purple")
    sns.distplot(train_proba_subtype[:,0], ax=ax[1])
    sns.distplot(train_proba_subtype[:,1], ax=ax[1])
    sns.distplot(train_proba_subtype[:,2], ax=ax[1])
    sns.distplot(train_proba_subtype[:,3], ax=ax[1])
    sns.distplot(train_proba_subtype[:,4], ax=ax[1])
    ax[0].set_title("Predicted probability of hemorrhage occurence in train")
    ax[1].set_title("Predicted probability of hemorrhage subtypes in train")
