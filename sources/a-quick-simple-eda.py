
import numpy as np
import pandas as pd
import cv2
import json
import os
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from matplotlib.collections import PatchCollection
from mpl_toolkits.mplot3d import Axes3D
DATASET_DIR = '/kaggle/input/pku-autonomous-driving/'
JSON_DIR = os.path.join(DATASET_DIR, 'car_models_json')
NUM_IMG_SAMPLES = 10 # The number of image samples used for visualization
df = pd.read_csv(os.path.join(DATASET_DIR, 'train.csv'))
df.head()
image_ids = np.array(df['ImageId'])
prediction_strings = np.array(df['PredictionString'])
prediction_strings = [
    np.array(prediction_string.split(' ')).astype(np.float32).reshape(-1, 7) \
    for prediction_string in prediction_strings
]
print('Image ID:', image_ids[0])
print('Annotations:\n', prediction_strings[0])
# https://raw.githubusercontent.com/ApolloScapeAuto/dataset-api/master/car_instance/car_models.py
models = {
    #           name                id
         'baojun-310-2017':          0,
            'biaozhi-3008':          1,
      'biaozhi-liangxiang':          2,
       'bieke-yinglang-XT':          3,
            'biyadi-2x-F0':          4,
           'changanbenben':          5,
            'dongfeng-DS5':          6,
                 'feiyate':          7,
     'fengtian-liangxiang':          8,
            'fengtian-MPV':          9,
       'jilixiongmao-2015':         10,
       'lingmu-aotuo-2009':         11,
            'lingmu-swift':         12,
         'lingmu-SX4-2012':         13,
          'sikeda-jingrui':         14,
    'fengtian-weichi-2006':         15,
               '037-CAR02':         16,
                 'aodi-a6':         17,
               'baoma-330':         18,
               'baoma-530':         19,
        'baoshijie-paoche':         20,
         'bentian-fengfan':         21,
             'biaozhi-408':         22,
             'biaozhi-508':         23,
            'bieke-kaiyue':         24,
                    'fute':         25,
                 'haima-3':         26,
           'kaidilake-CTS':         27,
               'leikesasi':         28,
           'mazida-6-2015':         29,
              'MG-GT-2015':         30,
                   'oubao':         31,
                    'qiya':         32,
             'rongwei-750':         33,
              'supai-2016':         34,
         'xiandai-suonata':         35,
        'yiqi-benteng-b50':         36,
                   'bieke':         37,
               'biyadi-F3':         38,
              'biyadi-qin':         39,
                 'dazhong':         40,
          'dazhongmaiteng':         41,
                'dihao-EV':         42,
  'dongfeng-xuetielong-C6':         43,
 'dongnan-V3-lingyue-2011':         44,
'dongfeng-yulong-naruijie':         45,
                 '019-SUV':         46,
               '036-CAR01':         47,
             'aodi-Q7-SUV':         48,
              'baojun-510':         49,
                'baoma-X5':         50,
         'baoshijie-kayan':         51,
         'beiqi-huansu-H3':         52,
          'benchi-GLK-300':         53,
            'benchi-ML500':         54,
     'fengtian-puladuo-06':         55,
        'fengtian-SUV-gai':         56,
'guangqi-chuanqi-GS4-2015':         57,
    'jianghuai-ruifeng-S3':         58,
              'jili-boyue':         59,
                  'jipu-3':         60,
              'linken-SUV':         61,
               'lufeng-X8':         62,
             'qirui-ruihu':         63,
             'rongwei-RX5':         64,
         'sanling-oulande':         65,
              'sikeda-SUV':         66,
        'Skoda_Fabia-2011':         67,
        'xiandai-i25-2016':         68,
        'yingfeinidi-qx80':         69,
         'yingfeinidi-SUV':         70,
              'benchi-SUR':         71,
             'biyadi-tang':         72,
       'changan-CS35-2012':         73,
             'changan-cs5':         74,
      'changcheng-H6-2016':         75,
             'dazhong-SUV':         76,
 'dongfeng-fengguang-S560':         77,
   'dongfeng-fengxing-SX6':         78
}
models_map = dict((y, x) for x, y in models.items())
cars = []
for prediction_string in prediction_strings:
    for car in prediction_string:
        cars.append(car)
cars = np.array(cars)
unique, counts = np.unique(cars[..., 0].astype(np.uint8), return_counts=True)
all_model_types = zip(unique, counts)

for i, model_type in enumerate(all_model_types):
    print('{}.\t Model type: {:<22} | {} cars'.format(i, models_map[model_type[0]], model_type[1]))
def plot_figures(
    sizes,
    pie_title,
    start_angle,
    bar_title,
    bar_ylabel,
    labels,
    explode,
    colors=None,
):
    fig, ax = plt.subplots(figsize=(14, 14))

    y_pos = np.arange(len(labels))
    barlist = ax.bar(y_pos, sizes, align='center')
    ax.set_xticks(y_pos, labels)
    ax.set_ylabel(bar_ylabel)
    ax.set_title(bar_title)
    if colors is not None:
        for idx, item in enumerate(barlist):
            item.set_color(colors[idx])

    def autolabel(rects):
        """
        Attach a text label above each bar displaying its height
        """
        for rect in rects:
            height = rect.get_height()
            ax.text(
                rect.get_x() + rect.get_width()/2., height,
                '%d' % int(height),
                ha='center', va='bottom', fontweight='bold'
            )

    autolabel(barlist)
    
    fig, ax = plt.subplots(figsize=(14, 14))
    
    pielist = ax.pie(sizes, explode=explode, labels=labels, autopct='%1.1f%%', shadow=False, startangle=start_angle, counterclock=False)
    ax.axis('equal')
    ax.set_title(pie_title)
    if colors is not None:
        for idx, item in enumerate(pielist[0]):
            item.set_color(colors[idx])

    plt.show()
plot_figures(
    counts,
    pie_title='The percentage of the number of cars of each model type',
    start_angle=170,
    bar_title='Distribution of cars of each model type',
    bar_ylabel='Frequency',
    labels=[label for label in unique],
    explode=np.zeros(len(unique))
)
# Get all json files
files = [file for file in os.listdir(JSON_DIR) if os.path.isfile(os.path.join(JSON_DIR, file))]

# For each json file, plot figure
for file in files:
    model_path = os.path.join(JSON_DIR, file)
    with open(model_path) as src:
        data = json.load(src)
        car_type = data['car_type']
        faces = data['faces']
        vertices = np.array(data['vertices'])
        triangles = np.array(faces) - 1

        fig = plt.figure(figsize=(16, 5))
        ax11 = fig.add_subplot(1, 2, 1, projection='3d')
        ax11.set_title('Model: {} | Type: {}'.format(file.split('.')[0], car_type))
        ax11.set_xlim([-2, 3])
        ax11.set_ylim([-3, 2])
        ax11.set_zlim([0, 3])
        ax11.view_init(30, -50)
        ax11.plot_trisurf(vertices[:,0], vertices[:,2], triangles, -vertices[:,1], shade=True, color='lime')
        
        ax12 = fig.add_subplot(1, 2, 2, projection='3d')
        ax12.set_title('Model: {} | Type: {}'.format(file.split('.')[0], car_type))
        ax12.set_xlim([-2, 3])
        ax12.set_ylim([-3, 2])
        ax12.set_zlim([0, 3])
        ax12.view_init(30, 40)
        ax12.plot_trisurf(vertices[:,0], vertices[:,2], triangles, -vertices[:,1], shade=True, color='lime')
def show_samples(samples):
    for sample in samples:
        fig, ax = plt.subplots(figsize=(18, 16))
        
        # Get image
        img_path = os.path.join(DATASET_DIR, 'train_images', '{}.{}'.format(sample, 'jpg'))
        img = cv2.imread(img_path, 1)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Get corresponding mask
        mask_path = os.path.join(DATASET_DIR, 'train_masks', '{}.{}'.format(sample, 'jpg'))
        mask = cv2.imread(mask_path, 0)

        patches = []
        contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            poly_patch = Polygon(contour.reshape(-1, 2), closed=True, linewidth=2, edgecolor='r', facecolor='r', fill=True)
            patches.append(poly_patch)
        p = PatchCollection(patches, match_original=True, cmap=matplotlib.cm.jet, alpha=0.3)

        ax.imshow(img/255)
        ax.set_title(sample)
        ax.add_collection(p)
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        plt.show()
# Randomly select samples
samples = image_ids[np.random.choice(image_ids.shape[0], NUM_IMG_SAMPLES, replace=False)]

# Show images and corresponding masks of too-far-away (not of interest) cars
show_samples(samples)
