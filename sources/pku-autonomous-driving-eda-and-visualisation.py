
import numpy as np
import pandas as pd
import cv2
import json
import os
import matplotlib
import matplotlib.pyplot as plt
Input = '../input/pku-autonomous-driving/'
file_list = os.listdir(Input)
print(file_list)
train = pd.read_csv(Input+'train.csv')
train.head()
print('Annotations for Image ',train['ImageId'][0],' are ',train['PredictionString'][0])
annotations = train['PredictionString'][0].split(' ')
print(len(annotations) / 7)

poses = []

for index in range(0,int(len(annotations)/7)):
    i = index*7
    poses.append(annotations[i:i+7])
print(poses)
#Read camera parameters 
fx = 2304.5479
fy = 2305.8757
cx = 1686.2379
cy = 1354.9849
camera_matrix = np.array([[fx, 0,  cx],
           [0, fy, cy],
           [0, 0, 1]], dtype=np.float32)
def convert2camera(worldCoord):
    x,y,z = worldCoord[0],worldCoord[1],worldCoord[2]
    return x * fx / z + cx, y * fy / z + cy
# Plot inline
#%matplotlib inline

x = []
y = []
img = cv2.imread(Input+'train_images/ID_8a6e65317.jpg')

for item in poses:
    coord = (float(item[4]),float(item[5]),float(item[6]))
    x.append(convert2camera(coord)[0])
    y.append(convert2camera(coord)[1])
   
plt.scatter(x,y, color='red', s=100);


plt.imshow(img)
plt.title('2D visualisation')
plt.show()
   
from math import sin, cos

# convert euler angle to rotation matrix
def euler_to_Rot(yaw, pitch, roll):
    Y = np.array([[cos(yaw), 0, sin(yaw)],
                  [0, 1, 0],
                  [-sin(yaw), 0, cos(yaw)]])
    P = np.array([[1, 0, 0],
                  [0, cos(pitch), -sin(pitch)],
                  [0, sin(pitch), cos(pitch)]])
    R = np.array([[cos(roll), -sin(roll), 0],
                  [sin(roll), cos(roll), 0],
                  [0, 0, 1]])
    return np.dot(Y, np.dot(P, R))
def draw_line(image, points):
    color = (255, 0, 0)
    cv2.line(image, tuple(points[0][:2]), tuple(points[3][:2]), color, 16)
    cv2.line(image, tuple(points[0][:2]), tuple(points[1][:2]), color, 16)
    cv2.line(image, tuple(points[1][:2]), tuple(points[2][:2]), color, 16)
    cv2.line(image, tuple(points[2][:2]), tuple(points[3][:2]), color, 16)
    return image


def draw_points(image, points):
    for (p_x, p_y, p_z) in points:
        cv2.circle(image, (p_x, p_y), int(1000 / p_z), (0, 255, 0), -1)
    return image
def visualize(img, coords):
    #Just the mean value of all type of car's dimension 
    x_l = 1.02
    y_l = 0.80
    z_l = 2.31
    
    img = img.copy()
    for point in coords:
        # Get values
        x, y, z = float(point[4]), float(point[5]), float(point[6])
        yaw, pitch, roll = -float(point[1]), -float(point[2]), -float(point[3])
        # Math
        Rt = np.eye(4)
        t = np.array([x, y, z])
        Rt[:3, 3] = t
        Rt[:3, :3] = euler_to_Rot(yaw, pitch, roll).T
        Rt = Rt[:3, :]
        P = np.array([[x_l, -y_l, -z_l, 1],
                      [x_l, -y_l, z_l, 1],
                      [-x_l, -y_l, z_l, 1],
                      [-x_l, -y_l, -z_l, 1],
                      [0, 0, 0, 1]]).T
        img_cor_points = np.dot(camera_matrix, np.dot(Rt, P))
        img_cor_points = img_cor_points.T
        img_cor_points[:, 0] /= img_cor_points[:, 2]
        img_cor_points[:, 1] /= img_cor_points[:, 2]
        img_cor_points = img_cor_points.astype(int)
        # Drawing
        img = draw_line(img, img_cor_points)
        img = draw_points(img, img_cor_points[-1:])
    
    return img
img = cv2.imread(Input+'train_images/ID_8a6e65317.jpg')

bbox = visualize(img,poses)

plt.imshow(bbox)
plt.title('3D visualisation')
plt.show()
#As we can see there're some inaccuracies in boxes
json_files = os.listdir(Input+'car_models_json')
print(len(json_files))
#Load the model
with open(Input+'car_models_json/fengtian-SUV-gai.json') as json_file:
    car_model_data = json.load(json_file)

for keys in enumerate(car_model_data):
    print(keys)
from mpl_toolkits.mplot3d import Axes3D

def plot_3d_car(model_json_file):
    with open(Input+f'car_models_json/{model_json_file}') as json_file:
        car_model_data = json.load(json_file)

    vertices = np.array(car_model_data['vertices'])
    faces = np.array(car_model_data['faces']) - 1
    car_type = car_model_data['car_type']
    x, y, z = vertices[:,0], vertices[:,2], -vertices[:,1]
    fig = plt.figure(figsize=(30, 10))
    ax = plt.axes(projection='3d')
    ax.plot_trisurf(x, y, faces, z,
                    cmap='viridis', edgecolor='none')
    ax.set_title(car_type)
    ax.view_init(30, 0)
    plt.show()
    fig = plt.figure(figsize=(30, 10))
    ax = plt.axes(projection='3d')
    ax.plot_trisurf(x, y, faces, z,
                    cmap='viridis', edgecolor='none')
    ax.set_title(car_type)
    ax.view_init(60, 0)
    plt.show()
    fig = plt.figure(figsize=(30, 10))
    ax = plt.axes(projection='3d')
    ax.plot_trisurf(x, y, faces, z,
                    cmap='viridis', edgecolor='none')
    ax.set_title(car_type)
    ax.view_init(-20, 180)
    plt.show()
    return
plot_3d_car('fengtian-SUV-gai.json')
mask_files = os.listdir(Input+'train_masks')
print('Number of mask images : ',len(mask_files))
train_images = os.listdir(Input+'train_images')
print('Number of train images : ',len(train_images))
mask = cv2.imread(Input+'train_masks/ID_8a6e65317.jpg')
image = cv2.imread(Input+'train_images/ID_8a6e65317.jpg')
plt.imshow(mask)
plt.title('Mask')
plt.show()

plt.imshow(image)
plt.title('Image')
plt.show()
fig, ax = plt.subplots(figsize=(20, 25))
plt.imshow(image)
plt.imshow(mask, cmap=plt.cm.viridis, interpolation='none', alpha=0.5)
plt.show()