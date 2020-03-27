
""" change working directory """
import os

if os.path.basename(os.getcwd()) == 'working':
    os.chdir('../input/tf-gan-code-20181007/transparent_latent_gan_kaggle_2018_1007/transparent_latent_gan_kaggle_2018_1007')
print('current working directory is {}'.format(os.getcwd()))

""" import packages """

import os
import glob
import sys
import numpy as np
import pickle
import tensorflow as tf
import PIL
import ipywidgets
import io

import src.tf_gan.generate_image as generate_image
import src.tf_gan.feature_axis as feature_axis
import src.tf_gan.feature_celeba_organize as feature_celeba_organize


""" load learnt feature axis directions """
path_feature_direction = './asset_results/pg_gan_celeba_feature_direction_40'

pathfile_feature_direction = glob.glob(os.path.join(path_feature_direction, 'feature_direction_*.pkl'))[-1]

with open(pathfile_feature_direction, 'rb') as f:
    feature_direction_name = pickle.load(f)

feature_direction = feature_direction_name['direction']
feature_name = feature_direction_name['name']
num_feature = feature_direction.shape[1]

feature_name = feature_celeba_organize.feature_name_celeba_rename
feature_direction = feature_direction_name['direction']* feature_celeba_organize.feature_reverse[None, :]


""" ========== start tf session and load GAN model ========== """

# path to model code and weight
path_pg_gan_code = './src/model/pggan'
path_model = './asset_model/karras2018iclr-celebahq-1024x1024.pkl'
sys.path.append(path_pg_gan_code)


""" create tf session """
yn_CPU_only = False

if yn_CPU_only:
    config = tf.ConfigProto(device_count = {'GPU': 0}, allow_soft_placement=True)
else:
    config = tf.ConfigProto(allow_soft_placement=True)
    config.gpu_options.allow_growth = True

sess = tf.InteractiveSession(config=config)

try:
    with open(path_model, 'rb') as file:
        G, D, Gs = pickle.load(file)
except FileNotFoundError:
    print('before running the code, download pre-trained model to project_root/asset_model/')
    raise

len_z = Gs.input_shapes[0][1]
z_sample = np.random.randn(len_z)
x_sample = generate_image.gen_single_img(z_sample, Gs=Gs)
""" ========== ipywigets gui interface ========== """

def img_to_bytes(x_sample):
    """ tool funcion to code image for using ipywidgets.widgets.Image plotting function """
    imgObj = PIL.Image.fromarray(x_sample)
    imgByteArr = io.BytesIO()
    imgObj.save(imgByteArr, format='PNG')
    imgBytes = imgByteArr.getvalue()
    return imgBytes

# a random sample of latent space noise
z_sample = np.random.randn(len_z)
# the generated image using this noise patter
x_sample = generate_image.gen_single_img(z=z_sample, Gs=Gs)

w_img = ipywidgets.widgets.Image(value=img_to_bytes(x_sample), fromat='png', 
                                 width=512, height=512,
                                 layout=ipywidgets.Layout(height='512px', width='512px')
                                )

class GuiCallback(object):
    """ call back functions for button click behaviour """
    counter = 0
    #     latents = z_sample
    def __init__(self):
        self.latents = z_sample
        self.feature_direction = feature_direction
        self.feature_lock_status = np.zeros(num_feature).astype('bool')
        self.feature_directoion_disentangled = feature_axis.disentangle_feature_axis_by_idx(
            self.feature_direction, idx_base=np.flatnonzero(self.feature_lock_status))

    def random_gen(self, event):
        self.latents = np.random.randn(len_z)
        self.update_img()

    def modify_along_feature(self, event, idx_feature, step_size=0.01):
        self.latents += self.feature_directoion_disentangled[:, idx_feature] * step_size
        self.update_img()

    def set_feature_lock(self, event, idx_feature, set_to=None):
        if set_to is None:
            self.feature_lock_status[idx_feature] = np.logical_not(self.feature_lock_status[idx_feature])
        else:
            self.feature_lock_status[idx_feature] = set_to
        self.feature_directoion_disentangled = feature_axis.disentangle_feature_axis_by_idx(
            self.feature_direction, idx_base=np.flatnonzero(self.feature_lock_status))
    
    def update_img(self):        
        x_sample = generate_image.gen_single_img(z=self.latents, Gs=Gs)
        x_byte = img_to_bytes(x_sample)
        w_img.value = x_byte

guicallback = GuiCallback()

step_size = 0.4
def create_button(idx_feature, width=128, height=40):
    """ function to built button groups for one feature """
    w_name_toggle = ipywidgets.widgets.ToggleButton(
        value=False, description=feature_name[idx_feature],
        tooltip='{}, Press down to lock this feature'.format(feature_name[idx_feature]),
        layout=ipywidgets.Layout(height='{:.0f}px'.format(height/2), 
                                 width='{:.0f}px'.format(width),
                                 margin='2px 2px 2px 2px')
    )
    w_neg = ipywidgets.widgets.Button(description='-',
                                      layout=ipywidgets.Layout(height='{:.0f}px'.format(height/2), 
                                                               width='{:.0f}px'.format(width/2),
                                                               margin='1px 1px 5px 1px'))
    w_pos = ipywidgets.widgets.Button(description='+',
                                      layout=ipywidgets.Layout(height='{:.0f}px'.format(height/2), 
                                                               width='{:.0f}px'.format(width/2),
                                                               margin='1px 1px 5px 1px'))
    
    w_name_toggle.observe(lambda event: 
                      guicallback.set_feature_lock(event, idx_feature))
    w_neg.on_click(lambda event: 
                     guicallback.modify_along_feature(event, idx_feature, step_size=-1 * step_size))
    w_pos.on_click(lambda event: 
                     guicallback.modify_along_feature(event, idx_feature, step_size=+1 * step_size))
    
    button_group = ipywidgets.VBox([w_name_toggle, ipywidgets.Box([w_neg, w_pos])],
                                  layout=ipywidgets.Layout(border='1px solid gray'))
    
    return button_group
  

list_buttons = []
for idx_feature in range(num_feature):
    list_buttons.append(create_button(idx_feature))

yn_button_select = True
def arrange_buttons(list_buttons, yn_button_select=True, ncol=4):
    num = len(list_buttons)
    if yn_button_select:
        feature_celeba_layout = feature_celeba_organize.feature_celeba_layout
        layout_all_buttons = ipywidgets.VBox([ipywidgets.Box([list_buttons[item] for item in row]) for row in feature_celeba_layout])
    else:
        layout_all_buttons = ipywidgets.VBox([ipywidgets.Box(list_buttons[i*ncol:(i+1)*ncol]) for i in range(num//ncol+int(num%ncol>0))])
    return layout_all_buttons
    

# w_button.on_click(on_button_clicked)
guicallback.update_img()
w_button_random = ipywidgets.widgets.Button(description='random face', button_style='success',
                                           layout=ipywidgets.Layout(height='40px', 
                                                               width='128px',
                                                               margin='1px 1px 5px 1px'))
w_button_random.on_click(guicallback.random_gen)

w_box = ipywidgets.Box([w_img, 
                         ipywidgets.VBox([w_button_random, 
                                         arrange_buttons(list_buttons, yn_button_select=True)])
                        ], layout=ipywidgets.Layout(height='628px', width='1024px')
                       )

print('INSTRUCTION: press +/- to adjust feature, toggle feature name to lock the feature')
display(w_box)
