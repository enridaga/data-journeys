
import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns
# Load training data

train = pd.read_csv('../input/train.csv')

# and remove nan values for now

train = train[train['ps_reg_03'] != -1]
train['ps_reg_03_int'] = train['ps_reg_03'].apply(lambda x: (40*x)**2)

train['ps_reg_03_int'].head(10)
# Actually convert ps_reg_03_int to integer

train['ps_reg_03_int'] = train['ps_reg_03_int'].apply(np.round).apply(int)
print("Unique values of ps_reg_03: ", len(train['ps_reg_03'].unique()))

print("Number of integer categories: ", len(train['ps_reg_03_int'].unique()))
fig, ax = plt.subplots(1,2,figsize=(10,5))

for i in range(2): ax[i].set_yscale('log') #Set y-axis to log-scale

train['ps_reg_03'].hist(ax=ax[0])

ax[0].set_xlabel('ps_reg_03')

train['ps_reg_03_int'].hist(ax=ax[1])

ax[1].set_xlabel('Integer')
print("Integer: ", 27*76+8)

print("ps_reg_03: ", np.sqrt(27*76+8)/40)
def recon(reg):

    integer = int(np.round((40*reg)**2)) # gives 2060 for our example

    for f in range(28):

        if (integer - f) % 27 == 0:

            F = f

    M = (integer - F)//27

    return F, M



# Using the above example to test

ps_reg_03_example = 1.13468057179

print("Federative Unit (F): ", recon(ps_reg_03_example)[0])

print("Municipality (M): ", recon(ps_reg_03_example)[1])
train['ps_reg_F'] = train['ps_reg_03'].apply(lambda x: recon(x)[0])

train['ps_reg_M'] = train['ps_reg_03'].apply(lambda x: recon(x)[1])

print(train[['ps_reg_03', 'ps_reg_F', 'ps_reg_M']].head(10))
fig, ax = plt.subplots(1,2,figsize=(10,5))

ax[0].set_yscale('log') #Set y-axis to log-scale only for M

train['ps_reg_M'].hist(ax=ax[0])

ax[0].set_xlabel('Municipality (M)')

train['ps_reg_F'].hist(ax=ax[1], bins=27)

ax[1].set_xlabel('Federative Unit (F)')