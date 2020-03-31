
# To manipulate data
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import random

# Standard plotly imports
import plotly.plotly as py
import plotly.graph_objs as go
import plotly.tools as tls
from plotly.offline import iplot, init_notebook_mode
import cufflinks

# Using plotly + cufflinks in offline mode
init_notebook_mode(connected=True)
cufflinks.go_offline(connected=True)

from functools import partial
from hyperopt import fmin, hp, tpe, Trials, space_eval, STATUS_OK, STATUS_RUNNING
from lightgbm import LGBMRegressor
import lightgbm

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from tqdm import tqdm_notebook
import warnings
warnings.filterwarnings("ignore")
df_train = pd.read_csv('../input/champs-scalar-coupling/train.csv')
# df_pot_energy = pd.read_csv('../input/potential_energy.csv')
# df_mul_charges = pd.read_csv('../input/mulliken_charges.csv')
# df_scal_coup_contrib = pd.read_csv('../input/scalar_coupling_contributions.csv')
# df_magn_shield_tensor = pd.read_csv('../input/magnetic_shielding_tensors.csv')
# df_dipole_moment = pd.read_csv('../input/dipole_moments.csv')
df_structure = pd.read_csv('../input/champs-scalar-coupling/structures.csv')
df_test = pd.read_csv('../input/champs-scalar-coupling/test.csv')

print(df_train.shape)
print("")
print(df_train.head())
print("")
print(df_train.nunique())
print("")
print(df_train.columns)
print(df_structure.shape)
print("")
print(df_structure.head())
print("")
print(df_structure.nunique())
print("")
print(df_structure.columns)
df_train['scalar_coupling_constant'].sample(200000).iplot(kind='hist', title='Scalar Coupling Constant Distribuition',
                                                          xTitle='Scalar Coupling value', yTitle='Probability', histnorm='percent' )
plt.figure(figsize=(15,10))

g = plt.subplot(211)
g = sns.countplot(x='type', data=df_train, )
g.set_title("Count of Different Molecule Types", fontsize=22)
g.set_xlabel("Molecular Type Name", fontsize=18)
g.set_ylabel("Count Molecules in each Type", fontsize=18)

g1 = plt.subplot(212)
g1 = sns.boxplot(x='type', y='scalar_coupling_constant', data=df_train )
g1.set_title("Count of Different Molecule Types", fontsize=22)
g1.set_xlabel("Molecular Type Name", fontsize=18)
g1.set_ylabel("Scalar Coupling distribution", fontsize=18)

plt.subplots_adjust(wspace = 0.5, hspace = 0.5,top = 0.9)

plt.show()
plt.figure(figsize=(15,10))

g = plt.subplot(211)
g = sns.countplot(x='atom_index_0', data=df_train, color='darkblue' )
g.set_title("Count of Atom index 0", fontsize=22)
g.set_xlabel("index 0 Number", fontsize=18)
g.set_ylabel("Count", fontsize=18)

g1 = plt.subplot(212)
g1 = sns.countplot(x='atom_index_1',data=df_train, color='darkblue' )
g1.set_title("Count of Atom index 1", fontsize=22)
g1.set_xlabel("index 1 Number", fontsize=18)
g1.set_ylabel("Count", fontsize=18)

plt.subplots_adjust(wspace = 0.5, hspace = 0.5,top = 0.9)

plt.show()
cross_index = ['atom_index_0','atom_index_1'] #seting the desired 

cm = sns.light_palette("green", as_cmap=True)
pd.crosstab(df_train[cross_index[0]], df_train[cross_index[1]]).style.background_gradient(cmap = cm)
plt.figure(figsize=(17,12))

g = plt.subplot(211)
g = sns.boxenplot(x='atom_index_0', y='scalar_coupling_constant', data=df_train, color='darkred' )
g.set_title("Count of Atom index 0", fontsize=22)
g.set_xlabel("index 0 Number", fontsize=18)
g.set_ylabel("Count", fontsize=18)

g1 = plt.subplot(212)
g1 = sns.boxenplot(x='atom_index_1', y='scalar_coupling_constant', data=df_train, color='darkblue' )
g1.set_title("Count of Atom index 1", fontsize=22)
g1.set_xlabel("index 1 Number", fontsize=18)
g1.set_ylabel("Scalar Coupling distribution", fontsize=18)

plt.subplots_adjust(wspace = 0.5, hspace = 0.5,top = 0.9)

plt.show()
scalar_index_cross = ['atom_index_0', 'atom_index_1'] #seting the desired 

cm = sns.light_palette("green", as_cmap=True)
pd.crosstab(df_train[scalar_index_cross[0]], df_train[scalar_index_cross[1]], 
            values=df_train['scalar_coupling_constant'], aggfunc=['mean']).style.background_gradient(cmap = cm)
def map_atom_info(df, atom_idx):
    df = pd.merge(df, df_structure, how = 'left',
                  left_on  = ['molecule_name', f'atom_index_{atom_idx}'],
                  right_on = ['molecule_name',  'atom_index'])
    
    df = df.drop('atom_index', axis=1)
    df = df.rename(columns={'atom': f'atom_{atom_idx}',
                            'x': f'x_{atom_idx}',
                            'y': f'y_{atom_idx}',
                            'z': f'z_{atom_idx}'})
    return df

df_train = map_atom_info(df_train, 0)
df_train = map_atom_info(df_train, 1)

df_test = map_atom_info(df_test, 0)
df_test = map_atom_info(df_test, 1)
## This is a very performative way to compute the distances
train_p_0 = df_train[['x_0', 'y_0', 'z_0']].values
train_p_1 = df_train[['x_1', 'y_1', 'z_1']].values
test_p_0 = df_test[['x_0', 'y_0', 'z_0']].values
test_p_1 = df_test[['x_1', 'y_1', 'z_1']].values

## linalg.norm, explanation:
## This function is able to return one of eight different matrix norms, 
## or one of an infinite number of vector norms (described below),
## depending on the value of the ord parameter.
df_train['dist'] = np.linalg.norm(train_p_0 - train_p_1, axis=1)
df_test['dist'] = np.linalg.norm(test_p_0 - test_p_1, axis=1)

df_train['dist_x'] = (df_train['x_0'] - df_train['x_1']) ** 2
df_test['dist_x'] = (df_test['x_0'] - df_test['x_1']) ** 2
df_train['dist_y'] = (df_train['y_0'] - df_train['y_1']) ** 2
df_test['dist_y'] = (df_test['y_0'] - df_test['y_1']) ** 2
df_train['dist_z'] = (df_train['z_0'] - df_train['z_1']) ** 2
df_test['dist_z'] = (df_test['z_0'] - df_test['z_1']) ** 2
df_train['dist'].sample(200000).iplot(kind='hist', title='Scalar Coupling Constant Distribuition',
                                                          xTitle='Scalar Coupling value', yTitle='Probability', histnorm='percent' )
g = sns.FacetGrid(df_train, col="type", col_wrap=2, height=4, aspect=1.5)
g.map(sns.violinplot, "dist");
plt.subplots_adjust(top=0.9)
g.fig.suptitle('Violin Distribution of Dist by Each Type', fontsize=25)
plt.figure(figsize=(14,12))

g1 = plt.subplot(211)
g1 = sns.boxenplot(x='atom_index_0', y='dist', data=df_train, color='blue' )
g1.set_title("Distance Distribution by Atom index 0", fontsize=22)
g1.set_xlabel("Index 0 Number", fontsize=18)
g1.set_ylabel("Distance Distribution", fontsize=18)

g2 = plt.subplot(212)
g2 = sns.boxenplot(x='atom_index_1', y='dist', data=df_train, color='green' )
g2.set_title("Distance Distribution by Atom index 1", fontsize=22)
g2.set_xlabel("Index 1 Number", fontsize=18)
g2.set_ylabel("Distance Distribution", fontsize=18)

plt.subplots_adjust(wspace = 0.5, hspace = 0.3,top = 0.9)

plt.show()
## graph from:
### https://www.kaggle.com/artgor/molecular-properties-eda-and-models

import networkx as nx

fig, ax = plt.subplots(figsize = (20, 12))
for i, t in enumerate(df_train['type'].unique()):
    train_type = df_train.loc[df_train['type'] == t]
    G = nx.from_pandas_edgelist(train_type, 'atom_index_0', 'atom_index_1', ['dist'])
    plt.subplot(2, 4, i + 1);
    nx.draw(G, with_labels=True);
    plt.title(f'Graph for type {t}')
fig, ax = plt.subplots(figsize = (20, 12))
for i, t in enumerate(df_train['type'].unique()):
    train_type = df_train.loc[df_train['type'] == t]
    bad_atoms_0 = list(train_type['atom_index_0'].value_counts(normalize=True)[train_type['atom_index_0'].value_counts(normalize=True) < 0.01].index)
    bad_atoms_1 = list(train_type['atom_index_1'].value_counts(normalize=True)[train_type['atom_index_1'].value_counts(normalize=True) < 0.01].index)
    bad_atoms = list(set(bad_atoms_0 + bad_atoms_1))
    train_type = train_type.loc[(train_type['atom_index_0'].isin(bad_atoms_0) == False) & (train_type['atom_index_1'].isin(bad_atoms_0) == False)]
    G = nx.from_pandas_edgelist(train_type, 'atom_index_0', 'atom_index_1', ['scalar_coupling_constant'])
    plt.subplot(2, 4, i + 1);
    nx.draw(G, with_labels=True);
    plt.title(f'Graph for type {t}')
number_of_colors = df_structure.atom.value_counts().count() # total number of different collors that we will use

# Here I will generate a bunch of hexadecimal colors 
color = ["#"+''.join([random.choice('0123456789ABCDEF') for j in range(6)])
             for i in range(number_of_colors)]
df_structure['color'] = np.nan

for idx, col in enumerate(df_structure.atom.value_counts().index):
    listcol = ['#C15477', '#7ECF7B', '#4BDBBD', '#338340', '#F9E951']
    df_structure.loc[df_structure['atom'] == col, 'color'] = listcol[idx]
## Building a 
trace1 = go.Scatter3d(
    x=df_structure[df_structure['molecule_name'] == 'dsgdb9nsd_000003'].x,
    y=df_structure[df_structure['molecule_name'] == 'dsgdb9nsd_000003'].y,
    z=df_structure[df_structure['molecule_name'] == 'dsgdb9nsd_000003'].z, 
    hovertext=df_structure[df_structure['molecule_name'] == 'dsgdb9nsd_000003'].atom,
    mode='markers', name="3 Atoms Molecule",visible=False, 
    marker=dict(
        size=10,
        color=df_structure[df_structure['molecule_name'] == 'dsgdb9nsd_000003'].color,                
    )
) 

trace2 = go.Scatter3d(
    x=df_structure[df_structure['molecule_name'] == 'dsgdb9nsd_002116'].x,
    y=df_structure[df_structure['molecule_name'] == 'dsgdb9nsd_002116'].y,
    z=df_structure[df_structure['molecule_name'] == 'dsgdb9nsd_002116'].z, 
    hovertext=df_structure[df_structure['molecule_name'] == 'dsgdb9nsd_002116'].atom,
    mode='markers', name="8 Atoms Molecule",visible=True, 
    marker=dict(symbol='circle',
                size=6,
                color=df_structure[df_structure['molecule_name'] == 'dsgdb9nsd_002116'].color,
                colorscale='Viridis',
                line=dict(color='rgb(50,50,50)', width=0.5)
               ),
    hoverinfo='text'
)

trace3 = go.Scatter3d(
    x=df_structure[df_structure['molecule_name'] == 'dsgdb9nsd_051136'].x,
    y=df_structure[df_structure['molecule_name'] == 'dsgdb9nsd_051136'].y,
    z=df_structure[df_structure['molecule_name'] == 'dsgdb9nsd_051136'].z, 
    hovertext=df_structure[df_structure['molecule_name'] == 'dsgdb9nsd_051136'].atom,
    mode='markers', name="17 Atoms Molecule",visible=False, 
    marker=dict(
        size=10,
        color=df_structure[df_structure['molecule_name'] == 'dsgdb9nsd_002116'].color,             
    )
) 

trace4 = go.Scatter3d(
    x=df_structure[df_structure['molecule_name'] == 'dsgdb9nsd_088951'].x,
    y=df_structure[df_structure['molecule_name'] == 'dsgdb9nsd_088951'].y,
    z=df_structure[df_structure['molecule_name'] == 'dsgdb9nsd_088951'].z, 
    hovertext=df_structure[df_structure['molecule_name'] == 'dsgdb9nsd_088951'].atom,
    mode='markers', name="25 Atoms Molecule",visible=False, 
    marker=dict(
        size=10,
        color=df_structure[df_structure['molecule_name'] == 'dsgdb9nsd_002116'].color,             
    )
) 
data = [trace1, trace2, trace3, trace4]

updatemenus = list([
    dict(active=-1,
         showactive=True,
         buttons=list([  
            dict(
                label = '3 Atoms',
                 method = 'update',
                 args = [{'visible': [True, False, False, False]}, 
                     {'title': 'Molecule with 3 Atoms'}]),
             
             dict(
                  label = '8 Atoms',
                 method = 'update',
                 args = [{'visible': [False, True, False, False]},
                     {'title': 'Molecule with 8 Atoms'}]),

            dict(
                 label = '17 Atoms',
                 method = 'update',
                 args = [{'visible': [False, False, True, False]},
                     {'title': 'Molecule with 17 Atoms'}]),

            dict(
                 label = '25 Atoms',
                 method = 'update',
                 args = [{'visible': [False, False, False, True]},
                     {'title': 'Molecule with 25 Atoms'}])
        ]),
    )
])


layout = dict(title="The distance between atoms in some molecules <br>(Select from Dropdown)<br> Molecule of 8 Atoms", 
              showlegend=False,
              updatemenus=updatemenus)

fig = dict(data=data, layout=layout)

iplot(fig)
## Function to reduce the DF size
def reduce_mem_usage(df, verbose=True):
    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
    start_mem = df.memory_usage().sum() / 1024**2    
    for col in df.columns:
        col_type = df[col].dtypes
        if col_type in numerics:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)  
            else:
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)    
    end_mem = df.memory_usage().sum() / 1024**2
    if verbose: print('Mem. usage decreased to {:5.2f} Mb ({:.1f}% reduction)'.format(end_mem, 100 * (start_mem - end_mem) / start_mem))
    return df
df_train['type_0'] = df_train['type'].apply(lambda x: x[0])
df_test['type_0'] = df_test['type'].apply(lambda x: x[0])

## All feature engineering references I got on Artgor's Kernel:
## https://www.kaggle.com/artgor/brute-force-feature-engineering

def create_features(df):
    df['molecule_couples'] = df.groupby('molecule_name')['id'].transform('count')
    df['molecule_dist_mean'] = df.groupby('molecule_name')['dist'].transform('mean')
    df['molecule_dist_min'] = df.groupby('molecule_name')['dist'].transform('min')
    df['molecule_dist_max'] = df.groupby('molecule_name')['dist'].transform('max')
    df['atom_0_couples_count'] = df.groupby(['molecule_name', 'atom_index_0'])['id'].transform('count')
    df['atom_1_couples_count'] = df.groupby(['molecule_name', 'atom_index_1'])['id'].transform('count')
    df[f'molecule_atom_index_0_x_1_std'] = df.groupby(['molecule_name', 'atom_index_0'])['x_1'].transform('std')
    df[f'molecule_atom_index_0_y_1_mean'] = df.groupby(['molecule_name', 'atom_index_0'])['y_1'].transform('mean')
    df[f'molecule_atom_index_0_y_1_mean_diff'] = df[f'molecule_atom_index_0_y_1_mean'] - df['y_1']
    df[f'molecule_atom_index_0_y_1_mean_div'] = df[f'molecule_atom_index_0_y_1_mean'] / df['y_1']
    df[f'molecule_atom_index_0_y_1_max'] = df.groupby(['molecule_name', 'atom_index_0'])['y_1'].transform('max')
    df[f'molecule_atom_index_0_y_1_max_diff'] = df[f'molecule_atom_index_0_y_1_max'] - df['y_1']
    df[f'molecule_atom_index_0_y_1_std'] = df.groupby(['molecule_name', 'atom_index_0'])['y_1'].transform('std')
    df[f'molecule_atom_index_0_z_1_std'] = df.groupby(['molecule_name', 'atom_index_0'])['z_1'].transform('std')
    df[f'molecule_atom_index_0_dist_mean'] = df.groupby(['molecule_name', 'atom_index_0'])['dist'].transform('mean')
    df[f'molecule_atom_index_0_dist_mean_diff'] = df[f'molecule_atom_index_0_dist_mean'] - df['dist']
    df[f'molecule_atom_index_0_dist_mean_div'] = df[f'molecule_atom_index_0_dist_mean'] / df['dist']
    df[f'molecule_atom_index_0_dist_max'] = df.groupby(['molecule_name', 'atom_index_0'])['dist'].transform('max')
    df[f'molecule_atom_index_0_dist_max_diff'] = df[f'molecule_atom_index_0_dist_max'] - df['dist']
    df[f'molecule_atom_index_0_dist_max_div'] = df[f'molecule_atom_index_0_dist_max'] / df['dist']
    df[f'molecule_atom_index_0_dist_min'] = df.groupby(['molecule_name', 'atom_index_0'])['dist'].transform('min')
    df[f'molecule_atom_index_0_dist_min_diff'] = df[f'molecule_atom_index_0_dist_min'] - df['dist']
    df[f'molecule_atom_index_0_dist_min_div'] = df[f'molecule_atom_index_0_dist_min'] / df['dist']
    df[f'molecule_atom_index_0_dist_std'] = df.groupby(['molecule_name', 'atom_index_0'])['dist'].transform('std')
    df[f'molecule_atom_index_0_dist_std_diff'] = df[f'molecule_atom_index_0_dist_std'] - df['dist']
    df[f'molecule_atom_index_0_dist_std_div'] = df[f'molecule_atom_index_0_dist_std'] / df['dist']
    df[f'molecule_atom_index_1_dist_mean'] = df.groupby(['molecule_name', 'atom_index_1'])['dist'].transform('mean')
    df[f'molecule_atom_index_1_dist_mean_diff'] = df[f'molecule_atom_index_1_dist_mean'] - df['dist']
    df[f'molecule_atom_index_1_dist_mean_div'] = df[f'molecule_atom_index_1_dist_mean'] / df['dist']
    df[f'molecule_atom_index_1_dist_max'] = df.groupby(['molecule_name', 'atom_index_1'])['dist'].transform('max')
    df[f'molecule_atom_index_1_dist_max_diff'] = df[f'molecule_atom_index_1_dist_max'] - df['dist']
    df[f'molecule_atom_index_1_dist_max_div'] = df[f'molecule_atom_index_1_dist_max'] / df['dist']
    df[f'molecule_atom_index_1_dist_min'] = df.groupby(['molecule_name', 'atom_index_1'])['dist'].transform('min')
    df[f'molecule_atom_index_1_dist_min_diff'] = df[f'molecule_atom_index_1_dist_min'] - df['dist']
    df[f'molecule_atom_index_1_dist_min_div'] = df[f'molecule_atom_index_1_dist_min'] / df['dist']
    df[f'molecule_atom_index_1_dist_std'] = df.groupby(['molecule_name', 'atom_index_1'])['dist'].transform('std')
    df[f'molecule_atom_index_1_dist_std_diff'] = df[f'molecule_atom_index_1_dist_std'] - df['dist']
    df[f'molecule_atom_index_1_dist_std_div'] = df[f'molecule_atom_index_1_dist_std'] / df['dist']
    df[f'molecule_atom_1_dist_mean'] = df.groupby(['molecule_name', 'atom_1'])['dist'].transform('mean')
    df[f'molecule_atom_1_dist_min'] = df.groupby(['molecule_name', 'atom_1'])['dist'].transform('min')
    df[f'molecule_atom_1_dist_min_diff'] = df[f'molecule_atom_1_dist_min'] - df['dist']
    df[f'molecule_atom_1_dist_min_div'] = df[f'molecule_atom_1_dist_min'] / df['dist']
    df[f'molecule_atom_1_dist_std'] = df.groupby(['molecule_name', 'atom_1'])['dist'].transform('std')
    df[f'molecule_atom_1_dist_std_diff'] = df[f'molecule_atom_1_dist_std'] - df['dist']
    df[f'molecule_type_0_dist_std'] = df.groupby(['molecule_name', 'type_0'])['dist'].transform('std')
    df[f'molecule_type_0_dist_std_diff'] = df[f'molecule_type_0_dist_std'] - df['dist']
    df[f'molecule_type_dist_mean'] = df.groupby(['molecule_name', 'type'])['dist'].transform('mean')
    df[f'molecule_type_dist_mean_diff'] = df[f'molecule_type_dist_mean'] - df['dist']
    df[f'molecule_type_dist_mean_div'] = df[f'molecule_type_dist_mean'] / df['dist']
    df[f'molecule_type_dist_max'] = df.groupby(['molecule_name', 'type'])['dist'].transform('max')
    df[f'molecule_type_dist_min'] = df.groupby(['molecule_name', 'type'])['dist'].transform('min')
    df[f'molecule_type_dist_std'] = df.groupby(['molecule_name', 'type'])['dist'].transform('std')
    df[f'molecule_type_dist_std_diff'] = df[f'molecule_type_dist_std'] - df['dist']

    df = reduce_mem_usage(df)
    
    return df
### %time
df_train = create_features(df_train)
df_test = create_features(df_test)
good_columns = [
'molecule_atom_index_0_dist_min',
'molecule_atom_index_0_dist_max',
'molecule_atom_index_1_dist_min',
'molecule_atom_index_0_dist_mean',
'molecule_atom_index_0_dist_std',
'dist',
'molecule_atom_index_1_dist_std',
'molecule_atom_index_1_dist_max',
'molecule_atom_index_1_dist_mean',
'molecule_atom_index_0_dist_max_diff',
'molecule_atom_index_0_dist_max_div',
'molecule_atom_index_0_dist_std_diff',
'molecule_atom_index_0_dist_std_div',
'atom_0_couples_count',
'molecule_atom_index_0_dist_min_div',
'molecule_atom_index_1_dist_std_diff',
'molecule_atom_index_0_dist_mean_div',
'atom_1_couples_count',
'molecule_atom_index_0_dist_mean_diff',
'molecule_couples',
'atom_index_1',
'molecule_dist_mean',
'molecule_atom_index_1_dist_max_diff',
'molecule_atom_index_0_y_1_std',
'molecule_atom_index_1_dist_mean_diff',
'molecule_atom_index_1_dist_std_div',
'molecule_atom_index_1_dist_mean_div',
'molecule_atom_index_1_dist_min_diff',
'molecule_atom_index_1_dist_min_div',
'molecule_atom_index_1_dist_max_div',
'molecule_atom_index_0_z_1_std',
'y_0',
'molecule_type_dist_std_diff',
'molecule_atom_1_dist_min_diff',
'molecule_atom_index_0_x_1_std',
'molecule_dist_min',
'molecule_atom_index_0_dist_min_diff',
'molecule_atom_index_0_y_1_mean_diff',
'molecule_type_dist_min',
'molecule_atom_1_dist_min_div',
'atom_index_0',
'molecule_dist_max',
'molecule_atom_1_dist_std_diff',
'molecule_type_dist_max',
'molecule_atom_index_0_y_1_max_diff',
'molecule_type_0_dist_std_diff',
'molecule_type_dist_mean_diff',
'molecule_atom_1_dist_mean',
'molecule_atom_index_0_y_1_mean_div',
'molecule_type_dist_mean_div',
'type']
for f in ['atom_index_0', 'atom_index_1', 'atom_1', 'type_0', 'type']:
    if f in good_columns:
        lbl = LabelEncoder()
        lbl.fit(list(df_train[f].values) + list(df_test[f].values))
        df_train[f] = lbl.transform(list(df_train[f].values))
        df_test[f] = lbl.transform(list(df_test[f].values))
# Threshold for removing correlated variables
threshold = 0.95

# Absolute value correlation matrix
corr_matrix = df_train.corr().abs()

# Getting the upper triangle of correlations
upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(np.bool))
# Select columns with correlations above threshold
to_drop = [column for column in upper.columns if any(upper[column] > threshold)]

print('There are %d columns to remove.' % (len(to_drop)))
df_train = df_train.drop(columns = to_drop)
df_test = df_test.drop(columns = to_drop)

print('Training shape: ', df_train.shape)
print('Testing shape: ', df_test.shape)
# Split the 'features' and 'income' data into training and testing sets
X_train, X_val, y_train, y_val = train_test_split(df_train.drop('scalar_coupling_constant',
                                                                axis=1), 
                                                  df_train['scalar_coupling_constant'], 
                                                  test_size = 0.10, 
                                                  random_state = 0)

df_val = pd.DataFrame({"type":X_val["type"]})
df_val['scalar_coupling_constant'] = y_val

X_train = X_train.drop(['id', 'atom_1','atom_0', 'type','molecule_name'], axis=1).values
y_train = y_train.values

X_val = X_val.drop(['id', 'atom_1','atom_0', 'type','molecule_name'], axis=1).values
y_val = y_val.values

X_test = df_test.drop(['id', 'atom_1','atom_0', 'type','molecule_name'], axis=1).values

print("Training set has {} samples.".format(X_train.shape[0]))
print("Validation set has {} samples.".format(X_val.shape[0]))
# Define searched space
hyper_space = {'objective': 'regression',
               'metric':'mae',
               'boosting':'gbdt',
               #'n_estimators': hp.choice('n_estimators', [25, 40, 50, 75, 100, 250, 500]),
               'max_depth':  hp.choice('max_depth', [5, 8, 10, 12, 15]),
               'num_leaves': hp.choice('num_leaves', [100, 250, 500, 650, 750, 1000,1300]),
               'subsample': hp.choice('subsample', [.3, .5, .7, .8, 1]),
               'colsample_bytree': hp.choice('colsample_bytree', [ .6, .7, .8, .9, 1]),
               'learning_rate': hp.choice('learning_rate', [.1, .2, .3]),
               'reg_alpha': hp.choice('reg_alpha', [.1, .2, .3, .4, .5, .6]),
               'reg_lambda':  hp.choice('reg_lambda', [.1, .2, .3, .4, .5, .6]),               
               'min_child_samples': hp.choice('min_child_samples', [20, 45, 70, 100])}
# Defining the Metric to score our optimizer
def metric(df, preds):
    df['diff'] = (df['scalar_coupling_constant'] - preds).abs()
    return np.log(df.groupby('type')['diff'].mean().map(lambda x: max(x, 1e-9))).mean()
lgtrain = lightgbm.Dataset(X_train, label=y_train)
lgval = lightgbm.Dataset(X_val, label=y_val)

def evaluate_metric(params):
    
    model_lgb = lightgbm.train(params, lgtrain, 500, 
                          valid_sets=[lgtrain, lgval], early_stopping_rounds=20, 
                          verbose_eval=500)

    pred = model_lgb.predict(X_val)

    score = metric(df_val, pred)
    
    print(score)
 
    return {
        'loss': score,
        'status': STATUS_OK,
        'stats_running': STATUS_RUNNING
    }
# Trail
trials = Trials()

# Set algoritm parameters
algo = partial(tpe.suggest, 
               n_startup_jobs=-1)

# Seting the number of Evals
MAX_EVALS= 15

# Fit Tree Parzen Estimator
best_vals = fmin(evaluate_metric, space=hyper_space, verbose=1,
                 algo=algo, max_evals=MAX_EVALS, trials=trials)

# Print best parameters
best_params = space_eval(hyper_space, best_vals)
print("BEST PARAMETERS: " + str(best_params))
model_lgb = lightgbm.train(best_params, lgtrain, 4000, 
                      valid_sets=[lgtrain, lgval], early_stopping_rounds=30, 
                      verbose_eval=500)

lgb_pred = model_lgb.predict(X_test)

df_test['scalar_coupling_constant'] = lgb_pred

df_test[['id', 'scalar_coupling_constant']].to_csv("molecular_struct_sub.csv", index=False)