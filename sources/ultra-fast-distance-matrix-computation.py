
# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from tqdm import tqdm_notebook

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
structures = pd.read_csv('../input/structures.csv')

structures
structures_idx = structures.set_index('molecule_name')

def get_dist_matrix(df_structures_idx, molecule):
    df_temp = df_structures_idx.loc[molecule]
    locs = df_temp[['x','y','z']].values
    num_atoms = len(locs)
    loc_tile = np.tile(locs.T, (num_atoms,1,1))
    dist_mat = np.sqrt((loc_tile - loc_tile.T)**2).sum(axis=1)
    return dist_mat
for molecule in tqdm_notebook(structures.molecule_name.unique()):
    get_dist_matrix(structures_idx, molecule)
xyz = structures[['x','y','z']].values

ss = structures.groupby('molecule_name').size()
ss = ss.cumsum()
ss
ssx = np.zeros(len(ss) + 1, 'int')
ssx[1:] = ss
ssx
molecule_id = 20
print(ss.index[molecule_id])
start_molecule = ssx[molecule_id]
end_molecule = ssx[molecule_id+1]
xyz[start_molecule:end_molecule]
structures_idx.loc['dsgdb9nsd_000022'][['x', 'y', 'z']].values
def get_fast_dist_matrix(xyz, ssx, molecule_id):
    start_molecule, end_molecule = ssx[molecule_id], ssx[molecule_id+1]
    locs = xyz[start_molecule:end_molecule]    
    num_atoms = end_molecule - start_molecule
    loc_tile = np.tile(locs.T, (num_atoms,1,1))
    dist_mat = np.sqrt((loc_tile - loc_tile.T)**2).sum(axis=1)
    return dist_mat
molecule_id = 2
molecule = ss.index[molecule_id]
print(molecule)
get_fast_dist_matrix(xyz, ssx, molecule_id)
get_dist_matrix(structures_idx, molecule)
for molecule_id in tqdm_notebook(range(structures.molecule_name.nunique())):
    get_fast_dist_matrix(xyz, ssx, molecule_id)
def ultra_fast_dist_matrices(xyz, ssx):
    for molecule_id in range(structures.molecule_name.nunique()):
        get_fast_dist_matrix(xyz, ssx, molecule_id)
### time ultra_fast_dist_matrices(xyz, ssx)
def sofast_dist(xyz, ssx, molecule_id):
    start_molecule, end_molecule = ssx[molecule_id], ssx[molecule_id+1]
    locs = xyz[start_molecule:end_molecule]     
    d=locs[:,None,:]-locs
    return np.sqrt(np.einsum('ijk,ijk->ij',d,d))

def sofast_dist_matrices(xyz, ssx):
    for molecule_id in range(structures.molecule_name.nunique()):
        sofast_dist(xyz, ssx, molecule_id)
### time sofast_dist_matrices(xyz, ssx)
from numba import jit
from math import sqrt

@jit
def numba_dist_matrix(xyz, ssx, molecule_id):
    start_molecule, end_molecule = ssx[molecule_id], ssx[molecule_id+1]
    locs = xyz[start_molecule:end_molecule]     
   # return locs
    num_atoms = end_molecule - start_molecule
    dmat = np.zeros((num_atoms, num_atoms))
    for i in range(num_atoms):
        for j in range(i+1, num_atoms):
            d = sqrt((locs[i,0] - locs[j,0])**2 + (locs[i,1] - locs[j,1])**2 + (locs[i,2] - locs[j,2])**2)
            dmat[i,j] = d
            dmat[j,i] = d
    return dmat

def numba_dist_matrices(xyz, ssx):
    for molecule_id in range(structures.molecule_name.nunique()):
        numba_dist_matrix(xyz, ssx, molecule_id)
molecule_id = 2
molecule = ss.index[molecule_id]
print(molecule)
numba_dist_matrix(xyz, ssx, molecule_id)
sofast_dist(xyz, ssx, molecule_id)
### time numba_dist_matrices(xyz, ssx)
from scipy.spatial.distance import pdist, squareform

def scipy_dist_matrix(xyz, ssx, molecule_id):
    start_molecule, end_molecule = ssx[molecule_id], ssx[molecule_id+1]
    locs = xyz[start_molecule:end_molecule]     
    cmat = pdist(locs)
    dmat = squareform(cmat, force='tomatrix')
    return dmat
scipy_dist_matrix(xyz, ssx, molecule_id)
def scipy_dist_matrices(xyz, ssx):
    for molecule_id in range(structures.molecule_name.nunique()):
        scipy_dist_matrix(xyz, ssx, molecule_id)
### time scipy_dist_matrices(xyz, ssx)
epsilon = 1e-5

def get_dist_matrix_assert(df_structures_idx, molecule):
    df_temp = df_structures_idx.loc[molecule]
    locs = df_temp[['x','y','z']].values
    num_atoms = len(locs)
    loc_tile = np.tile(locs.T, (num_atoms,1,1))
    dist_mat = np.sqrt((loc_tile - loc_tile.T)**2).sum(axis=1)
    assert np.abs(dist_mat[0,1] - np.linalg.norm(locs[0] - locs[1])) < epsilon
    return dist_mat

for molecule in tqdm_notebook(structures.molecule_name.unique()[660:]):
    try:
        get_dist_matrix_assert(structures_idx, molecule)
    except: 
        print('assertion error on', molecule)
        break
        


@jit
def numba_dist_matrix_ssert(xyz, ssx, molecule_id):
    start_molecule, end_molecule = ssx[molecule_id], ssx[molecule_id+1]
    locs = xyz[start_molecule:end_molecule]     
   # return locs
    num_atoms = end_molecule - start_molecule
    dmat = np.zeros((num_atoms, num_atoms))
    for i in range(num_atoms):
        for j in range(i+1, num_atoms):
            d = sqrt((locs[i,0] - locs[j,0])**2 + (locs[i,1] - locs[j,1])**2 + (locs[i,2] - locs[j,2])**2)
            dmat[i,j] = d
            dmat[j,i] = d
    assert np.abs(dmat[0,1] - np.linalg.norm(locs[0] - locs[1])) < epsilon
    return dmat

for molecule_id in range(structures.molecule_name.nunique()):
    numba_dist_matrix_ssert(xyz, ssx, molecule_id)
