
import matplotlib.pyplot as plt
import numpy as np

# Definition of the cutoff function
def fc(Rij, Rc):
    y_1 = 0.5*(np.cos(np.pi*Rij[Rij<=Rc]/Rc)+1)
    y_2 = Rij[Rij>Rc]*0
    y = np.concatenate((y_1,y_2))
    return y

# Define x
x = np.arange(0, 11, 0.01)

# Plot the function with different cutoff radii
Rc_range = [5.0, 6.0, 7.0, 8.0, 9.0, 10.0]

fig = plt.figure(figsize=(10,8))

for Rc in Rc_range:
    plt.plot(x, fc(x,Rc), label=f'Rc={Rc}')

plt.axis([0, 11, 0, 1.1])
plt.xticks(range(11))
plt.yticks(np.arange(0, 1.1, 0.1))
plt.xlabel('Distance from center of atom')
plt.ylabel('Cutoff Function')
plt.legend()
    
plt.show()
import ase.visualize
from ase.build import molecule

# Create the methanol molecule
methanol = molecule('CH3OH')

ase.visualize.view(methanol, viewer="x3d")
max(methanol.get_all_distances().flatten())
Rc = 3.0
methanol.get_chemical_symbols()
all_dist = methanol.get_all_distances()
print('This is the distance matrix:')
print(all_dist)

dist_from_C = all_dist[0]
print('')
print('Distances from carbon atom to every other atoms:\n', dist_from_C)
G1_C = fc(dist_from_C, Rc).sum()
print('G1 for the carbon atom:', G1_C)
# Number of atoms in the molecules
natom = len(methanol.get_chemical_symbols())

# Definition of a vectorized cutoff function
def fc_vect(Rij, Rc):
    return np.where(Rij <= Rc, 0.5 * (np.cos(np.pi * Rij/Rc)+1), 0).sum(1)

# Calculate G1
G1 = fc_vect(all_dist, Rc)

print(G1)
# Define the G2 function
def get_G2(Rij, eta, Rs):
     return np.exp(-eta*(Rij-Rs)**2) * fc(Rij, Rc)
    
# Set a list of six eta/Rs tuples
p = [(0.4, 0.2),(0.4, 0.5),(0.4, 1.0),(0.5, 2.0),(0.5, 3.0),(0.5, 4.0)]

# Compute the six G2 corresponding to the six eta/Rs tuples
G2 = np.zeros((natom, len(p)))
for i in range(natom):
    for j, (eta, Rs) in enumerate(p):
        G2[i,j] =  get_G2(all_dist[i], eta, Rs).sum()
    
print(G2)
import pandas as pd
import borisdee_kaggle_functions as bd

# Load all relevant files
raw_struct = pd.read_csv('../input/acsf-up-to-g4/structures_with_g4.csv')
raw_charges = pd.read_csv('../input/champs-scalar-coupling/mulliken_charges.csv')
raw_test = pd.read_csv('../input/champs-scalar-coupling/test.csv')

# We need to free a bit of memory for Kaggle servers. Kudos to artgor for this function.
raw_struct = bd.reduce_mem_usage(raw_struct)
raw_charges = bd.reduce_mem_usage(raw_charges)
raw_test = bd.reduce_mem_usage(raw_test)
# Create train and test sets from the structure file containing ACSF
raw_train = raw_struct[raw_struct['molecule_name'].isin(raw_charges['molecule_name'].unique())]
raw_train.reset_index(drop=True, inplace=True)
test = raw_struct[raw_struct['molecule_name'].isin(raw_test['molecule_name'].unique())]
display(raw_train.head(), test.head())
# Drop useless columns
columns_to_drop = ['Unnamed: 0', 'molecule_name', 'atom_index', 'atom', 'x', 'y', 'z']
raw_train = raw_train.drop(columns_to_drop, axis=1)
test = test.drop(columns_to_drop, axis=1)

# Add Mulliken charges to training set
raw_train['mulliken_charge'] = raw_charges['mulliken_charge']

# Create train and cv sets
train = raw_train.sample(frac=0.80, random_state=2019)
cv = raw_train.drop(train.index)

print('Shape of train set:', train.shape)
print('Shape of cv set:', cv.shape)
# Hypothesis Testing
df1 = pd.DataFrame(train['mulliken_charge'])
df2 = pd.DataFrame(cv['mulliken_charge'])
bd.check_statistics(df1,df2)
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.metrics import mean_absolute_error

target = 'mulliken_charge'

# Test set 
X_test = test

# Train set
X_train = train.drop(target, axis=1)    
y_train = train[target]

# CV set
X_cv = cv.drop(target, axis=1)
y_cv = cv[target]
  
# Extra Tree
reg = ExtraTreesRegressor(n_estimators=8, max_depth=20, n_jobs=4)
reg.fit(X_train, y_train)
pred_train = reg.predict(X_train)
pred_cv = reg.predict(X_cv)
pred_test = reg.predict(X_test)

print('MAE on train set: %.2E.' %mean_absolute_error(y_train, pred_train)) 
print('MAE on cv set: %.2E.' %mean_absolute_error(y_cv, pred_cv))
print('')

# Plotiplot
plt.plot(y_cv,pred_cv,'o')
plt.plot([-1,1],[-1,1]) # perfect fit line
plt.xlabel('True')
plt.ylabel('Predicted')
plt.show()
tmp = raw_struct[raw_struct['molecule_name'].isin(raw_test['molecule_name'].unique())]
mulliken_charges_test=pd.DataFrame()
mulliken_charges_test['molecule_name'] = tmp['molecule_name']
mulliken_charges_test['atom_index'] = tmp['atom_index']
mulliken_charges_test['mulliken_charge'] = pred_test
mulliken_charges_test[mulliken_charges_test['molecule_name'] == 'dsgdb9nsd_000004']
# Need to reload the initial structure file for my function to work properly...
struct = pd.read_csv('../input/champs-scalar-coupling/structures.csv')

bd.view('dsgdb9nsd_000004', struct)
mulliken_charges_test.to_csv('mulliken_charges_test_set.csv', index=False)