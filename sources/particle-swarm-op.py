
# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.
import random


#function that models the problem
def sigmoid(x):
  return 1 / (1 +np.exp(-x))
def objective_function (position):
    if position>=0.5:
        cost=1
    else:
        cost=0
    return cost
def fitness_function(position_vector):
    toplam=0
    for i in position_vector:
        cost=objective_function(i)
        toplam=toplam+cost
    return toplam
#Some variables to calculate the velocity
W = 0.5
c1 = 0.5
c2 = 0.9
target = 1

n_iterations = 100
target_error = 0.001
n_particles = 1000

position_vector = np.array([(-1) ** (bool(random.getrandbits(1))) * random.random()for _ in range(n_particles)])
pbest_position = position_vector
pbest_fitness_value = np.array([0.0 for _ in range(n_particles)])
gbest_fitness_value = 0.0
gbest_position = np.array([0.0])

velocity_vector = ([np.array([0]) for _ in range(n_particles)])
iteration = 1
while iteration < n_iterations:
    
    for i in range(n_particles):
        fitness_cadidate = objective_function(position_vector[i])
        #print(fitness_cadidate, ' ', particle_position_vector[i])
        
        if(pbest_fitness_value[i] < fitness_cadidate):
            pbest_fitness_value[i] = fitness_cadidate
            pbest_position[i] = position_vector[i]

        if(gbest_fitness_value < fitness_cadidate):
            gbest_fitness_value = fitness_cadidate
            gbest_position = position_vector[i]
    print("gbest is ",gbest_position, "in iteration number ", iteration)
    if(target*n_particles-fitness_function(position_vector) < target_error):
        break
    
    for i in range(n_particles):
        new_velocity = (W*velocity_vector[i]) + (c1*random.random()) * (pbest_position[i] - position_vector[i]) + (c2*random.random()) * (gbest_position-position_vector[i])
        new_position = new_velocity + position_vector[i]
        new_position=sigmoid(new_position)
        position_vector[i] = new_position
        
    iteration = iteration + 1
print("The best position is ",fitness_function(position_vector), "in iteration number ", iteration)