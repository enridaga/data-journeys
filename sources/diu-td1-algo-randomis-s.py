
# -*- coding: utf-8 -*-
"""
Estimation de PI par Tirage aléatoire
=====================================
@author: Eric
La surface d'un cercle C de rayon R est : PI * R * R
La surface du carré inglobant C est     :  4 * R * R
Le nombre de tirages aléatoires tombant dans le cercle donne une estimation de PI / 4
"""
import random
import math

# import numpy as np
import matplotlib.pyplot as plt

nb_points = 200

for quadrant in [1, 2, 3, 4]:
    nb = 0
    tab_pi_val = []
    x_plot = []
    y_plot = []
    
    if quadrant % 2 == 0:
        coef_x = 1.0
    else:
        coef_x = -1.0
    
    if quadrant <= 2 :
        coef_y = 1.0
    else:
        coef_y = -1.0
        
        
    for i in range(1,nb_points):
        x = coef_x * random.random()
        y = coef_y * random.random()
        x_plot.append(x)
        y_plot.append(y)
        interieur = ((x*x+y*y)<= 1.0)
        nb = nb + interieur
        pi_val =  4.0 * nb / i
        tab_pi_val.append( pi_val )
    
    
    ax1 = plt.subplot(2,1,1,aspect='equal')
    # fig = plt.figure()
    # ax = plt.axes()
    
    plt.plot(x_plot, y_plot, '.')
    
#    ax1.set_aspect(1)
    plt.axis([-1,1,-1,1])
    plt.grid(linestyle='--')
    
    cercle = plt.Circle((0,0),1, color='pink')
    cercle.set_edgecolor('g')
    cercle.set_linewidth(1)
    
    ax1.add_artist(cercle)

    ax2 = plt.subplot(2,1,2)
    ax2.plot(tab_pi_val)
    plt.ylim(2.0,4.5)
    

ax2.axhline(y=math.pi,color='r')

plt.show()
Vous pouvez être critique sur la précision de cette méthode