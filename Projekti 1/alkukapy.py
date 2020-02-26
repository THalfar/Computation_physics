#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 25 12:56:39 2020

@author: fobia
"""
mu = 1.25663706212e-6

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import simps
from mpl_toolkits.mplot3d import Axes3D
from numba import jit
import time

x = np.linspace(-3, 3 , 11)
y = np.linspace(-3, 3 , 11)
z = np.linspace(-3, 3 , 11)

X, Y, Z = np.meshgrid(x, y, z)

Bx = np.zeros_like(X)
By = np.zeros_like(X)
Bz = np.zeros_like(X)

maara = 123
dt = 2*np.pi / maara

sade = 2
kokomaara = maara*3

varaukset = np.zeros((kokomaara, 3))

for i in range(maara):    
    varaukset[i,:] = (-2,sade*np.sin(dt*i),sade*np.cos(dt*i))

fig = plt.figure(figsize  = (13,13))
ax = fig.add_subplot(111, projection='3d')
ax.plot(varaukset[:maara,0], varaukset[:maara,1], varaukset[:maara,2], 'r') 
ax.plot((varaukset[0,0], varaukset[maara-1,0]) , (varaukset[maara-1,1], varaukset[0,1]), 
        (varaukset[maara-1,2], varaukset[0,2]), 'r') 

for i in range(maara, maara*2):    
    varaukset[i,:] = (2, sade*np.sin(dt*(i-maara)),sade*np.cos(dt*(i-maara)))    
    
ax.plot(varaukset[maara:maara*2,0], varaukset[maara:maara*2,1], varaukset[maara:maara*2,2], 'r') 
ax.plot((varaukset[maara,0], varaukset[maara*2-1,0]) , (varaukset[maara*2-1,1], varaukset[maara,1]), 
        (varaukset[maara*2-1,2], varaukset[maara,2]), 'r') 

for i in range(maara*2, maara*3):    
    varaukset[i,:] = (3*np.sin(dt*(i-maara*2)),3*np.cos(dt*(i-maara*2)), -3)    
    
ax.plot(varaukset[maara*2:,0], varaukset[maara*2:,1], varaukset[maara*2:,2], 'r') 
ax.plot((varaukset[-2,0], varaukset[-1,0]) , (varaukset[-1,1], varaukset[-2,1]), 
        (varaukset[-1,2], varaukset[-2,2]), 'r') 


# ax.plot(varaukset[maara:,0], varaukset[maara:,1], varaukset[maara:,2])    




pikkunuolet = np.zeros((kokomaara, 3))

#TODO muista se viimeinen kun teet tästä funktion!
for i in range(0, kokomaara-1):
        
    pikkunuolet[i,0] = varaukset[i+1,0] - varaukset[i,0]
    pikkunuolet[i,1] = varaukset[i+1,1] - varaukset[i,1]
    pikkunuolet[i,2] = varaukset[i+1,2] - varaukset[i,2]
# viimeiseen nuoleen tarvitaan eka koordi  
pikkunuolet[maara-1,0] = varaukset[0,0] - varaukset[maara-1,0]
pikkunuolet[maara-1,1] = varaukset[0,1] - varaukset[maara-1,1]
pikkunuolet[maara-1,2] = varaukset[0,2] - varaukset[maara-1,2]

# @jit  
def laskeB(X,Y,Z, varaukset, pikkunuolet):  
    
    for i in range(X.shape[0]):
    
        for j in range(X.shape[1]):
    
            for k in range(X.shape[2]):
                
                for q in range(len(varaukset)):
                    
                    r = varaukset[q,:] - (X[i,j,k], Y[i, j, k], Z[i,j,k]) 
                    dl = pikkunuolet[q,:]
                    B = np.cross(dl, r) / np.linalg.norm(r)**3
                    Bx[i,j,k] += B[0]
                    By[i,j,k] += B[1]
                    Bz[i,j,k] += B[2]
                    
    return Bx / (4*np.pi), By / (4*np.pi), Bz / (4*np.pi)
                

start_time = time.time()
Bx, By, Bz = laskeB(X,Y,Z,varaukset, pikkunuolet)
print ("Magworld ", time.time() - start_time, "s to run")

ax.quiver(X,Y,Z,Bx,By,Bz)
    
    