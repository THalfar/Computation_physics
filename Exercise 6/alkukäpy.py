#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 13 14:52:50 2020

@author: halfar
"""

import numpy as np
from scipy.sparse import diags
import matplotlib.pyplot as plt
from scipy.integrate import simps


N = 23
grid = np.linspace(0,1,N)

h = grid[1] - grid[0]
A = np.zeros((N,N))
b = np.zeros((N,))

diag = np.repeat(2/h, N)
neig = np.repeat(-1/h, N)
diago = [diag, neig, neig]

A = diags(diago, [0, -1, 1]).toarray()
A = A[1:-1, 1:-1]

for i in range(1, b.shape[0]-1):
    b[i] = np.pi/h * (grid[i-1] + grid[i+1] - 2*grid[i])*np.cos(np.pi*grid[i]) \
        + 1/h * (2*np.sin(np.pi*grid[i]) - np.sin(np.pi*grid[i-1]) - np.sin(np.pi*grid[i+1]))
        
b = b[1:-1]
a = np.linalg.solve(A,b)      

# fingrid solution is a values because hat function is 1 at these points
# so a*u_i = a
fin_sol = np.zeros((N,))
fin_sol[1:-1] = a
# known solution
val_sol = np.sin(np.pi * grid)

fig = plt.figure(figsize = (15,10))
plt.plot(grid, val_sol, 'r', label = "Right values")
plt.plot(grid, fin_sol, 'bo', label = "Fin values")
plt.legend()
plt.title("Plot right solution vs. finite element solution")
plt.xlabel("Grid value")
plt.ylabel("$\Phi$")


#%%
N1 = 10
N2 = 23
N = N1+N2

grid = np.linspace(0,0.5,N1)
grid = np.concatenate((grid, np.linspace(0.6,1,N2)))

def hattu(x, idx, h): # korjaa gridi tähän
    
    arvo = 0.0
    
    if x >= (idx-1)*h and x < (idx)*h: # TODO kysy! miten rajat?! 
        arvo = (x - (idx-1)*h) / h
        
    elif x > (idx)*h and x <= (idx+1)*h:
        arvo = ((idx+1)*h - x) / h
        
    return arvo

def Dhattu(x, idx, h):
       
    if x >= (idx-1)*h and x < (idx)*h:
        return 1 / h
        
    elif x >= (idx)*h and x < (idx+1)*h:
        return -1 / h
        
    return 0.0
    
N_tiheys = 1000
tiheys = np.linspace(0,1,N_tiheys)
A = np.zeros((N,N))

for i in range(1, N-1):
    
    for j in range(1,N-1):
        
        h = grid[j] - grid[j-1]
        
        ueka = np.zeros(len(tiheys))
        for ii in range(len(tiheys)):
            ueka[ii] = Dhattu(tiheys[ii], i, h)

        
        utoka = np.zeros(len(tiheys))    
        for ii in range(len(tiheys)):
            utoka[ii] = Dhattu(tiheys[ii], j, h)
        
        tulo = ueka*utoka    
        A[i,j] = simps(tulo, tiheys)
            
        



    
    

    
    