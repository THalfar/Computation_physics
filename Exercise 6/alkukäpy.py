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

N = 101 # jaollinen 2

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

def hattu(x, idx, h):
    
    arvo = 0.0
    
    if x > (idx-1)*h and x <= (idx)*h:
        arvo = (x - (idx-1)*h) / h
        
    elif x > (idx)*h and x <= (idx+1)*h:
        arvo = ((idx+1)*h - x) / h
        
    return arvo

def Dhattu(x, idx, h):
    
    
    
    if x > (idx-1)*h and x <= (idx)*h:
        return 1 / h
        
    elif x > (idx)*h and x <= (idx+1)*h:
        return -1 / h
        
    return 0.0
    
    

papa = []
tiheys = np.linspace(0,1,400)
for i in range(len(tiheys)):
    papa.append(Dhattu(tiheys[i], 1, h))

papa = np.array(papa)
    
papa2 = []
for i in range(len(tiheys)):
    papa2.append(Dhattu(tiheys[i], 10, h))
papa2 = np.array(papa2)    
    
tata = papa2*papa




    
    

    
    