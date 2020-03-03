#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar  3 15:39:44 2020

@author: halfar
"""

import numpy as np
from scipy.integrate import simps
import matplotlib.pyplot as plt


def toAlfa(basis, grid = 41):
    
    det = np.linalg.det(basis) # assuming basis is full i.e. rank is same as dimensio
    
    alpha_x = np.linspace(0,1,grid)
    alpha_y = np.linspace(0,1,grid)
    
    alpha = np.array([alpha_x, alpha_y])        
    A = basis.dot(alpha)
    
    xx = A[0,:]
    yy = A[1,:]
    
    integrande = np.zeros((len(xx), len(yy)))
    
    for i in range(len(xx)):
        for j in range((len(yy))):
            
            integrande[i,j] = (xx[i] + yy[j]) * np.exp(-0.5*np.sqrt(xx[i]**2 + yy[j]**2))
    
    integrande *= det #wronskin determinantti lieneepi
    
    return integrande, alpha_x, alpha_y
   
        
def integration(integrande, alpha_x, alpha_y):
    
    dy = np.zeros(len(alpha_y))
        
    for i in range(len(alpha_y)):
        dy[i] = simps(integrande[i], alpha_x)
        
    inte = simps(dy, alpha_y)

    return inte    


def plotting(a1, a2):
    
    x = np.linspace(-0.1,2,41)
    y = np.linspace(-0.1,1.25,41)
    
    integrande = np.zeros((41,41))
    
    for i in range(41):
        for j in range(41):
            integrande[i,j] = (x[i] + y[j]) * np.exp(-0.5*np.sqrt(x[i]**2 + y[j]**2))
            
    fig = plt.figure(figsize = (13,13))
    ax = fig.add_subplot(111)
    CF = ax.contourf(x, y, integrande)
    fig.colorbar(CF, cmap = 'plasma')
    # Plot the area omega using basis vectors
    ax.plot([0,a1[0]],[0, a1[1]], 'r')
    ax.plot([0, a2[0]], [0, a2[1]], 'r'  )        
    ax.plot([a2[0], a2[0] + a1[0]],  [a2[1], a2[1]], 'r')
    ax.plot([a1[0], a1[0] + a2[0]],   [0, a2[1]], 'r')
    
    ax.set_title("Function value")
    ax.set_xlabel("X-axis")
    ax.set_ylabel("Y-axis")
    

    
        
  

    

def main():
    a1 = [1.2, 0]
    a2 = [0.6, 1]
   
    A = np.array([a1, a2]) # basis
    
    integrande, alpha_x, alpha_y = toAlfa(A)
    integral = integration(integrande, alpha_x, alpha_y)
    print(integral)
    
    exact_value = 0.925713

    plotting(a1,a2)
if __name__=="__main__":
    main()
        
        
    
