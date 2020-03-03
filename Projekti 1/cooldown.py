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
    """
    Transfer function to alpha space using given basis and calculate
    function values in this space

    Parameters
    ----------
    basis : np.array
        Basis vectors in array 
    grid : int, optional
        Gridsize. The default is 41.

    Returns
    -------
    integrande : np.array
        Integrand values at alpha space
    alpha_x : np.array
        First axis grid values in alphaspace
    alpha_y : np.array
        Second axis grid values in alphaspace

    """
    
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
    """
    Integrates in 2D over alphaspace

    Parameters
    ----------
    integrande : np.array
        Values of function at different space points
    alpha_x : np.array
          First alpha space vector 
    alpha_y : np.array
          Second alpha space vector

    Returns
    -------
    inte : float
        Value of integral 

    """
    
    dy = np.zeros(len(alpha_y))
        
    for i in range(len(alpha_y)):
        dy[i] = simps(integrande[i], alpha_x)
        
    inte = simps(dy, alpha_y)

    return inte    


def plotting(a1, a2, grid = 42):
    """
    Plots a given function 2D contour and the integration area

    Parameters
    ----------
    a1 : list
        First basis vector of integration area
    a2 : list
        Second basis vector of integration area
    grid : int, optional
        gridsize. The default is 42.

    Returns
    -------
    None.

    """
    
    x = np.linspace(-0.1,2,grid)
    y = np.linspace(-0.1,1.25,grid)
    
    integrande = np.zeros((grid,grid))
    
    for i in range(grid):
        for j in range(grid):
            integrande[i,j] = (x[i] + y[j]) * np.exp(-0.5*np.sqrt(x[i]**2 + y[j]**2))
            
    fig = plt.figure(figsize = (13,13))
    ax = fig.add_subplot(111)
    CF = ax.contourf(x, y, integrande)
    fig.colorbar(CF, cmap = 'plasma', label = "Value")
    # Plot the area omega using basis vectors
    ax.plot([0,a1[0]],[0, a1[1]], 'r')
    ax.plot([0, a2[0]], [0, a2[1]], 'r'  )        
    ax.plot([a2[0], a2[0] + a1[0]],  [a2[1], a2[1]], 'r')
    ax.plot([a1[0], a1[0] + a2[0]],   [0, a2[1]], 'r')
    
    ax.set_title("Function value contour plot")
    ax.set_xlabel("X-axis")
    ax.set_ylabel("Y-axis")
    

def main():
    
    a1 = [1.2, 0]
    a2 = [0.6, 1]
    A = np.array([a1, a2]) # basis
    
    integrande, alpha_x, alpha_y = toAlfa(A)
    integral = integration(integrande, alpha_x, alpha_y)
        
    exact_value = 0.925713
    print("Integral error to known value is: {:.4f}".format(np.abs(exact_value - integral)))
    print("Integral error to known value is: {:.2%}".format(np.abs(exact_value - integral) / exact_value))
    plotting(a1,a2)
    
if __name__=="__main__":
    main()
        
        
    
