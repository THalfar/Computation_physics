# -*- coding: utf-8 -*-
"""
Created on Wed Feb  5 16:42:04 2020

@author: halfar

"""

import numpy as np
from numba import jit
import time
import matplotlib.pyplot as plt
import seaborn as sns

@jit
def jacobi(phi, tol = 1e-6, step = 0.25):
    """ Implementation of Jacob method
    
    Parameters
    ----------
    phi : numpy array
        Phi starting grid
    tol : float
        Tolerance when stop
    step : float
        Step size

    Returns
    -------
    phi_first : numpy array
        Phi grid after relaxation

    """
    
    phi_bool = np.zeros_like(phi, dtype = bool)
    phi_bool = np.where(phi != 0, True, False)
    
    sum_first = np.sum(phi)    
    phi_first = np.copy(phi)
    phi_last = np.copy(phi)
    
    overtol = True
    
    while overtol:
        
        for i in range(1, phi.shape[0]-1):
            
            for j in range(1,phi.shape[1]-1):
                
                if phi_bool[i,j] == True:
                    continue
                
                phi_last[i,j] = step * (phi_first[i-1,j] + phi_first[i+1,j] + phi_first[i,j-1] + phi_first[i,j+1])
                
        sum_last = np.sum(phi_last)
        
        if np.abs(sum_last - sum_first) < tol:
            overtol = False
            
        sum_first = sum_last
        
        phi_first = np.copy(phi_last)
        phi_last = np.copy(phi)
        
    return phi_first

@jit    
def SOR(phi, tol = 1e-6, omega = 1.8 ):
    """ Implementation of Simultaneous Over Relaxation SOR
    

    Parameters
    ----------
    phi : numpy array
        Starting values of phi
    tol : float, optional
        Tolerance when stop. The default is 1e-6.
    omega : float, optional
        omega value. The default is 1.8.

    Returns
    -------
    phi_first : numpy array
        Phi array after relaxation

    """
    
    phi_bool = np.zeros_like(phi, dtype = bool)
    phi_bool = np.where(phi != 0, True, False)
    
    sum_first = np.sum(phi)    
    phi_first = np.copy(phi)
    phi_last = np.copy(phi)
    
    overtol = True
    
    while overtol:
        
        for i in range(1, phi.shape[0]-1):
            
            for j in range(1,phi.shape[1]-1):
                
                if phi_bool[i,j] == True:
                    continue
                
                phi_last[i,j] = (1-omega) * phi_first[i,j] +  omega/4 * (phi_last[i-1,j] + phi_first[i+1,j] + phi_last[i,j-1] + phi_first[i,j+1])
                
        sum_last = np.sum(phi_last)
        
        if np.abs(sum_last - sum_first) < tol:
            overtol = False
            
        sum_first = sum_last
        
        phi_first = np.copy(phi_last)
        phi_last = np.copy(phi)
        
    return phi_first

def two_plates(size = 41):    
    
    x = np.linspace(-1,1,size) 
    y = np.linspace(-1,1,size)      
    X, Y = np.meshgrid(x,y) # needed for quiver axis
        
    # Find indices for plates that grid size can be changed    
    x1_index = int(np.where(np.abs(x+0.3) < 2/size)[0])    
    y_index_last = int(np.where(np.abs(y-0.5) < 2/size)[0])
    y_index_first = int(np.where(np.abs(y+0.5) < 2/size)[0])
    x2_index = int(np.where(np.abs(x-0.3) < 2/size)[0])
        
    phi = np.zeros_like(X) 
    phi[y_index_first:y_index_last, x1_index] = -1
    phi[y_index_first:y_index_last, x2_index] = 1
    
    phi = SOR(phi)

    fig = plt.figure(figsize = (15,15))
    ax = fig.add_subplot(111)
    plt.imshow(np.flipud(phi), cmap='jet', interpolation='spline16') # y-starts from -1, need to flipflop 
    ax.set_title("Potential using SOR with gridsize: {}".format(size))
    ticks = range(0,size,5)
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)
    tickvalues = x[ticks]
    ax.set_xticklabels(tickvalues)
    ax.set_yticklabels(-tickvalues)
    ax.set_xlabel("x-axis")
    ax.set_ylabel("y-axis")
    plt.show()   

    U, V = np.gradient(phi)
    fig = plt.figure(figsize = (15,15))
    ax = fig.add_subplot(111)    
    ax.quiver(X,Y,-V,-U)
    ax.set_title("Electric field using SOR with gridsize: {}".format(size))
    ax.set_xlabel("x-axis")
    ax.set_ylabel("y-axis")    
    plt.show()   

def face(size = 41):
    x = np.linspace(-1,1,size) 
    y = np.linspace(-1,1,size)    
    X,Y = np.meshgrid(x,y)
   
    phi = np.zeros_like(X) 
    phi[30,10] = 1.5
    phi[30,30] = 1.5
    phi[10,10:30] = -1.5
    phi[20:30,20] = -1
    
    phi = SOR(phi)

    fig = plt.figure(figsize = (15,15))
    ax = fig.add_subplot(111)
    
    # Because grid y-axis starts from -1 need to flip that its right way..     
    plt.imshow(np.flipud(phi), cmap='jet', interpolation='spline16')    
    ax.set_title("Potential using SOR with gridsize: {}".format(size))
    ticks = range(0,size,5)
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)
    tickvalues = x[ticks]
    ax.set_xticklabels(tickvalues)
    ax.set_yticklabels(-tickvalues)
    ax.set_xlabel("x-axis")
    ax.set_ylabel("y-axis")
    plt.show()   

    U, V = np.gradient(phi)
    fig = plt.figure(figsize = (15,15))
    ax = fig.add_subplot(111)    
    ax.quiver(X,Y,-V,-U)
    ax.set_title("Electric field using SOR with gridsize: {}".format(size))
    ax.set_xlabel("x-axis")
    ax.set_ylabel("y-axis")
    plt.show()   
    
    
    
    
def main():
    SOR(np.zeros((9,9)))# compile..
    
    start_time = time.time()
    gridsizes = [21,41,81] # TODO its only works with grids that have a value close to x=-0.3 etc... 
    for size in gridsizes:
        two_plates(size)
        print ("SOR did take compiled with Numba gridsize {}: {}s".format(size ,time.time() - start_time))
    
    face()

if __name__=="__main__":
    main()    