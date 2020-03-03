#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 26 12:11:12 2020

@author: halfar
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import simps
from mpl_toolkits.mplot3d import Axes3D
from numba import jit
import time

mu = 1.25663706212e-6

def calBcircle(X, Y, Z, Bx, By, Bz, place, axis, radius, I, pieces, ax = None):
    
    dt = 2*np.pi / pieces
    
    # Initialize the wire elements from which calculate magnetic field
    wire_elements = np.zeros((pieces, 3))
    
    # Place coordinates where make an current circle
    x, y, z = place
     
    # Use axis parameter to 
    if axis == 'x':        
        for i in range(pieces):
            wire_elements[i,:] = (x, y + radius*np.cos(dt*i), z + radius*np.sin(dt*i))
    
    elif axis == 'y':
         for i in range(pieces):
            wire_elements[i,:] = (x + radius*np.cos(dt*i), y  , z + radius*np.sin(dt*i))
    
    else:        
         for i in range(pieces):
            wire_elements[i,:] = (x + radius*np.cos(dt*i), y + radius*np.sin(dt*i), z)
            
    if ax != None:
        ax.plot(wire_elements[:,0], wire_elements[:,1], wire_elements[:,2], 'r')
        ax.plot((wire_elements[-1,0], wire_elements[0,0]) , (wire_elements[-1,1], wire_elements[0,1]), 
            (wire_elements[-1,2], wire_elements[0,2]), 'r') 
    
    
    small_arrows = np.zeros((pieces, 3))
    
    for i in range(0, pieces-1):
            
        small_arrows[i,0] =  wire_elements[i+1,0] - wire_elements[i,0]
        small_arrows[i,1] =  wire_elements[i+1,1] - wire_elements[i,1]
        small_arrows[i,2] =  wire_elements[i+1,2] - wire_elements[i,2]
        
    # viimeiseen nuoleen tarvitaan eka koordi  
    small_arrows[-1,0] = wire_elements[0,0] - wire_elements[-1,0]
    small_arrows[-1,1] = wire_elements[0,1] - wire_elements[-1,1]
    small_arrows[-1,2] = wire_elements[0,2] - wire_elements[-1,2]
    
  
    
    for i in range(X.shape[0]):

        for j in range(X.shape[1]):
    
            for k in range(X.shape[2]):
                
                for q in range(len(wire_elements)):
                    
                    r = (X[i,j,k], Y[i, j, k], Z[i,j,k]) - wire_elements[q,:]
                    dl = small_arrows[q,:]
                    B = np.cross(dl, r) / np.linalg.norm(r)**3
                    Bx[i,j,k] += (B[0] * I) / (4*np.pi)
                    By[i,j,k] += (B[1] * I) / (4*np.pi)
                    Bz[i,j,k] += (B[2] * I) / (4*np.pi)
                    
    return Bx, By, Bz
    

def test_algorithm():
    """
    Test the algorithm using gridspace 21 and comparing analytical 
    and numerical result using error abs sum and plotting
    this to logarithmical scale comparing different elements 
    amount -> If plot is linear, error is polynomial and algorithm
    is working.

    Returns
    -------
    None.

    """

    space = np.geomspace(3, 123, 10, dtype = int)  
    
    r = 2
    i = 1
    
    x = np.linspace(-3, 3 , 21)
    y = np.linspace(-3, 3 , 21)
    z = np.linspace(-3, 3 , 21)
    
    X, Y, Z = np.meshgrid(x, y, z)
        
    Bright = np.zeros(21)    
    for j in range(21):        
        # Analytical result is known at x-axis center
        Bright[j] = (i*r**2) / (2 * (y[j]**2 + r**2)**(3/2) )
              
    errors = []
    for n in space:
        
        Bx = np.zeros_like(X)
        By = np.zeros_like(X)
        Bz = np.zeros_like(X)
    
        Bx, By, Bz = calBcircle(X,Y,Z,Bx,By,Bz, (0,0,2), 'x', r, i, n)
        # Take value along axis where analytical function is known
        Bx_num = Bx[10, :, 10]
        errors.append(np.sum(np.abs(Bx_num - Bright)))
    # Plot the errors as function of line element amount    
    plt.scatter(space, np.array(errors), facecolor = 'red')
    plt.yscale('log')
    plt.xscale('log')
    plt.title("Error plot algorithm with grid size 21")
    plt.xlabel("Number of segments in circle")
    plt.ylabel("Abs. error sum")


def main():
    
    # test_algorithm()
    gridkoko = 11
    x = np.linspace(-4, 4 , gridkoko)
    y = np.linspace(-4, 4 , gridkoko)
    z = np.linspace(-4, 4 , gridkoko)
    
    X, Y, Z = np.meshgrid(x, y, z)
    
    Bx = np.zeros_like(X)
    By = np.zeros_like(X)
    Bz = np.zeros_like(X)

    fig = plt.figure(figsize  = (13,13))
    ax = fig.add_subplot(111, projection='3d')
    
    # start_time = time.time()
    
    Bx, By, Bz = calBcircle(X,Y,Z,Bx,By,Bz, (1,2,3), 'z', 1, 1, 42, ax)
    Bx, By, Bz = calBcircle(X,Y,Z,Bx,By,Bz, (3,1,2), 'x', 2, 1, 42, ax)
        
    # Bx, By, Bz = calBcircle(X,Y,Z,Bx,By,Bz, -2, 'y', 2, 1, 42, ax)
    # Bx, By, Bz = calBcircle(X,Y,Z,Bx,By,Bz, 3, 'y', 3, 1, 42, ax)
        
    # Bx, By, Bz = calBcircle(X,Y,Z,Bx,By,Bz, -3, 'z', 2, 1, 42, ax)
    # # Bx, By, Bz = calBcircle(X,Y,Z,Bx,By,Bz, 3, 'z', 2, 1, 42, ax)
    
    # # Bx, By, Bz = calBcircle(X,Y,Z,Bx,By,Bz, 2, 'x', 1, 1, 42, ax)
    # # Bx, By, Bz = calBcircle(X,Y,Z,Bx,By,Bz, -2, 'x', 1, 1, 42, ax)
    
    # # Bx, By, Bz = calBcircle(X,Y,Z,Bx,By,Bz, 2, 'z', 1, 1, 42, ax)
    # # Bx, By, Bz = calBcircle(X,Y,Z,Bx,By,Bz, -2, 'z', 1, 1, 42, ax)
        
    # # Bx, By, Bz = calBcircle(X,Y,Z,Bx,By,Bz, 2, 'y', 1, -1, 42, ax)
    # # Bx, By, Bz = calBcircle(X,Y,Z,Bx,By,Bz, -2, 'y', 1, 1, 42, ax)
    
    lognorm = lambda x: np.sign(x) * np.log(np.abs(x) + 1)    
    ax.quiver(X,Y,Z,lognorm(Bx),lognorm(By),lognorm(Bz),  alpha = 0.6 )
    
    # print ("Ringworld ", time.time() - start_time, "s to run")
    
    # fig = plt.figure(figsize  = (13,13))
    # plt.streamplot(X[:,:,5], Y[:,:, 5], Bx[:,:,5], By[:,:,5])
    
        
    

if __name__=="__main__":
    main()
        
        
    
        
    