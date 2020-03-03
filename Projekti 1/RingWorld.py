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
import time

MU = 1.25663706212e-6

def calBcircle(X, Y, Z, Bx, By, Bz, place, axis, radius, I, pieces, ax = None):
    """
    Calculates the magnetic field of a circle using small wire elements
    repsenting circle

    Parameters
    ----------
    X : np.array
        X-axis meshgrid
    Y : np.array
        Y-axis meshgrid
    Z : np.array
        Z-axis meshgrid
    Bx : np.array
        B-field meshgrid of x-axis
    By : np.array
        B-field meshgrid of y-axis
    Bz : np.array
        B-field meshgrid of y-axis
    place :  (x,y,z) tuplet
        Place where put the circle midpoint
    axis : char
        What is the axis of rotation
    radius : float
        Radius of the circle
    I : Current
        Current in the circle
    pieces : int
        How many wire elements in circle
    ax : matplot axis, optional
        Plot to this axis if given

    Returns
    -------
    Bx : np.array
        Updated meshgrid values of B-field in x-axis
    By : np.array
        Updated meshgrid values of B-field in y-axis
    Bz : np.array
        Updated meshgrid values of B-field in x-axis

    """
    # Parametrization variable    
    dt = 2*np.pi / pieces
    
    # Initialize the wire elements from which calculate magnetic field
    wire_elements = np.zeros((pieces, 3))
    
    # Place coordinates where make an current circle
    x, y, z = place
     
    # Use axis parameter to determine in which axis the circle rotates
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
                    # Calculate the distance from wire_element middlepoint
                    r = (X[i,j,k], Y[i, j, k], Z[i,j,k]) - wire_elements[q,:] 
                    + (small_arrows[q,:] / 2)
                        
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
    this to logarithmical scale comparing different ring elements 
    amount -> If plot is linear, error is polynomial and algorithm
    is working.

    Returns
    -------
    None.

    """
    # Grid spacing from where calculate errors
    space = np.geomspace(10, 50, 10, dtype = int) 
    
    r = 2 # radius
    i = 1 # current
    
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
    
        Bx, By, Bz = calBcircle(X,Y,Z,Bx,By,Bz, (0,0,0), 'x', r, i, n)
        # Take value along axis where analytical function is known
        Bx_num = Bx[10, :, 10]
        errors.append(np.sum(np.abs(Bx_num - Bright)))
        
    errors = np.array(errors)   
    # Plot the errors as function of line element amount    
    plt.figure(figsize = (13,13))    
    plt.scatter(space, errors, facecolor = 'red')    
    plt.yscale('log')
    plt.xscale('log')
    plt.title("Error plot algorithm with grid size 21")
    plt.xlabel("Number of segments in circle")
    plt.ylabel("Abs. error sum")


def two_circle(gridsize = 11, elements = 42):
    """
    Plot two circle along x-axis as required in assigment

    Parameters
    ----------
    gridsize : int, optional
        What is the gridsize. The default is 11.

    Returns
    -------
    None.

    """
    # Need to have axis zero point to have a good slice of ringworld for streamplot
    if gridsize % 2 == 0:
        print("Gridsize must be uneven!")
        return 
    
    x = np.linspace(-3, 3 , gridsize)
    y = np.linspace(-3, 3 , gridsize)
    z = np.linspace(-3, 3 , gridsize)
    
    X, Y, Z = np.meshgrid(x, y, z)
    
    Bx = np.zeros_like(X)
    By = np.zeros_like(X)
    Bz = np.zeros_like(X)
        
    fig = plt.figure(figsize  = (13,13))
    ax = fig.add_subplot(111, projection='3d')
    
    start_time = time.time()
    
    # We have as assigment says two circle that have radius = 1 and
    # are 4 units away from each other with current 1 Ampere
    Bx, By, Bz = calBcircle(X,Y,Z,Bx,By,Bz, (-2,0,0), 'x', 2, 1, elements, ax)
    Bx, By, Bz = calBcircle(X,Y,Z,Bx,By,Bz, (2,0,0), 'x', 2, 1, elements, ax)
    
    print ("Ringworld with two ring and {} grid with {} elements per ring did take: {:.2f} s to run "
           .format(gridsize, elements, time.time() - start_time))
        
    # Using logarithm to scale the arraow length
    log = lambda x: np.sign(x) * np.log(np.abs(x) + 1)    
    ax.quiver(X,Y,Z,log(Bx),log(By),log(Bz),  alpha = 0.6 )
    ax.set_xlabel("X-axis [m]")
    ax.set_ylabel("Y-axis [m]")
    ax.set_zlabel("Z-axis [m]")
    ax.set_title("Ringworld of two ring with mu=1, log arrow length, gridsize {} and {} elements per ring".format(gridsize, elements))
    plt.show()
    
    # Convert magnetic field to teslas by multiplying with MU
    Bx *= MU
    By *= MU
    Bz *= MU
    
    fig = plt.figure(figsize  = (13,13))
    ax = fig.gca()
    # take index where axis is zero (so can use different gridsizes automatically)
    index = np.where(x==0)[0].item()
    norm = np.sqrt(Bx[:,:,index]**2 + By[:,:,index]**2 + Bz[:,:,index]**2)
    strm = ax.streamplot(X[:,:,index], Y[:,:, index], Bx[:,:,index], By[:,:,index], 
                         linewidth = 2, color = norm, cmap='plasma')
    clb = plt.colorbar(strm.lines)
    clb.set_label("Magnetic field strength [T]")
    ax.set_xlabel("X-axis [m]")
    ax.set_ylabel("Y-axis [m]")
    ax.set_title("Stream plot from slice at z=0 and gridsize {} and {} elements per ring".format(gridsize, elements))
    plt.show()
    
   
def ringworld(gridsize = 11, elements = 42):
    """
    Calculates four rings that magnifys each other magnetic field 
    using currents oppositely directed

    Parameters
    ----------
    gridsize : int, optional
        Gridsize where calculate B-field. The default is 11.

    Returns
    -------
    None.

    """  
    # Need to have axis zero point to have a good slice of ringworld for streamplot
    if gridsize % 2 == 0:
        print("Gridsize must be uneven!")
        return 
    
    x = np.linspace(-2.5, 2.5 , gridsize)
    y = np.linspace(-2.5, 2.5 , gridsize)
    z = np.linspace(-2.5, 2.5 , gridsize)
    
    X, Y, Z = np.meshgrid(x, y, z)
    
    Bx = np.zeros_like(X)
    By = np.zeros_like(X)
    Bz = np.zeros_like(X)
        
    fig = plt.figure(figsize  = (13,13))
    ax = fig.add_subplot(111, projection='3d')
    
    start_time = time.time()
    # Initialize rings..
    Bx, By, Bz = calBcircle(X,Y,Z,Bx,By,Bz, (1.1,0,0), 'y', 0.8, 1, elements, ax)
    Bx, By, Bz = calBcircle(X,Y,Z,Bx,By,Bz, (-1.1,0,0), 'y', 0.8, -1, elements, ax)
    Bx, By, Bz = calBcircle(X,Y,Z,Bx,By,Bz, (0,1.1,0), 'x', 0.8, 1, elements, ax)
    Bx, By, Bz = calBcircle(X,Y,Z,Bx,By,Bz, (0,-1.1,0), 'x', 0.8, -1, elements, ax)
    
    print ("Ringworld with four ring and {} grid with {} elements per ring did take: {:.2f} s to run "
           .format(gridsize, elements, time.time() - start_time))
    # using logarithm to try prune the arrows
    log = lambda x: np.sign(x) * np.log(np.abs(x) + 1)    
    ax.quiver(X,Y,Z,log(Bx),log(By),log(Bz),  alpha = 0.6 )
    ax.set_xlabel("X-axis [m]")
    ax.set_ylabel("Y-axis [m]")
    ax.set_zlabel("Z-axis [m]")
    ax.set_title("Ringworld of four ring with mu=1, log arrow length, gridsize {} and {} elements per ring".format(gridsize, elements))
    plt.show()
    
    # Convert magnetic field to teslas by multiplying with MU
    Bx *= MU
    By *= MU
    Bz *= MU
        
    fig = plt.figure(figsize  = (13,13))
    ax = fig.gca()
    idx = np.where(x==0)[0].item()
    norm = np.sqrt(Bx[:,:,idx]**2 + By[:,:,idx]**2 + Bz[:,:,idx]**2)
    strm = ax.streamplot(X[:,:,idx], Y[:,:, idx], Bx[:,:,idx], By[:,:,idx], linewidth = 2, color = norm, cmap='plasma')
    clb = plt.colorbar(strm.lines)
    clb.set_label("Magnetic field strength [T]")
    ax.set_xlabel("X-axis [m]")
    ax.set_ylabel("Y-axis [m]")
    ax.set_title("Stream plot from slice at z=0 and gridsize {} and {} elements per ring".format(gridsize, elements))
    plt.show()
    
    

def main():
    
    test_algorithm()
    # two_circle()
    # ringworld()


if __name__=="__main__":
    main()
        
        
    
        
    