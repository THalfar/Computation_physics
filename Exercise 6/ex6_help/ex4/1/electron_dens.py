""" 
------ EXERCISE 4 - problems 3 and 4 --------
----- FYS-4096 - Computational Physics ------

Calculates electron density along a line between
two points in simulation cells given in files
dft_chargedensity1.xsf and dft_chargedensity2.xsf
in problem 3 the line is between [0.1,0.1,2.8528]
and [4.45,4.45,2.8528]. In probelm 4 there are two lines
first one is between [-1.4466,1.3073,3.2115] and 
[1.4361,3.1883,1.3542]. Second one is between 
[2.9996,2.1733,2.1462] and [8.7516,2.1733,2.1462]

It would seem that the line in problem 3 goes
along a side of the cubic through the diagonal of said side

In problem 4 the first line could go almost through one O
atom and near a V atom.

Second line could go from near a V atom over another
V atom and then some 

:function: electron_density: calculates the electron density with help from
                             read_xsf_example.py and interpolates it with
                             spline_class.py
"""

import read_xsf_example as r_xsf
from spline_class import *
import numpy as np
import matplotlib.pyplot as plt
import math

def electron_dens(filename, r0, r1, N):
    """
    Calculates the electron density on an interpolated spline line
    :param: filename: file from which the data is read
    :param: r0      : starting point of the line
    :param: r1      : ending point of the line
    :param: N       : number of points on the interpolated line
    
    :return:        : electron density values on the line
    """    
    
    # rho:     electron density in unit volume (one for each point)
    # lattice: lattice vectors                          (3 by 3)
    # grid:    grid size in lattice vector diection     (3 by 1)
    # shift:   unknown
    rho, lattice, grid, shift = r_xsf.read_example_xsf_density(filename)
    
    # separate the lattice vectors
    a1 = lattice[:,0]
    a2 = lattice[:,1]
    a3 = lattice[:,2]
    
    # starting and ending points as vectors in the simulation lattice grid
    a_i = np.array([[r0.dot(a1)/(np.linalg.norm(a1)**2)*grid[0]], \
                    [r0.dot(a2)/(np.linalg.norm(a2)**2)*grid[1]], \
                    [r0.dot(a3)/(np.linalg.norm(a3)**2)*grid[2]]])
    a_f = np.array([[r1.dot(a1)/(np.linalg.norm(a1)**2)*grid[0]], \
                    [r1.dot(a2)/(np.linalg.norm(a2)**2)*grid[1]], \
                    [r1.dot(a3)/(np.linalg.norm(a3)**2)*grid[2]]])
    
    
    # check if the line goes over the simulation cell
    # i.e. if we need to produce another cell
    n1 = math.ceil(max(abs(a_i[0])/grid[0],abs(a_f[0])/grid[0],[1])[0])
    n2 = math.ceil(max(abs(a_i[1])/grid[1],abs(a_f[1])/grid[1],[1])[0])
    n3 = math.ceil(max(abs(a_i[2])/grid[2],abs(a_f[2])/grid[2],[1])[0])
    
    # spline interpolation, if some of the observed points are
    # on the negative side
    if any(r0<0) or any(r1<0):
        
        # if there is negative directions create more cells
        rho = np.concatenate((rho,rho), axis = 0)
        rho = np.concatenate((rho,rho), axis = 1)
        rho = np.concatenate((rho,rho), axis = 2)
        
        # if lines go over simulation cell, create more
        if n1 > 1:
            for i in range(2*(n1-1)):
                rho = np.concatenate((rho,rho), axis = 0)
        
        if n2 > 1:
            for i in range(2*(n2-1)):
                rho = np.concatenate((rho,rho), axis = 1)
        
        if n3 > 1:
            for i in range(2*(n3-1)):
                rho = np.concatenate((rho,rho), axis = 2)
        
        # create spline class for interpolation
        spl = spline(x=n1*np.linspace(-grid[0]+1,grid[0]-1,2*n1*grid[0]),\
                     y=n2*np.linspace(-grid[1]+1,grid[1]-1,2*n2*grid[1]),\
                     z=n3*np.linspace(-grid[2]+1,grid[2]-1,2*n3*grid[2]),\
                     f=rho, dims = 3)
    
    
    else:
        # if lines go over simulation cell, create more
        if n1 > 1:
            for i in range(n1-1):
                rho = np.concatenate((rho,rho), axis = 0)
        
        if n2 > 1:
            for i in range(n2-1):
                rho = np.concatenate((rho,rho), axis = 1)
        
        if n3 > 1:
            for i in range(n3-1):
                rho = np.concatenate((rho,rho), axis = 2)
                
        # create spline class for interpolation
        spl = spline(x=n1*np.linspace(0,grid[0]-1,n1*grid[0]),\
                     y=n2*np.linspace(0,grid[1]-1,n2*grid[1]),\
                     z=n3*np.linspace(0,grid[2]-1,n3*grid[2]),\
                     f=rho, dims = 3)
    
    # create line along which to interpolate
    x = np.linspace(a_i[0],a_f[0],N)
    y = np.linspace(a_i[1],a_f[1],N)
    z = np.linspace(a_i[2],a_f[2],N)
    F = np.zeros((N,1))
    
    # interpolation
    for i in range(N):
        F[i]=spl.eval3d(x[i],y[i],z[i])
                   
    return F
    
      

def main():
    # NOTE: This calculation is slow

    # ----------------- PROBLEM 3 ---------------------
    # file from which data is gathered
    filename = 'dft_chargedensity1.xsf'
    
    # starting and ending points
    r0 = np.array([0.1,0.1,2.8528])
    r1 = np.array([4.45,4.45,2.8528])
    
    print(f"Running electrondensity calculation on {filename} \n" \
          f"between points {r0}, {r1}")
          
    # Calculations
    F = electron_dens(filename,r0,r1,500)
    
    # plot the figure
    fig = plt.figure()
    title = 'Electron density on line r0-r1'
    fig.suptitle(title)
    
    ax1 = fig.add_subplot(311)
    
    ax1.plot(np.linspace(0,1,500),F ,'b')
    
    #  axis labels
    ax1.set_xlabel('r')
    ax1.set_ylabel('electron density')
    
    
    ax1.set_title("problem 3 line")
    
    # ----------------- PROBLEM 4 ---------------------
    # file from which data is gathered
    filename = 'dft_chargedensity2.xsf'
    
    # starting and ending points
    r0 = np.array([-1.4466,1.3073,3.2115])
    r1 = np.array([1.4361,3.1883,1.3542])
    print(f"Running electrondensity calculation on {filename} \n" \
          f"between points {r0}, {r1}")
    
    # Calculations
    F = electron_dens(filename,r0,r1,500)
    
    ax2 = fig.add_subplot(312)
    
    ax2.plot(np.linspace(0,1,500),F ,'b')
    
    #  axis labels
    ax2.set_xlabel('r')
    ax2.set_ylabel('electron density')
    
    ax2.set_title("problem 4 first line")
    
    # starting and ending points
    r0 = np.array([2.9996,2.1733,2.1462])
    r1 = np.array([8.7516,2.1733,2.1462])
    print(f"Running electrondensity calculation on {filename} \n" \
          f"between points {r0}, {r1}")
    
    # Calculations
    F = electron_dens(filename,r0,r1,500)
    
    ax3 = fig.add_subplot(313)
    
    ax3.plot(np.linspace(0,1,500),F ,'b')
    
    #  axis labels
    ax3.set_xlabel('r')
    ax3.set_ylabel('electron density')
    
    ax3.set_title("problem 4 second line")
    
    # show the figure
    plt.show()

if __name__ == "__main__":
    main()
