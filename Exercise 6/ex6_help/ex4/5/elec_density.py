""" 
Problem 3 and 4 of exercise set 4
"""

import read_xsf_example as r_xsf
from spline_class import *
from numpy import *
import matplotlib.pyplot as plt
import math

def electron_density(filename, r0, r1, N = 500):
    """
    Calculate the electron density at 
    a given line in the lattice
	
    :param: filename: given filename to read data from
    :param: r0:       line starting point
    :param: r1:       line end point
    :param: N:        Number of points along line
    """  
    
    # Get the lattice values
    rho, lattice, grid, shift = r_xsf.read_example_xsf_density(filename)
    
    # define the lattice vectors
    a1 = lattice[:,0]
    a2 = lattice[:,1]
    a3 = lattice[:,2]
    
    # start and end points as vectors in the lattice grid
    a_i = array([[r0.dot(a1)/(linalg.norm(a1)**2)*grid[0]], \
                    [r0.dot(a2)/(linalg.norm(a2)**2)*grid[1]], \
                    [r0.dot(a3)/(linalg.norm(a3)**2)*grid[2]]])
                    
    a_f = array([[r1.dot(a1)/(linalg.norm(a1)**2)*grid[0]], \
                    [r1.dot(a2)/(linalg.norm(a2)**2)*grid[1]], \
                    [r1.dot(a3)/(linalg.norm(a3)**2)*grid[2]]])
    
    
    # check if the line goes outside the simulation cell, and create
    # more if needed
    nx = math.ceil(max(abs(a_i[0])/grid[0],abs(a_f[0])/grid[0],[1])[0])
    ny = math.ceil(max(abs(a_i[1])/grid[1],abs(a_f[1])/grid[1],[1])[0])
    nz = math.ceil(max(abs(a_i[2])/grid[2],abs(a_f[2])/grid[2],[1])[0])
    
    # Interpolate values in the negative side, if we have to
    # calculate values there
    if any(r0<0) or any(r1<0):
        
        # Create more cells, if there is negative directions
        rho = concatenate((rho,rho), axis = 0)
        rho = concatenate((rho,rho), axis = 1)
        rho = concatenate((rho,rho), axis = 2)
        
        # if line goes over the cell, create more
        if nx > 1:
            for i in range(2*(nx-1)):
                rho = concatenate((rho,rho), axis = 0)
        
        if ny > 1:
            for i in range(2*(ny-1)):
                rho = concatenate((rho,rho), axis = 1)
        
        if nz > 1:
            for i in range(2*(nz-1)):
                rho = concatenate((rho,rho), axis = 2)
        
        spl3d = spline(x=nx*linspace(-grid[0]+1,grid[0]-1,2*nx*grid[0]),\
                     y=ny*linspace(-grid[1]+1,grid[1]-1,2*ny*grid[1]),\
                     z=nz*linspace(-grid[2]+1,grid[2]-1,2*nz*grid[2]),\
                     f=rho, dims = 3)
    
    
    else:
        # if lines go over simulation cell, create more
        if nx > 1:
            for i in range(nx-1):
                rho = concatenate((rho,rho), axis = 0)
        
        if ny > 1:
            for i in range(ny-1):
                rho = concatenate((rho,rho), axis = 1)
        
        if nz > 1:
            for i in range(nz-1):
                rho = concatenate((rho,rho), axis = 2)
                
        # create spline class for interpolation
        spl3d = spline(x=nx*linspace(0,grid[0]-1,nx*grid[0]),\
                     y=ny*linspace(0,grid[1]-1,ny*grid[1]),\
                     z=nz*linspace(0,grid[2]-1,nz*grid[2]),\
                     f=rho, dims = 3)
    
    # create line along which to interpolate
    x = linspace(a_i[0],a_f[0],N)
    y = linspace(a_i[1],a_f[1],N)
    z = linspace(a_i[2],a_f[2],N)
    
    # Interpolate values
    F = zeros((N,1))
    for i in range(N):
        F[i] = spl3d.eval3d(x[i],y[i],z[i])
                   
    return F
    
def plot_electron_density(filename,r0,r1,N,name): 
    """
     Plot electron density at given start and end
     point. Save the figure at given name 
	
    :param: filename: given filename to read data from
    :param: r0:       line starting point
    :param: r1:       line end point
    :param: N:        Number of points along line
    :param: name:     Save figure as this name
    """     
        
    print("Electron density along line from r0 = {} to r1 = {}".format(r0,r1))
    F = electron_density(filename,r0,r1,N)
    
    # plot the electron density
    fig = plt.figure()
    title = 'Electron density'
    fig.suptitle(title)
    ax = fig.add_subplot(111)
    ax.plot(linspace(0,1,N),F)
    
    ax.set_xlabel('|r-r0|/|r1-r0|')
    ax.set_ylabel('electron density')    
    fig.savefig(name,dpi=200)
    plt.show()
    
    
def main():
    #Plot for problem 3
    filename = 'dft_chargedensity1.xsf'
    print("Using data from", filename)
    # start and end points
    
    r0 = array([0.1,0.1,2.8528])
    r1 = array([4.45,4.45,2.8528])
    plot_electron_density(filename,r0,r1,500,'Electron_density_ex3')
    #-----------------------
    
    # Plots for problem 4
    filename = 'dft_chargedensity2.xsf'
    print("Using data from", filename)
    
    # start and end points 1
    # This part takes ages to complete,
    # the flaw seems to be in negative coordinates
    
    r0 = array([-1.4466,1.3073,3.2115])
    r1 = array([1.4361,3.1883,1.3542])
    plot_electron_density(filename,r0,r1,500,'Electron_density_ex4_1')
    
    # start and end points 2
    r0 = array([2.9996,2.1733,2.1462])
    r1 = array([8.7516,2.1733,2.1462])
    plot_electron_density(filename,r0,r1,500,'Electron_density_ex4_2')

if __name__ == "__main__":
    main()
