# -*- coding: utf-8 -*-
"""
Exercise sheet 4 Assignement 2

This code reads the densit matrix and lattice of two molecule in and outputs the 
number of electrons in a unit cell and their reciprocal lattice vectors.

rho = electron density matrix
lattice = lattice parameters defining the unit cell
grid = dimensions

Created on Thu Jan 30 17:47:51 2020

@author: mvleko
"""

import numpy as np
from scipy.integrate import simps
from read_xsf_example import read_example_xsf_density


def number_of_electrons(rho,lattice,grid):
    #This function calculates the number of electrons in dimensions given by 
    #the grid for orthogonal lattice vectors.
    
    #The lattice parameters are saved in the diagonal elements of lattice
    a=lattice[0,0]
    b=lattice[1,1]
    c=lattice[2,2]
    
    #Grid defines the dimensions+1
    dx=a/(grid[0]-1)
    dy=b/(grid[1]-1)
    dz=c/(grid[2]-1)
     
    #Integration over the electron density over the unit cell volume gives the number of electrons.
    IX = simps(rho, dx=dx)
    IY = simps(IX, dx=dy)
    I = simps(IY, dx=dz)
    return I

def number_of_electrons2(rho,lattice,grid):
    #This function calculates the number of electrons in dimensions given by 
    #the grid for non-orthogonal lattice vectors.
    
    #We integrate the charge density over vectors from 0 to 1 and multiply it 
    #with the determinante of the original integration volume to get the correct scaling.
    a=np.linspace(0,1.0,grid[0])
    b=np.linspace(0,1.0,grid[1])
    c=np.linspace(0,1.0,grid[2])
 
    dx=a[1]-a[0]
    dy=b[1]-b[0]
    dz=c[1]-c[0]
     
    #Integration over orthonormal volume * the determinant of the original 
    #integration volume gives the number of electrons for linear transformations.
    IX = simps(rho, dx=dx, axis=0)
    IY = simps(IX, dx=dy, axis=0)
    I = simps(IY, dx=dz)*np.linalg.det(np.transpose(lattice))
    return I


def main():
    
    #First example
    print('File 1:')
    filename = 'dft_chargedensity1.xsf'
    rho, lattice, grid, shift = read_example_xsf_density(filename)      #read in the file
    N1=number_of_electrons(rho,lattice,grid)
    print(' Electrons in unit cell: ',int(np.round(N1)))
    
    #Reciprocal lattice vector according to lecture slides p.5: B^t*A=2*pi*Identity
    #A=transpose(lattice)
    B1=np.transpose(2*np.pi*np.dot(np.identity(3), np.linalg.inv(np.transpose(lattice))))
    print(' Reciprocal lattice vectors:\n',B1,'\n')
    
    
    #Second example
    print('File 2:')
    filename = 'dft_chargedensity2.xsf'
    rho2, lattice2, grid2, shift = read_example_xsf_density(filename)
    N2=number_of_electrons2(rho2,lattice2,grid2)    
    print(' Electrons in unit cell: ',int(np.round(N2)))
    
    B2=np.transpose(2*np.pi*np.dot(np.identity(3), np.linalg.inv(np.transpose(lattice2))))
    print(' Reciprocal lattice vectors:\n', B2)
    
if __name__=="__main__":
    main()
   
    
    
    
    