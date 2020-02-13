# -*- coding: utf-8 -*-
"""
Exercise sheet 4 assignement 3

This code calculates the electron density of a molecule along a line between 
point r0 and r1 and gives its plot. The electron density is given by a VESTA file.
It uses the interpolation class 'spline_class'.

Created on Thu Jan 30 19:14:14 2020

@author: mvleko
"""


import numpy as np
from read_xsf_example import read_example_xsf_density
from spline_class import spline
import matplotlib.pyplot as plt


def main():  
    #Read in the file
    filename = 'dft_chargedensity1.xsf'
    rho, lattice, grid, shift = read_example_xsf_density(filename)

    #Create data in initial coordinates    
    a=np.linspace(0,lattice[0,0],grid[0])
    b=np.linspace(0,lattice[1,1],grid[1])
    c=np.linspace(0,lattice[2,2],grid[2])
    
    #Create the interpolated data
    spline_data=spline(x=a,y=b,z=c,f=rho,dims=3)
    
    #Calculate the interpolation for the new coordinates        
    r0=np.array([0.1,0.1,2.8528])       #Given in the exercise
    r1=np.array([4.45,4.45,2.8528])  
    t=np.linspace(0,1,500)
    result=[]
    for i in range(500):
        r = r0+t[i]*(r1-r0)             #This is the new interpolation vector at index i
        inter=spline_data.eval3d(r[0],r[1],r[2])[0][0][0]
        result.append(inter)            #Store the values in a list
    
    length=np.linalg.norm(r1-r0)        #Length of the interpoaltion line
    
    #Plot the electron density as a function of t  
    plt.figure()
    plt.plot(t*length,result)
    plt.title('Electron density interpolation')
    plt.xlabel('Distance [Angstrom]')
    plt.xlim([0,1*length])
    plt.ylabel('Electron density')
    plt.grid()
    plt.show()
    
    
if __name__=="__main__":
    main()
    