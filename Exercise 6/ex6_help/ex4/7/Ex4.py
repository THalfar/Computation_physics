# -*- coding: utf-8 -*-
"""
Exercise sheet 4 assignement 4

Interpolate the charge density of sample molecules along two lines. 
The two lines are specified by 'vec'.

Created on Thu Jan 30 19:14:14 2020

@author: mvleko
"""

import numpy as np
from read_xsf_example import read_example_xsf_density
from spline_class import spline
import matplotlib.pyplot as plt


def main():  
    #Read in the file
    filename = 'dft_chargedensity2.xsf'
    rho, lattice, grid, shift = read_example_xsf_density(filename)
    A_inv=np.linalg.inv(np.transpose(lattice))  #The inverse of our matrix A

    #Create data in initial coordinates  in orthonormal basis   
    a=np.linspace(0,1,grid[0])
    b=np.linspace(0,1,grid[1])
    c=np.linspace(0,1,grid[2])
    
    spline_data=spline(x=a,y=b,z=c,f=rho,dims=3)  
    
    #Input the values of the lines from starting to end point, along which the interpolation should be performed.
    vec=np.array([[-1.4466,1.3073,3.2115],  #start line 1
         [1.4361,3.1883,1.3542],            #end line 1
         [2.9996,2.1733,2.1462],            #start line 2
         [8.7516,2.1733,2.1462]])           #end line 2
    
    t=np.linspace(0,1,500)
    
    counter=0
    for j in range(0,3,2):  #This gives the value 0 in the first iteration and 2 in the second.
                            #(This loop is included because the spline in line 30 takes forever and I don't want to run that part twice)
        
        #Choose the vectors start and end points, so that we can construct the interpolation line r0+t[i]*(r1-r0):        
        r0=vec[j]
        r1=vec[j+1]          
        
        result=[]
        
        #Calculate the interpolation
        for i in range(500):
            r = np.mod(A_inv.dot(r0+t[i]*(r1-r0)),1)    #This maps our vector according to r=A*alpha (r is our vector, A the transpose(lattice) and alpha our new coordinates)
                                                        #The modulo is needed in case our interpolation vector is not in the unit cell anymore (see lecture slides week 4 p.3)
            inter=spline_data.eval3d(r[0],r[1],r[2])[0][0][0]
            result.append(inter)
        
        #Plot the figures
        counter+=1
        length=np.linalg.norm(r1-r0)        #Length of the interpolation line
        
        plt.figure()
        plt.plot(t*length,result)
        plt.title('Interpolation along line {}'.format(counter))
        plt.xlabel('Distance [Angstrom]')
        plt.xlim([0,1*length])
        plt.ylabel('Electron density')
        plt.grid()
        plt.show()
    
    
if __name__=="__main__":
    main()