#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb 16 16:49:11 2020

@author: halfar
"""

import numpy as np
from scipy.sparse import diags
import matplotlib.pyplot as plt
from scipy.integrate import simps
from numba import jit


#%% Problem 2 a
def problem2a(N=23, plot = False):

    grid = np.linspace(0,1,N)
    
    h = grid[1] - grid[0]
    A = np.zeros((N,N))
    b = np.zeros((N,))
    # Use analytically known values
    diag = np.repeat(2/h, N)
    neig = np.repeat(-1/h, N)
    diago = [diag, neig, neig]
    A = diags(diago, [0, -1, 1]).toarray()
    A = A[1:-1, 1:-1]
    # Use for b vector analytical known values
    for i in range(1, b.shape[0]-1):
        b[i] = np.pi/h * (grid[i-1] + grid[i+1] - 2*grid[i])*np.cos(np.pi*grid[i]) \
            + 1/h * (2*np.sin(np.pi*grid[i]) - np.sin(np.pi*grid[i-1]) - np.sin(np.pi*grid[i+1]))
            
    b = b[1:-1]
    a = np.linalg.solve(A,b)      
    # fingrid solution is a values because hat function is 1 at these points
    # so a*u_i = a
    fin_sol = np.zeros((N,))
    fin_sol[1:-1] = a
    # known solution
    val_sol = np.sin(np.pi * grid)
    
    if plot:
        fig = plt.figure(figsize = (15,10))
        plt.plot(grid, val_sol, 'r', label = "Right values")
        plt.plot(grid, fin_sol, 'bo', label = "FEM values")
        plt.legend()
        plt.title("Plot right solution vs. finite element solution")
        plt.xlabel("Grid value")
        plt.ylabel("$\Phi$")
        plt.show()
    
    return A

#%%
@jit
def hat(idx, grid, x):
    """
    hat function 

    Parameters
    ----------
    idx : int 
        index of hat function
    grid : np.array
        grid of hat function places
    x : np.array
        real space where calculating hat values

    Returns
    -------
    out : np.array
        Values of hat function in real space x
    """
    
    h_first = grid[idx] - grid[idx-1]
    h_last = grid[idx+1] - grid[idx]
    # Using boolean to get fast code, that this can take whole x axis in
    out = (x > grid[idx-1])*(x <= grid[idx]) * ((x - grid[idx-1])/h_first) + \
        (x > grid[idx])*(x <= grid[idx+1]) * ((grid[idx+1] - x)/h_last) 
        
    return out

@jit
def Dhat(idx, grid, x):
    """
    hat function derivate

    Parameters
    ----------
    idx : int
        index of hat function
    grid : np.array
        array where hat is defined
    x : np.array
        real space to where calculate hat function dericate

    Returns
    -------
    out : np.array
        Values of hat function derivate

    """
    
    h_first = grid[idx] - grid[idx-1]
    h_last = grid[idx+1] - grid[idx]
    # using b oolean to get fast code, that this can get x axis in
    out = (x > grid[idx-1])*(x <= grid[idx]) * (1/h_first) + \
        (x > grid[idx])*(x <= grid[idx+1]) * ( -1/h_last)
        
    return out


def FEMhat(grid, x, phi):
    """
    Finited Element Method with hat base functions

    Parameters
    ----------
    grid : np.array
        grid of base function values
    x : np.array
        real space where calculate FEM values
    phi : np.array
        second derivate of unkown function

    Returns
    -------
    out : np.array
        FEM result array

    """
    
    N = len(grid)
    A = np.zeros((N,N))
    # Calculate A matrix values 
    for i in range(1, N-1):    
        for j in range(1, N-1):
        
            ueka = Dhat(j, grid, x)
            utoka = Dhat(i, grid, x)
            A[i,j] = simps(ueka*utoka, x)
            
    A = A[1:-1, 1:-1]
        
    b = np.zeros((N,1))
    for i in range(N):    
        b[i] =  simps(hat(i, grid,x)*phi, x)
    b = b[1:-1]

    a = np.linalg.solve(A,b)      
    a = np.vstack((phi[0],a,phi[-1]))
    
    out = np.zeros(len(x))
    for i, ind in enumerate(a):
        out += ind*hat(i, grid, x)
        
    return out, A
   
# plot few different FEM base functions     
def plot_FEM(spacing = [3,5,7,21], x_space = 1000):
    
    x = np.linspace(0,1,x_space)
    phi_oik = np.sin(np.pi*x)
    phi = np.pi**2 * np.sin(np.pi * x)

    plt.figure(figsize=(15,10))      
    for n in spacing:
        grid = np.linspace(0,1,n)
        testa, _ = FEMhat(grid,x, phi)
        plt.plot(x, testa, label = "Basis num: {}".format(n) )        
        
    plt.plot(x, phi_oik, label = "Right value")
    plt.legend()
    plt.xlabel("X-axis")
    plt.ylabel("$\Phi$")
    plt.title("Plot of few FEM")
    plt.show()   
 
# plot how error behaves with different amount of base functions    
def plot_errors():

    x = np.linspace(0,1,1000)
    phi_oik = np.sin(np.pi*x)
    phi = np.pi**2 * np.sin(np.pi * x)
    
    plt.figure(figsize=(15,10)) 
    
    spacing = np.geomspace(3,100, 10, dtype = int)
    
    errors = []
    
    for n in spacing:
        grid = np.linspace(0,1,n)
        testa, _ = FEMhat(grid,x, phi)
        errors.append(np.mean(np.abs(testa-phi_oik)))
    
    plt.figure(figsize=(15,10))    
    plt.plot(spacing, errors, 'ro')
    plt.yscale('log')
    plt.xscale('log')
    plt.xlabel("Base function count")
    plt.ylabel("Mean abs. error")
    plt.title("Error behavior of different base function count ")
    plt.show()
 
# Compare right A with a FEM A    
def problem2c(spacing = 8):
    
    x = np.linspace(0,1,1000)
    phi = np.pi**2 * np.sin(np.pi * x)
    
    grid = np.linspace(0,1,spacing)
    _ , testa = FEMhat(grid,x, phi)
    right = problem2a(spacing)
    
    print("Right A matrix:")    
    print("")
    print(right)
    print("FEM matrix A:")
    print("")
    print(testa)

   
def main():
    problem2a(plot = True)
    plot_FEM()
    plot_errors()
    problem2c()
    
if __name__=="__main__":
    main()    