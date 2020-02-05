# -*- coding: utf-8 -*-
"""
Created on Wed Feb  5 15:17:01 2020

@author: halfar
"""

import numpy as np
from numba import jit
import time
import matplotlib.pyplot as plt



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
def gauss_seidel(phi, tol = 1e-6, step = 0.25):
    """ Implementation of Gauss-Seidel method

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
                
                phi_last[i,j] = step * (phi_last[i-1,j] + phi_first[i+1,j] + phi_last[i,j-1] + phi_first[i,j+1])
                
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
    
    
    
# Simple testing of jacobi
def test_jacobi():    
    x = np.linspace(0,1,SIZE) 
    y = np.linspace(0,1,SIZE)   
    
    X, Y = np.meshgrid(x,y)
    phi = np.zeros_like(X) 
    phi[:,0] = 1
    Z = jacobi(phi)
    
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_wireframe(X,Y,Z,rstride=1,cstride=1)
    ax.set_title("Jacob")
    ax.set_xlabel("x-axis")
    ax.set_ylabel("y-axis")
    plt.show()

# Simple testing of gauss-seidel    
def test_gauss_seidel():    
    x = np.linspace(0,1,SIZE) 
    y = np.linspace(0,1,SIZE)   
    
    X, Y = np.meshgrid(x,y)
    phi = np.zeros_like(X) 
    phi[:,0] = 1
    Z = gauss_seidel(phi)
    
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_wireframe(X,Y,Z,rstride=1,cstride=1)
    ax.set_title("Gauss-Seidel")
    ax.set_xlabel("x-axis")
    ax.set_ylabel("y-axis")
    plt.show()    
    
# Simple testing of SOR    
def test_SOR():    
    x = np.linspace(0,1,SIZE) 
    y = np.linspace(0,1,SIZE)   
    
    X, Y = np.meshgrid(x,y)
    phi = np.zeros_like(X) 
    phi[:,0] = 1
    Z = SOR(phi)
    
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_wireframe(X,Y,Z,rstride=1,cstride=1)
    ax.set_title("SOR")
    ax.set_xlabel("x-axis")
    ax.set_ylabel("y-axis")
    plt.show()    
    
SIZE = 50 # size of grid
def main():
    # first time ever with Numba :) i first let it compile these and then to the calculations with bigger data
    # Gives a lot of warning, but it works! Order of degree faster at least!
    jacobi(np.zeros((9,9))) 
    gauss_seidel(np.zeros((9,9))) 
    SOR(np.zeros((9,9))) 
    
    # Test Jacobi
    start_time = time.time()
    test_jacobi()
    print ("Jacobi did take compiled with Numba: {}s".format(time.time() - start_time))
    
    # Test Gauss-seidel    
    start_time = time.time()
    test_gauss_seidel()
    print ("Gauss-Seidel did take compiled with Numba: {}".format(time.time() - start_time))
    
    # Test SOR
    start_time = time.time()
    test_SOR()
    print ("SOR did take compiled with Numba: {}".format(time.time() - start_time))
    
 

if __name__=="__main__":
    main()