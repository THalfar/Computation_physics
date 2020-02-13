""" 
---------- EXERCISE 5 - problem 3 -----------
----- FYS-4096 - Computational Physics ------

Solves a partial differential equation with different
methods [jacobi, gauss-seidel,sor] and plots the resulting
surfaces.

:function: jacobi        creates an estimate for the partial with jacobi method
:function: gauss_seidel  creates an estimate for the partial with 
                         gauss-seidel method
:function: sor           creates an estimate for the partial with sor method
:function: solve_partial uses estimation function as many times until tolerance
                         is met
:function: main          creates the plots and calls everything needed                         
"""

import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt


def jacobi(phi, rho, h, N):
    """
    Algorithm to solve the 2d partial differential 
    new estimation is calculateed with jacobi method
    see equations in for example in FYS-4096 materials
    :param: phi    last estimation
    :param: rho    some function
    :param: h      grid separation
    :param: N      grid size
    
    :return: new estimation of the solution
    """
    phi_new = 1.0*phi
    for i in range(1,N-1):
        for j in range(1, N-1):
            phi_new[i, j] = 1.0/4.0*(phi[i+1, j] + phi[i-1, j] + \
                            phi[i, j+1] + phi[i, j-1] + h**2*rho[i, j])
    return phi_new

def gauss_seidel(phi, rho, h, N):
    """
    Algorithm to solve the 2d partial differential 
    new estimation is calculateed with gauss-seidel method
    see equations in for example in FYS-4096 materials
    :param: phi    last estimation
    :param: rho    some function
    :param: h      grid separation
    :param: N      grid size
    
    :return: new estimation of the solution
    """
    phi2 = 1.0*phi
    for i in range(1, N - 1):
        for j in range(1, N - 1):
            phi2[i, j] = 1.0 / 4.0 * (phi2[i + 1, j] + phi2[i - 1, j] + \
                         phi2[i, j + 1] + phi2[i, j - 1] + h ** 2 * rho[i, j])
    return phi2

def sor(phi, rho, h, N, omega=1.8):
    """
    Algorithm to solve the 2d partial differential 
    new estimation is calculateed with sor method
    see equations in for example in FYS-4096 materials
    :param: phi    last estimation
    :param: rho    some function
    :param: h      grid separation
    :param: N      grid size
    :param: omega  constant for sor method
    
    :return: new estimation of the solution
    """
    phi2 = 1.0 * phi
    for i in range(1, N - 1):
        for j in range(1, N - 1):
            phi2[i, j] = (1-omega)*phi2[i, j] + omega / 4.0 * (phi2[i + 1, j] +\
                         phi2[i - 1, j] + phi2[i, j + 1] + phi2[i, j - 1] +    \
                         h ** 2 * rho[i, j])
    return phi2

def solve_partial(N,method="jacobi",tol=1e-3):
    """
    Solves 2d partial differential equation in one of 
    three methods
    :param: N       grid size
    :param: method  method to be calculated with [jacobi, gaussseidel, sor]
    :param: tol     tolerance the calculation is done with
    
    :return: returns the value of the differential at every point and the 
             grid along x and y 
    """
    phi = np.zeros((N, N))
    x = np.linspace(0, 1, N)
    y = np.linspace(0, 1, N)
    rho = 0*np.ones((N, N))
    h = x[1]-x[0]
    phi[-1, :] = 1
    
    # choose update method
    if method=="jacobi": next = jacobi
    if method=="gaussseidel": next = gauss_seidel
    if method=="sor": next = sor
    
    # calculate PDE for the grid until it changes less than tol
    loops = 0
    while True:
        loops += 1
        
        # next position
        phi2 = next(phi, rho, h, N)
        if np.max(np.abs(phi2-phi)) < tol:
            break
        phi = phi2
        
    print(f"{method} looped {loops} times")
    return phi, x, y

def main():
    
    # number of grid points
    N = 21
    fig = plt.figure()
    
    # solve the partial for every point and then plots it for jacobi method
    phi, x, y = solve_partial(N, "jacobi")
    X, Y = np.meshgrid(x, y)
    ax1 = fig.add_subplot(131, projection='3d')
    ax1.plot_wireframe(X, Y, phi, rstride=1, cstride=1)
    ax1.set_title('Jacobi')
    
    # solve the partial for every point and then plots it for gauss-seidel method
    phi, x, y = solve_partial(N, "gaussseidel")
    X, Y = np.meshgrid(x, y)
    ax2 = fig.add_subplot(132, projection='3d')
    ax2.plot_wireframe(X, Y, phi, rstride=1, cstride=1)
    ax2.set_title('Gauss-Seidel')
    
    # solve the partial for every point and then plots it for sor method
    phi, x, y = solve_partial(N,"sor")
    X, Y = np.meshgrid(x, y)
    ax3 = fig.add_subplot(133, projection='3d')
    ax3.plot_wireframe(X, Y, phi, rstride=1, cstride=1)
    ax3.set_title('SOR')
    plt.show()

if __name__ == "__main__":
    main()
