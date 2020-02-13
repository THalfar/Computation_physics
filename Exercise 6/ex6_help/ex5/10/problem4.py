"""
Poisson equation and relaxation methods

Related to FYS-4096 Computational Physics
exercise 5 assignments.

By Roman Goncharov on February 2020
"""
import numpy as np
import matplotlib.pyplot  as plt



N = 21
L = 1
Phi0 = np.zeros((N,N))

"""Boundary conditions"""
def boundaries(Phi0):
    Phi0[7,6:15] = 1.
    Phi0[13,6:15] = -1.
    
def gauss_seidel_mod(Phi,tol):
    """
    Here is the same gauss-seidel method, except the new boundaries
    see problem3.py
    """
    boundaries(Phi)
    count = 0
    err = 2*tol
    while err > tol and count < 10:
        err = 0.
        phi = np.copy(Phi)
        for i in range(1,N-1):
            for j in range(1,N-1):
                Phi[i,j] = (phi[i+1,j] + Phi[i-1,j] + phi[i,j+1] + Phi[i,j-1])/4
                err += abs(Phi[i,j]-phi[i,j])
        count += 1
        boundaries(Phi)
    print('There were ',count,'Gauss-Seidel iterations')
    return Phi




def main():
    """
    All the necessary actions and plotting  perfomed in main()
    """
    """Field initiylization"""
    Phi = gauss_seidel_mod(Phi0,1e-6)
    
    Ex=np.zeros((N,N))
    Ey=np.zeros((N,N))
    dx = 0.01
    for i in range(1,N-1):
        for j in range(1,N-1):
            Ex[i,j]=-(Phi[i,j+1]-Phi[i,j-1])/(2*dx)
            Ey[i,j]=-(Phi[i+1,j]-Phi[i-1,j])/(2*dx)
    ###

    """plotting"""
    x = np.linspace(-L,L,N)
    y = np.linspace(-L,L,N)
    X,Y = np.meshgrid(x,y)


    fig, ax = plt.subplots(figsize=(9,9))

    ax.quiver(Y,X,Ey,Ex,scale=1000)
    ax.set_xlabel(r'$x$')
    ax.set_ylabel(r'$y$')

    plt.show()

if __name__=="__main__":
    main()
