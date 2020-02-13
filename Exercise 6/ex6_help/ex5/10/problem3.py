"""
Poisson equation and relaxation methods

Related to FYS-4096 Computational Physics
exercise 5 assignments.

By Roman Goncharov on February 2020
"""
import numpy as np
import matplotlib.pyplot  as plt
from mpl_toolkits.mplot3d import Axes3D

N = 21 # spacing for solution
L = 1 # given box size

Phi = np.zeros((N,N))

def boundaries(Phi):
    """Boundary conditions"""
    Phi[0,:] = 0
    Phi[N-1,:] = 0
    Phi[:,0] = 0
    Phi[:,N-1] = 1
###

def jacobi(Phi,tol):

    """
    Jacobi method for solving Laplace equation
    For details see Week 4 FYS-4096 Computational Physics lecture slides
    Phi = given 2D function
    """
    count = 0 # counter
    err = tol+1 #to start the loop
    boundaries(Phi)
    
    while err > tol and count < 10000:
        err = 0.
        phi = np.copy(Phi)
        for i in range(1,N-1):
            for j in range(1,N-1):
                Phi[i,j] = (phi[i+1,j] + phi[i-1,j] + phi[i,j+1] + phi[i,j-1])/4
                err += abs(Phi[i,j]-phi[i,j])
        boundaries(Phi) #keep the boundaries in safety
        count += 1
    print('There were ',count,'Jacobi iterations')
    return Phi

def gauss_seidel(Phi,tol):

    """
    Gauss-Seidel  method for solving Laplace equation
    For details see Week 4 FYS-4096 Computational Physics lecture slides
    Phi = given 2D function
    """
    count = 0
    err = tol+1
    boundaries(Phi)
    while err > tol and count < 10000:
        err = 0.
        phi = np.copy(Phi)
        for i in range(1,N-1):
            for j in range(1,N-1):
                Phi[i,j] = (phi[i+1,j] + Phi[i-1,j] + phi[i,j+1] + Phi[i,j-1])/4
                err += abs(Phi[i,j]-phi[i,j])
        boundaries(Phi)
        count += 1
    print('There were ',count,'Gauss-Seidel iterations')
    return Phi

def sor(Phi,tol):
    """
    SOR method for solving Laplace equation
    For details see Week 4 FYS-4096 Computational Physics lecture slides
    Phi = given 2D function
    """
    count = 0
    omega = 1.8
    err = tol+1
    boundaries(Phi)
    while err > tol and count < 10000:
        err = 0.
        phi = np.copy(Phi)
        for i in range(1,N-1):
            for j in range(1,N-1):
                Phi[i,j] =(1-omega)*phi[i,j]+omega*(phi[i+1,j] + Phi[i-1,j] + phi[i,j+1] + Phi[i,j-1])/4
                err += abs(Phi[i,j]-phi[i,j])
        boundaries(Phi)
        count += 1
    print('There were ',count,'SOR iterations')
    return Phi

def main():
    """
    All the necessary calculations and plotting  perfomed in main()
    """
    
    """function Phi calculation"""
    Phi1 = jacobi(Phi,1e-8)
    Phi2 = gauss_seidel(Phi,1e-8)
    Phi3 = sor(Phi,1e-8)
    
    """plotting"""
    x = np.linspace(0,L,N)
    y = np.linspace(0,L,N)
    X,Y = np.meshgrid(x,y)

    fig1 = plt.figure()
    ax1 = fig1.add_subplot(111, projection='3d')
    ax1.plot_wireframe(X,Y,Phi1,rstride=1,cstride=1)
    ax1.set_title('Jacobi method')

    fig2 = plt.figure()
    ax2 = fig2.add_subplot(111, projection='3d')
    ax2.plot_wireframe(X,Y,Phi2,rstride=1,cstride=1)
    ax2.set_title('Gauss-Seidel method')

    fig3 = plt.figure()
    ax3 = fig3.add_subplot(111, projection='3d')
    ax3.plot_wireframe(X,Y,Phi3,rstride=1,cstride=1)
    ax3.set_title('SOR method')
    plt.show()
        
    
if __name__=="__main__":
    main()
