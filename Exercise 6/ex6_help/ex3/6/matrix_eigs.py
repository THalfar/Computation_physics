"""
FYS-4096 Computational physics 

1. Add code to function 'largest_eig'
- use the power method to obtain 
  the largest eigenvalue and the 
  corresponding eigenvector of the
  provided matrix

2. Compare the results with scipy's eigs
- this is provided, but you should use
  that to validating your power method
  implementation

Hint: 
  dot(A,x), A.dot(x), A @ x could be helpful for 
  performing matrix operations

"""


import numpy as np
import matplotlib.pyplot as plt
import scipy.sparse as sp
import scipy.sparse.linalg as sla
from scipy.integrate import simps


def eigenvalue(A,x):
    """ Calculate the eigenvalue for matrix A when eigenvector x is known
    
    Args:
        A (ndarray): Matrix
        x (ndarray): Eigenvector
    
    Returns:
        float: Eigenvalue
    """

    #Ax = A.dot(x)
    #return x.dot(Ax)
    Ax = A @ x
    return x @ Ax

def largest_eig(A, precision = 0.00001):
    """ Calculate the largest eigenvalue and the corresponding eigenvector of
    matrix A using the power method.
    
    Args:
        A (ndarray): Matrix
        precision (float, optional): Maximum error for eigenvalue
    
    Returns:
        eig_new (float): Largest eigenvalue of matrix A
        x_new (ndarray): Eigenvector corresponding to the largest eigenvalue
        n_iterations (int): Number of performed iterations
    """

    # Start with a random guess for eigenvector (size same as number of columns
    # in A.
    x_n = np.random.rand(A.shape[1])
    eig_n = eigenvalue(A, x_n)
    n_iterations = 0

    # Iterate new eigenvalues and -vectors until desired precision is obtained.
    while True:
        n_iterations += 1

        # make a new approximation for the eigenvector
        Ax = A @ x_n
        x_n1 = Ax / np.linalg.norm(Ax)  

        # make a new approximation for the eigenvalue based on new eigenvector
        eig_n1 = eigenvalue(A, x_n1)
        if np.abs(eig_n - eig_n1) < precision:
            break

        x_n = x_n1
        eig_n = eig_n1

    return eig_n1, x_n1, n_iterations


def main():
    grid = np.linspace(-5,5,100)
    grid_size = grid.shape[0]
    dx = grid[1]-grid[0]
    dx2 = dx*dx
    
    # make test matrix
    H0 = sp.diags(
        [
            -0.5 / dx2 * np.ones(grid_size - 1),
            1.0 / dx2 * np.ones(grid_size) - 1.0/(abs(grid)+0.8),
            -0.5 / dx2 * np.ones(grid_size - 1)
        ],
        [-1, 0, 1])
    
    # use scipy to calculate the largest eigenvalue
    # and corresponding vector
    eigs, evecs = sla.eigsh(H0, k=1, which='LA')

    # use your power method to calculate the same
    l,vec,iterations=largest_eig(H0,10**-10)
    
    # see how they compare
    print('largest_eig estimate: ', l)
    print('scipy eigsh estimate: ', eigs[0])
    
    # eigsh eigen vector
    psi0=evecs[:,0]
    norm_const=simps(abs(psi0)**2,x=grid)
    psi0=psi0/norm_const
    
    # largest_eig eigen vector 
    psi0_=vec
    norm_const=simps(abs(psi0_)**2,x=grid)
    psi0_=psi0_/norm_const
    
    plt.style.use('seaborn-bright')
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(grid,abs(psi0)**2,label='scipy eig')
    ax.plot(grid,abs(psi0_)**2,'r--',label='largest_eig')
    ax.set_xlabel(r'$x$')
    ax.set_ylabel('Largest eigenvector squared')
    ax.legend(loc=0)
    fig.savefig('problem3.pdf', dpi=200)
    plt.show()


if __name__=="__main__":
    main()
