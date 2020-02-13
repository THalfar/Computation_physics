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


from numpy import *
from matplotlib.pyplot import *
import scipy.sparse as sp
import scipy.sparse.linalg as sla
from scipy.integrate import simps

# This seems not a good method. Convergence takes a long time in some cases!
def largest_eig(A,tol=1e-12):
    """
    Uses power method to find the biggest eigenvalue and corresponding
    eigen vector

    Parameters
    ----------
    A : scipy.sparse matrix
        Matrix from which the eigenvalues and vector is calculated
    tol : float
        Tolerance when stop iterating. The default is 1e-12.

    Returns
    -------
    eig_value : float 
        Maximum eigenvalue of matrix A
    eig_vector : np.array
        Corresponding eigenvector of maximum eigenvalue

    """
    # Initialize a random starting point
    # Because random, it might take a long time to converge
    x = np.random.rand(A.shape[0],1)
    
    while True:
        x_cand = A.dot(x) / linalg.norm(A.dot(x)) # Calculate next canditate
        #if canditate norm change below tolerance stop iterating
        if linalg.norm(x-x_cand) < tol: 
            eig_vector = x_cand
            eig_value = linalg.norm(A.dot(x_cand)) /linalg.norm(x_cand)
            break
        
        else:
            x = x_cand # continue iteration with new x value
   
    return eig_value, eig_vector


def main():
    grid = linspace(-5,5,100)
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
    l,vec=largest_eig(H0)
    
    # see how they compare
    print('largest_eig estimate: ', l)
    print('scipy eigsh estimate: ', eigs)
    
    # eigsh eigen vector
    psi0=evecs[:,0]
    norm_const=simps(abs(psi0)**2,x=grid)
    psi0=psi0/norm_const
    
    # largest_eig eigen vector 
    psi0_=vec[:,0]
    norm_const=simps(abs(psi0_)**2,x=grid)
    psi0_=psi0_/norm_const
    
    plot(grid,abs(psi0)**2,label='scipy eig. vector squared')
    plot(grid,abs(psi0_)**2,'r--',label='largest_eig vector squared')
    legend(loc=0)
    show()


if __name__=="__main__":
    main()
