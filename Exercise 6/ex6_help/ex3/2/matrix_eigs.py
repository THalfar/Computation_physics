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


def largest_eig(A,tol=1e-12):
    """
    :param: A     matrix whose eigen vector and (max) eigenvalue is searched
    :param: tol   tolerance of the convergence
    
    :return: eig_value, eig_vector   largest eigen_value approximated with
                                     power method and corresponding
                                     eigen_vector
    """
   
   # create original estimate that's linearly dependent on 
   # every eigenvector
    eig_vector = ones((A.shape[1],1))
    eig_vector = eig_vector/linalg.norm(eig_vector)
    
    # initiate some values
    dif = 10
    nth = A
    eig_value = 10
        
    while dif > tol:
        
        last_eig = eig_value
        # A^n
        nth = dot(nth,A)
        
        # new eigenvector approximation thats normalized
        eig_vector = A.dot(eig_vector)
        eig_vector = eig_vector/linalg.norm(eig_vector)
        
        eig_value = 0
        i = 0
        
        # eigen value then is approximated to be the mean of 
        # A*v divided elementwise with v
        for x in A.dot(eig_vector):
            eig_value += x/eig_vector[i]
            i += 1
        
        eig_value = eig_value/(i)
        
        # absolute difference
        dif = abs(last_eig - eig_value)
        
    
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
