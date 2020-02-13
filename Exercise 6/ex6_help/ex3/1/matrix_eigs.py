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


def largest_eig(A, tol=1e-12, max_iterations=100000):
    """
    Calculates the largest eigenvalue using power method 
    """
    # cast to numpy array
    A = A.toarray()

    # random starting vector
    eig_vector = np.random.rand(A.shape[1])

    # placeholder, needed to compare last two results
    eig_value = 0
    for N in range(1,max_iterations):
        # calculate the matrix-by-vector product Ab
        b_k1 = np.dot(A, eig_vector)

        # save previous result to estimate convergence 
        prev = eig_value

        # calculate the norm
        eig_value = np.linalg.norm(b_k1)

        # re normalize the vector
        eig_vector = b_k1 / eig_value

        # stop when the error between last two results is small
        if(abs(prev - eig_value) < tol):
            print("Largest eigenvalue found with ", N, " iterations")
            eig_vector = np.reshape(eig_vector, [len(eig_vector), 1])
            return eig_value, eig_vector
    print("Max iterations (", N, ") reached, tolerance not reached")
    eig_vector = np.reshape(eig_vector, [len(eig_vector), 1])
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
    print('scipy eigsh estimate: ', eigs[0])
    
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
