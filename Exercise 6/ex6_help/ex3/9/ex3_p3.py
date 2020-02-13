""" Problem 3: find the largest eigenvalue/eigenvector. """

# Note: uses code from the provided matrix_eigs.py.
from numpy import *
from matplotlib.pyplot import *
import scipy.sparse as sp
import scipy.sparse.linalg as sla
from scipy.integrate import simps


def largest_eig(A, tol=1e-12):
    """ Find the largest eigenvalue and the corresponding eigenvector.

    Use the power method to calculate the largest eigenvector of the
    input matrix and the Rayleigh quotient to calculate the corresponding
    eigenvalue.

    Parameters
    ----------
    A : array
        Input matrix
    tol : tolerance (maximum difference between final iterations).

    Returns
    -------
    float
        The largest eigenvalue of matrix A.
    array
        The corresponding eigenvector.

    """

    # Initial guess for the eigenvector:
    x = np.random.rand(A.shape[0])

    eig_value_last = 0
    eig_value = np.nan
    error = np.inf
    iterations = 0

    # Power method: x converges to the eigenvector at large n.
    # Estimate error by the difference in eigenvalues between iterations.
    # Continue until error is less than tolerance.
    while error > tol:
        x = A.dot(x) / np.linalg.norm(A.dot(x))
        # Calculate corresponding eigenvalue using the Rayleigh quotient.
        eig_value = A.dot(x).dot(x) / x.dot(x)
        error = abs(eig_value - eig_value_last)
        eig_value_last = eig_value
        iterations += 1

    # Convert eig_vector to the required form.
    eig_vector = np.array([x]).transpose()

    return [eig_value], eig_vector


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
    print('Difference: ', str(eigs - l))
    
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
