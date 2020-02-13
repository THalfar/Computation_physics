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


def largest_eig(A, tol=1e-12):
    """
    Calculates largest eigenvalue and corresponding eigenvector with power method.
    :param A: matrix
    :param tol: error tolerance to stop calculating
    :return: eigenvalue and vector
    """
    max_iters = 1000
    A = A.toarray()
    # starting vector
    eig_vector = np.ones(A.shape[1])
    for i in range(max_iters):
        last_iter = eig_vector
        eig_vector = A @ eig_vector
        # normalize the vector
        eig_vector_norm = np.linalg.norm(eig_vector)
        eig_vector = eig_vector/eig_vector_norm
        diff =np.absolute(eig_vector - last_iter)
        # checks if all elements are withing tolerance of the step
        for i in range(len(diff)):
            # if not within tolerance continue iterations
            if diff[i] > tol:
                break
            # if we have found eigenvalue within tolerance calculate eigenvalue
            elif i == len(diff):
                eig_value = np.dot(np.dot(A,eig_vector),eig_vector)/np.dot(eig_vector,eig_vector)
                return eig_value, eig_vector
    print("max iters reached but here are values anwyay: ")
    eig_value = np.dot(np.dot(A, eig_vector), eig_vector) / np.dot(eig_vector, eig_vector)
    return eig_value, eig_vector

def main():
    grid = linspace(-5, 5, 100)
    grid_size = grid.shape[0]
    dx = grid[1] - grid[0]
    dx2 = dx * dx

    # make test matrix
    H0 = sp.diags(
        [
            -0.5 / dx2 * np.ones(grid_size - 1),
            1.0 / dx2 * np.ones(grid_size) - 1.0 / (abs(grid) + 0.8),
            -0.5 / dx2 * np.ones(grid_size - 1)
        ],
        [-1, 0, 1])

    # use scipy to calculate the largest eigenvalue
    # and corresponding vector
    eigs, evecs = sla.eigsh(H0, k=1, which='LA')

    # use your power method to calculate the same
    l, vec = largest_eig(H0)

    # see how they compare
    print('largest_eig estimate: ', l)
    print('scipy eigsh estimate: ', eigs)

    # eigsh eigen vector
    psi0 = evecs[:, 0]
    norm_const = simps(abs(psi0) ** 2, x=grid)
    psi0 = psi0 / norm_const

    # largest_eig eigen vector
    psi0_ = vec # used to be vec[: , 0] but my code already produces 1 dimensional vector
    norm_const = simps(abs(psi0_) ** 2, x=grid)
    psi0_ = psi0_ / norm_const

    plot(grid, abs(psi0) ** 2, label='scipy eig. vector squared')
    plot(grid, abs(psi0_) ** 2, 'r--', label='largest_eig vector squared')
    legend(loc=0)
    show()


if __name__ == "__main__":
    main()
