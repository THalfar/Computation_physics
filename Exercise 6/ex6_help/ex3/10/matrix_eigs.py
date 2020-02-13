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

    eig_vector = np.random.rand(A.shape[0], 1)
    eig_vector = eig_vector / np.linalg.norm(eig_vector)

    MAX_ITERS = 1e5
    current_iteration = 0
    difference = 1
    B = A  # B will become A^n at the while-loop, initializing as A

    eig_value = 0

    print("Starting largest_eig. Max iterations is set at {:.0f}.".format(MAX_ITERS))

    while difference > tol and current_iteration  < MAX_ITERS:
        old_eig_value = eig_value
        B = B.dot(A) # B^n * A = B^(n+1) (A == B)

        # Eigenvector approximation: (A.v)/|A.v|
        eig_vector = A.dot(eig_vector)
        eig_vector = eig_vector / np.linalg.norm(eig_vector)

        # Eigenvalue is calculated as mean of division A.v / v
        eig_value = np.mean(A.dot(eig_vector)/eig_vector)

        # Difference to last eigen value (for loop stop criteria: this needs to be lower than specified tolerance)
        difference = np.abs(eig_value - old_eig_value)

        # Increase counter and print info every 500th iteration of the loop
        current_iteration += 1

        if(current_iteration % 500 == 0):
            print("Iteration {:}, difference is now {:}".format(current_iteration, difference))

    print("Largest estimate found! Iteration {:}, difference to last {:} < tol ({})".
          format(current_iteration, difference, tol))

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
