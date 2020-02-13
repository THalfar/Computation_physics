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

"""


from numpy import *
from matplotlib.pyplot import *
import scipy.sparse as sp
import scipy.sparse.linalg as sla
from scipy.integrate import simps


def largest_eigs(A,tol=1e-9):
    """
    Simple power method code needed here

    and provided.
    """
    N=A.shape[0]
    u=random.rand(N,1)
    u=u/linalg.norm(u)
    uold=1.0*u
    dtol=100
    iters=0
    while dtol>tol and iters<100000:
        u=A.dot(u)
        u=u/linalg.norm(u)
        dtol=linalg.norm(u-uold)/N
        uold=1.0*u
        iters+=1
    print(iters)
    return mean(A.dot(u)/u),u
    #return eig_value, eig_vector

def main():
    grid = linspace(-5,5,100)
    grid_size = grid.shape[0]
    dx = grid[1]-grid[0]
    dx2 = dx*dx
    
    H0 = sp.diags(
        [
            -0.5 / dx2 * ones(grid_size - 1),
            1.0 / dx2 * ones(grid_size) - 1.0/(abs(grid)+0.8),
            -0.5 / dx2 * ones(grid_size - 1)
        ],
        [-1, 0, 1])
    
    eigs, evecs = sla.eigsh(H0, k=1, which='LA')
    
    l,vec=largest_eigs(H0)
    
    print('largest_eig estimate: ', l)
    print('scipy eigsh estimate: ', eigs)
    
    psi0=evecs[:,0]
    norm_const=simps(abs(psi0)**2,x=grid)
    psi0=psi0/norm_const
    print(norm_const)

    print(shape(vec))
    psi0_=vec[:,0]
    norm_const=simps(abs(psi0_)**2,x=grid)
    psi0_=psi0_/norm_const
    print(norm_const)

    plot(grid,abs(psi0)**2,label='scipy eig. vector squared')
    plot(grid,abs(psi0_)**2,'r--',label='largest_eig vector squared')
    legend(loc=0)
    show()


if __name__=="__main__":
    main()
