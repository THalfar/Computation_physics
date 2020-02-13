"""
Determining the number of electrons in the simulation

Related to FYS-4096 Computational Physics
exercise 4 assignments.

By Roman Goncharov on January 2020
"""
from numpy import *
from matplotlib.pyplot import *
import scipy.sparse as sp
import scipy.sparse.linalg as sla
from scipy.integrate import simps

"""
Importing read from file funcion that returns:
- electron density,
- lattice,
- grid (shape of the electon density matrix),
- shift-parameter
"""
from read_xsf_example import read_example_xsf_density

def recip_vec(lattice):
    """
    Reciprocal vectors calculation. 
    For the details see Week 4 FYS-4096 Computational Physics lecture slides
    
    :param lattice: Lattice matrix
    :return: Reciprocal matrix consisting all 
             the reciprocal vectors that printed out
    """
    n, m = lattice.shape
    identity(m)
    B = 2*pi*transpose(linalg.inv(transpose(lattice)))
    for i in range(m):
        print(i+1,"reciprocal vector ", B[:,i])
    return B

def part_num(rho,lattice):
    """
    Numer of particles calculation. 
    It is an cell integral of electron density by origin vector
    Here the fractional distance is used 
    (for details see Week 4 FYS-4096 Computational Physics lecture slides)
    """
    x= linspace(0,1,rho.shape[0])
    y= linspace(0,1,rho.shape[1])
    z= linspace(0,1,rho.shape[2])
   
    num =round(simps(simps(simps(rho*linalg.det(lattice), dx=x[1]-x[0],axis=0), dx=y[1]-y[0],axis=0),dx=z[1]-z[0]))
    return num
    
              
def main():
    """
    All the necessary calculations perfomed in main()
    """
    ###reading of the first XSF-file###
    filename = 'dft_chargedensity1.xsf'
    rho, lattice, grid, shift = read_example_xsf_density(filename)
    print("Lattice matrix 1: ",lattice)
    B = recip_vec(lattice)
    print(part_num(rho,lattice))
    ###end for the first file###

    ###reading of the second XSF-file###
    filename = 'dft_chargedensity2.xsf'
    rho, lattice, grid, shift = read_example_xsf_density(filename)
    print("Lattice matrix 2: ",lattice)
    B = recip_vec(lattice)
    print(part_num(rho,lattice))
    ###end for the first file###

    
    
    
if __name__=="__main__":
    main()
