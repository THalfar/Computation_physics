"""
Problem 2 
1.2.2020
Marika Honkanen
"""
from numpy import *
from scipy.integrate import simps

from ex4_help.read_xsf_example import read_example_xsf_density

"""
input:
    lattice -- matrix of lattice vectors in real space
    rho -- matrix which includes electron density in every point of simulation cell
output:
    rho_dz -- number of electrons in simulation cell (float)
working:
The main idea is to calculate 3D integral over a simulation cell.
The integrand is, of course, electron density in the cell.
Because of non-orthogonal lattice vectors, they are converted
into very weird alpha space. After converting, integrand is 
density times determinant of lattice matrix.
3D integral is calculated using Simpson's method.
"""


def particle_density(lattice, rho):
     lattice_tr = transpose(lattice)
     determinant = linalg.det(lattice_tr)
     integrand = rho*determinant

     alfa_x = linspace(0,1,shape(rho)[0])
     alfa_y = linspace(0,1,shape(rho)[1])
     alfa_z = linspace(0,1,shape(rho)[2])

     rho_dx = zeros((len(alfa_y), len(alfa_z)))
     for i in range(len(alfa_y)):
             for j in range(len(alfa_z)):
                 rho_dx[i][j] = simps(integrand[:,i,j],alfa_x)
    
     rho_dy = zeros(len(alfa_z))
     for i in range(len(alfa_z)):
             rho_dy[i] = simps(rho_dx[:,i], alfa_y)
     rho_dz = simps(rho_dy,alfa_z)
     return(rho_dz)

"""
input:
    lattice -- matrix of lattice vectors in real space
output:
    B -- matrix of lattice vectors in reciprocal space
working:
Calculate reciprocal lattice vectors from real space lattice
vectors.
Eq: B_t*A = 2*pi*I, 
where B_t is transpose of lattice vectors in reciprocal space
and A is lattice vectors and I is identity matrix
"""

def reciprocal_space(lattice):
     lattice_tr = transpose(lattice)
     lattice_inv = linalg.inv(lattice_tr)
     I = identity(shape(lattice)[0])
     B = transpose(I*lattice_inv*2*pi)
     return(B)

"""
input:
   filename -- name of given file
output:
   ---
working:
Execute reciprocal_space and particle_density functions for given file.
"""


def execute_problem2(filename):
    rho, lattice, grid, shift = read_example_xsf_density(filename)
    particle_density(lattice, rho)
    reciprocal_space(lattice)
    print('For', filename,':')
    print('Number of electrons:', particle_density(lattice,rho))
    print('Reciprocal lattice vectors:')
    print(reciprocal_space(lattice))

def main():
    #filename = input("Give the name of xsf file: ")
    filename = 'ex4_help/dft_chargedensity1.xsf'
    execute_problem2(filename)
    filename = 'ex4_help/dft_chargedensity2.xsf'
    execute_problem2(filename)
    
if __name__=="__main__":
    main()

