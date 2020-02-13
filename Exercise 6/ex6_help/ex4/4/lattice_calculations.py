"""
Calculating numbers of electrons and reciprocal lattices.
"""

from read_xsf_example import read_example_xsf_density
from scipy.integrate import simps
import numpy as np

def count_electrons(filename):
    # Function that integrates electron density over an area to get the number
    # of electrons, and returns it.
    
    # Read the file.
    rho, lattice, grid, shift = read_example_xsf_density(filename)

    # Define the axis points.
    x = np.linspace(0, lattice[0, 0], grid[0])
    y = np.linspace(0, lattice[1, 1], grid[1])
    z = np.linspace(0, lattice[2, 2], grid[2])

    # Arrays for storing the intermediate integral values.
    ints1d = np.zeros(grid[0])
    ints2d = np.zeros([grid[0], grid[1]])

    # Integarte the electron density.
    for i in range(grid[0]):
        for j in range(grid[1]):
            ints2d[i, j] = simps(rho[i, j, :], z)
        ints1d[i] = simps(ints2d[i, :], y)
    return simps(ints1d, x)
    
def reciprocal_lattice_vectors(filename):
    # A function that calculates and returns the reciprocal lattice vectors.
    
    # Read the file.
    rho, lattice, grid, shift = read_example_xsf_density(filename)
    
    # Solve the matrix equation.
    mat = np.linalg.solve((10**-10)*lattice, [[2*np.pi, 0, 0],[0, 2*np.pi, 0],[0, 0, 2*np.pi]])
    return np.transpose(mat)

def main():
    
    # Lattice 1
    print("Number of electrons in lattice 1:")
    print(count_electrons("dft_chargedensity1.xsf"))
    print("Reciprocal lattice:")
    print(reciprocal_lattice_vectors("dft_chargedensity1.xsf"))

    print("number of electrons in lattice 2:")
    print(count_electrons("dft_chargedensity2.xsf"))
    print("Reciprocal lattice:")
    print(reciprocal_lattice_vectors("dft_chargedensity2.xsf"))
    
main()