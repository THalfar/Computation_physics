"""
Problem 2 of exercise set 4
Calculate number of electrons in simulation cell,
and the reciprocal lattices
Calculations is done for files dft_chargedensity1.xsf
and dft_chargedensity2.xsf

"""

import read_xsf_example as rxsf
import numpy as np

def lattice_analysis(filename):
    """
    Calculate the number of electrons in cell,
	and display the number of electrons and the
	reciprocal lattice vector
	
    :param: filename: given filename to read data from
    """ 
    rho, lattice, grid, shift = rxsf.read_example_xsf_density(filename)
    # Unit volume of block is V = (a x b) dot c, normalized with grid size
    volume = np.dot(np.cross(lattice[0,:]/grid[0], lattice[1,:]/grid[1]), lattice[2,:]/grid[2])
    # Each block has electron density time volume electrons in it, so total electron amount is the
    # sum of all the electrons in blocks.
    Ne = np.sum(rho)*volume
    
    # reciprocal lattice is defined by B^T = 2pi*inv(A)
    reciLattice = np.transpose(2*np.pi*np.linalg.inv(lattice))
    
    print(f"Total electrons in {filename}: {Ne}")
    print("Reciprocal lattice vectors", reciLattice[0, :], reciLattice[1, :], reciLattice[2, :])
    
    
def main():
    lattice_analysis('dft_chargedensity1.xsf')
    lattice_analysis('dft_chargedensity2.xsf')

if __name__ == "__main__":
    main()
