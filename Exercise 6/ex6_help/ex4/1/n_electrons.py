""" 
-------- EXERCISE 4 - problem 2 ------------
----- FYS-4096 - Computational Physics -----

Calculates number of electrons in simulated lattice cell
and reciprocal lattice vectors for files 
dft_chargedensity1.xsf and dft_chargedensity2.xsf

:function: calc_nelectron: calculates the number of electrons with help from
           read_xsf_example.py
:function: calc_vol      : calculates the unit volume of the simulated lattice
:function: rec_lattice   : forms the reciprocal lattice vectors
"""

import read_xsf_example as r_xsf
import numpy as np

def calc_nelectron(filename):
    """
    Calculates the number of electrons in the simulated lattice cell
    
    :param: filename : filename for the lattice cell (.xsf - file)
    
    :return:           Number of electrons in the simulated cell    
    """
    
    # rho:     electron density in unit volume (one for each point)
    # lattice: lattice vectors                          (3 by 3)
    # grid:    grid size in lattice vector diection     (3 by 1)
    # shift:   unknown
    rho, lattice, grid, shift = r_xsf.read_example_xsf_density(filename)

    # solve the unit volume of lattice
    unit_vol = calc_vol(lattice,grid)
    
    # Sum over all rho*unit volume
    
    return np.sum(rho)*unit_vol, lattice
    
def calc_vol(lattice,grid):
    """
    Unit volume calculated with V = (a x b) dot c
    with a,b,c normalized with grid size
    
    :param: lattice: lattice vectors
    :param: grid   : grid size in all directions
    
    :return:  unit volume of lattice
    """
    
    return np.dot(np.cross(lattice[:,0]/grid[0], lattice[:,1]/grid[1]), \
                  lattice[:,2]/grid[2])
    
def rec_lattice(lattice):
    """
    Calculates reciprocal lattice vectors
    defined by [b1 b2 b3]^T = 2pi[a1 a2 a3]^-1
    
    :param: lattice: lattice vectors (column vectors)
    
    :return:       : returns reciprocal lattice vectors
    """
    
    return np.transpose(2*np.pi*np.linalg.inv(lattice))
    

def main():
    
    # Calculate for first file
    filename = 'dft_chargedensity1.xsf'
    
    N_e, lattice = calc_nelectron(filename)
    reciprocal = rec_lattice(lattice)
    
    print(f"Number of electrons in {filename}: {N_e}")
    print(f"Reciprocal lattice vectors are: {reciprocal[0, :]},"\
          f" {reciprocal[1, :]}, {reciprocal[2, :]}")
    print()
    
    # Calculate for second file
    filename = 'dft_chargedensity2.xsf'
    N_e, lattice = calc_nelectron(filename)
    reciprocal = rec_lattice(lattice)
    
    print(f"Number of electrons in {filename}: {N_e}")
    print(f"Reciprocal lattice vectors are: {reciprocal[0, :]},"\
          f" {reciprocal[1, :]}, {reciprocal[2, :]}")
          

if __name__ == "__main__":
    main()
