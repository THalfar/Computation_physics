"""
For Computational physics
Problem 2
"""

from numpy import *
from scipy import integrate

def read_example_xsf_density(filename):
    lattice=[]
    density=[]
    grid=[]
    shift=[]
    i=0
    start_reading = False
    with open(filename, 'r') as f:
        for line in f:
            if "END_DATAGRID_3D" in line:
                start_reading = False
            if start_reading and i==1:
                grid=array(line.split(),dtype=int)
            if start_reading and i==2:
                shift.append(array(line.split(),dtype=float))
            if start_reading and i==3:
                lattice.append(array(line.split(),dtype=float))
            if start_reading and i==4:
                lattice.append(array(line.split(),dtype=float))
            if start_reading and i==5:
                lattice.append(array(line.split(),dtype=float))
            if start_reading and i>5:            
                density.extend(array(line.split(),dtype=float))
            if start_reading and i>0:
                i=i+1
            if "DATAGRID_3D_UNKNOWN" in line:
                start_reading = True
                i=1
    
    rho=zeros((grid[0],grid[1],grid[2]))
    ii=0
    for k in range(grid[2]):
        for j in range(grid[1]):        
            for i in range(grid[0]):
                rho[i,j,k]=density[ii]
                ii+=1

    # convert density to 1/Angstrom**3 from 1/Bohr**3
    a0=0.52917721067
    a03=a0*a0*a0
    rho/=a03
    return rho, array(lattice), grid, shift


def amount_of_electrons(rho,lattice,grid):
    """
    Counts the amount of electrons from 3D-integration of the electron density distribution
    over the volume of the cell.
    """

    # Later we are going to do a change of variables, hence the
    # boundaries from 0 to 1.
    x = linspace(0, 1.0, grid[0])
    y = linspace(0, 1.0, grid[1])
    z = linspace(0, 1.0, grid[2])

    # Integrating over volume of the cell.
    int_dx = integrate.simps(rho, dx = x[1]-x[0] ,axis =  0)
    int_dy = integrate.simps(int_dx,dx=y[1]-y[0] , axis = 0)

    # Change of variable in integration with determinant of Jacobian matrix.
    # Jacobian matrix in this case is the lattice matrix.
    # This is crucial in the second case, where the cell is distorted.
    I = integrate.simps(int_dy,dx = z[1]-z[0] )*linalg.det(lattice)
    return I


def main():

    # For the first case.
    print()
    filename1 = 'dft_chargedensity1.xsf'
    rho1, lattice1, grid1, shift1 = read_example_xsf_density(filename1)
    print(grid1)

    # Amount of the electrons from integral.
    Integral1 = amount_of_electrons(rho1, lattice1, grid1)
    print("Amount of the electrons in the first case:", Integral1)
    print()

    # Counts the reciprocal vectors from B^T*A = 2*pi*I.
    A1 = transpose(lattice1)
    B_transpose1 = 2*pi*linalg.inv(A1)
    # reciprocal lattice vector
    B1 = transpose(B_transpose1)

    print("Reciprocal lattice vectors for the 1. case")
    print()
    print(B1)


    # For the second case.
    filename2 = 'dft_chargedensity2.xsf'
    rho2, lattice2, grid2, shift2 = read_example_xsf_density(filename2)
    A2 = transpose(lattice2)

    print()
    Integral2 = amount_of_electrons(rho2,lattice2,grid2)
    print("Amount of the electrons in the second case:", Integral2)
    print()

    # Counts the reciprocal vectors from B^T*A = 2*pi*I.
    A2 = transpose(lattice2)
    B_transpose2 = 2*pi*linalg.inv(A2)
    # reciprocal lattice vector
    B2 = transpose(B_transpose2)
    print("Reciprocal lattice vectors for the 2. case")
    print()
    print(B2)

    
if __name__=="__main__":
    main()



