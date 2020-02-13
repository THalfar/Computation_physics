from numpy import *
import math

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

def num_of_electrons(rho,lattice):
    """
    calculates the num of electrons in simulation by calculating avarage electron density and multiplying it
    with volume
    :param rho: matrix of electron densities
    :param lattice: lattice vectors
    :return: number of electrons
    """
    # volume from vector triple product
    volume = dot(lattice[0],cross(lattice[1],lattice[2]))
    return rho.mean()*volume

def reciprocal_lattice(lattice):
    """
    calculates reciprocal lattices from normal lattice vectors
    with formula B^(T)=2piIA^(-1)
    :param lattice: lattice vectors
    :return:
    """
    return transpose(2*math.pi*identity(3)*linalg.inv(lattice))

def main():
    filename = 'dft_chargedensity1.xsf'
    rho, lattice, grid, shift = read_example_xsf_density(filename)
    print("# of electrons 1: ",num_of_electrons(rho, lattice))
    print("reciprocal lattice 1: ",reciprocal_lattice(lattice))
    filename = 'dft_chargedensity2.xsf'
    rho, lattice, grid, shift = read_example_xsf_density(filename)
    print("# of electrons 2: ",num_of_electrons(rho, lattice))
    print("reciprocal lattice 2: ", reciprocal_lattice(lattice))

if __name__=="__main__":
    main()



