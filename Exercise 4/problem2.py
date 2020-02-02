from numpy import *
from scipy.integrate import simps

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

def grid_cell_integration(rho, lattice):
    """
    Counts electron number using known density grid and corresponding lattice
    E_count = V / n * sum(grid) like in monte carlo integration, but now
    go over the grid. 

    Parameters
    ----------
    rho : np.array
        grid of electron densities
    lattice : np.array
        vectors that determine one cell

    Returns
    -------
    electron count in this unit cell

    """
    total_density = sum(rho)
    volume = linalg.det(lattice) #volume of unit cell -> det of lattice vectors
    grid_count = rho.size
    
    return (volume / grid_count) * total_density
    
def lattice2reciprocal(lattice):
    """
    Calculate reciprocal lattice using B' A = 2pi I
    -> B = 2pi * inv(A)'

    Parameters
    ----------
    lattice : np.array
        lattice vectors

    Returns
    -------
    B : np.array
        reciprocal lattice vectors

    """
    
    B = 2*pi*linalg.inv(lattice)
    B = transpose(B)
    return B
   
def main():
    filename = 'dft_chargedensity1.xsf'
    rho, lattice, grid, shift = read_example_xsf_density(filename)
    electron_count1 = grid_cell_integration(rho, lattice)
    recip1 = lattice2reciprocal(lattice)
    print("Electron count in cell 1: {}".format(electron_count1))
    print("Reciprocal lattice for cell 1:")
    print(recip1)
    
    print()
    
    filename = 'dft_chargedensity2.xsf'
    rho, lattice, grid, shift = read_example_xsf_density(filename)
    electron_count2 = grid_cell_integration(rho, lattice)
    recip2 = lattice2reciprocal(lattice)
    print("Electron count in cell 2: {}".format(electron_count2))
    print("Reciprocal lattice for cell 2:")
    print(recip2)

    
if __name__=="__main__":
    main()



