"""
Electron density along a line (1/2)

Related to FYS-4096 Computational Physics
exercise 4 assignments.

By Roman Goncharov on January 2020
"""
from matplotlib.pyplot import *
from numpy import *

"""
Importing read from file funcion that returns:
- electron density,
- lattice,
- grid (shape of the electon density matrix),
- shift-parameter
"""
from read_xsf_example import read_example_xsf_density

from spline_class import *



def main():
    """
    All the necessary calculations perfomed in main()
    """
    filename = 'dft_chargedensity1.xsf'
    rho, lattice, grid, shift = read_example_xsf_density(filename)
    
    n = 500 # number of points in between
    """
    Defining grid with shape of electorn density matrix
    in range of norm of lattice vectors
    """
    x = linspace(0, lattice[0,0], rho.shape[0])
    y = linspace(0, lattice[1,1], rho.shape[1])
    z = linspace(0, lattice[2,2], rho.shape[2])

    """
    Line points from assignment

    Line function in parametric form:  r(t) = r_0 + t*(r_1-r_0)
    """
    r0 = array([0.10000,0.10000,2.85280])
    r1 = array([4.45000,4.45000,2.85280])
    t = linspace(0,1,n)


    """Interpolation along a line"""

    rho_spline = spline(x=x,y=y,z=z,f=rho,dims=3)
    
    rho_line = zeros((n,))
    for i in range(n):
        xx = r0+t[i]*(r1-r0)
        rho_line[i] =rho_spline.eval3d(xx[0],xx[1],xx[2])
    


    """
    Plotting. Here the calculated values ploted as a function 
    of electron density along the line \rho(x)
    
    It can be compared with the Line Profile perfomed by VESTA
    """
    ###plotting of calculated results###
    fig = figure() 
    ax = fig.add_subplot(111)
    ax.plot(t*linalg.norm(r1-r0), rho_line,label=r"Electron density along a line")
    ax.legend(loc=0)
    ax.set_xlabel(r'$x$')
    ax.set_ylabel(r'$\rho$')
    # end of plotting




    ###ploting of VESTA results###
    print("XCrySDen XSF file (fractional points)")
    print("Point 1: 0.02196  0.02196  0.50000")
    print("Point 2: 0.97703  0.97703  0.50000")
    print("Number of points: 500")

    
    fig1 = figure()
    ax1 = fig1.add_subplot(111)
    X, Y = [], []
    for line in open('problem3.txt', 'r'):
        values = [float(s) for s in line.split()]
        X.append(values[0])
        Y.append(values[1])

    ax1.plot(X, Y,label=r"Electron density along a line (VESTA)")
    ax1.legend(loc=0)
    ax1.set_xlabel(r'$x$')
    ax1.set_ylabel(r'$\rho$')
    # end of plotting
    
    show()

    
if __name__=="__main__":
    main()
