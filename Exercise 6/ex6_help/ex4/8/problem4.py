"""
Electron density along a line (2/2)

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
    filename = 'dft_chargedensity2.xsf'
    rho, lattice, grid, shift = read_example_xsf_density(filename)
    n = 500 # number of points in between

    """
    Line points from assignment

    Line function in parametric form:  r(t) = r_0 + t*(r_1-r_0)
    """
    r0 = array([-1.44660, 1.30730,  3.21150])
    r1 = array([ 1.43610,  3.18830,  1.35420])

    r00 = array([2.99960,  2.17330,  2.14620])
    r11 = array([8.75160,  2.17330,  2.14620])

    t = linspace(0,1,n)

    """
    Since the lattice matrix is not a diagonal,
    we use fractional distances
    (for details see Week 4 FYS-4096 Computational Physics lecture slides)
    """
    new_lat = linalg.inv(transpose(lattice)) # A⁻¹ 

    """fractional vectors"""
    x = linspace(0,1, rho.shape[0])
    y = linspace(0,1, rho.shape[1])
    z = linspace(0,1, rho.shape[2])

    """Interpolation along a line"""
    rho_spline = spline(x=x,y=y,z=z,f=rho,dims=3)

    rho_line1 = zeros((n,))
    for i in range(n):
        xx = new_lat.dot(r0)+t[i]*new_lat.dot((r1-r0))
        rho_line1[i] =rho_spline.eval3d(xx[0],xx[1],xx[2])


    rho_line2 = zeros((n,))
    
    """ 
    Here we have to redefine one of the given point-vectors 
    by taking modulus (periodic movement)
    (for details see Week 4 FYS-4096 Computational Physics lecture slides)
    """
    unit = ones(3)
    for i in range(n):
        xx = mod(new_lat.dot(r00)+t[i]*new_lat.dot((r11-r00)),unit)
        rho_line2[i] =rho_spline.eval3d(xx[0],xx[1],xx[2])


    
    """
    Plotting. Here the calculated values ploted as a function 
    of electron density along the line \rho(x)
    
    It can be compared with the Line Profile perfomed by VESTA
    """
    ###plotting of calculated results (1/2)###
    fig0 = figure() 
    ax0 = fig0.add_subplot(111)
    ax0.plot(t*linalg.norm(r1-r0), rho_line1,label=r"Electron density along the first line")
    ax0.legend(loc='upper right')
    ax0.set_xlabel(r'$x$')
    ax0.set_ylabel(r'$\rho$')
    # end of plotting
    
    ###plotting of calculated results (2/2)###
    fig00 = figure() 
    ax00 = fig00.add_subplot(111)
    ax00.plot(t*linalg.norm(r11-r00), rho_line2,label=r"Electron density along the second line")
    ax00.legend(loc='upper right')
    ax00.set_xlabel(r'$x$')
    ax00.set_ylabel(r'$\rho$')
    # end of plotting

    ###ploting of VESTA results (1/2)###
    fig = figure() 
    ax = fig.add_subplot(111)
    print("XCrySDen XSF file 1  (fractional points)")
    print("Point 1: 0.10620  0.28809  0.70860")
    print("Point 2:  0.40050  0.70261  0.29880")
    print("Number of points: 500")
    X, Y = [], []
    for line in open('problem4_1.txt', 'r'):
        values = [float(s) for s in line.split()]
        X.append(values[0])
        Y.append(values[1])

    ax.plot(X, Y,label=r"Electron density along the first line (VESTA)")
    ax.legend(loc='upper right')
    ax.set_xlabel(r'$x$')
    ax.set_ylabel(r'$\rho$')
    # end of plotting
    
    ###ploting of VESTA results (2/2)###
    fig1 = figure() 
    ax1 = fig1.add_subplot(111)
    print("XCrySDen XSF file 2  (fractional points)")
    print("Point 1:   0.76053  0.47893  0.47355")
    print("Point 2: 1.76053  0.47893  0.47355")
    print("Number of points: 500")
    XX, YY = [], []
    for line in open('problem4_2.txt', 'r'):
        values = [float(s) for s in line.split()]
        XX.append(values[0])
        YY.append(values[1])

    ax1.plot(XX, YY,label=r"Electron density along the second line (VESTA)")
    ax1.legend(loc='upper right')
    ax1.set_xlabel(r'$x$')
    ax1.set_ylabel(r'$\rho$')
    # end of plotting
    
    show()
    
if __name__=="__main__":
    main()
