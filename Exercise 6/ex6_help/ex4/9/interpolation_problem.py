"""
problem 3 and 4
1.2.2020
Marika Honkanen
"""
from numpy import *
from matplotlib import pyplot as plt

from ex4_help.read_xsf_example import read_example_xsf_density
from ex4_help.spline_class import spline

"""
input:
    lattice -- lattice matrix
    rho -- electron density matrix
    point0 -- an endpoint in line in which density will be interpolated
    point1 -- the other endpoint in line in which density will be interpolated
output:
    t -- a vector from 0 to 1 with 500 steps. This vector is a given line
         converted into annoying alpha-space(?) (linspace)
    density -- electron density along t (linspace)
working:
Calculate 3D interpolation from given density matrix along line between given points.
x y and z are lattice vectors of cell in alpha space 
(i.e. vectors are orthogonal and lenght is 1).
Lattice vectors and density matrix are given to spline function 
to generate data which will be used for interpolation.
r contains 500 points on given line in real space and alpha is same line
converted to alpha space.
Finally, line points in alpha space are given to evl3d function to 
interpolate density in line.
"""


def interpolate(lattice, rho, point0, point1):

    x = linspace(0, 1, shape(rho)[0])
    y = linspace(0, 1, shape(rho)[1])
    z = linspace(0, 1, shape(rho)[2])
    #X, Y, Z = meshgrid(x,y,z)
    spl3d = spline(x=x,y=y,z=z,f=rho,dims=3)

    t = linspace(0,1,500)
    r = zeros((len(t),len(point1)))
    alpha = zeros_like(r)
    inv_lattice = linalg.inv(transpose(lattice))

    for i in range(500):
         r[i] = point0 + t[i]*(point1-point0)
         alpha[i] = mod(inv_lattice.dot(r[i]),1)

    X = alpha[:,0]
    Y = alpha[:,1]
    Z = alpha[:,2]
    
    density = zeros_like(X)
    for i in range(len(X)):
         density[i] = spl3d.eval3d(X[i],Y[i],Z[i])
    
    return t, density  
"""
input:
    t -- a vector from 0 to 1 with 500 steps. This vector is a given line
         converted into alpha-space(?) (linspace)
    density -- electron density along t (linspace) 
output:
    ---
working:
To plot density as function of t. To save fig with given name.
"""
    
def plotting(t,density, fig_name):
    plt.plot(t, density)
    plt.xlabel('t')
    plt.ylabel('electron density [1/Å³]')
    plt.savefig(fig_name)
    plt.clf()

"""
Execute the problem 4
"""

def problem4():
    filename = 'ex4_help/dft_chargedensity2.xsf'
    r0 = array([2.9996,2.1733,2.1462])
    r1 = array([8.7516,2.1733,2.1462])
    fig_name = 'problem4_second.png'
    rho, lattice, grid, shift = read_example_xsf_density(filename)    
    t, density = interpolate(lattice, rho, r0, r1)
    plotting(t, density, fig_name)
    
    r0 = array([-1.4466,1.3073,3.2115])
    r1 = array([1.4361,3.1883,1.3542])
    fig_name = 'problem4_first.png'
    t, density = interpolate(lattice, rho, r0, r1)
    plotting(t, density, fig_name)

"""
Execute the problem 3
"""

def problem3():
    filename = 'ex4_help/dft_chargedensity1.xsf'
    r0 = array([0.1,0.1,2.8528])
    r1 = array([4.45,4.45,2.8528])
    fig_name = 'problem3.png'
    rho, lattice, grid, shift = read_example_xsf_density(filename)    
    t, density = interpolate(lattice, rho, r0, r1)
    plotting(t, density, fig_name)

     
def main():
    problem4()
    problem3()          
    
if __name__=="__main__":
    main()

