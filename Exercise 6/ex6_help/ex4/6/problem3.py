"""
For problem 3
Should determine electron density along line
from (0.1, 0.1, 2.8528) to (4.45, 4.45, 2.8528).
"""

from read_xsf_example import read_example_xsf_density
from spline_class import *
from numpy import *
from matplot import *


def r(t):
    r0 = array([0.1, 0.1 ,2.8528])
    r1 = array([4.45, 4.45, 2.8528])
    return r0 + t*(r1-r0)

def main():

    filename1 = 'dft_chargedensity1.xsf'
    rho1, lattice1, grid1, shift1 = read_example_xsf_density(filename1)

    # Creates a spline-object
    x1  = linspace(0, lattice1[0][0], grid1[0])
    y1 = linspace(0,lattice1[1][1], grid1[1])
    z1 = linspace(0,lattice1[2][2], grid1[2])

    electron_spline = spline(x=x1,y=y1,z=z1,f=rho1,dims=3)

    # t must be from 0 to 1, so we are on the line.
    # I took only 50 points, because it took so long with 500.
    t = linspace(0, 1, 50)
    j = 0

    x = []
    y = []
    z = []

    while j < 50:
        x.append(r(t[j])[0])
        y.append(r(t[j])[1])
        z.append(r(t[j])[2])

        j = j + 1

    # Evaluate the electrn density in the line
    evaluated = electron_spline.eval3d(x,y,z)

    # Let's draw some picture
    X,Y = meshgrid(x,y)

    pcolor(X, Y, evaluated[...,int(len(z)/2)])
    title('Interpolated')
    show()


main()