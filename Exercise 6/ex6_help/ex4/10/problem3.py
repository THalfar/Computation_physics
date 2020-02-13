from read_xsf_example import read_example_xsf_density
from spline_class import *
import numpy as np
import matplotlib.pyplot as plt

def interpolation_to_line(rho,grid,lattice,x1,y1,z1,x2,y2,z2, n):
    """

    :param x1: starting point x-coordinate
    :param y1: starting point y-coordinate
    :param z1: starting point z-coordinate
    :param x2: ending point x-coordinate
    :param y2: ending point y-coordinate
    :param z2: ending point z-coordinate
    :param n: number of points
    :return:
    """
    # coordinates for each density point
    x =np.linspace(0, lattice[0][0], grid[0])
    y = np.linspace(0, lattice[0][0], grid[1])
    z = np.linspace(0, lattice[0][0], grid[2])
    spl3d = spline(x=x, y=y, z=z, f=rho, dims=3)
    # coordinates on line
    xx = np.linspace(x1, x2, n)
    yy = np.linspace(y1, y2, n)
    zz = np.linspace(z1, z2, n)
    # values on line
    F = spl3d.eval3d(xx, yy, zz)
    # creating new figure
    fig1d = plt.figure()
    ax1d = fig1d.add_subplot(111)
    # plotting, only taking diagonal elements form eval3d since it produces matrix of all
    # combinations of x, y and z.
    F_diag = np.zeros(len(xx))
    for i in range(len(xx)):
        F_diag[i] = F[i][i][i]
    ax1d.plot(xx, F_diag, label= "interpolation")
    # plot legend
    ax1d.legend(loc=0)

    plt.show()


def main():
    filename = 'dft_chargedensity1.xsf'
    rho, lattice, grid, shift = read_example_xsf_density(filename)
    # runs too slowly to use 500 points for some reason i couldn't figure out
    interpolation_to_line(rho,grid,lattice, 0.1, 0.1, 2.8528, 4.45, 4.45, 2.8528, 50)

if __name__=="__main__":
    main()
