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
    xx_original = np.linspace(x1, x2, n)
    yy = np.linspace(y1, y2, n)
    zz = np.linspace(z1, z2, n)
    # changing periodic functions
    xx = split_by_periodicity(lattice[0][0], xx_original)
    yy = split_by_periodicity(lattice[1][1], yy)
    zz = split_by_periodicity(lattice[2][2], zz)
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
    ax1d.plot(xx_original, F_diag, label= "interpolation")
    # plot legend
    ax1d.legend(loc=0)

    plt.show()

def change_of_base(lattice):
    """
    calculate change of basis matrix for transformation into quadrilateral base
    :param lattice: lattice vectors in matrix
    :return: change of basis matrix
    """
    # making the quadrilateral base
    square_lattice = np.identity(3)
    for i in range(3):
        square_lattice[i][i] = lattice[i][i]
    # calculating change of basis matrix
    return square_lattice, linalg.inv(square_lattice)@lattice

def change_coordinates(change_matrix, coords):
    """
    changing points from different basis into other basis
    :param change_matrix: change of basis matrix
    :param coords: coordinates for point
    :return:
    """
    return change_matrix @ coords

def split_by_periodicity(interval,coord):
    """
    changes coordinates to periodic coordinates
    :param interval: lenght  of periodic
    :param coord: coordinates
    :return: new periodic coordinates
    """
    new_coord = np.zeros(len(coord))
    for i in range(len(coord)):
        new_coord[i] = coord[i] % interval
    return new_coord

def main():
    filename = 'dft_chargedensity2.xsf'
    rho, lattice, grid, shift = read_example_xsf_density(filename)
    square_lattice, change_matrix = change_of_base(lattice)

    # first line
    start = [-1.4466, 1.3073, 3.2115]
    end = [1.4361, 3.1883, 1.3542]
    start = change_coordinates(change_matrix, start)
    end = change_coordinates(change_matrix, end)
    # runs too slowly to use 500 points for some reason i couldn't figure out
    interpolation_to_line(rho, grid, square_lattice, start[0], start[1], start[2], end[0], end[1], end[2], 5)

    # second line
    start = [2.9996, 2.1733, 2.1462]
    end = [8.7516, 2.1733, 2.1462]
    start = change_coordinates(change_matrix, start)
    end = change_coordinates(change_matrix, end)
    # runs too slowly to use 500 points for some reason i couldn't figure out
    interpolation_to_line(rho, grid, square_lattice, start[0], start[1], start[2], end[0], end[1], end[2], 5)

if __name__=="__main__":
    main()
