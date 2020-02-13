""" Excercise 4: problems 1-4. """

import read_xsf_example
import numpy as np
import spline_class as spline
import matplotlib.pyplot as plt


def problem_2(rho, lattice, grid, name):
    """ Calculate total amount of electrons and reciprocal lattice vectors """

    # calculate volume of the cell using triple product
    volume = np.dot(np.cross(lattice[0], lattice[1]), lattice[2])
    # calculate number of elements in cell
    elements = grid[0]*grid[1]*grid[2]
    volume_element = volume / elements
    total_electrons = volume_element * np.sum(rho)
    print('Total electrons in ', name, '=', str(total_electrons))

    # calculate reciprocal lattice vectors
    b1 = 2*np.pi*(np.cross(lattice[1], lattice[2]) /
                  np.dot(lattice[0], np.cross(lattice[1], lattice[2])))

    b2 = 2*np.pi*(np.cross(lattice[2], lattice[0]) /
                  np.dot(lattice[1], np.cross(lattice[2], lattice[0])))

    b3 = 2*np.pi*(np.cross(lattice[0], lattice[1]) /
                  np.dot(lattice[2], np.cross(lattice[0], lattice[1])))

    print('Reciprocal lattice vectors for ', name, ' are:', b1, '; ', b2,
          '; ', b3)


def plot_electron_density(spl3d, lattice, line, points, title):
    """ Problems 3 and 4: plot electron densities """

    # Calculate line coordinates in terms of lattice vectors
    r0_a = np.linalg.solve(lattice, line[0])
    r1_a = np.linalg.solve(lattice, line[1])

    line_x = np.linspace(r0_a[0], r1_a[0], points)
    line_y = np.linspace(r0_a[1], r1_a[1], points)
    line_z = np.linspace(r0_a[2], r1_a[2], points)

    # interpolate value of rho at each point on the line
    line_f = []
    for i in range(0, len(line_x)):
        f = spl3d.eval3d(line_x[i], line_y[i], line_z[i])
        line_f.append(f[0][0][0])

    # calculate x axis in terms of distance
    line_dist = np.sqrt((line_x-line_x[0])**2 + (line_y-line_y[0])**2 + (
            line_z-line_z[0])**2)

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_title(title)
    ax.plot(line_dist, line_f)

    plt.show()


if __name__ == '__main__':

    # read files
    file_xsf_1 = 'dft_chargedensity1.xsf'
    file_xsf_2 = 'dft_chargedensity2.xsf'

    rho_1, lattice_1, grid_1, shift_1 = read_xsf_example.\
        read_example_xsf_density(file_xsf_1)

    rho_2, lattice_2, grid_2, shift_2 = read_xsf_example.\
        read_example_xsf_density(file_xsf_2)

    # problem 2
    problem_2(rho_1, lattice_1, grid_1, file_xsf_1)
    problem_2(rho_2, lattice_2, grid_2, file_xsf_2)

    # problem 3
    x = np.linspace(0, 1, grid_1[0])
    y = np.linspace(0, 1, grid_1[1])
    z = np.linspace(0, 1, grid_1[2])

    spl3d_1 = spline.spline(x=x, y=y, z=z, f=rho_1, dims=3)
    line_1 = [[0.1, 0.1, 2.8528], [4.45, 4.45, 2.8528]]
    plot_electron_density(spl3d_1, lattice_1, line_1, 500,
                          file_xsf_1 + '\n' + str(line_1))

    # problem 4
    x = np.linspace(0, 1, grid_2[0])
    y = np.linspace(0, 1, grid_2[1])
    z = np.linspace(0, 1, grid_2[2])

    spl3d_2 = spline.spline(x=x, y=y, z=z, f=rho_2, dims=3)

    line_2 = [[-1.4466, 1.3073, 3.2115], [1.4361, 3.1883, 1.3542]]
    plot_electron_density(spl3d_2, lattice_2, line_2, 500,
                          file_xsf_2 + '\n' + str(line_2))

    line_3 = [[2.9996, 2,1733, 2,1462], [8.7516, 2.1733, 2,1462]]
    plot_electron_density(spl3d_2, lattice_2, line_3, 500,
                          file_xsf_2 + '\n' + str(line_3))
