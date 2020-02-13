"""
Computational Physics,exercise 5 problem 4

Arttu Hietalahti, 6.2.2020
"""



import numpy as np
from matplotlib.pyplot import *
from scipy.integrate import odeint
from mpl_toolkits.mplot3d import Axes3D
from spline_class import spline


def sor_update(x, y, phi_matrix, omega, h, epsilon, rho):
    # SOR method from lecture slides
    return (1-omega)*phi_matrix[x, y] + \
           omega/4*(phi_matrix[x+1, y] + phi_matrix[x-1, y] + phi_matrix[x, y+1] + phi_matrix[x, y-1] + h**2*rho/epsilon)


def solve_poisson_sor_method(initial_phi_matrix, boundary_matrix, rho_mat, h=0., epsilon=0., tol=1e-8, max_iters=10000, omega=1.8):
    """
        Solves the poisson equation by iterating with the sor method.
    :param initial_phi_matrix: initial phi conditions
    :param boundary_matrix: boundaries which will not be modified in calculations.
    :param rho_mat: charge matrix
    :param h: step size (assumed equal for x and y)
    :param epsilon: permittivity
    :param tol: determines result accuracy
    :param max_iters: loop quits after max_iters
    :param omega: omega in SOR method
    :return: solved phi matrix
    """

    phi_matrix = np.copy(initial_phi_matrix)  # phi_matrix will contain the solution

    for i in range(max_iters):
        old_phi_matrix = np.copy(phi_matrix)  # make a copy for tolerance check

        for x_idx in range(1, phi_matrix.shape[0]):
            for y_idx in range(1, phi_matrix.shape[1]):

                # only update points which are not boundaries. Boundary matrix has 0 for non-boundary points.
                if boundary_matrix[x_idx, y_idx] == 0:
                    phi_matrix[x_idx, y_idx] = sor_update(x_idx, y_idx, phi_matrix, omega, h, epsilon, rho_mat[x_idx, y_idx])

        # tolerance is reached if maximum change in phi is less than tol
        #  > 0 condition ensures that the loop will not exit immediately at the beginning
        if 0 < np.amax(np.abs(np.subtract(old_phi_matrix, phi_matrix))) < tol:
            print("solve_poisson_sor_method: tolerance " + str(tol) + " reached after " + str(i) + " iterations.")
            return phi_matrix

    print("solve_poisson_sor_method: solution did not converge below tolerance " + str(tol) +
          " in maximum amount of iterations (" + str(max_iters) + ")")
    return phi_matrix


def main():
    gridpoints = 21  # this must be form N*20 + 1 to include the plates and work correctly!
    if np.mod(gridpoints - 1, 20) != 0:
        print("Error: gridpoints must of form N*20 + 1")
        return

    tol = 1e-10  # tolerance for convergence
    max_iters = 5000  # maximum amount of 'updates' allowed (prevents infinite loops)
    epsilon = 8.8e-12  # permittivity, in case charges are present

    x_min, x_max = (-1, 1)
    y_min, y_max = (-1, 1)
    blue_plate_x, blue_plate_y_min, blue_plate_y_max = -0.3, -0.5, 0.5
    red_plate_x, red_plate_y_min, red_plate_y_max = 0.3, -0.5, 0.5

    x = np.linspace(x_min, x_max, gridpoints)
    y = np.linspace(y_min, y_max, gridpoints)
    h = x[1]-x[0]

    X, Y = np.meshgrid(x, y, indexing='ij')  # for plotting

    # initialize phi matrix with given boundary conditions
    initial_phi_matrix = np.zeros([gridpoints, gridpoints])  # first index is x, second index is y

    blue_plate = np.transpose(np.ones((gridpoints-1) // 2))
    red_plate = -1*blue_plate

    # get blue and red place indices
    blue_plate_x_idx = int((blue_plate_x - x_min) * gridpoints // 2)
    blue_plate_y_min_idx = int((blue_plate_y_min - y_min) * gridpoints // 2)
    blue_plate_y_max_idx = int(gridpoints + (blue_plate_y_max - y_max) * gridpoints // 2)

    red_plate_x_idx = int((red_plate_x - x_min) * gridpoints // 2)
    red_plate_y_min_idx = int((red_plate_y_min - y_min) * gridpoints // 2)
    red_plate_y_max_idx = int(gridpoints + (red_plate_y_max - y_max) * gridpoints // 2)

    # add blue and red plate to the phi matrix
    initial_phi_matrix[blue_plate_x_idx, blue_plate_y_min_idx:blue_plate_y_max_idx] = blue_plate
    initial_phi_matrix[red_plate_x_idx, red_plate_y_min_idx:red_plate_y_max_idx] = red_plate

    rho_mat = np.zeros([gridpoints, gridpoints])  # use zeros for rho

    boundary_matrix = np.zeros([gridpoints, gridpoints])  # boundary matrix is the same form as initial phi matrix
    # value is 1 for boundary points, 0 for others.

    # add square boundaries and plates in the boundary matrix
    boundary_matrix[blue_plate_x_idx, blue_plate_y_min_idx:blue_plate_y_max_idx] = 1.
    boundary_matrix[red_plate_x_idx, red_plate_y_min_idx:red_plate_y_max_idx] = 1.
    boundary_matrix[:, 0] = 1.
    boundary_matrix[:, -1] = 1.
    boundary_matrix[0, :] = 1.
    boundary_matrix[-1, :] = 1.

    solved_phi_matrix = solve_poisson_sor_method(initial_phi_matrix, boundary_matrix, rho_mat, h, epsilon, tol=tol,
                                                 max_iters=max_iters, omega=1.8)

    rcParams.update({'font.size': 13})

    phi_gradient = np.array(np.gradient(solved_phi_matrix, h))
    e_field = -1*phi_gradient
    e_field_x = e_field[0, :, :]
    e_field_y = e_field[1, :, :]


    fig, ax = subplots()
    quiver(X, Y, e_field_x, e_field_y, angles='xy', scale_units='xy', scale=20)

    # plot blue and red plates
    blue_plate_y = np.linspace(blue_plate_y_min, blue_plate_y_max, 100)
    red_plate_y = np.linspace(red_plate_y_min, red_plate_y_max, 100)
    blue_plate_x = blue_plate_x*np.ones_like(blue_plate_y)
    red_plate_x = red_plate_x * np.ones_like(red_plate_y)

    plot(blue_plate_x, blue_plate_y, 'b-', linewidth=5)
    plot(red_plate_x, red_plate_y, 'r-',linewidth=5)

    # plot boundaries (left, right, up, down)
    lbound_x, lbound_y = x_min * np.ones(100), np.linspace(y_min, y_max, 100)
    rbound_x, rbound_y = x_max * np.ones(100), np.linspace(y_min, y_max, 100)
    ubound_x, ubound_y = np.linspace(x_min, x_max, 100), y_max * np.ones(100)
    dbound_x, dbound_y = np.linspace(x_min, x_max, 100), y_min * np.ones(100)
    plot(lbound_x, lbound_y, 'k-', linewidth=3)
    plot(rbound_x, rbound_y, 'k-', linewidth=3)
    plot(ubound_x, ubound_y, 'k-', linewidth=3)
    plot(dbound_x, dbound_y, 'k-', linewidth=3)
    ax.set_aspect('equal', 'box')
    title('E-field visualization in xy-plane. Blue plate is at potential 1, red plate is at potential -1\n '
          'boundaries are at potential 0')
    xlabel('x')
    ylabel('y')

    # interpolate potential field with spline and plot it
    spl2d=spline(x=x,y=y,f=solved_phi_matrix,dims=2)
    xx = np.linspace(x_min, x_max, 200)
    yy = np.linspace(y_min, y_max, 200)

    Z = spl2d.eval2d(xx, yy)
    XX, YY = np.meshgrid(xx, yy, indexing='ij')
    fig, ax = subplots(1, 1)
    c = ax.pcolor(XX, YY, Z, cmap='RdBu')
    colorbar(c)
    xlabel('x')
    ylabel('y')
    title('Electric potential (spline interpolation)')
    ax.set_aspect('equal', 'box')

    show()


if __name__ == "__main__":
    main()