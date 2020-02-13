"""
Computational Physics,exercise 5 problem 3

Arttu Hietalahti, 6.2.2020
"""



import numpy as np
from matplotlib.pyplot import *
from scipy.integrate import odeint
from mpl_toolkits.mplot3d import Axes3D


def jacobi_update(x, y, old_phi_matrix, h, epsilon, rho):
    # Jacobi method from lecture slides
    return 1/4*(old_phi_matrix[x+1, y] + old_phi_matrix[x-1, y] + old_phi_matrix[x, y+1] + old_phi_matrix[x, y-1] + h**2*rho/epsilon)


def gauss_seidel_update(x, y, phi_matrix, h, epsilon, rho):
    # Gauss-Seidel method from lecture slides
    return 1/4*(phi_matrix[x+1, y] + phi_matrix[x-1, y] + phi_matrix[x, y+1] + phi_matrix[x, y-1] + h**2*rho/epsilon)


def sor_update(x, y, phi_matrix, omega, h, epsilon, rho):
    # SOR method from lecture slides
    return (1-omega)*phi_matrix[x, y] + \
           omega/4*(phi_matrix[x+1, y] + phi_matrix[x-1, y] + phi_matrix[x, y+1] + phi_matrix[x, y-1] + h**2*rho/epsilon)


def solve_poisson_jacobi(initial_phi_matrix, boundary_matrix, rho_mat, h=0., epsilon=0., tol=1e-8, max_iters=10000):
    """
        Solves the poisson equation by iterating with the Jacobi method.
    :param initial_phi_matrix: initial phi conditions
    :param boundary_matrix: contains boundary points (1 for boundaries, 0 for non-boundaries)
    :param rho_mat: charge matrix
    :param h: step size (assumed equal for x and y)
    :param epsilon: permittivity
    :param tol: determines result accuracy
    :param max_iters: loop quits after max_iters
    :return: solved phi matrix
    """

    phi_matrix = np.copy(initial_phi_matrix)  # phi_matrix will contain the solution

    for i in range(max_iters):
        old_phi_matrix = np.copy(phi_matrix)  # make a copy for tolerance check

        for x_idx in range(phi_matrix.shape[0]):
            for y_idx in range(phi_matrix.shape[1]):

                # update only non-boundary points
                if boundary_matrix[x_idx, y_idx] == 0:
                    # jacobi method uses only the information from previous phi matrix
                    phi_matrix[x_idx, y_idx] = jacobi_update(x_idx, y_idx, old_phi_matrix, h, epsilon, rho_mat[x_idx, y_idx])

        # tolerance is reached if maximum change in phi is less than tol
        #  > 0 condition ensures that the loop will not exit immediately at the beginning
        if 0 < np.amax(np.abs(np.subtract(old_phi_matrix, phi_matrix))) < tol:
            print("solve_poisson_jacobi: tolerance " + str(tol) + " reached after " + str(i) + " iterations.")
            return phi_matrix

    print("solve_poisson_jacobi: solution did not converge below tolerance " + str(tol) +
          " in maximum amount of iterations (" + str(max_iters) + ")")
    return phi_matrix


def solve_poisson_gauss_seidel(initial_phi_matrix, boundary_matrix, rho_mat, h=0., epsilon=0., tol=1e-8, max_iters=10000):
    """
        Solves the poisson equation by iterating with the gauss-seidel method.
    :param initial_phi_matrix: initial phi conditions
    :param boundary_matrix: contains boundary points (1 for boundaries, 0 for non-boundaries)
    :param rho_mat: charge matrix
    :param h: step size (assumed equal for x and y)
    :param epsilon: permittivity
    :param tol: determines result accuracy
    :param max_iters: loop quits after max_iters
    :return: solved phi matrix
    """

    phi_matrix = np.copy(initial_phi_matrix)  # phi_matrix will contain the solution

    for i in range(max_iters):
        old_phi_matrix = np.copy(phi_matrix)  # make a copy for tolerance check

        for x_idx in range(phi_matrix.shape[0]):
            for y_idx in range(phi_matrix.shape[1]):

                # update only non-boundary points
                if boundary_matrix[x_idx, y_idx] == 0:
                    # gauss seideluses the already updated values for [i-1, j] and [i, j-1]
                    phi_matrix[x_idx, y_idx] = gauss_seidel_update(x_idx, y_idx, phi_matrix, h, epsilon,
                                                                   rho_mat[x_idx, y_idx])

        # tolerance is reached if maximum change in phi is less than tol
        #  > 0 condition ensures that the loop will not exit immediately at the beginning
        if 0 < np.amax(np.abs(np.subtract(old_phi_matrix, phi_matrix))) < tol:
            print("solve_poisson_gauss_seidel: tolerance " + str(tol) + " reached after " + str(i) + " iterations.")
            return phi_matrix

    print("solve_poisson_gauss_seidel: solution did not converge below tolerance " + str(tol) +
          " in maximum amount of iterations (" + str(max_iters) + ")")
    return phi_matrix


def solve_poisson_sor_method(initial_phi_matrix, boundary_matrix,
                             rho_mat, h=0., epsilon=0., tol=1e-8, max_iters=10000, omega=1.8):
    """
        Solves the poisson equation by iterating with the sor method.
    :param initial_phi_matrix: initial phi conditions
    :param boundary_matrix: contains boundary points (1 for boundaries, 0 for non-boundaries)
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

        # iterate through the matrix, except the boundaries
        for x_idx in range(phi_matrix.shape[0]):
            for y_idx in range(phi_matrix.shape[1]):

                # update only non-boundary points
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


def test_all_methods():
    # function tests problem 3 with all methods and plots the solutions

    # parameters for poisson grid, etc.

    # these three affect the performance most
    gridpoints = 40  # grid points per axis
    tol = 1e-6  # tolerance for convergence
    max_iters = 5000  # maximum amount of 'updates' allowed (prevents infinite loops)

    L_x = 1
    L_y = 1
    h = L_x/gridpoints
    epsilon = 8.8e-12
    phi0 = 1
    x_vec = np.linspace(0, L_x, gridpoints)
    y_vec = np.linspace(0, L_y, gridpoints)
    X, Y = np.meshgrid(x_vec, y_vec)  # for plotting

    # initialize phi matrix with given boundary conditions
    initial_phi_matrix = np.zeros([gridpoints, gridpoints])  # first index is x, second index is y
    initial_phi_matrix[:, -1] = phi0 * np.ones(gridpoints)

    zero_rho_mat = np.zeros([gridpoints, gridpoints])

    # for testing non-zero rho behaviour
    rho_mat = np.copy(zero_rho_mat)
    rho_mat[gridpoints//2, gridpoints//2] = 3e-8

    boundary_matrix = np.zeros([gridpoints, gridpoints])  # boundary matrix is the same form as initial phi matrix
    # value is 1 for boundary points, 0 for others.

    # add boundaries in the boundary matrix
    boundary_matrix[:, 0] = 1.
    boundary_matrix[:, -1] = 1.
    boundary_matrix[0, :] = 1.
    boundary_matrix[-1, :] = 1.

    solved_phi_matrix1 = solve_poisson_jacobi(initial_phi_matrix, boundary_matrix, zero_rho_mat, h, epsilon,
                                              tol=tol, max_iters=max_iters)
    solved_phi_matrix2 = solve_poisson_gauss_seidel(initial_phi_matrix, boundary_matrix,
                                                    zero_rho_mat, h, epsilon, tol=tol, max_iters=max_iters)
    solved_phi_matrix3 = solve_poisson_sor_method(initial_phi_matrix, boundary_matrix,
                                                  zero_rho_mat, h, epsilon, tol=tol, max_iters=max_iters, omega=1.8)
    solved_phi_matrix4 = solve_poisson_sor_method(initial_phi_matrix, boundary_matrix, rho_mat, h, epsilon, tol=tol,
                                                  max_iters=max_iters, omega=1.8)

    rcParams.update({'font.size': 13})

    fig = figure()
    ax1 = fig.add_subplot(221, projection='3d')
    ax1.plot_wireframe(X, Y, initial_phi_matrix, rstride=1, cstride=1)
    ax1.set_title("Initial condition")
    ax2 = fig.add_subplot(222, projection='3d')
    ax2.plot_wireframe(X, Y, solved_phi_matrix1, rstride=1, cstride=1)
    ax2.set_title("Solution (Jacobi)")
    ax3 = fig.add_subplot(223, projection='3d')
    ax3.plot_wireframe(X, Y, solved_phi_matrix2, rstride=1, cstride=1)
    ax3.set_title("Solution (Gauss-Seidel)")
    ax4 = fig.add_subplot(224, projection='3d')
    ax4.plot_wireframe(X, Y, solved_phi_matrix3, rstride=1, cstride=1)
    ax4.set_title("Solution (SOR-method)")

    # change viewing angle
    ax1.view_init(20, 135)
    ax2.view_init(20, 135)
    ax3.view_init(20, 135)
    ax4.view_init(20, 135)

    fig2 = figure()
    ax = fig2.add_subplot(111, projection='3d')
    ax.plot_wireframe(X, Y, solved_phi_matrix4, rstride=1, cstride=1)
    ax.set_title("Testing with some point charges")
    ax.view_init(20, 135)


    show()


def main():
    test_all_methods()


if __name__ == "__main__":
    main()