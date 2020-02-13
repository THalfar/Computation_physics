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
    gridpoints = 41  # this must be form N*20 + 1 to include the plates and work correctly!
    if np.mod(gridpoints - 1, 20) != 0:
        print("Error: gridpoints must of form N*20 + 1")
        return

    tol = 1e-23  # tolerance for convergence
    max_iters = 5000  # maximum amount of 'updates' allowed (prevents infinite loops)
    epsilon = 8.8e-12  # permittivity, in case charges are present
    q = 1.602176634e-19  # elementary charge

    x_min, x_max = (-1, 1)
    y_min, y_max = (-1, 1)

    x = np.linspace(x_min, x_max, gridpoints)
    y = np.linspace(y_min, y_max, gridpoints)
    h = x[1]-x[0]

    X, Y = np.meshgrid(x, y, indexing='ij')  # for plotting

    fig, ax = subplots()
    ax.set_xticks(np.arange(x_min, x_max, h))
    ax.set_yticks(np.arange(y_min, y_max, h))
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(x_min, x_max)
    ax.set_title("Set point charges using mouse. Left click = add positive charge, right click = add negative charge\n "
          "You may add or remove multiple charges for each point. continue by closing figure.")
    ax.grid(True)
    ax.set_aspect('equal', 'box')

    rho_mat = np.zeros([gridpoints, gridpoints])  # rho will contain the charges (negative or positive)

    clicked = False

    # update rho_mat by user mouse clicks until the figure is closed.
    def onclick(event):
        ix, iy = event.xdata, event.ydata
        if str(ix) == 'None' or str(iy) == 'None':
            return

        ix_index = int(np.round((ix - x_min) / h))
        iy_index = int(np.round((iy - y_min) / h))

        ix = x[ix_index]
        iy = y[iy_index]
        if str(event.button) == "MouseButton.LEFT" or event.button == 1:  # same thing on different PC's
            rho_mat[ix_index, iy_index] += q
        elif str(event.button) == "MouseButton.RIGHT" or event.button == 3:
            rho_mat[ix_index, iy_index] -= q
        else:
            return

        tot_point_charge = np.copy(rho_mat[ix_index, iy_index])
        # different colors for different charges
        if tot_point_charge > 0:
            color = 'r'
        elif tot_point_charge < 0:
            color = 'b'
        else:
            color = 'w'

        transparency = min(np.abs(tot_point_charge)/q * 0.1, 1)
        print("[" + str(np.round(ix, decimals=3)) + ", " + str(np.round(iy, decimals=3)) +
              "] Total charge: " + str(int(tot_point_charge / q)) + "q")

        ax.scatter(ix, iy, c='w', s=120)  # this allows color changes for the point (from red to blue etc..)
        ax.scatter(ix, iy, c=color, alpha=transparency, s=120)

        fig.canvas.draw()

    fig.canvas.mpl_connect('button_press_event', onclick)
    show()

    if np.count_nonzero(rho_mat) == 0:
        print("Error: You must add point charges!")
        return

    # initialize phi matrix with given boundary conditions
    initial_phi_matrix = np.zeros([gridpoints, gridpoints])  # first index is x, second index is y

    boundary_matrix = np.zeros([gridpoints, gridpoints])  # boundary matrix is the same form as initial phi matrix
    # value is 1 for boundary points, 0 for others.

    # add square boundaries
    boundary_matrix[:, 0] = 1.
    boundary_matrix[:, -1] = 1.
    boundary_matrix[0, :] = 1.
    boundary_matrix[-1, :] = 1.

    print("Starting SOR method iteration...")
    solved_phi_matrix = solve_poisson_sor_method(initial_phi_matrix, boundary_matrix, rho_mat, h, epsilon, tol=tol,
                                                 max_iters=max_iters, omega=1.8)

    rcParams.update({'font.size': 13})

    phi_gradient = np.array(np.gradient(solved_phi_matrix, h))
    e_field = -1*phi_gradient
    e_field_x = e_field[0, :, :]
    e_field_y = e_field[1, :, :]

    quiverfig, quiver_ax = subplots()
    quiver_ax.quiver(X, Y, e_field_x, e_field_y, angles='xy', scale_units='xy')

    # add point charge markers
    for i in range(rho_mat.shape[0]):
        for j in range(rho_mat.shape[1]):
            tot_point_charge = rho_mat[i, j]
            transparency = min(np.abs(tot_point_charge) / q * 0.25, 1)
            if tot_point_charge > q/2:
                quiver_ax.scatter(x_min + h*i, y_min + h*j, alpha=transparency, c='r', s=250)
            elif tot_point_charge < -q/2:
                quiver_ax.scatter(x_min + h * i, y_min + h * j, alpha=transparency, c='b', s=250)

    # plot boundaries (left, right, up, down)
    lbound_x, lbound_y = x_min * np.ones(100), np.linspace(y_min, y_max, 100)
    rbound_x, rbound_y = x_max * np.ones(100), np.linspace(y_min, y_max, 100)
    ubound_x, ubound_y = np.linspace(x_min, x_max, 100), y_max * np.ones(100)
    dbound_x, dbound_y = np.linspace(x_min, x_max, 100), y_min * np.ones(100)
    quiver_ax.plot(lbound_x, lbound_y, 'k-', linewidth=3)
    quiver_ax.plot(rbound_x, rbound_y, 'k-', linewidth=3)
    quiver_ax.plot(ubound_x, ubound_y, 'k-', linewidth=3)
    quiver_ax.plot(dbound_x, dbound_y, 'k-', linewidth=3)
    quiver_ax.set_aspect('equal', 'box')
    title('E-field visualization in xy-plane with custom point charges. Red = positive, blue = negative')
    xlabel('x')
    ylabel('y')

    # interpolate potential field with spline and plot it
    spl2d=spline(x=x,y=y,f=solved_phi_matrix,dims=2)
    xx = np.linspace(x_min, x_max, gridpoints*5)
    yy = np.linspace(y_min, y_max, gridpoints*5)

    Z = spl2d.eval2d(xx, yy)
    XX, YY = np.meshgrid(xx, yy, indexing='ij')
    fig, ax = subplots(1, 1)

    # colormap scaling
    if np.min(Z) >= 0:
        vmin = 0
        vmax = np.max(Z)
        cmap = 'Reds'
    elif np.max(Z) <= 0:
        vmax = 0
        vmin = np.min(Z)
        cmap = 'Blues'
    else:
        vmax = max(np.abs(np.min(Z)), np.max(Z))
        vmin = -vmax
        cmap = 'RdBu_r'

    c = ax.pcolor(XX, YY, Z, vmin=vmin, vmax=vmax, cmap=cmap)
    # c = ax.pcolor(X, Y, solved_phi_matrix, vmin=vmin, vmax=vmax, cmap=cmap)
    colorbar(c)
    xlabel('x')
    ylabel('y')
    title('Electric potential (spline interpolation)')
    ax.set_aspect('equal', 'box')

    show()


if __name__ == "__main__":
    main()