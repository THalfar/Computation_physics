""" Problem 2: estimation of values on a straight line using 2D interpolation. """

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import linear_interp


def fun(x, y):
    """ Function to generate the experimental data for this problem. """
    return (x + y)*np.exp(-np.sqrt(x**2 + y**2))


def fun_line(x):
    """ Returns the points at which function "fun" is to be evaluated. """
    return np.where(x >= 0, np.sqrt(1.75)*x, np.nan)


# Define grid and generate experimental data.
x0 = -2
x1 = 2
y0 = -2
y1 = 2
x_exp = np.linspace(x0, x1, 30)
y_exp = np.linspace(y0, y1, 30)
x_exp_mesh, y_exp_mesh = np.meshgrid(x_exp, y_exp)
z_exp_mesh = fun(x_exp_mesh, y_exp_mesh)

# Initialize interpolation with experimental data.
lin2d = linear_interp.linear_interp(x=x_exp, y=y_exp, f=z_exp_mesh, dims=2)

# Generate line points:
x0_line = 0
x1_line = x1
x_line = np.linspace(x0_line, x1_line, 100)
y_line = fun_line(x_line)
z_line_list = []  # to store output values at the line points

# Iterate over all points in the line and interpolate z values.
for i in range(0, len(x_line)):
    x_line_mesh, y_line_mesh = np.meshgrid(
        np.array([x_line[i]]), np.array([y_line[i]]))
    z_line_mesh = lin2d.eval2d(x_line[i], y_line[i])
    z_line_list.append(z_line_mesh[0][0])


def plot_3d():
    """ Plot experimental data in 3D (for testing). """
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_wireframe(x_exp, y_exp_mesh, z_exp_mesh)
    for i in range(0, len(x_line)):
        x_line_mesh, y_line_mesh = np.meshgrid(
            np.array([x_line[i]]), np.array([y_line[i]]))
        z_line_mesh = lin2d.eval2d(x_line[i], y_line[i])
        ax.scatter(x_line_mesh, y_line_mesh, zs=z_line_mesh, color='red')


def plot_2d():
    """ Plot 1D values from the interpolation and the exact function. """

    # Generate exact values at the line points.
    z_line_exact = fun(x_line, y_line)

    fig, ax = plt.subplots()
    ax.plot(x_line, z_line_list, '--', label='interpolated')
    ax.plot(x_line, z_line_exact, '-', label='exact')

    ax.legend()


if __name__ == '__main__':
    plot_2d()
    plot_3d()
    plt.show()
