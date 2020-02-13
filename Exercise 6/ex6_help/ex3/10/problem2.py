"""
FYS-4096 Computational Physics
Exercise 3
Problem 2

2D interpolation
"""

# imports
from ex3_lib import *
import numpy as np
from matplotlib import cm
import matplotlib.pyplot as plt

from linear_interp import *
from spline_class import *


def linefun(x):
    return np.sqrt(1.75)*x


def main():
    """
    Main function
    :return: -
    """

    # Creating the values and grid
    xs, ys = grid(-2, 2, -2, 2, 30, 30)
    x, y = np.meshgrid(xs, ys)

    # Plot the problem's domain in 3D:
    """fig = plt.figure()
    ax = fig.gca(projection='3d')

    ax.plot_surface(x, y, fun(x, y), cmap=cm.viridis, alpha=0.75)
    ax.plot(x[x >= 0], linefun(x[x >= 0]), 'r-')

    plt.xlabel('x')
    plt.ylabel('y')
    plt.show()"""

    # Borrowed the example from linear_interp.py and modified a little:
    fig2d = plt.figure()
    ax2d = fig2d.add_subplot(221, projection='3d')
    ax2d2 = fig2d.add_subplot(222, projection='3d')
    ax2d3 = fig2d.add_subplot(223)
    ax2d4 = fig2d.add_subplot(224)

    x = np.linspace(-2.0, 2.0, 11)
    y = np.linspace(-2.0, 2.0, 11)
    X, Y = np.meshgrid(x, y)
    Z = fun(X, Y)

    # Variables for plotting the line
    linex = np.linspace(0, 2, 100)
    liney = linefun(linex)
    linex = linex[liney <= 2]
    liney = liney[liney <= 2]

    # Plotting the original surface
    ax2d.plot_surface(X, Y, Z, cmap=cm.viridis, alpha=0.75)
    ax2d.plot(x[x >= 0], linefun(x[x >= 0]), 'r-')
    ax2d3.pcolor(X, Y, Z)
    ax2d3.plot(linex, liney, 'r-')

    # Initialize interpolation schemas
    lin2d = linear_interp(x=x, y=y, f=Z, dims=2)
    spl2d = spline(x=x, y=y, f=Z, dims=2)

    # Plotting params for interpolations
    x = np.linspace(-2.0, 2.0, 100)
    y = np.linspace(-2.0, 2.0, 100)
    X, Y = np.meshgrid(x, y)
    Z = lin2d.eval2d(x, y)

    # Plotting interpolated functions
    ax2d2.plot_surface(X, Y, Z, cmap=cm.viridis, alpha=0.75)
    ax2d2.plot(x[x >= 0], linefun(x[x >= 0]), 'r-')
    ax2d4.pcolor(X, Y, Z)
    ax2d4.plot(linex, liney, 'r-')

    # Plot 1D interpolated values vs. exact value
    fig2 = plt.figure()

    lininterpx = np.empty_like(linex)
    splinterpx = np.empty_like(linex)
    for i in range(linex.size):
        lininterpx[i] = lin2d.eval2d(linex[i], liney[i])
        splinterpx[i] = spl2d.eval2d(linex[i], liney[i])

    plt.plot(linex, lininterpx, '-', label='linear interpolation')
    plt.plot(linex, splinterpx, '-', label='spline interpolation')
    plt.plot(linex, fun(linex, linefun(linex)), '-', label='exact value')
    plt.xlabel('x')
    plt.ylabel('f(x, y) along y=sqrt(1.75)*x')
    plt.legend()
    plt.show()

    return 0


if __name__=="__main__":
    main()
