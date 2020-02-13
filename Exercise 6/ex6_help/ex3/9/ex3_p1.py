""" Problem 1: calculating a double integral using scipy's Simpson rule. """

import numpy as np
from scipy.integrate import simps
from scipy.integrate import dblquad


def fun(x, y):
    """ Function to be integrated in this problem. """
    return (x + y)*np.exp(-np.sqrt(x**2 + y**2))


# Define limits for integration:
x_0 = 0
x_1 = 2
y_0 = -2
y_1 = 2


def calculate_integral_dblquad():
    """ Calculate integral using scipy dblquad (for testing). """
    return dblquad(fun, x_0, y_0, lambda x: y_0, lambda x: y_1)


def calculate_integral(n_grid):
    """ Calculate integral using scipy's Simpson rule.

    Parameters
    ----------
    n_grid : int
        Number of grid points for both x and y directions.

    Returns
    -------
    float
       Integral value.

    """

    # Define grid.
    x_range = np.linspace(x_0, x_1, n_grid)
    y_range = np.linspace(y_0, y_1, n_grid)
    dy = y_range[1] - y_range[0]

    integral_total = 0

    # Divide volume in n_grid slices over y and calculate 1D integral over x
    # for each slice, then multiply by grid spacing to get volume. Add slices
    # to get total volume. First and last slices are cut in half.
    for y in y_range:
        z_slice = fun(x_range, y)
        integral_1d = simps(z_slice, x_range)
        integral_2d_slice = dy * integral_1d
        if y == y_range[0] or y == y_range[-1]:
            integral_2d_slice = integral_2d_slice / 2
        integral_total += integral_2d_slice

    return integral_total


if __name__ == '__main__':
    """ Calculate specified integral with different grids and with dblquad. """
    n = 100
    integral_simps = calculate_integral(n)
    print('Integral, Simpson (N =', str(n), '): ', str(integral_simps))

    n = 1000
    integral_simps = calculate_integral(n)
    print('Integral, Simpson (N =', str(n), '): ', str(integral_simps))

    integral_scipy = calculate_integral_dblquad()
    print('Integral, scipy dblquad: ', str(integral_scipy))

