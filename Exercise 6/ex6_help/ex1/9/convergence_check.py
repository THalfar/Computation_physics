"""
Plot convergence diagrams for different numerical methods of derivative and
integral approximation.
"""

from typing import Callable
import math

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import num_calculus
from monte_carlo_integration import monte_carlo_integration

def compute_derivatives(test_func: Callable, x_point: float, grid_spacing: list) -> tuple:
    """
    Compute the first and second derivatives of the given 1D function at the given
    point, using different values of grid spacing.

    :param test_func: Function for which derivatives are approximated
    :param x_point: Point in x-axis where derivative is approximated
    :param num_grids: List of grid spacing values to use
    :returns: Tuple of first derivate approximations and second derivative
    approximations in respective indices of the tuple.
    """

    # With each grid spacing value, get the approximation of first and second
    # derivatives
    first_derivatives = [num_calculus.first_derivative(test_func, x_point, dx)
                         for dx in grid_spacing]
    second_derivatives = [num_calculus.second_derivative(test_func, x_point, dx)
                         for dx in grid_spacing]

    return first_derivatives, second_derivatives

def compute_convergences(num_grids: int=100):
    """
    Compute convergences of numerical approximations to first and second
    derivatives. Multiple functions will be tested. Plots the results.

    :param num_grids: Number of different grid spacings to test per approximation
    """

    # Create grid spacing values for convergence check
    grid_spacing = [1/x for x in range(1, num_grids)]
    mc_iterations = range(1, num_grids)
    mc_blocks = 25

    # Let's test first and second derivatives of some interesting functions.
    # Start by making tuples with content of each index being
    # 0: function
    # 1: test point in x
    # 2: true value of first derivative
    # 3: true value of second derivative
    # 4: integration x start
    # 5: integration x stop
    # 6: true integral value

    # First derivative of sin(x) is cos(x) and second is -sin(x). Set x=3/4*pi
    # and compute true solutions to derivatives and integral from zero to pi
    test_sin = (math.sin, 3/4*math.pi, math.cos(3/4*math.pi),
                -1.*math.sin(3/4*math.pi), 0.0, math.pi, 2.0)
    # Try with a different function: exp(-2x). First derivative is -2*exp(-2x)
    # and second derivative is 4x*exp(-2x). Set x=2 and compute true values for
    # derivates and set integration range from -1 to 1, true result is approx.
    # 27.290
    def expfun(x: float):
        return math.exp(-2.0*x)
    expfun.__name__ = 'exp(-2x)'
    test_exp = (expfun, 2.0, -2.0*math.exp(-2*2), 4.0*math.exp(-2*2), -1., 1.0,
                3.6269)
    # Wrap all tests in a list
    test_functions = [test_sin, test_exp]
    # Initialize a result list, which will be the absolute error between the
    # numerical estimate and true value
    errors = []
    for test in test_functions:

        # Computing the approximations
        first, second = compute_derivatives(test[0], test[1], grid_spacing)
        # Unpack the tuple from MC integration on the fly ([0] at the end)
        integrals = [monte_carlo_integration(test[0], test[4], test[5], mc_blocks, n)[0]
                    for n in mc_iterations]

        # Compute absolute errors for each grid spacing value
        first_derivative_errors = [abs(test[2] - f) for f in first]
        second_derivative_errors = [abs(test[3] - s) for s in second]
        integral_errors = [abs(test[6] - i) for i in integrals]

        # Errors will be a list of tuples, where each list element is a
        # different function and index 0 of the tuple is the list of errors of first
        # derivative, while index 1 of the tuple is the list of errors of second
        # derivative approximation.
        errors.append((first_derivative_errors, second_derivative_errors,
                       integral_errors))

    # Make a pretty plot of convergences. Not making a separate function this
    # time because so many variables from the convergence check are needed
    methods = ['First derivative', 'Second derivative', 'Monte Carlo Integration']
    sns.set(palette='pastel', context='notebook')
    # Make a subplot for each method we are testing
    fig, axes = plt.subplots(len(methods), 1)
    fig.suptitle('Method convergences')
    # Iterate over the number of subplots / methods
    for ind, ax in enumerate(axes.flatten()):
        ax.set_title(methods[ind])
        # Derivatives are plotted differently than the MC method
        if ind in [0, 1]:
            for test_i, test in enumerate(test_functions):
                ax.plot(grid_spacing, errors[test_i][ind], label=test[0].__name__)
            ax.invert_xaxis()
            ax.set_xlabel('Grid spacing')
        else:
            for test_i, test in enumerate(test_functions):
                ax.plot(mc_iterations, errors[test_i][ind], label=test[0].__name__)
            ax.set_xlabel('Iterations')
        ax.set_ylabel('Absolute error')
        ax.legend()
    plt.tight_layout()
    plt.show()

def main():
    """
    doc
    """

    compute_convergences()

if __name__ == "__main__":
    main()
