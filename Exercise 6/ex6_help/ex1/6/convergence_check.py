import numpy as np
import matplotlib.pyplot as plt
from num_calculus import first_derivative, second_derivative, \
     riemann_sum, trapedzoid_sum, simpson_sum, monte_carlo_integration

""" FYS-4096 Computational Physics: Exercise 1 """
""" Author: Santeri Saariokari """


def calculate_derivative_errors():
    # Calculates the numerical first and second order derivatives
    # and compares them to analytical values
    f = lambda x: np.sin(x)
    f_dot = lambda x: np.cos(x)
    f_dot2 = lambda x: -np.sin(x)
    x = 1.5
    errors1 = []
    errors2 = []
    dxs = np.linspace(1e-6, 1e-1, 1000)
    for dx in dxs:
        errors1.append(abs(first_derivative(f, x, dx) - f_dot(x)))
        errors2.append(abs(second_derivative(f, x, dx) - f_dot2(x)))
    plot_derivative_errors(dxs, errors1, errors2)


def calculate_integral_errors():
    # Calculates numerical integrals and compares them to analytical values
    correct_integral = 2
    errors_riemann = []
    errors_trapezoid = []
    errors_simpson = []
    grid_sizes = np.linspace(40, 1000, 10, dtype=int)
    for N in grid_sizes:
        x = np.linspace(0, np.pi, N)
        f = np.sin(x)
        errors_riemann.append(
            abs(riemann_sum(x, f) - correct_integral))

        errors_trapezoid.append(
            abs(trapedzoid_sum(x, f) - correct_integral))

        errors_simpson.append(
            abs(simpson_sum(x, f) - correct_integral))
    plot_integral_errors(
        grid_sizes, errors_riemann, errors_trapezoid, errors_simpson)


def calculate_monte_carlo_errors():
    # Calculates numerical integral using monte carlo simulation
    # and compares the values to analytical answer
    correct_integral = 2
    errors = []
    iteration_amounts = np.linspace(40, 1000, 10, dtype=int)
    for iters in iteration_amounts:
        integral, dI = monte_carlo_integration(np.sin, 0., np.pi/2, 10, iters)
        errors.append(2*dI)  # 2SEM
    plot_monte_carlo_error(iteration_amounts, errors)


def plot_derivative_errors(x, error, error2):
    # Plots the first and second order derivative errors in same figure
    fig = plt.figure()
    ax = fig.add_subplot(111)

    # plot and add label if legend desired
    ax.plot(x, error, label='error of first derivative')
    ax.plot(x, error2, label='error of second derivative')

    # plot legend
    ax.legend(loc=0)

    # set axes labels and limits
    ax.set_xlabel(r'$dx$')
    ax.set_ylabel(r'Absolute error')
    ax.set_xlim(x.min(), x.max())
    fig.tight_layout(pad=1)

    # save figure as pdf with 200dpi resolution
    fig.savefig('derivative_errors.pdf', dpi=200)
    plt.show()


def plot_integral_errors(x, error, error2, error3):
    # Plots the 3 numerical integration method errors in same figure
    fig = plt.figure()
    ax = fig.add_subplot(111)

    # plot and add label if legend desired
    ax.plot(x, error,  'r', label='error of Riemann sum')
    ax.plot(x, error2, 'b--', label='error of trapezoid sum')
    ax.plot(x, error3, label='error of Simpson rule')

    # plot legend
    ax.legend(loc=0)

    # set axes labels and limits
    ax.set_xlabel(r'$dx$')
    ax.set_ylabel(r'Absolute error')
    ax.set_xlim(x.min(), x.max())
    fig.tight_layout(pad=1)
    plt.grid('on')

    # save figure as pdf with 200dpi resolution
    fig.savefig('integral_errors.pdf', dpi=200)
    plt.show()


def plot_monte_carlo_error(x, error):
    # Plots the Monte Carlo integration errors as a function of iterations
    fig = plt.figure()
    ax = fig.add_subplot(111)

    # plot and add label if legend desired
    ax.plot(x, error, label='error of Monte Carlo integration')

    # plot legend
    ax.legend(loc=0)

    # set axes labels and limits
    ax.set_xlabel(r'Number of iterations')
    ax.set_ylabel(r'Absolute error')
    ax.set_xlim(x.min(), x.max())
    fig.tight_layout(pad=1)
    plt.grid('on')

    # save figure as pdf with 200dpi resolution
    fig.savefig('monte_carlo_errors.pdf', dpi=200)
    plt.show()


if __name__ == "__main__":
    calculate_derivative_errors()
    calculate_integral_errors()
    calculate_monte_carlo_errors()
