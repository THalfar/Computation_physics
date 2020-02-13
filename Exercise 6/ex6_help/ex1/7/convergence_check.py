#!/usr/bin/python3
# Exercise1 / Arttu Hietalahti 262981

import numpy as np
import matplotlib.pyplot as plt
from num_calculus import first_derivative, second_derivative, trapezoid


# my_plot1 for drawing absolute error of functions first_derivative and second_derivative
def my_plot1():
    fig = plt.figure()
    ax = fig.add_subplot(111)

    #  calculate numerical derivatives for sine function at x = pi/2 with different spacings
    #  correct first derivative is 0
    #  correct second derivative is -1
    x = np.pi/2
    f = np.sin(x)

    correct_der1 = 0
    correct_der2 = -1
    dx_spacing = np.linspace(0.001, 1, 100)

    first_derivative_abs_err = []
    second_derivative_abs_err = []

    # save absolute error for each spacing into array first_derivative_abs_err
    for i in range(dx_spacing.shape[0]):
        err = abs(first_derivative(np.sin, x, dx_spacing[i]) - correct_der1)
        first_derivative_abs_err.append(err)

    # same for second derivative
    for i in range(dx_spacing.shape[0]):
        err = abs(second_derivative(np.sin, x, dx_spacing[i]) - correct_der2)
        second_derivative_abs_err.append(err)

    # plot and add label if legend desired
    ax.plot(dx_spacing, first_derivative_abs_err, label=r'first_derivative abs error')
    ax.plot(dx_spacing, second_derivative_abs_err, label=r'second_derivative abs error')

    ax.legend(loc=0)

    # set axes labels and limits
    ax.set_xlabel(r'x spacing')
    ax.set_ylabel(r'Absolute error')
    ax.set_xlim(dx_spacing.min(), dx_spacing.max())
    ax.set_ylim(bottom=0)

    plt.title('Absolute error as a function of x spacing. \nEvaluated derivative of function sin(x) at x=pi/2')
    fig.tight_layout(pad=1)
    # save figure as pdf with 200dpi resolution
    fig.savefig('derivative_error.pdf', dpi = 200)
    plt.show()


# my_plot2 for drawing absolute error of trapezoid function
def my_plot2():
    fig = plt.figure()
    # - or, e.g., fig = plt.figure(figsize=(width, height))
    # - so-called golden ratio: width=height*(sqrt(5.0)+1.0)/2.0
    ax = fig.add_subplot(111)

    #  integral to evaluate is sin(x) through range [0, pi]
    #  correct integral value is 2
    correct_int = 2
    trapezoid_abs_err = []

    # calculate trapezoid abs error for function sin(x) through range [0, pi] for each x spacing
    x_point_amount = np.arange(1000, 10, -1)
    x_spacings = np.pi/x_point_amount
    for n in x_point_amount:
        x = np.linspace(0, np.pi, n)
        f = np.sin(x)
        err = abs(trapezoid(x, f) - correct_int)
        trapezoid_abs_err.append(err)

    np.flipud(x_spacings)  # flip x_spacings to match trapezoid_abs_err
    # plot absolute error as a function of x spacing
    ax.plot(x_spacings, trapezoid_abs_err, label=r'trapezoid integral abs error')

    ax.legend(loc=0)
    ax.set_xlabel(r'x spacing')
    ax.set_ylabel(r'Absolute error')
    ax.set_xlim(x_spacings.min(), x_spacings.max())
    ax.set_ylim(bottom=0)

    plt.title('Absolute error as a function of x spacing. \nEvaluated integral: sin(x) through range [0, pi]')
    fig.tight_layout(pad=1)
    # save figure as pdf with 200dpi resolution
    fig.savefig('trapezoid_error.pdf', dpi=200)
    plt.show()


def main():
    my_plot1()
    my_plot2()


if __name__ == "__main__":
    main()
