# Luukas Kuusela, luukas.kuusela@tuni.fi 253061, Spring 2020
# Computational Physics exercise 1

import numpy as np
import matplotlib.pyplot as plt
from num_calculus import *


# Functions used for testing
def test_fun1(x):
    return np.sin(x)


def test_fun1_diff(x):
    # Analytic derivative of sin(x)
    return np.cos(x)


def test_fun1_diff2(x):
    # Analytic second derivative of sin(x)
    return -np.sin(x)


def test_fun2(x):
    return np.exp(x)*np.cos(-x)


def test_fun2_diff(x):
    # Analytic derivative of exp(x)*cos(-x)
    return np.exp(x)*(np.cos(x)-np.sin(x))


def test_fun2_diff2(x):
    # Analytic second derivative of exp(x)*cos(-x)
    return -2*np.exp(x)*(np.sin(x))


def test_fun2_val():
    # Analytic integral of exp(x)*cos(-x) from 0 to pi/2
    return (np.exp(np.pi)*np.sinh(np.pi))



def my_plot():

    size = 100
    x = np.linspace(0, np.pi / 2, size)

    # point at which derivative is considered
    xx = np.pi/3

    dx = np.linspace(0.001, 0.1, size)

    # error of first derivative for fun1
    error_f1_diff = np.zeros(size)
    for i in range(len(dx)):
        error_f1_diff[i] = np.abs(test_fun1_diff(xx) - first_derivative(test_fun1, xx, dx[i]))

    # error of first derivative for fun2
    error_f2_diff = np.zeros(size)
    for i in range(len(dx)):
        error_f2_diff[i] = np.abs(test_fun2_diff(xx) - first_derivative(test_fun2, xx, dx[i]))

    fig1 = plt.figure()
    # - or, e.g., fig = plt.figure(figsize=(width, height))
    # - so-called golden ratio: width=height*(sqrt(5.0)+1.0)/2.0
    ax1 = fig1.add_subplot(111)

    # plot and add label if legend desired
    ax1.plot(dx, error_f1_diff, label=r'$f(x)=\sin(x)$')
    ax1.plot(dx, error_f2_diff, label=r'$f(x)=\exp(x)*cos(-x)$')

    # plot legend
    ax1.legend(loc=0)

    # set axes labels and limits
    ax1.set_xlabel(r'$dx$')
    ax1.set_ylabel(r"Absolute error")
    ax1.set_title(r"Absolute error of f'(x) at x=pi/3")
    ax1.set_xlim(dx.min(), dx.max())
    fig1.tight_layout(pad=1)

    # save figure as pdf with 200dpi resolution
    fig1.savefig('testfile.pdf', dpi=200)

    # error of second derivative for fun1
    error_f1_diff2 = np.zeros(size)
    for i in range(len(dx)):
        error_f1_diff2[i] = np.abs(test_fun1_diff2(xx) - second_derivative(test_fun1, xx, dx[i]))

    # error of second derivative for fun2
    error_f2_diff2 = np.zeros(size)
    for i in range(len(dx)):
        error_f2_diff2[i] = np.abs(test_fun2_diff2(xx) - second_derivative(test_fun2, xx, dx[i]))

    fig2 = plt.figure()
    # - or, e.g., fig = plt.figure(figsize=(width, height))
    # - so-called golden ratio: width=height*(sqrt(5.0)+1.0)/2.0
    ax2 = fig2.add_subplot(111)

    # plot and add label if legend desired
    ax2.plot(dx, error_f1_diff2, label=r'$f(x)=\sin(x)$')
    ax2.plot(dx, error_f2_diff2, label=r'$f(x)=\exp(x)*cos(-x)$')

    # plot legend
    ax2.legend(loc=0)

    # set axes labels and limits
    ax2.set_xlabel(r'$dx$')
    ax2.set_ylabel(r"Absolute error")
    ax2.set_title(r"Absolute error of f''(x) at x=pi/3")
    ax2.set_xlim(dx.min(), dx.max())
    fig2.tight_layout(pad=1)

    # save figure as pdf with 200dpi resolution
    fig2.savefig('testfile2.pdf', dpi=200)


    # Integral
    # Grid spacing is given by b-a/n where a is start point, b is end point and n is the number of grid points

    # Container for error of trapezoid rule
    error_trapz = np.zeros(size)
    # Container for spacing dx
    dx = np. zeros(size)

    curr_point = 0
    for i in range(10, size):
        # Make new grid spacing
        x = np.linspace(0, 2*np.pi, i)
        error_trapz[curr_point] = np.abs(trapezoid_rule(x, test_fun2(x)) - test_fun2_val())
        # Store used grid spacing
        dx[curr_point] = (2*np.pi)/i
        curr_point += 1

    fig3 = plt.figure()
    # - or, e.g., fig = plt.figure(figsize=(width, height))
    # - so-called golden ratio: width=height*(sqrt(5.0)+1.0)/2.0
    ax3 = fig3.add_subplot(111)

    # plot and add label if legend desired
    ax3.plot(dx, error_trapz, label=r'$f(x)=\exp(x)*cos(-x)$')

    # plot legend
    ax3.legend(loc=0)

    # set axes labels and limits
    ax3.set_xlabel(r'$dx$')
    ax3.set_ylabel(r"Absolute error ")
    ax3.set_title(r"Absolute error of trapezoid rule")
    ax3.set_xlim(dx.min(), dx.max())
    fig3.tight_layout(pad=1)

    # save figure as pdf with 200dpi resolution
    fig3.savefig('testfile3.pdf', dpi=200)
    plt.show()


def main():
    my_plot()


if __name__=="__main__":
    main()