""" visual convergence check for derivative functions and riemann sum function in file num_calculus.py"""
import numpy as np
import matplotlib.pyplot as plt
from num_calculus import first_derivative
from num_calculus import second_derivative
from num_calculus import riemann_sum

# plots function. parameters are x coordinates, y coordinates (function values) and filename
# for the figure
def my_plot(x,f, filename):
    fig = plt.figure()
    ax = fig.add_subplot(111)

    # plot and add label if legend desired
    ax.plot(x, f, label=r'error')
    # plot legend
    ax.legend(loc=0)
    # set axes labels and limits
    ax.set_xlabel(r'dx')
    ax.set_ylabel(r'absolute error')
    ax.set_xlim(x.min(), x.max())
    fig.tight_layout(pad=1)
    # save figure as pdf with 200dpi resolution
    fig.savefig(filename, dpi = 200)
    plt.show()

# Plots convergence of first derivate with different intervals. Parameters are function, point and analytically
# calculated real value.
def first_derivative_convergence_figure(fun, point, real_value):
    # making different intervals and empty array for y coordinates
    dx = np.linspace(1, 10e-3, 100)
    y = np.zeros((len(dx)))
    # calculating difference to analytical value for all intervals
    for i in range(len(dx)):
        diff=abs(real_value - first_derivative(fun, point, dx[i]))
        y[i]=diff
    # plotting figure
    my_plot(dx, y, "first_derivative_error_figure.pdf")

# Plots convergence of second derivate with different intervals. Parameters are function, point and analytically
# calculated real value.
def second_derivative_convergence_figure(fun, point, real_value):
    # making different intervals and empty array for y coordinates
    dx = np.linspace(1, 10e-3, 100)
    y = np.zeros((len(dx)))
    # calculating difference to analytical value for all intervals
    for i in range(len(dx)):
        diff=abs(real_value - second_derivative(fun, point, dx[i]))
        y[i]=diff
    # plotting figure
    my_plot(dx, y, "second_derivative_error_figure.pdf")

# Plots convergence of riemann sum with different intervals. Parameters are function, start point ,end point and
# analytically calculated real value.
def riemann_sum_convergence_figure(fun, xmin, xmax, real_value):
    y = np.zeros(51)
    dx = np.zeros(51)
    # different number of gaps for intervals
    for i in range(50,100):
        # making x coordinate and interval
        x = np.linspace(1, 10e-3, i)
        interval = x[0]-x[1]
        dx[i-50] = interval
        # calculating difference and y coordinate.
        diff = abs(real_value - riemann_sum(x, fun(x)))
        y[i-50] = diff
    # plotting figure
    my_plot(dx, y, "riemann_sum_error_figure.pdf")

# defines test function and calls all the figures.
def test_convergence_figure():
    def fun(x):
        return np.sin(x)
    print(first_derivative_convergence_figure(fun, 0, 1))
    print(second_derivative_convergence_figure(fun, np.pi/2, -1))
    print(riemann_sum_convergence_figure(fun,0,np.pi/2,1))

def main():
    test_convergence_figure()

if __name__=="__main__":
    main()