""" File containing numerical derivation and integration functions."""
import numpy as np


# numerical first order derivative, parameters are function, point and dx
def first_derivative(function, x, dx):
    # using definition of derivative
    return (function(x + dx) - function(x)) / dx


def test_first_derivative():
    print("testing first derivative")

    # basic polynom
    def fun(x):
        return 3 * x ** 2

    print("this should be 12: ", end="")
    print(first_derivative(fun, 2, 10e-5))

    # harder polynom
    def fun2(x):
        return 3 * x ** 5 - 2 * x + 4

    print("this should be 238: ", end="")
    print(first_derivative(fun2, 2, 10e-5))


# numerical second order derivative, parameters are function, point and dx
def second_derivative(function, x, dx):
    # formula 7 from week 1 lecture slides
    return (function(x + dx) + function(x - dx) - 2 * function(x)) / (dx * dx)


def test_second_derivative():
    print("testing second derivative")

    # basic polynom
    def fun3(x):
        return 3 * x ** 2

    print("this should be 6: ", end="")
    print(second_derivative(fun3, 2, 10e-5))

    # harder polynom
    def fun4(x):
        return 3 * x ** 5 - 2 * x + 4

    print("this should be 480: ", end="")
    print(second_derivative(fun4, 2, 10e-5))


# riemann sum for uniformal grid, takes grid (array) and numpy function as parameters
def riemann_sum(x, function):
    sum = 0
    # uniform grid so dx is constant
    dx = x[1] - x[0]
    # calculating the sum
    for i in range(len(x)):
        # formula 8 from week 1 lecture slides
        sum += dx * function[i]
    return sum


def test_riemann_sum():
    print("testing riemann sum")
    # creating grid and sin function with numpy.
    x = np.linspace(0, np.pi / 2, 100)
    f = np.sin(x)
    print("this should be 1: ", end="")
    print(riemann_sum(x, f))
    # creating grid and exp function with numpy.
    x = np.linspace(0, 1, 100)
    g = np.exp(x)
    print("this should be 1.7183: ", end="")
    print(riemann_sum(x, g))


# trapezoid rule integration for uniformal grid, takes grid (array) and numpy function as parameters
def trapezoid_integral(x, function):
    sum = 0
    # uniform grid so dx is constant
    dx = x[1] - x[0]
    # calculating the sum with formula 9 from week 1 lecture slides
    for i in range(len(x) - 1):
        sum += dx * (function[i] + function[i + 1])
    sum = 1 / 2 * sum
    return sum


def test_trapezoid_integral():
    print("testing trapezoid integral")
    # creating grid and sin function with numpy.
    x = np.linspace(0, np.pi / 2, 100)
    f = np.sin(x)
    print("this should be 1: ", end="")
    print(trapezoid_integral(x, f))
    # creating grid and exp function with numpy.
    x = np.linspace(0, 1, 100)
    g = np.exp(x)
    print("this should be 1.7183: ", end="")
    print(trapezoid_integral(x, g))


# simpson integration for uniformal grid, takes grid (array) and numpy function as parameters
def simpson_integral(x, function):
    sum = 0
    # uniform grid so dx is constant
    dx = x[1] - x[0]
    # calculating the sum with formula 10 from week 1 lecture slides
    # for even grid
    if len(x) % 2 == 0:
        for i in range(len(x) // 2 - 1):
            sum += (function[2 * i] + 4 * function[2 * i + 1] + function[2 * i + 2])
        sum = dx / 3 * sum
        return sum
    # for odd grid with formule 11 from week 1 lecture slides
    else:
        for i in range((len(x) - 1) // 2 - 1):
            sum += (function[2 * i] + 4 * function[2 * i + 1] + function[2 * i + 2])
        sum = dx / 3 * sum
        delta_sum = dx / 12 * (-function[len(x) - 3] + 8 * function[len(x) - 2] + 5 * function[len(x) - 1])
        sum += delta_sum
        return sum


def test_simpson_integral():
    print("testing simpson integral")
    # creating grid and sin function with numpy. odd grid.
    x = np.linspace(0, np.pi / 2, 99)
    f = np.sin(x)
    print("this should be 1: ", end="")
    print(simpson_integral(x, f))
    # creating grid and exp function with numpy. even grid.
    x = np.linspace(0, 1, 100)
    g = np.exp(x)
    print("this should be 1.7183: ", end="")
    print(simpson_integral(x, g))


# numerical monte carlo integration, takes function, start and end points on x axis, number of blocks
# and number of iterations for each block as parameters.
def monte_carlo_integration(fun, xmin, xmax, blocks, iters):
    # creating empty array for block values
    block_values = np.zeros((blocks,))
    # length of the integral interval
    L = xmax - xmin
    # looping through all the blocks
    for block in range(blocks):
        # looping though all the iterations inside block
        for i in range(iters):
            # calculating random x value from the integral interval
            x = xmin + np.random.rand() * L
            # adding value of function at given x to block_values
            block_values[block] += fun(x)
        # dividing block by number of iterations to get mean
        block_values[block] /= iters
    # value of integral is mean of the block values multiplied by lenght of interval
    I = L * np.mean(block_values)
    # calculating error estimate for integral
    dI = L * np.std(block_values) / np.sqrt(blocks)
    return I, dI


def test_monte_carlo_integration():
    print("testing meonte_carlo_integration")

    # testing sin function
    def func(x):
        return np.sin(x)

    print("this should be 1: ", end="")
    I, dI = monte_carlo_integration(func, 0., np.pi / 2, 10, 100)
    print(I, '+ / -', 2 * dI)

    # testing exp function
    def func(x):
        return np.exp(x)

    print("this should be 1.7183: ", end="")
    I, dI = monte_carlo_integration(func, 0., 1, 10, 100)
    print(I, '+ / -', 2 * dI)


def main():
    test_first_derivative()
    test_second_derivative()
    test_riemann_sum()
    test_trapezoid_integral()
    test_simpson_integral()
    test_monte_carlo_integration()


if __name__ == "__main__":
    main()
