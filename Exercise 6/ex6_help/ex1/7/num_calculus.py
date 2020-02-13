#!/usr/bin/python3
# Exercise1 / Arttu Hietalahti 262981
import numpy as np


def first_derivative(function, x, dx):
    return (function(x+dx) - function(x))/dx


def second_derivative(function, x, dx):
    # returns rate of change of first derivative at point x
    der_x = first_derivative(function, x, dx)  # first derivative at x
    der_xdx = first_derivative(function, x+dx, dx)  # first derivative at x + dx
    return (der_xdx-der_x)/dx


def test_first_derivative():
    print('Test for first_derivative function')
    x = 1  # x for which derivative is  calculated
    dx = 0.001  # dx used in numerical derivation
    tol = 0.005  # tolerance for testing
    correct_der = 6*x  # derivative of fun1
    num_der = first_derivative(fun1, x, dx)
    abs_err = np.abs(num_der-correct_der)

    print('Numerical derivative at x=' + str(x) + ' is:  ' + str(num_der))
    print('Analytical derivative at x=' + str(x) + ' is:  ' + str(correct_der))
    print('Absolute error is:  ' + str(abs_err))

    if abs_err < tol:
        print("Test passed! (tolerance: " + str(tol) + ")")
    else:
        print("Test failed! (tolerance: " + str(tol) + ")")


def test_second_derivative():
    print('Test for second_derivative function')
    x = 1  # x for which derivative is  calculated
    dx = 0.001  # dx used in numerical derivation
    tol = 0.005  # tolerance for testing
    correct_der2 = 6  # 2nd derivative of fun1

    num_der = second_derivative(fun1, x, dx)
    abs_err = np.abs(num_der - correct_der2)

    print('Numerical 2nd derivative at x=' + str(x) + ' is:  ' + str(num_der))
    print('Analytical 2nd derivative at x=' + str(x) + ' is:  ' + str(correct_der2))
    print('Absolute error is:  ' + str(abs_err))

    if abs_err < tol:
        print("Test passed! (tolerance: " + str(tol) + ")")
    else:
        print("Test failed! (tolerance: " + str(tol) + ")")


# function calculates riemann sum
# x is the points in x-axis (uniform grid)
# f is the function values corresponding to x
def riemann_sum(x, f):
    step = x[1]-x[0]  # uniform step
    rsum = 0
    for i in range(x.shape[0] - 1):
        rsum += f[i]*step  # add rectangle to the riemann sum

    return rsum


def riemann_test():
    print('Test for riemann_sum function')
    # test riemann_sum by integrating fun1 through range 0 to 2
    x = np.linspace(0, 2, 1000)
    f = fun1(x)
    tol = 0.05
    correct_int = 8  # integral of fun1 through range 0 to 2
    riemann_int = riemann_sum(x, f)

    abs_err = np.abs(correct_int-riemann_int)

    print('Riemann integral of fun1 through range 0 to 2 is ' + str(riemann_int))
    print('Analytical integral of fun1 through range 0 to 2 is ' + str(correct_int))
    print('Absolute error is:  ' + str(abs_err))

    if abs_err < tol:
        print("Test passed! (tolerance: " + str(tol) + ")")
    else:
        print("Test failed! (tolerance: " + str(tol) + ")")


# function calculates integral using the trapezoid rule
# x is the array of points in x-axis (uniform grid)
# f is the array of function values corresponding to x
def trapezoid(x, f):
    step = x[1] - x[0]  # uniform step

    trapz_sum = 0
    for i in range(x.shape[0] - 1):
        trapz_sum += 1/2*(f[i] + f[i+1])*step  # add trapezoid to the sum

    return trapz_sum


def trapezoid_test():
    print('Test for trapezoid function')
    # test trapezoid function by integrating fun1 through range 0 to 2
    x = np.linspace(0, 2, 1000)
    f = fun1(x)
    tol = 0.05
    correct_int = 8  # integral of fun1 through range 0 to 2
    trapz_int = trapezoid(x, f)

    abs_err = np.abs(correct_int-trapz_int)

    print('Trapezoid integral of fun1 through range 0 to 2 is ' + str(trapz_int))
    print('Analytical integral of fun1 through range 0 to 2 is ' + str(correct_int))
    print('Absolute error is:  ' + str(abs_err))

    if abs_err < tol:
        print("Test passed! (tolerance: " + str(tol) + ")")
    else:
        print("Test failed! (tolerance: " + str(tol) + ")")


def simpson_integral(x, f):
    step = x[1] - x[0]  # uniform step

    n = x.shape[0] - 1  # number of intervals in x
    simpson_int = 0
    if n % 2 == 1:  # uneven number of intervals
        for i in range(n//2):
            simpson_int += step/3 * (f[2*i] + 4*f[2*i+1] + f[2*i+2])
        simpson_int += step/12 * (-1*f[n-2] + 8*f[n-1] + 5*f[n])  # last term for uneven number of intervals
    else:  # even number of intervals
        for i in range(n//2):
            simpson_int += step/3 * (f[2*i] + 4*f[2*i+1] + f[2*i+2])  # common form simpson rule

    return simpson_int


def simpson_test():
    print('Test for simpson function')
    # test simpson function by integrating fun1 through range 0 to 2
    x = np.linspace(0, 2, 1000)
    f = fun1(x)
    tol = 0.05
    correct_int = 8  # integral of fun1 through range 0 to 2
    simpson_int = simpson_integral(x, f)

    abs_err = np.abs(correct_int-simpson_int)

    print('Simpson integral of fun1 through range 0 to 2 with step is ' + str(simpson_int))
    print('Analytical integral of fun1 through range 0 to 2 is ' + str(correct_int))
    print('Absolute error is:  ' + str(abs_err))

    if abs_err < tol:
        print("Test passed! (tolerance: " + str(tol) + ")")
    else:
        print("Test failed! (tolerance: " + str(tol) + ")")


def monte_carlo_integration(fun, xmin, xmax, blocks, iters):
    block_values = np.zeros((blocks,))  # blocks will each contain average value of function in range
    L = xmax-xmin  # width of range
    for block in range(blocks):
        for i in range(iters):  # iterate multiple times per block
            x = xmin + np.random.rand()*L  # random x point in range
            block_values[block] += fun(x)  # add function value at x to block value
        block_values[block]/=iters  # calculate average function value in range
    I = L*np.mean(block_values)  # calculate integral through L and average of block values (average values of function)
    dI = L*np.std(block_values)/np.sqrt(blocks)  # average error
    return I, dI


def monte_carlo_test():
    print('Test for monte_carlo_integration function')
    # test monte carlo function by integrating fun1 through range 0 to 2
    tol = 0.05
    correct_int = 8  # integral of fun1 through range 0 to 2
    mc_int, dI = monte_carlo_integration(fun1, 0, 2, 100, 1000)

    abs_err = np.abs(correct_int - mc_int)

    print('Monte Carlo integral of fun1 through range 0 to 2 is ' + str(mc_int))
    print('Analytical integral of fun1 through range 0 to 2 is ' + str(mc_int))
    print('Absolute error is:  ' + str(abs_err))

    if abs_err < tol:
        print("Test passed! (tolerance: " + str(tol) + ")")
    else:
        print("Test failed! (tolerance: " + str(tol) + ")")


def fun1(x): return 3*x**2


def func(x): return np.sin(x)


def main():
    test_first_derivative()
    print('\n')
    test_second_derivative()
    print('\n')
    riemann_test()
    print('\n')
    trapezoid_test()
    print('\n')
    simpson_test()
    print('\n')
    monte_carlo_test()
    print('\n')

    I, dI = monte_carlo_integration(func, 0., np.pi/2, 10, 100)
    print('Monte Carlo result for func through [0, pi/2]: ', I, '+/-', 2*dI)


if __name__ == "__main__":
    main()