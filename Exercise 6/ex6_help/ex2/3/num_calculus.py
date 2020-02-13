"""Many functions for numerically estimating derivates and integrals
of single argument python functions. Includes first and second derivatives,
left riemann sum and trapezoid rule, simpson's rule and monte carlo integrals. Also includes tests for the functions."""

import numpy as np


def first_derivative(function, x, dx):
    # A function that takes a given single argument function
    # and estimates its first derivative at a given point x using 
    # given dx as the difference in the difference quontient.

    return (function(x + dx)-function(x))/dx


def test_first_derivative():
    # A function that tests first_derivative(), comparing values
    # gotten from the function with known answers. Returns 1 if
    # the function is working properly and 0 otherwise.

    def fun1(x): return np.sin(x)
    def fun2(x): return 3*x**2
    if np.abs(first_derivative(fun1, 1, 0.001) - np.cos(1)) > 0.01:
        return 0
    elif np.abs(first_derivative(fun2, 10, 0.001) - 60) > 0.01:
        return 0
    else:
        return 1


def second_derivative(function, x, dx):
    # A function that takes a given single argument function
    # and estimates its second derivative at a given point x using 
    # given dx as the difference in the difference quontient.
    
    return (function(x + dx)+function(x - dx)-2*function(x))/dx**2


def test_second_derivative():
    # A function that tests second_derivative(), comparing values
    # gotten from the function with known answers. Returns 1 if
    # the function is working properly and 0 otherwise.

    def fun1(x): return np.sin(x)
    def fun2(x): return 3*x**3
    if np.abs(second_derivative(fun1, 1, 0.001) + np.sin(1)) > 0.001:
        return 0
    elif np.abs(second_derivative(fun2, 10, 0.001) - 180) > 0.001:
        return 0
    else:
        return 1


def riemann_sum(x, function):
    # A function that calculates the left riemann sum of a given function
    # using its values over linspace x.

    diff = x[1] - x[0]
    sum = 0
    for i in range(len(x) - 1):
        sum += function(x[i])*diff
    return sum


def test_riemann_sum():
    # A function that tests riemann_sum(), comparing values
    # gotten from the function with known answers. Returns 1 if
    # the function is working properly and 0 otherwise.

    points = np.linspace(0, 1, 1000)
    def fun1(x): return x**3
    if np.abs(riemann_sum(points, fun1) - 1/4) > 0.001:
        return 0
    else:
        return 1


def trapezoid_rule(x, function):
    # A function that estimates the integral of a given function over a
    # given linspace using trapezoid rule.

    diff = x[1] - x[0]
    sum = 0
    for i in range(len(x) - 1):
        sum += 0.5*(function(x[i])+function(x[i+1]))*diff
    return sum


def test_trapezoid_rule():
    # A function that tests trapezoid_rule(), comparing values
    # gotten from the function with known answers. Returns 1 if
    # the function is working properly and 0 otherwise.

    points = np.linspace(0, 1, 1000)
    def fun1(x): return x**3
    if np.abs(trapezoid_rule(points, fun1) - 1/4) > 0.0001:
        return 0
    else:
        return 1

def simpson_rule( x, function ):
    # A function that estimates the integral of a given function over a
    # given linspace with odd number of ponts using Simpson rule.

    # The function checks that x has even number of intervals.
    if len(x) % 2 == 0:
        return None

    # If there were even number of intervals, the function estimates
    # the integral.
    diff = x[1] - x[0]
    sum = 0
    for i in range(int(len(x)/2) -1):
        sum += (function(x[2*i])+4*function(x[2*i+1])+function(x[2*i+2]))
    return diff/3*sum
    
def test_simpson_rule():
    # A function that tests simpson_rule(), comparing values
    # gotten from the function with known answers. Returns 1 if
    # the function is working properly and 0 otherwise.

    points = np.linspace(0, 1, 1001)
    points2 = np.linspace(0, 1, 10)
    def fun1(x): return x**3
    if np.abs(simpson_rule(points, fun1) - 1/4) > 0.01:
        return 0
    # With odd number of intervals the function should return None
    elif simpson_rule(points2, fun1) is not None:
        return 0
    else:
        return 1

def monte_carlo_integration(fun, xmin, xmax, blocks, iters):
    # A function that integrates a given function fun from
    # xmin to xmax using monte carlo integration with a given number
    # of blocks with a given number of iterations.

    block_values = np.zeros((blocks,))
    L = xmax - xmin
    for block in range(blocks):
        for i in range(iters):
            x = xmin + np.random.rand()*L
            block_values[block] += fun(x)
        block_values[block] /= iters
    I = L*np.mean(block_values)
    dI = L*np.std(block_values)/np.sqrt(blocks)
    return I,dI

def test_monte_carlo():
    # A function that tests monte_carlo_integral(), comparing values
    # gotten from the function with known answers. Returns 1 if
    # the function is working properly and 0 otherwise.

    def fun1(x): return np.sin(x)
    def fun2(x): return x**3
    if np.abs(monte_carlo_integration(fun1, 0, np.pi/2, 10, 1000)[0] - 1) > 0.01:
        return 0
    elif np.abs(monte_carlo_integration(fun2, 0, 1, 10, 1000)[0] -1/4) > 0.01:
        return 0
    else:
        return 1

def gradient(fun, coordinates):
    # A function for calculating the gradient of a N-dimensional function at a point
    # given by a list of coordinates. Returns the gradient in a list.

    gradient = np.zeros(len(coordinates))
    for i in range(len(coordinates)):
        new_co = coordinates
        # Create a help function where all but number i of the variables is constant
        def new_fun(x):
            new_co[i] = x
            return fun(new_co)
        gradient[i] = first_derivative(new_fun, coordinates[i], 0.00001)
    return gradient

def test_gradient():
    # A function that tests gradient(), comparing values
    # gotten from the function with known answers. Returns 1 if
    # the function is working properly and 0 otherwise.

    coordinates = [1, 1, 1, 1]
    def fun(co): return co[0] + co[1] + co[2]**2 + co[3]**3

    grad = gradient(fun, coordinates)
    ans = [1, 1, 2, 3]
    for i in range(4):
        if np.abs(grad[i] - ans[i]) > 0.01:
            return 0
    return 1

def steepest_descent(fun, starting_coordinates):
    # A function that finds a minimum of a given N-dimensional function using
    # steepest descent method, starting at a given value.

    co = starting_coordinates
    while np.linalg.norm(gradient(fun, co)) > 0.001:
        grad = gradient(fun, co)
        const = 1 / (np.linalg.norm(grad) + 1)
        for i in range(len(co)):
            co[i] -= const*grad[i]

    return co

def test_steepest_descent():
    # A function that tests steepest_descent(), comparing values
    # gotten from the function with known answers. Returns 1 if
    # the function is working properly and 0 otherwise.

    def fun(co): return (co[0]-1)**2 +(co[1]-2)**2
    minimum = steepest_descent(fun, [0, 0])
    if np.abs(minimum[0] - 1) > 0.01:
        return 0
    if np.abs(minimum[1] - 2) > 0.01:
        return 0
    return 1


def main():
    # A function that tests that all functions included are working
    # properly.

    first_derivative_test = test_first_derivative()
    second_derivative_test = test_second_derivative()
    riemann_sum_test = test_riemann_sum()
    trapezoid_rule_test = test_trapezoid_rule()
    simpson_rule_test = test_simpson_rule()
    monte_carlo_test = test_monte_carlo()

    if first_derivative_test == 0:
        print("First_derivative() is not working properly.")
    if second_derivative_test == 0:
        print("Second_derivative() is not working properly.")
    if riemann_sum_test == 0:
        print("riemann_sum() is not working properly.")
    if trapezoid_rule_test == 0:
        print("trapezoid_rule() is not working properly.")
    if simpson_rule_test == 0:
        print("simpson_rule() is not working properly.")
    if monte_carlo_test == 0:
        print("monte_carlo_integration() is not working properly.")
    if test_gradient() == 0:
        print("gradient() is not working properly.")
    if test_steepest_descent() == 0:
        print("steepest_descent() is not working properly.")
    if (first_derivative_test == 1 and second_derivative_test == 1 and
            riemann_sum_test == 1 and trapezoid_rule_test == 1 and
            simpson_rule_test == 1 and monte_carlo_test == 1) and \
            test_gradient() == 1 and test_steepest_descent() == 1:
        print("Everything is working as intended.")


if __name__ == "__main__":
    main()
