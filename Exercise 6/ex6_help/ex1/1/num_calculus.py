# Luukas Kuusela, luukas.kuusela@tuni.fi 253061, Spring 2020
# Computational Physics exercise 1
# Primary source:
# [1] https://moodle.tuni.fi/pluginfile.php/414057/mod_resource/content/1/week1.pdf

import numpy as np


# Test function for "first derivative" using fixed values
def test_first_derivative():

    x = 0.8
    dx = 0.001
    error_limit = 0.01

    # Analytic derivative for test function 3*x**2
    test_val = 6 * x

    first_der = first_derivative(test_fun, x, dx)
    print("First derivative: ", first_der)

    if abs(first_der - test_val) < error_limit:
        print("First derivative result is correct")
    else:
        print("First derivative result is wrong")


# Test function for "second derivative" using fixed values
def test_second_derivative():
    x = 0.8
    dx = 0.001
    error_limit = 0.01

    # Analytic second derivative for test function 3*x**2
    test_val = 6

    second_der = second_derivative(test_fun, x, dx)
    print("Second derivative: ", second_der)

    if abs(second_der - test_val) < error_limit:
        print("Second derivative result is correct")
    else:
        print("Second derivative result is wrong")


# Test function for "riemann_sum" using fixed values
def test_riemann_sum():
    x = np.linspace(0, np.pi / 2, 100)
    test_sum = riemann_sum(x, test_int_fun(x))
    error_limit = 0.01
    test_val = test_int_val()
    print("Riemann sum: ", test_sum)

    if abs(test_sum - test_val) < error_limit:
        print("Riemann sum result is correct")
    else:
        print("Riemann sum result is wrong")


# Test function for "trapezoid_rule" using fixed values
def test_trapezoid_rule():
    x = np.linspace(0, np.pi / 2, 100)
    test_sum = trapezoid_rule(x, test_int_fun(x))
    error_limit = 0.01
    test_val = test_int_val()
    print("Trapezoid rule: ", test_sum)

    if abs(test_sum - test_val) < error_limit:
        print("Trapezoid rule result is correct")
    else:
        print("Trapezoid rule result is wrong")


# Test function for "simpson_rule" using fixed values
def test_simpson_rule():
    x = np.linspace(0, np.pi / 2, 100)
    test_sum = simpson_rule(x, test_int_fun(x))

    error_limit = 0.01
    test_val = test_int_val()
    print("Simpson rule: ", test_sum)

    if abs(test_sum - test_val) < error_limit:
        print("Simpson rule result is correct")
    else:
        print("Simpson rule result is wrong")


# Test function for "monte_carlo_integration" using fixed values
def test_monte_carlo():
    integral, dI = monte_carlo_integration(np.sin, 0., np.pi/2, 10, 100)
    print("Monte Carlo: ", integral, '+/-', 2 * dI)
    correct_value = 1

    if abs(integral - correct_value) < abs(dI):
        print("Monte Carlo result is correct")
    else:
        print("Monte Carlo result is wrong")


# Function returns the numerical estimate for first derivative of function given as parameter 'function'
# at point x + dx
def first_derivative(function, x, dx):
    # Let’s use two points symmetrically at distance dx away from x.
    # formula from [1]
    return (function(x+dx)-function(x-dx))/(2*dx)


# Function returns the numerical estimate for second derivative of function given as parameter 'function'
# at point x+ dx
def second_derivative(function, x, dx):
    # Formula from [1]
    return (function(x+dx)+function(x-dx)-2*function(x))/(dx**2)


# Calculates integral using riemann sum, integrand function
# values are given on a uniform grid, x is an array of values
# and f is an array of function values
def riemann_sum(x, f):
    sum = 0

    # Sum from 0 to N, f(xi)*delta_x where, delta_x is x[i+1]-x[i]
    # formula from [1]]
    for i in range(0, len(x)-1):
        sum += (x[i+1]-x[i])*f[i]
    return sum


# Calculates integral using trapezoid rule, integrand function
# values are given on a uniform grid, x is an array of values
# and f is an array of function values
def trapezoid_rule(x, f):
    sum = 0

    # formula from [1]]
    for i in range(0, len(x)-1):
        sum += (f[i]+f[i+1])*(x[i+1]-x[i])
    return sum/2


# Calculates integral using simpson rule, integrand function
# values are given on a uniform grid, x is an array of values
# and f is an array of function values
def simpson_rule(x, f):
    sum = 0

    # spacing in an uniform grid is give as difference between any two neighbouring points
    h = x[1]-x[0]

    even = False

    # check if there is an even number of spacings, this is when there is an uneven number of points
    if len(x) % 2 != 0:
        even = True
        # we use int() to make N an integer
        N = int((len(x)/2)-1)

    else:
        # -1 to make the number of spacings even, last slice is added in the end
        N = int(((len(x))/2)-1)

    # formula from [1]]
    for i in range(0, N):
        sum += f[2*i]+4*f[2*i+1]+f[2*i+2]
    I = (h/3)*sum

    # uneven number of spacings, add the last slice
    N = len(x)-1
    if not even:
        # formula from [1]
        I += (h/12)*(-f[len(x)-3]+8*f[len(x)-2]+5*f[len(x)-1])
    return I


# Approximates the integral of fun using Monte Carlo at points xmin - xmax
def monte_carlo_integration(fun,xmin,xmax,blocks,iters):

    # Create an empty array to store values in
    block_values=np.zeros((blocks,))

    # Width of integration
    L = xmax-xmin

    for block in range(blocks):
        for i in range(iters):
            # Get random point x from within the integration area
            x = xmin+np.random.rand()*L
            # Add function value at x to corresponding slot in container
            block_values[block] += fun(x)
        # Get the block average by dividing values by amount
        block_values[block] /= iters
    # Area is given by width * (average) height
    I = L*np.mean(block_values)
    # Error estimate for the integral using equation 22 in [1]
    dI = L*np.std(block_values)/np.sqrt(blocks)
    return I,dI


# Function used for testing integrals, if this is changed
# change also the correct values in tst functions
def test_int_fun(x):
    return np.sin(x)


# Correct value for integral used for testing integrals, if this is changed
# change also the function used for testing
def test_int_val():
    # integral of sin(x) from 0 to pi/2
    return 1


# We define the function to test derivatives here
# if this is changed, change also the correct values used in derivaive test functions
def test_fun(x):
    return 3 * x ** 2


def main():

    # First and second derivative

    test_first_derivative()
    test_second_derivative()

    # Integrals
    test_riemann_sum()
    test_trapezoid_rule()
    test_simpson_rule()
    test_monte_carlo()


if __name__=="__main__":
    main()