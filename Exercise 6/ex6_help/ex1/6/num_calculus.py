import numpy as np

""" FYS-4096 Computational Physics: Exercise 1 """
""" Author: Santeri Saariokari """


def first_derivative(function, x, dx):
    # calculates the first derivative based on equation (5) in [1]
    return (function(x + dx) - function(x - dx)) / (2 * dx)

def gradient(functions, x, dx):
    grad = []
    for i in []:
        grad.append(functions[i](x, dx))
    return grad

def second_derivative(function, x, dx):
    # calculates the second derivative based on equation (7) in [1]
    return (function(x - dx) - 2 * function(x) + function(x + dx)) / dx**2


def riemann_sum(x, f):
    # Approximates the integral of f at points x using Riemann sum
    # eq (8) in [1]
    integral = 0
    dx = x[1] - x[0]
    for i in f:
        integral += dx * i
    return integral


def trapedzoid_sum(x, f):
    # Approximates the integral of f at points x using trapezoid rule
    # eq (9) in [1]
    integral = 0
    h = x[1] - x[0]
    for i in range(len(f) - 1):
        integral += h * (f[i] + f[i+1]) / 2
    return integral


def simpson_sum(x, f):
    # Approximates the integral of f at points x using Simpson rule
    # eq (10) in [1]
    integral = 0
    h = x[1] - x[0]

    # int() works as floor for floating numbers
    for i in range(0, int(len(f) / 2) - 1):
        # eq (10) in [1]
        integral += f[2*i] + 4 * f[2*i+1] + f[2*i+2]

    integral *= h / 3

    # even amount of points means uneven amount of intervals and vice versa
    if len(f) % 2 == 0:
        # eq (11) in [1]
        integral += (-f[-3] + 8 * f[-2] + 5 * f[-1]) * h / 12
    return integral


def monte_carlo_integration(fun, xmin, xmax, blocks, iters):
    # Approximates the integral of fun at points xmin - xmax
    # using Monte Carlo integration

    # Empty container for results
    block_values = np.zeros((blocks,))
    # Width of integration area
    L = xmax - xmin
    for block in range(blocks):
        for i in range(iters):
            # Get random point x from integration area
            x = xmin + np.random.rand() * L
            # Add function value at x to corresponding slot in container
            block_values[block] += fun(x)
        # get the block average by dividing values by amount
        block_values[block] /= iters
    # Area = width * (average) height
    integral = L * np.mean(block_values)
    # Error estimate for the integral using equation (22) in [1]
    dI = L * np.std(block_values) / np.sqrt(blocks)
    return integral, dI


def test_function(x):
    # hardcoded test function
    return np.sin(x)


def test_vector_function(x):
    # hardcoded test function
    return [lambda x: np.sin(x), lambda x: np.cos(x), lambda x: np.exp(x)]


def test_first_derivative():
    # test function for "first_derivative" using fixed values
    correct_val = 1
    test_x = 0
    test_dx = 0.001
    result = first_derivative(test_function, test_x, test_dx)
    print("x: ", test_x, ", dx: ", test_dx,
          ", 1. derivative: ", result)
    print("Error is ", abs(correct_val - result), "\n")

def test_gradient():
    # test function for "gradient" using fixed values
    correct_val = 1
    test_x = 0
    test_dx = 0.001
    print(test_vector_function)
    result = gradient(test_vector_function, test_x, test_dx)
    print(result)
    #print("x: ", test_x, ", dx: ", test_dx,
    #      ", 1. derivative: ", result)
    #print("Error is ", abs(correct_val - result), "\n")


def test_second_derivative():
    # test function for "second_derivative" using fixed values
    correct_val = 0
    test_x = 0
    test_dx = 0.001
    result = second_derivative(test_function, test_x, test_dx)
    print("x: ", test_x, ", dx: ", test_dx,
          ", 2. derivative: ", result)
    print("Error is ", abs(correct_val - result), "\n")


def test_riemann_sum():
    # test function for "riemann_sum" using fixed values
    x = np.linspace(0, np.pi/2, 100)
    f = np.sin(x)
    integral = riemann_sum(x, f)
    print("Riemann: ", integral)


def test_trapezoid_sum():
    # test function for "trapezoid_sum" using fixed values
    x = np.linspace(0, np.pi/2, 100)
    f = np.sin(x)
    integral = trapedzoid_sum(x, f)
    print("Trapezoid: ", integral)


def test_simpson_sum():
    # test function for "simpson_sum" using fixed values
    x = np.linspace(0, np.pi/2, 100)
    f = np.sin(x)
    integral = simpson_sum(x, f)
    print("Simpson: ", integral)


def test_monte_carlo():
    # test function for "monte_carlo_integration" using fixed values
    integral, dI = monte_carlo_integration(np.sin, 0., np.pi/2, 10, 100)
    # reliability estimate of 95% for 2SEM
    print("Monte Carlo: ", integral, '+/-', 2 * dI)
    correct_value = 1
    if(abs(integral - correct_value) < abs(2 * dI)):
        print("Monte Carlo result is correct")
    else:
        print("Monte Carlo result is wrong")


def main():
    # starts tests
    test_first_derivative()
    test_gradient()
    test_second_derivative()
    test_riemann_sum()
    test_trapezoid_sum()
    test_simpson_sum()
    test_monte_carlo()


if __name__ == "__main__":
    main()

# source material
# [1] https://moodle.tuni.fi/pluginfile.php/414057/mod_resource/content/1/week1.pdf
