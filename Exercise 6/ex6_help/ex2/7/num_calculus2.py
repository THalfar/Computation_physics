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
    print('Analytical integral of fun1 through range 0 to 2 is ' + str(correct_int))
    print('Absolute error is:  ' + str(abs_err))

    if abs_err < tol:
        print("Test passed! (tolerance: " + str(tol) + ")")
    else:
        print("Test failed! (tolerance: " + str(tol) + ")")


def fun1(x): return 3*x**2


def fun2(x, y): return x**2*y**2


def fun3(x, y, z): return x**2*y**2*z**2


def func(x): return np.sin(x)


def num_gradient(function, pos, step):
    """
    function returns N-dimensional gradient of function at position with accuracy set by step
    :param function: function to evaluate gradient for
    :param pos: position at which gradient is evaluated (as a list)
    :param step: step used for numerical derivative in each dimension
    :return: grad: the numerical gradient as a list (or a number in 1D case)
    """
    if not isinstance(pos, list):
        # 1D case separately
        grad = first_derivative(function, pos, step)
    else:
        # 2 or more dimensions
        dims = len(pos)
        grad = []
        for i in range(dims):
            pos2 = list(pos)  # copy position
            pos2[i] += step  # add step to dimension i
            der = (function(*pos2) - function(*pos))/step  # function derivative in dimension i
            grad.append(der)  # add derivative to gradient list

    return grad


def num_gradient_test():
    print("Test for num_gradient function")
    print("_______________1D CASE_______________")
    pos = 1
    correct_grad1 = 6  # correct gradient of fun1 at pos = 1
    step = 0.00001
    num_grad1 = num_gradient(fun1, pos, step)

    print('Function: 3*x^2  ||  Position: x = ' + str(pos) + ' || Step size: ' + str(step))
    print('Numerical gradient: ' + str(num_grad1))
    print('Analytical gradient: ' + str(correct_grad1) + '\n')

    print("_______________2D CASE_______________")
    pos = [4, 5]
    correct_grad2 = [200, 160]  # correct gradient of fun2 at pos = [4, 5]
    step = 0.00001
    num_grad2 = num_gradient(fun2, pos, step)

    print('Function: x^2*y^2  ||  Position: [x, y] = ' + str(pos) + ' || Step size: ' + str(step))
    print('Numerical gradient: ' + str(num_grad2))
    print('Analytical gradient: ' + str(correct_grad2) + '\n')

    print("_______________3D CASE_______________")
    pos = [1, 2, 3]
    correct_grad3 = [72, 36, 24]  # correct gradient of fun3 at pos = [1, 2, 3]
    step = 0.00001
    num_grad3 = num_gradient(fun3, pos, step)

    print('Function: x^2*y^2*z^2  ||  Position: [x, y, z] = ' + str(pos) + ' || Step size: ' + str(step))
    print('Numerical gradient: ' + str(num_grad3))
    print('Analytical gradient: ' + str(correct_grad3) + '\n')


def minimum_by_gradient_descent(function, startpos, step_param = 1.0, grad_step=1e-8, tol=1e-6):
    """
        finds the minimum position of function using gradient descent method (from lecture slides, week 2)
    :param function: function to evaluate minimum for
    :param startpos: position to start the search
    :param step_param: used in the gradient descent step (a term in lecture slides)
    :param grad_step: how accurately the numerical gradient is calculated each time
    :param tol: how small the gradient norm must be to accept position as minimum
    :return: minimum position of function
    """
    pos = np.array(startpos)  # start search from start position
    while True:
        print(pos)
        gradient = np.array(num_gradient(function, list(pos), grad_step))
        gradient_norm = np.linalg.norm(gradient)

        # exit condition is if the gradient norm is close enough to zero
        if gradient_norm < tol:
            return pos

        step_n = step_param/(gradient_norm + 1)  # nth step lecture slides formula (31)
        pos = np.subtract(pos, np.multiply(step_n, gradient))  # lecture slides


def minimum_by_gradient_descent_test():
    # simple test function for gradient descent function

    print("Testing function minimum_by_gradient_descent")

    def fun4(x, y): return x * y

    start_point = [5, 5]
    correct_min_pos = [0, 0]
    correct_min_value = fun4(*correct_min_pos)

    gradient_desc_min_pos = minimum_by_gradient_descent(fun4, start_point, step_param=1.0, tol=1e-8)
    gradient_desc_min_value = fun4(*gradient_desc_min_pos)

    print("\nSearched minimum for f = x*y from start position [x, y] = [5, 5]")
    print("Tolerance: gradient norm must be less than 1e-8\n")
    print("Gradient descent min coordinates: [x, y] = " + str(gradient_desc_min_pos))
    print("Analytical min coordinates: [x, y] = " + str(correct_min_pos))
    print("Gradient descent min value: " + str(gradient_desc_min_value))
    print("Analytical min value: " + str(correct_min_value))


def run_tests():
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
    num_gradient_test()
    print('\n')
    minimum_by_gradient_descent_test()
    print('\n')


def main():
    run_tests()


if __name__ == "__main__":
    main()