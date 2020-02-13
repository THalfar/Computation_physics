""" num_calculus.py is a lib of numerical calculus functions"""
import numpy as np
import math
from monte_carlo import monte_carlo_integration

class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

def first_derivative( function, x, dx ):
    # Central difference
    return (function(x+dx)-function(x-dx)) / (2*dx)

def second_derivative(function, x, dx):
    # Central difference again (basicly forward and backward difference combined to get 2nd order derivative)
    return (function(x+dx)-2*function(x)+function(x-dx)) / (dx**2)

def test_first_derivative():
    print("Testing 1st derivative")
    try:
        print(bcolors.OKGREEN+"Starting test 1: f(x) = x at x = 1.")
        assert np.round(first_derivative(test_function_1, 1, 0.0001), 3) == 1
        print("Passed test function 1: f(x) = x at x = 1")
        print(bcolors.ENDC)

        print(bcolors.OKGREEN+"Starting test 2: f(x) = x^2 at x = 1.")
        assert np.round(first_derivative(test_function_2, 1, 0.0001), 3) == 2
        print("Passed test function 2: f(x) = x^2 at x = 1")
        print(bcolors.ENDC)

        print(bcolors.OKGREEN+"Starting test 3: f(x) = 5 at x = 1.")
        assert np.round(first_derivative(test_function_3, 1, 0.0001), 3) == 0
        print("Passed test function 3: f(x) = 5 at x = 1")
        print(bcolors.ENDC)

        print(bcolors.OKGREEN+"Starting test 4: f(x) = 1/x at x = 1.")
        assert np.round(first_derivative(test_function_4, 1, 0.0001), 3) == -1
        print("Passed test function 4: f(x) = 1/x at x = 1")
        print(bcolors.ENDC)

        print(bcolors.OKGREEN+"Starting test 5: f(x) = 1/x at x = 2.")
        assert np.round(first_derivative(test_function_4, 2, 0.0001), 3) == -1/4
        print("Passed test function 4: f(x) = 1/x at x = 2")
        print(bcolors.ENDC)

        print(bcolors.OKGREEN+"Starting test 6: f(x) = ln(x) at x = 1.")
        assert np.round(first_derivative(test_function_5, 1, 0.0001), 3) == 1
        print("Passed test function 5: f(x) = ln(x)")
        print(bcolors.ENDC)
    except AssertionError as e:
        print("The test failed.")


def test_second_derivative():
    print("Testing 2nd derivative")
    try:
        print(bcolors.OKGREEN+"Starting test 7: f(x) = x at x = 1.")
        assert np.round(second_derivative(test_function_1, 1, 0.0001), 3) == 0
        print("Passed test function 7: f(x) = x at x = 1")
        print(bcolors.ENDC)

        print(bcolors.OKGREEN+"Starting test 8: f(x) = x^2 at x = 1.")
        assert np.round(second_derivative(test_function_2, 1, 0.0001), 3) == 2
        print("Passed test function 8: f(x) = x^2 at x = 1")
        print(bcolors.ENDC)

        print(bcolors.OKGREEN+"Starting test 9: f(x) = 5 at x = 1.")
        assert np.round(second_derivative(test_function_3, 1, 0.0001), 3) == 0
        print("Passed test function 9: f(x) = 5 at x = 1")
        print(bcolors.ENDC)

        print(bcolors.OKGREEN+"Starting test 10: f(x) = 1/x at x = 1.")
        assert np.round(second_derivative(test_function_4, 1, 0.0001), 3) == 2
        print("Passed test function 10: f(x) = 1/x at x = 1")
        print(bcolors.ENDC)

        print(bcolors.OKGREEN+"Starting test 11: f(x) = 1/x at x = 2.")
        assert np.round(second_derivative(test_function_4, 2, 0.0001), 3) == 1/4
        print("Passed test function 11: f(x) = 1/x at x = 2")
        print(bcolors.ENDC)

        print(bcolors.OKGREEN+"Starting test 12: f(x) = ln(x) at x = 1.")
        assert np.round(second_derivative(test_function_5, 1, 0.0001), 3) == -1
        print("Passed test function 12: f(x) = ln(x)")
        print(bcolors.ENDC)
    except AssertionError as e:
        print(bcolors.FAIL+bcolors.BOLD+"The test failed."+bcolors.ENDC)

def test_function_1(x):
    return x

def test_function_2(x):
    return x**2

def test_function_3(x):
    return 5

def test_function_4(x):
    return 1/x

def test_function_5(x):
    return math.log(x)

def test_function_6(x):
    return np.sin(x)

def test_function_7(x):
    return np.cos(x)

def test_function_8(x):
    return 2*x

def riemann_sum(x, f):
    """
    Riemann sum function (midpoint)
    :param x: uniform grid to sum over
    :param f: functions values to be summed
    :return: Riemann's sum
    """
    mpsum = 0
    dx = np.diff(x)
    for i in range(len(x)-1):
        mpsum += dx[i]*f(x[i]+dx[i]/2)
    return mpsum


def trapezoid(x, f):
    """
    Trapezoidal integration
    :param x: uniform grid to sum over
    :param f: functions values to be summed
    :return: Trapezoidal rule sum
    """

    mpsum = 0
    dx = np.diff(x)
    for i in range(len(x)-1):
        mpsum += dx[i]/2*(f(x[i])+f(x[i+1]))
    return mpsum

def simpson(x, f):
    """
    Simpson's rule integration
    :param x: uniform grid to sum over (n = even)
    :param f: functions values to be summed
    :return: Simpson's rule sum
    """

    mpsum = 0
    dx = np.diff(x)
    for i in range(len(x)-1):
        mpsum += dx[i]/6 * (f(x[i]) + 4*f((x[i]+x[i+1])/2) + f(x[i+1]))

    return mpsum

def test_riemann_sum():
    print("Testing Riemann sum")

    # All left sums

    try:
        x = np.linspace(0, np.pi / 2, 100)

        print(bcolors.OKGREEN+"Starting test 13: f(x) = sin(x) over 0...1/2pi")
        assert np.round(riemann_sum(x, test_function_6), 3) == 1
        print("Passed test function 13: f(x) = sin(x)")
        print(bcolors.ENDC)


        x = np.linspace(0, np.pi*6, 100)
        f = np.sin(x)

        print(bcolors.OKGREEN+"Starting test 14: f(x) = cos(x) over 0...6pi")
        assert np.round(riemann_sum(x, test_function_7), 3) == 0
        print("Passed test function 14: f(x) = cos(x)")
        print(bcolors.ENDC)

        x = np.linspace(0, 5, 100)
        f = 2*x

        print(bcolors.OKGREEN+"Starting test 15: f(x) = 2x over 0...5")
        assert np.round(riemann_sum(x, test_function_8), 3) == 25
        print("Passed test function 15: f(x) = 2x")
        print(bcolors.ENDC)
    except AssertionError:
        print(bcolors.FAIL+bcolors.BOLD+"The test failed."+bcolors.ENDC)

def test_trapezoid_rule():
    print("Testing Trapezoidal rule")

    # All left sums

    try:
        x = np.linspace(0, np.pi / 2, 100)

        print(bcolors.OKGREEN+"Starting test 16: f(x) = sin(x) over 0...1/2pi")
        assert np.round(trapezoid(x, test_function_6), 3) == 1
        print("Passed test function 16: f(x) = sin(x)")
        print(bcolors.ENDC)


        x = np.linspace(0, np.pi*6, 100)

        print(bcolors.OKGREEN+"Starting test 17: f(x) = cos(x) over 0...6pi")
        assert np.round(trapezoid(x, test_function_7), 3) == 0
        print("Passed test function 17: f(x) = cos(x)")
        print(bcolors.ENDC)

        x = np.linspace(0, 5, 100)

        print(bcolors.OKGREEN+"Starting test 18: f(x) = 2x over 0...5")
        assert np.round(trapezoid(x, test_function_8), 3) == 25
        print("Passed test function 18: f(x) = 2x")
        print(bcolors.ENDC)
    except AssertionError:
        print(bcolors.FAIL+bcolors.BOLD+"The test failed."+bcolors.ENDC)

def test_simpson():
    print("Testing Simpson's rule")

    # All left sums

    try:
        x = np.linspace(0, np.pi / 2, 100)

        print(bcolors.OKGREEN+"Starting test 19: f(x) = sin(x) over 0...1/2pi")
        assert np.round(simpson(x, test_function_6), 3) == 1
        print("Passed test function 19: f(x) = sin(x)")
        print(bcolors.ENDC)


        x = np.linspace(0, np.pi*6, 100)

        print(bcolors.OKGREEN+"Starting test 20: f(x) = cos(x) over 0...6pi")
        assert np.round(simpson(x, test_function_7), 3) == 0
        print("Passed test function 20: f(x) = cos(x)")
        print(bcolors.ENDC)

        x = np.linspace(0, 5, 100)

        print(bcolors.OKGREEN+"Starting test 21: f(x) = 2x over 0...5")
        assert np.round(simpson(x, test_function_8), 3) == 25
        print("Passed test function 21: f(x) = 2x")
        print(bcolors.ENDC)
    except AssertionError:
        print(bcolors.FAIL+bcolors.BOLD+"The test failed."+bcolors.ENDC)

def test_mc():
    print("Testing Monte Carlo integration")

    # All left sums

    try:
        print(bcolors.OKGREEN+"Starting test 22: f(x) = sin(x) over 0...1/2pi")
        assert np.abs(np.round(monte_carlo_integration(test_function_6, 0, np.pi/2, 100, 100)[0], 2) - 1) <= 0.2
        print("Passed test function 22: f(x) = sin(x)")
        print(bcolors.ENDC)

        print(bcolors.OKGREEN+"Starting test 23: f(x) = cos(x) over 0...6pi")
        assert np.abs(np.round(monte_carlo_integration(test_function_7, 0, 6*np.pi, 100, 100)[0], 2)) <= 0.5
        print("Passed test function 23: f(x) = cos(x)")
        print(bcolors.ENDC)

        print(bcolors.OKGREEN+"Starting test 24: f(x) = 2x over 0...5")
        assert np.abs(np.round(monte_carlo_integration(test_function_8, 0, 5, 100, 100)[0], 2)-25) <= 0.5
        print("Passed test function 24: f(x) = 2x")
        print(bcolors.ENDC)
    except AssertionError:
        print("Test failed.")


def main():
    # Runs very simple (and not so well defined) tests. Could be much more accurate... e.g. no rounding.
    test_first_derivative()
    test_second_derivative()
    test_riemann_sum()
    test_trapezoid_rule()
    test_simpson()
    test_mc()


if __name__=="__main__":
    main()


