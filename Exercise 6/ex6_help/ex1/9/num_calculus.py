"""
Functions for numerical approximation
"""

import math
from typing import Callable

import numpy as np

def first_derivative(function: Callable, x_point: float, x_step: float) -> float:
    """
    Numerically estimate the first derivative of the given function.

    :param function: A function for which the derivative is approximated.
    :param x_point: X value where to evaluate derivative
    :param x_step: X step size to use for the evaluation. Smaller value will
    give an estimate with less error.
    :returns: An approximation of the derivative at x_point, as a float
    """

    # The formula for approximating with error O(h^2) is evaluated and returned.
    # The error grows exponentially with step size x_step.
    return (function(x_point + x_step) - function(x_point - x_step)) / (2 * x_step)

def second_derivative(function: Callable, x_point: float, x_step: float):
    """
    Numerically estimate the second derivative of the given function.

    :param function: A function for which the derivative is approximated.
    :param x_point: X value where to evaluate derivative
    :param x_step: X step size to use for the evaluation. Smaller value will
    give an estimate with less error.
    :returns: An approximation of the derivative at x_point, as a float
    """
    return (function(x_point + x_step) + function(x_point - x_step) -
            2*function(x_point)) / x_step**2

def riemann_sum(function: Callable, num_points: int, x_start: float, x_stop: float) -> float:
    """
    Implements Riemann sum formula for approximating an intergral numerically.

    The Riemann sum assumes that the integrated function is (approximately) a
    constant in each small interval. Hence accuracy should improve with
    increasing N (num_points).

    :param function: Function for which the integral is evaluated
    :param num_points: Number of approximation intervals. An integer is expect.
    If a float is given, will truncate to an integer
    :param x_start: X coordinate for beginning of integration interval.
    :param x_stop: X coordinate for ending of integration interval.
    """

    # Make sure an integer amount of approximation points is given
    num_points = int(num_points)

    # Then form an array of evaluation points in uniform grid.
    # This way, we will always use the point at the beginning (left) of an interval.
    points = np.linspace(x_start, x_stop, num_points)
    # We can also easily get delta x form the resulting array
    delta_x = points[1] - points[0]

    # The formula is simple enough to be computed and returned on one line.
    # Use Python's list comprehension to evaluate function at evaluation
    # points in the given range, then sum the points weighted by step size.
    return sum([function(x_i)*delta_x for x_i in points])

def trapezoid_rule(function: Callable, num_points: int, x_start: float, x_stop: float) -> float:
    """
    Implements Trapezoid rule formula for approximating an intergral numerically.

    Accuracy should improve with increasing N (num_points).

    :param function: Function for which the integral is evaluated
    :param num_points: Number of approximation intervals. An integer is expect.
    If a float is given, will truncate to an integer
    :param x_start: X coordinate for beginning of integration interval.
    :param x_stop: X coordinate for ending of integration interval.
    """

    # Make sure an integer amount of approximation points is given
    num_points = int(num_points)

    # Then form an array of evaluation points and find the step size (delta x)
    points = np.linspace(x_start, x_stop, num_points)
    delta_x = points[1] - points[0]

    # Use Python's list comprehension to iterate over evaluation points x_i.
    # At each point, effectively we add the mean of the function in the
    # interval, but division by 2 can be taken out of the summation. Intervals
    # are weighted with step size (delta x).
    return 0.5 * sum([(function(x_i) + function(x_i+delta_x))*delta_x for x_i in points])

def simpson_rule(function: Callable, num_points: int, x_start: float, x_stop: float) -> float:
    """
    Implements Simpson rule formula for approximating an intergral numerically.

    Accuracy should improve with increasing N (num_points).

    :param function: Function for which the integral is evaluated
    :param num_points: Number of approximation intervals. An integer is expect.
    If a float is given, will truncate to an integer
    :param x_start: X coordinate for beginning of integration interval.
    :param x_stop: X coordinate for ending of integration interval.
    """

    # Make sure an integer amount of approximation points is given
    num_points = int(num_points)

    # Then form an array of evaluation points and find the step size (delta x)
    points = np.linspace(x_start, x_stop, num_points)
    delta_x = points[1] - points[0]

    # We need to consider whether there is an even or odd number of intervals.
    # If we divide the number of points in two and truncate to integer, we get a
    # minimum range that we always need to compute anyway.
    half_points = int(num_points / 2)

    # To make the following formulation a bit prettier (shorter) we can evaluate function
    # at every point already.
    values = [function(x) for x in points]

    # Calculate an intermediate result with an even grid. We now iterate only
    # until N/2 - 1 points, and this is accounted for in the range() function.
    # Use the index of current iteration point to access correct coordinates
    # from the value array.
    even_grid_sum = (delta_x / 3) * sum(
        [values[2 * ind] + 4 * values[2 * ind + 1] + values[2 * ind + 2]
         for ind in range(half_points - 1)])
    # This can be checked with the modulus operator
    if num_points % 2 == 0: # Even

        return even_grid_sum

    # Add the last slice, where N is the index of the last point, we subtract 1
    # from the number of points due to zero-based indexing
    N = num_points - 1
    return even_grid_sum + (delta_x / 12) * (
        -1 * function(points[N - 2]) + 8 * function(points[N - 1]) + 5 *
        function(points[N]))

def test_first_derivative(relative_tolerance: float = 0.01) -> bool:
    """
    Test the first_derivative function.

    :param relative_tolerance: Tolerance for testing the accuracy of
    first_derivative function. A relative tolerance is used. Defaults to 0.01 (1%).
    :returns: Boolean True if first_derivative returns a correct approximation,
    else return False
    """

    # First define a function that is used for testing. Let's use something for
    # which the analytical solution of first derivative is available.
    # I'll use the lambda formalism because it is a neat and readable way of
    # defining simple functions, for example here we will use f(x) = x^2 + x.
    test_fun = lambda x: x**2 + x

    # Let's test first_derivative at x = 1 and use a very small step.
    # The first derivative of this function is 2x + 1.
    # At x = 1, the first derivative of our test_fun is 3.
    x_test = 1.
    x_step = 1e-2
    true_result = 3.

    # Get the estimate from our approximation:
    estimate = first_derivative(test_fun, x_test, x_step)

    # Since our approximation cannot be assumed to give a perfect result, we
    # cannot do a simple comparison of numerical values with '=='. Instead we
    # need to use a comparison function that accepts a tolerance. We will return
    # the output of the comparison directly
    return math.isclose(true_result, estimate, rel_tol=relative_tolerance)

def test_second_derivative(relative_tolerance: float = 0.01) -> bool:
    """
    Test the second_derivative function.

    :param relative_tolerance: Tolerance for testing the accuracy of
    second_derivative function. A relative tolerance is used. Defaults to 0.01 (1%).
    :returns: Boolean True if second_derivative returns a correct approximation,
    else return False
    """

    # Define a test function for which analytical solution is known.
    # I'll use f(x) = x^3 + x^2 + x. The first derivative is
    # 3x^2 + 2x + 1 and the second is 6x + 2. At test point x=1 the second
    # derivative should evaluate to 8.
    test_fun = lambda x: x**3 + x**2 + x
    x_test = 1.
    x_step = 1e-2
    true_result = 8.

    # Return the result of comparison with a tolerance
    return math.isclose(true_result,
                        second_derivative(test_fun, x_test, x_step),
                        rel_tol=relative_tolerance)

def test_riemann_sum(relative_tolerance: float = 0.01) -> bool:
    """
    Test the riemann_sum function.

    :param relative_tolerance: Tolerance for testing the accuracy of
    the function. A relative tolerance is used. Defaults to 0.01 (1%).
    :returns: Boolean True if function returns a correct approximation,
    else return False
    """

    # Define a test function, use something for which the analytical result is
    # known. For example integrating sin(x) from zero to pi should give 2.
    num_points = 1000
    x_start = 0
    x_stop = math.pi
    true_result = 2.

    # Return the result of comparison with a tolerance
    return math.isclose(true_result,
                        riemann_sum(math.sin, num_points, x_start, x_stop),
                        rel_tol=relative_tolerance)

def test_trapezoid_rule(relative_tolerance: float = 0.01) -> bool:
    """
    Test the trapezoid_rule function.

    :param relative_tolerance: Tolerance for testing the accuracy of
    the function. A relative tolerance is used. Defaults to 0.01 (1%).
    :returns: Boolean True if function returns a correct approximation,
    else return False
    """

    # Define a test function, use something for which the analytical result is
    # known. For example integrating sin(x) from zero to pi should give 2.
    num_points = 1000
    x_start = 0.
    x_stop = math.pi
    true_result = 2.

    # Return the result of comparison with a tolerance
    return math.isclose(true_result,
                        trapezoid_rule(math.sin, num_points, x_start, x_stop),
                        rel_tol=relative_tolerance)

def test_simpson_rule(relative_tolerance: float = 0.01) -> bool:
    """
    Test the simpson_rule function.

    :param relative_tolerance: Tolerance for testing the accuracy of
    the function. A relative tolerance is used. Defaults to 0.01 (1%).
    :returns: Boolean True if function returns a correct approximation,
    else return False
    """

    # Define a test function, use something for which the analytical result is
    # known. For example integrating sin(x) from zero to pi should give 2.
    num_points = 99
    x_start = 0
    x_stop = math.pi
    true_result = 2.

    # Return the result of comparison with a tolerance
    return math.isclose(true_result,
                        simpson_rule(math.sin, num_points, x_start, x_stop),
                        rel_tol=relative_tolerance)

def main():
    """
    doc
    """
    first_deriv_test_res = test_first_derivative(1e-4)
    print("Test result for function first_derivate() is {}".format(
        first_deriv_test_res))

    second_deriv_test_res = test_second_derivative(1e-4)
    print("Test result for function second_derivate() is {}".format(
        second_deriv_test_res))

    riemann_test_res = test_riemann_sum(1e-4)
    print("Test result for function riemann_sum() is {}".format(
        riemann_test_res))

    trapezoid_test_res = test_trapezoid_rule(1e-4)
    print("Test result for function trapezoid_rule() is {}".format(
        trapezoid_test_res))

    simpson_test_res = test_simpson_rule(1e-4)
    print("Test result for function simpson_rule() is {}".format(
        simpson_test_res))

if __name__ == "__main__":
    main()
