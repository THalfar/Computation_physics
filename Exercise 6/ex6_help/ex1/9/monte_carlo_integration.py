"""
Implement Monte Carlo numerical integration method. Includes test and example
usage.
"""

import math
from typing import Callable

# Import numpy for efficient vector calculations
import numpy as np

def monte_carlo_integration(fun: Callable, xmin: float, xmax: float, blocks:
                            int, iters: int) -> tuple:
    """
    Perform monte carlo integration for the given function in the desired range.
    Returns a reliability estimate in addition to the intergral value estimate.

    :param fun: Python callable function
    :param xmin: Start of integration range
    :param xmax: End of integration range
    :param blocks: Number of times to perform the integral estimation. Many
    blocks are needed, so that the central limit theorem can be used for
    ensemble average (expectation value) calculation.
    :param iters: Number of iterations per block to perform sampling in the
    range (xmin, xmax). Higher number should give greater coverage of the range.
    :returns: Tuple of (I, dI) where I is the mean of obtained
    integration results and dI is the standard deviation of the results for
    reliability estimation.
    """

    # Initialize an array for saving results for integral evaluation
    # Each entry will contain the result obtained after 'iters' samples
    block_values = np.zeros((blocks,))

    # L is used to scale the results correctly
    L = xmax - xmin

    for block in range(blocks): # Repeat the integration 'blocks' times
        for i in range(iters): # Run 'iters' iterations per block

            # Get a random sample point in the user defined range.
            # np.random.rand() returns values in range (0, 1) so scale it with L
            x = xmin+np.random.rand()*L

            # Add the obtained value to the set of values in this block
            block_values[block]+=fun(x)

        # Divide the estimates in the current block with number of iterations to
        # get their average
        block_values[block]/=iters

    # Get the ensemble average as the final estimate of the integral
    I = L*np.mean(block_values)

    # Calculate the standard deviation of integral evaluations to find a statistical error
    # estimate SEM
    dI = L*np.std(block_values)/np.sqrt(blocks)

    # Return the integral value and error estimate
    return I, dI

def func(x):
    """
    Wrapper for evaluating a function for testing Monte Carlo integration.
    """
    return np.sin(x)

def test_monte_carlo_integration(relative_tolerance: float = 0.01,
                                 SEM_threshold: float = 0.005) -> bool:
    """
    Test for Monte Carlo (MC) integration. Returns boolean True if test is passed,
    else False. The relative tolerance can be given as a parameter, it defaults
    to 0.1 which accepts 1% difference in approximation result compared to
    analytical result. A threshold for SEM is also given as input, and for the
    test to pass SEM needs to be lower than the threshold.

    :param relative_tolerance: Float, > 0, for comparing the MC approximation to
    known analytical result.
    :returns: True if passed, else False
    """

    # Our test function is a sin function.
    x_start = 0.
    x_stop = math.pi
    blocks = 50
    iterations = 1000
    true_result = 2.

    # Return the result of comparison with a tolerance
    I, SEM = monte_carlo_integration(func, x_start, x_stop, blocks, iterations)
    print('Mean estimate is {m} and SEM is {s}'.format(m=I, s=SEM))
    return math.isclose(true_result, I, rel_tol=relative_tolerance) and (
        SEM < SEM_threshold)

def main():
    """
    Run Monte Carlo integration for our previously defined function and test the
    result againts a known value.
    """
    mc_test_result = test_monte_carlo_integration()
    print("Test result for Monte Carlo integration is {}".format(
        mc_test_result))

if __name__ == "__main__":
    main()
