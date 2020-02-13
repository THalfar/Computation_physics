"""
FYS-4096 Computational Physics
Exercise 3
Problem 1

Numerical integration in 2D with Numpy's Simpson's rule implementation
"""


# imports
from ex3_lib import *
from scipy.integrate import simps
import matplotlib.pyplot as plt

def eval_simpson(n):
    """
    Evaluates the Simpson's rule integral with simple n x n grid
    :param n: Grid spacing (number of intervals in both directions, x and y)
    :return: Integral value with the given grid spacing
    """
    xs, ys = grid(0, 2, -2, 2, n, n)
    xint_val = np.zeros_like(xs)

    xint_val[:] = [simps(fun(x, ys), xs) for x in xs]
    int_val = simps(xint_val, ys)
    print("Evaluated Simpson integral with {:} intervals (x & y): {:f}".format(n, int_val))

    return int_val

def main():
    """
    Main function
    :return: -
    """

    # Some values for N:
    n_values = np.concatenate([np.linspace(2, 9, 8).astype(int),
                               np.linspace(10, 90, 9).astype(int),
                               np.linspace(100, 900, 9).astype(int),
                               np.linspace(1000, 10000, 10).astype(int)])

    # Evaluate the integral with N
    integral_values = [eval_simpson(n) for n in n_values]

    # Relative change to previous value
    int_change = np.abs(np.diff(integral_values)) / n_values[1:]

    # Plot convergence
    plt.figure()
    plt.loglog(n_values[1:], int_change*100, 'bx-')
    plt.xlabel('Number of intervals (x and y)')
    plt.ylabel('Integral relative change to previous value (%)')
    plt.title('Convergence plot')
    plt.show()

    return 0



if __name__=="__main__":
    main()
