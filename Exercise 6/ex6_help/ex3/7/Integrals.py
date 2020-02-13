"""
Calculating some integrals.
"""

import numpy as np
from scipy.integrate import simps

# Problem 1:

# Define the function to integrate.
def f(x, y): return (x + y) * np.exp(-1*np.sqrt(x**2 + y**2))


def int_2d(x, y):
    # Function to integrate f over a grid determined by linspaces x and y.

    ints = np.zeros(len(y))
    # Calculate the integral dx for all points in y, 'collapsing' the integral into one dimension.
    for i in range(len(y)):
        ints[i] = simps(f(x, y[i]), x)

    # Integrate and return the 'collapsed' values over y.
    return simps(ints, y)


def main():

    # Grid 1: 10 by 10
    x = np.linspace(0, 2, 10)
    y = np.linspace(-2, 2, 10)

    print("With a 10 by 10 grid:")
    print(int_2d(x, y))

    # Grid 2: 100 by 100
    x = np.linspace(0, 2, 100)
    y = np.linspace(-2, 2, 100)

    print("With a 100 by 100 grid:")
    print(int_2d(x, y))

    # Grid 1: 1000 by 1000
    x = np.linspace(0, 2, 1000)
    y = np.linspace(-2, 2, 1000)

    print("With a 1000 by 1000 grid:")
    print(int_2d(x, y))

    # Grid 4: 10000 by 10000
    x = np.linspace(0, 2, 10000)
    y = np.linspace(-2, 2, 10000)

    print("With a 10000 by 10000 grid:")
    print(int_2d(x, y))

main()