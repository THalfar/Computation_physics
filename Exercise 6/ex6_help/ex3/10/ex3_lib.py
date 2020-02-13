"""
FYS-4096 Computational Physics
Exercise 3

This file contains some helper functions that are needed in multiple problems during this exercise.
Avoiding copy-paste by importing this to problem solution files.
"""

import numpy as np

def fun(x, y):
    """
    Function to be integrated
    :param x: x to evaluate
    :param y: y to evaluate
    :return: function value
    """
    return (x+y)*np.exp(-np.sqrt(x**2+y**2))

def grid(xmin, xmax, ymin, ymax, nx, ny):
    """
    Quick helper function to generate two linear spaced vectors for grid
    :param xmin: minimum x value
    :param xmax: maximum x value
    :param ymin: minimum y value
    :param ymax: maximum y value
    :param nx: number of x intervals
    :param ny: number of y intervals
    :return:
    """
    x = np.linspace(xmin, xmax, nx)
    y = np.linspace(ymin, ymax, ny)
    return x, y