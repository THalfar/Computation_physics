"""
A function that finds indices in an array with values just below or at a requested value.
"""

import numpy as np

def index_search(array, value):
    # A function that returns the index of a given value or if the value is not
    # in the array, the index just below it. If the value is outside the array,
    # returns None.

    # Return none if the value is outside the array.
    if value < array[0] or value > array[-1]:
        return None

    for i in range(len(array)):
        if array[i] >= value:
            return i

def test_index_search():

    array = np.linspace(0,100,1000)
    if index_search(array, 50) == 500:
        return True
    else:
        return False

def many_dim_index_search(r_0, h, dim):
    # I have no idea what my goal is.
    pass

def test_many_dim_index_search():
    r_0 = 1e-5
    dim = 100
    r_max = 100
    h = np.log(r_max/(r_0 + 1))/(dim - 1)
    r = np.zeros(dim)
    r[0] = 0
    for i in range(1, dim):
        r[i] = r_0*(np.exp(i*h) - 1)

def main():
    if test_index_search():
        print("index_search() is working properly.")
    else:
        print("index_search() is not working properly.")

    test_many_dim_index_search()

main()