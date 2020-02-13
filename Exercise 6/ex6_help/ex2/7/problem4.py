# Arttu Hietalahti 262981

import numpy as np


def get_interpolation_indices_linear(x, point):
    """
        calculates the indexes needed in interpolation
        eg. the previous and next indices around off-grid point pos
    :param x: grid (must be linear!)
    :param point: off-grid point
    :return: left idx and right idx between which point lies
    """
    step = x[1] - x[0]
    idx_left = point // step  # linear grid allows using this to find left idx
    return int(idx_left), int(idx_left + 1)


def get_interpolation_indices_nonlinear(x, point):
    """
        function finds left and right side interpolation indices
        using the bisection method
    :param x: grid (nonlinear)
    :param point: off-grid point
    :return: left and right side interpolation indices for point
    """

    if not x[0] < point < x[-1]:
        print("Error: cannot interpolate outside grid range!")
        return np.empty(0)  # return empty array if point is out of grid range

    left_idx = 0    # start bisection indexing from start and end of grid
    right_idx = len(x) - 1
    while True:

        # this condition means that we have found indices between which point lies
        if right_idx == left_idx + 1:
            return int(left_idx), int(right_idx)

        halfway_idx = left_idx + (right_idx - left_idx)//2  # halfway between left_idx and right_idx

        # halfway index is the new left or right index depending on corresponding x value
        if x[halfway_idx] > point:
            right_idx = halfway_idx
        else:
            left_idx = halfway_idx


def create_nonlinear_grid(r_0, rmax, dim):
    """
        Function generates a nonlinear grid
        Based on exercise 2 problem 4B formulas

    :param r_0: used in calculating h
    :param rmax: used in calculating h
    :param dim: used in calculating h and is also the length of the created grid.
    :return: a nonlinear grid as a numpy array
    """

    h = np.log10(rmax / r_0 + 1) / (dim - 1)
    r = np.empty([dim, 1])
    r[0] = 0.
    for i in range(1, dim):
        r[i] = r_0 * (np.exp(i * h) - 1)
    return np.array(r)


def test_linear_index_search():
    print("Testing linear interpolation index search")
    x = np.arange(0, 100, 0.5)  # [0, 0.5, 1, 1.5, ..., 100]
    testpoints = [0.2, 25.99999, 50.0000001, 80.42, 99.21, 10.3]
    for testpoint in testpoints:
        print("_________________________________")
        print("Testpoint " + str(testpoint))
        print("Grid used: [0, 0.5, 1, 1.5, ..., 100]")
        interp_indices = get_interpolation_indices_linear(x, testpoint)
        print("Interpolation indices found: " + str(interp_indices))
        print("Grid values at indices: " + str(x[interp_indices[0]]) + "  " + str(x[interp_indices[1]]))

        # test passed if testpoint value lies between the two adjacent values.
        if x[interp_indices[0]] < testpoint < x[interp_indices[1]]:
            print("TEST PASSED!")
        else:
            print("TEST FAILED!")
    print("\n")


def test_nonlinear_index_search():

    print("Testing non-linear interpolation index search")
    x = create_nonlinear_grid(r_0=1e-5, rmax=100, dim=100000)  # grid to test
    testpoints = [1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2]  # testpoints to search interpolation indices for
    for testpoint in testpoints:
        print("_________________________________")
        print("Testpoint " + str(testpoint))
        interp_indices = get_interpolation_indices_nonlinear(x, testpoint)
        print("Interpolation indices found: " + str(interp_indices))
        print("Grid values at indices: " + str(x[interp_indices[0]]) + "  " + str(x[interp_indices[1]]))

        # test passed if testpoint value lies between the two adjacent grid values.
        if x[interp_indices[0]] < testpoint < x[interp_indices[1]]:
            print("TEST PASSED!")
        else:
            print("TEST FAILED!")
    print("\n")


def main():
    test_linear_index_search()
    test_nonlinear_index_search()
    

if __name__ == "__main__":
    main()
