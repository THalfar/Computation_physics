"""
FYS-4096 problem 4
Contains functions to find closest index i for x[i] ~ x0
Linear case has constant runtime in relation to grid size
"bisection" is a recursive binary search function suited for this case
fallback method is slow, O(n)? complexity (I think).
"""

import numpy as np

def linear_index(grid, x0):
    """
    Assuming linear grid, distance between grid points is x[i+1]-x[i],
    thus arbitrary x0 is closest to the point x[j] ~ x0/(x[i+1]-x[i])
    if this isn't the closest point, fallback method is used
    :param grid: np.linspace() grid
    :param x0: float or int inside the bounds of the grid
    :return: int index where x[i] <= x0 < x[i+1]
    """
    l = grid[2]-grid[1]
    index = int(x0//l)  # // floor division
    if np.abs(x0-grid[index]) < l:
        return index
    else:
        print("Something happened, calling fallback function")
        return fallback_index(grid, x0)


def bisect_index(grid, x0):
    """
    Actually a binary search algorithm, log n average performance according to wikipedia
    Compares grid center to center +1 and center -1, returns center or center -1 if x0
    is in between the points. Else calls itself with halved grid
    :param grid: arbitrary grid
    :param x0: float or int inside the bounds of the grid
    :return: int index where x[i] <= x0 < x[i+1]
    """
    center = np.size(grid)//2
    if x0 >= grid[center]:
        if x0 <= grid[center+1]:
            return center
        return bisect_index(grid[center:], x0)+center  # +center for correct offset
    else:
        if x0 >= grid[center-1]:
            return center-1
        return bisect_index(grid[:center], x0)

def fallback_index(grid, x0):
    """
    Iterates through grid to find index closest to x0, scales in n
    :param grid: arbitrary grid
    :param x0: float or int inside the bounds of the grid
    :return: int index where x[i] <= x0 < x[i+1]
    """
    i0 = 0
    for i1 in range(np.size(grid)):
        # if x[i-1] is closer to x0 than x[i] return i-1
        if np.abs(x0 - grid[int(i0)]) < np.abs(x0 - grid[int(i1)]):
            return i0
        else:
            i0 = i1

def grid_generator(r0, h, dim):
    # Problem 4 b) grid generator
    r = np.zeros(dim)
    r[0] = 0
    for i in range(1,dim):
        r[i] = r0*(np.exp(i*h)-1)
    return r

def test_linear():
    grid = np.linspace(0, 100, 201)
    x0 = [3, 3.25, np.pi, 50.1234121234, 80.854734]
    ans = [6, 6, 6, 100, 161]
    for x, a in zip(x0, ans):
        if a != linear_index(grid, x):
            print(a, linear_index(grid, x))
            print("Linear function test failed")
            return
    print("Test succesful for linear function")

def test_bisection():
    grid = np.linspace(0, 100, 201)
    x0 = [3, 3.25, np.pi, 50.1234121234, 80.854734]
    ans = [6, 6, 6, 100, 161]
    for x, a in zip(x0, ans):
        if a != bisect_index(grid, x):
            print(a, bisect_index(grid, x))
            print("Bisection function test failed")
            return
    dim = 100
    rmax = 100
    r0 = 1e-5
    h = np.log(rmax/r0+1)/(dim-1)
    grid = grid_generator(r0, h, dim)
    if bisect_index(grid, 0.15) != 59: # Checked correct value from debugger
        print("Bisection function test failed")
    print("Test succesful for bisection function")

def test_fallback():
    grid = np.linspace(0, 100, 200)
    x0 = [3, 3.25, np.pi, 50.1234121234, 80.854734]
    ans = [6, 6, 6, 100, 161]
    for x, a in zip(x0, ans):
        if a != fallback_index(grid, x):
            print(a, fallback_index(grid, x))
            print("Fallback test failed")
            return
    print("Test succesful for fallback function")


def main():
    test_linear()
    test_bisection()
    test_fallback()

if __name__ == "__main__":
    main()